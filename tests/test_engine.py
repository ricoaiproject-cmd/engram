"""engine.py の単体テスト。

FakeEmbedder + モック db/store + now 注入で時間を制御する。
db / store は NotImplementedError スタブのため、unittest.mock で差し替える。
統合フェーズで db/store が完成したら mock を外してエンドツーエンドテストに昇格できる。
"""

from __future__ import annotations

import time
import math
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest

from engram.config import Settings
from engram.embedder import FakeEmbedder
from engram.engine import MemoryEngine, _hit_to_dict
from engram.models import MemoryRecord, RecallHit

# ---------------------------------------------------------------------------
# 共通フィクスチャ
# ---------------------------------------------------------------------------

DAY = 86400.0
NOW = 1_750_000_000.0  # 固定の「現在時刻」


def _settings(tmp_path: Path) -> Settings:
    """テスト用 Settings(一時ディレクトリを使用)。"""
    return Settings(
        memories_dir=tmp_path / "memories",
        data_dir=tmp_path / "data",
        dup_threshold=0.92,
        deep_score_threshold=0.35,
        candidate_k=20,
        correction_min_importance=7,
        consolidate_min_age_days=14,
        consolidate_cluster_sim=0.75,
        colink_increment=0.1,
        colink_max=1.0,
        reinforce_weight=1.0,
        reinforce_strength_max=3.0,
        recall_hit_weight=0.3,
        create_alpha=2.0,
        w_relevance=0.6,
        w_activation=0.25,
        w_importance=0.15,
    )


def _fake_record(
    id: str = "01ABC",
    type: str = "knowledge",
    importance: int = 5,
    tier: str = "hot",
    content: str = "テスト記憶",
    tags: list[str] | None = None,
    path: str = "/tmp/test.md",
    content_hash: str | None = None,
) -> MemoryRecord:
    """テスト用ダミー MemoryRecord。content_hash 未指定なら content から自動生成。"""
    import hashlib
    h = content_hash if content_hash is not None else hashlib.sha256(content.strip().encode()).hexdigest()
    return MemoryRecord(
        id=id,
        type=type,
        created="2026-06-11T09:00:00+09:00",
        importance=importance,
        tags=tags or [],
        source="test",
        tier=tier,
        links=[],
        content=content,
        path=Path(path),
        content_hash=h,
    )


def _fake_db_mem(
    id: str = "01ABC",
    type: str = "knowledge",
    importance: int = 5,
    tier: str = "hot",
    path: str = "/tmp/test.md",
    content_hash: str = "",
    created_at: float = NOW - DAY,
) -> dict:
    """DB の get_memory / all_memories が返す dict 形式。"""
    return {
        "id": id,
        "type": type,
        "importance": importance,
        "tier": tier,
        "path": path,
        "content_hash": content_hash,
        "created_at": created_at,
    }


def _build_engine(tmp_path: Path, *, embedder=None):
    """モック db / store でエンジンを構築するヘルパー。"""
    settings = _settings(tmp_path)
    embedder = embedder or FakeEmbedder(dim=64)
    db = MagicMock()
    store = MagicMock()

    # デフォルトの振る舞い(多くのテストで必要)
    db.vector_search.return_value = []
    db.keyword_search.return_value = []
    db.get_events.return_value = {}
    db.all_memories.return_value = []
    db.get_links.return_value = []
    db.get_embeddings.return_value = {}
    db.get_memory.return_value = None

    return MemoryEngine(settings=settings, store=store, db=db, embedder=embedder), db, store


# ---------------------------------------------------------------------------
# remember / recall 往復テスト
# ---------------------------------------------------------------------------

class TestRememberRecall:
    def test_remember_then_recall(self, tmp_path):
        """remember で保存した記憶が recall で取得できること。"""
        engine, db, store = _build_engine(tmp_path)

        # store.create のモック
        record = _fake_record(id="MEM001", content="Python の非同期処理について")
        store.create.return_value = record
        db.vector_search.return_value = []  # 重複なし

        result = engine.remember(
            "Python の非同期処理について",
            type="knowledge",
            importance=7,
            now=NOW,
        )
        assert result["status"] == "created"
        assert result["id"] == "MEM001"

        # db.upsert_memory が呼ばれたことを確認
        db.upsert_memory.assert_called_once()

        # create イベントが記録されたことを確認
        db.add_event.assert_called_once()
        args = db.add_event.call_args
        assert args[0][0] == "MEM001"
        assert args[0][1] == "create"

        # recall のセットアップ: このノードを返す
        db.vector_search.return_value = [("MEM001", 0.95)]
        db.keyword_search.return_value = [("MEM001", -1.0)]
        db.all_memories.return_value = [_fake_db_mem(
            id="MEM001", importance=7, created_at=NOW - DAY
        )]
        db.get_events.return_value = {
            "MEM001": [(NOW - 100, 2.4)]  # create イベント相当
        }
        store.read.return_value = record

        recall_result = engine.recall("Python 非同期", now=NOW)
        assert recall_result["mode"] == "fast"
        hits = recall_result["hits"]
        assert len(hits) > 0
        assert hits[0]["id"] == "MEM001"

    def test_remember_stores_create_event_weight(self, tmp_path):
        """importance に応じた初期符号化ブーストで create イベントが記録されること。"""
        engine, db, store = _build_engine(tmp_path)

        record = _fake_record(id="MEM002", importance=10)
        store.create.return_value = record
        db.vector_search.return_value = []

        engine.remember("重要な記憶", type="knowledge", importance=10, now=NOW)

        # create_event_weight(10, alpha=2.0) = 1 + 2.0 * (10/10) = 3.0
        event_call = db.add_event.call_args
        assert pytest.approx(event_call[0][2], abs=1e-6) == 3.0


# ---------------------------------------------------------------------------
# 重複検知テスト
# ---------------------------------------------------------------------------

class TestDuplicate:
    def test_duplicate_reinforced(self, tmp_path):
        """cos >= dup_threshold の場合 duplicate_reinforced を返すこと。"""
        engine, db, store = _build_engine(tmp_path)

        # 既存記憶がほぼ同じベクトルで返る
        db.vector_search.return_value = [("EXIST001", 0.95)]  # >= 0.92

        result = engine.remember(
            "Pythonの非同期処理",
            type="knowledge",
            importance=5,
            now=NOW,
        )

        assert result["status"] == "duplicate_reinforced"
        assert result["id"] == "EXIST001"
        # store.create は呼ばれない
        store.create.assert_not_called()
        # reinforce イベントが記録される
        db.add_event.assert_called_once()
        assert db.add_event.call_args[0][1] == "reinforce"

    def test_no_duplicate_below_threshold(self, tmp_path):
        """cos < dup_threshold の場合は新規作成されること。"""
        engine, db, store = _build_engine(tmp_path)

        db.vector_search.return_value = [("EXIST001", 0.85)]  # < 0.92
        record = _fake_record(id="NEW001")
        store.create.return_value = record

        result = engine.remember("全然違う内容", type="knowledge", importance=5, now=NOW)
        assert result["status"] == "created"
        store.create.assert_called_once()


# ---------------------------------------------------------------------------
# reinforce 後に順位が上がるテスト
# ---------------------------------------------------------------------------

class TestReinforce:
    def test_reinforce_raises_score(self, tmp_path):
        """reinforce 後に recall スコアが上昇すること。"""
        engine, db, store = _build_engine(tmp_path)

        mem = _fake_db_mem(id="MEM_HOT", importance=5, created_at=NOW - 7 * DAY)
        db.all_memories.return_value = [mem]
        db.vector_search.return_value = [("MEM_HOT", 0.8)]
        db.keyword_search.return_value = []
        record = _fake_record(id="MEM_HOT", content="強化される記憶")
        store.read.return_value = record

        # reinforce 前のスコア
        db.get_events.return_value = {"MEM_HOT": [(NOW - 7 * DAY, 1.5)]}
        result_before = engine.recall("強化", now=NOW, record_hits=False)
        score_before = result_before["hits"][0]["score"] if result_before["hits"] else 0.0

        # reinforce イベントを追加してスコアを再計算
        db.get_memory.return_value = mem
        engine.reinforce(["MEM_HOT"], strength=2.0, now=NOW)

        # reinforce 後のイベントでスコアを確認
        db.get_events.return_value = {
            "MEM_HOT": [
                (NOW - 7 * DAY, 1.5),
                (NOW, 2.0),  # reinforce weight * strength
            ]
        }
        result_after = engine.recall("強化", now=NOW + 1, record_hits=False)
        score_after = result_after["hits"][0]["score"] if result_after["hits"] else 0.0

        assert score_after >= score_before

    def test_reinforce_unknown_ids(self, tmp_path):
        """存在しない id は unknown_ids として列挙されること。"""
        engine, db, store = _build_engine(tmp_path)
        db.get_memory.return_value = None

        result = engine.reinforce(["GHOST001", "GHOST002"])
        assert set(result["unknown_ids"]) == {"GHOST001", "GHOST002"}
        assert result["reinforced"] == []

    def test_reinforce_creates_colink(self, tmp_path):
        """複数 id を同時に reinforce すると co_recall リンクが作られること。"""
        engine, db, store = _build_engine(tmp_path)
        mem_a = _fake_db_mem(id="A")
        mem_b = _fake_db_mem(id="B")
        db.get_memory.side_effect = lambda id_: {"A": mem_a, "B": mem_b}.get(id_)

        engine.reinforce(["A", "B"], now=NOW)

        # co_recall リンクが両方向に作られる
        link_calls = db.add_link.call_args_list
        kinds = [c[0][2] for c in link_calls]
        assert kinds.count("co_recall") == 2


# ---------------------------------------------------------------------------
# correct フローテスト
# ---------------------------------------------------------------------------

class TestCorrect:
    def _setup_correct(self, tmp_path):
        engine, db, store = _build_engine(tmp_path)
        old_record = _fake_record(
            id="OLD001", content="誤った情報", importance=5, tier="hot"
        )
        db.get_memory.return_value = _fake_db_mem(
            id="OLD001", importance=5, tier="hot", path="/tmp/old.md"
        )
        store.read.return_value = old_record

        new_record = _fake_record(id="NEW001", content="正しい情報", importance=7)
        store.create.return_value = new_record
        db.vector_search.return_value = []  # 重複なし

        return engine, db, store, old_record, new_record

    def test_correct_creates_new_and_supersedes_old(self, tmp_path):
        """correct で旧記憶が superseded になり新記憶が作られること。"""
        engine, db, store, old_record, new_record = self._setup_correct(tmp_path)

        result = engine.correct(
            "OLD001",
            corrected_content="正しい情報",
            reason="検証で誤りが判明",
            now=NOW,
        )

        assert result["status"] == "corrected"
        assert result["old_id"] == "OLD001"
        new_id = result["new_id"]

        # 旧記憶が superseded になる
        db.set_tier.assert_called_with("OLD001", "superseded")
        store.set_tier.assert_called_with(old_record, "superseded")

        # superseded_by リンクが張られる
        link_calls = db.add_link.call_args_list
        superseded_calls = [c for c in link_calls if c[0][2] == "superseded_by"]
        assert len(superseded_calls) >= 1
        assert superseded_calls[0][0][0] == "OLD001"

    def test_correct_new_memory_has_correction_tag(self, tmp_path):
        """訂正後の新記憶に correction タグが付くこと。"""
        engine, db, store, _, _ = self._setup_correct(tmp_path)
        engine.correct("OLD001", "正しい情報", "理由", now=NOW)

        # store.create の引数に tags: ["correction", ...] が含まれる
        create_kwargs = store.create.call_args[1]
        assert "correction" in create_kwargs.get("tags", [])

    def test_correct_new_importance_raised(self, tmp_path):
        """訂正後の記憶は correction_min_importance 以上になること。"""
        engine, db, store, old_record, _ = self._setup_correct(tmp_path)
        # importance=3 の低い記憶を訂正
        old_record.importance = 3
        db.get_memory.return_value = _fake_db_mem(id="OLD001", importance=3)

        engine.correct("OLD001", "正しい情報", "理由", now=NOW)

        create_kwargs = store.create.call_args[1]
        assert create_kwargs["importance"] >= 7  # correction_min_importance

    def test_correct_not_found(self, tmp_path):
        """存在しない id に correct すると not_found が返ること。"""
        engine, db, store, _, _ = self._setup_correct(tmp_path)
        db.get_memory.return_value = None

        result = engine.correct("GHOST", "正しい内容", "理由")
        assert result["status"] == "not_found"

    def test_correct_old_not_in_fast_recall(self, tmp_path):
        """訂正後、旧記憶(superseded)は fast recall(tier=hot のみ)に出ないこと。"""
        engine, db, store, _, _ = self._setup_correct(tmp_path)
        # fast recall は tier=hot のみ対象
        # superseded の記憶は all_memories(tiers=["hot"]) に含まれない
        db.all_memories.return_value = []
        db.vector_search.return_value = []
        db.keyword_search.return_value = []

        result = engine.recall("誤った情報", mode="fast", now=NOW)
        hit_ids = [h["id"] for h in result["hits"]]
        assert "OLD001" not in hit_ids

    def test_correct_deep_note_on_superseded(self, tmp_path):
        """deep recall で superseded 記憶に note が付くこと。"""
        engine, db, store, old_record, _ = self._setup_correct(tmp_path)

        # deep recall セットアップ: superseded な OLD001 を返す
        old_mem = _fake_db_mem(id="OLD001", tier="superseded", importance=5)
        db.all_memories.return_value = [old_mem]
        db.vector_search.return_value = [("OLD001", 0.8)]
        db.keyword_search.return_value = []
        db.get_links.return_value = [("OLD001", "NEW001", "superseded_by", 1.0)]
        db.get_embeddings.return_value = {}
        old_record.tier = "superseded"
        store.read.return_value = old_record

        result = engine.recall("誤った情報", mode="deep", now=NOW)
        old_hits = [h for h in result["hits"] if h["id"] == "OLD001"]
        if old_hits:
            assert "NEW001" in old_hits[0]["note"]


# ---------------------------------------------------------------------------
# co_recall リンク形成テスト
# ---------------------------------------------------------------------------

class TestCoRecallLink:
    def test_co_recall_links_formed_on_reinforce(self, tmp_path):
        """3件同時 reinforce で3ペアの co_recall リンクが形成されること。"""
        engine, db, store = _build_engine(tmp_path)

        ids = ["A", "B", "C"]
        for id_ in ids:
            db.get_memory.side_effect = lambda i: _fake_db_mem(id=i)

        # 実際は side_effect を dict で
        mem_map = {i: _fake_db_mem(id=i) for i in ids}
        db.get_memory.side_effect = lambda i: mem_map.get(i)

        engine.reinforce(ids, now=NOW)

        # A-B, A-C, B-C の3ペア × 双方向 = 6 co_recall リンク
        link_calls = db.add_link.call_args_list
        co_recall_calls = [(c[0][0], c[0][1]) for c in link_calls if c[0][2] == "co_recall"]
        assert len(co_recall_calls) == 6


# ---------------------------------------------------------------------------
# forget テスト
# ---------------------------------------------------------------------------

class TestForget:
    def test_forget_sets_trash_tier(self, tmp_path):
        """forget でゴミ箱移動 + DB tier=trash になること。"""
        engine, db, store = _build_engine(tmp_path)

        record = _fake_record(id="DEL001")
        db.get_memory.return_value = _fake_db_mem(id="DEL001")
        store.read.return_value = record

        result = engine.forget("DEL001")

        assert result["status"] == "forgotten"
        store.move_to_trash.assert_called_once_with(record)
        db.set_tier.assert_called_once_with("DEL001", "trash")

    def test_forget_not_found(self, tmp_path):
        """存在しない id に forget すると not_found が返ること。"""
        engine, db, store = _build_engine(tmp_path)
        db.get_memory.return_value = None

        result = engine.forget("GHOST")
        assert result["status"] == "not_found"


# ---------------------------------------------------------------------------
# consolidation_candidates のクラスタリングテスト
# ---------------------------------------------------------------------------

class TestConsolidationCandidates:
    def test_clusters_similar_episodes(self, tmp_path):
        """類似エンベディングの episode がクラスタされること。"""
        engine, db, store = _build_engine(tmp_path)
        embedder = engine.embedder

        # 同じテキストの episode x2(埋め込みが同一 → cos=1.0)
        text = "今日の進捗: PR レビュー"
        ep_ids = ["EP001", "EP002"]
        ep_mems = [
            _fake_db_mem(
                id=eid,
                type="episode",
                tier="hot",
                created_at=NOW - 20 * DAY,  # min_age_days=14 より古い
            )
            for eid in ep_ids
        ]
        db.all_memories.return_value = ep_mems

        # embeddings: 同一ベクトル(cos=1.0 >= 0.75)
        vec = embedder.embed_docs([text])[0]
        db.get_embeddings.return_value = {eid: vec for eid in ep_ids}

        for eid in ep_ids:
            store.read.return_value = _fake_record(id=eid, content=text)

        result = engine.consolidation_candidates(now=NOW)
        clusters = result["clusters"]
        assert len(clusters) >= 1
        assert len(clusters[0]["ids"]) == 2

    def test_no_cluster_for_young_episodes(self, tmp_path):
        """min_age_days より新しい episode はクラスタ候補に出ないこと。"""
        engine, db, store = _build_engine(tmp_path)
        embedder = engine.embedder

        text = "今日の進捗"
        ep_ids = ["EP_NEW1", "EP_NEW2"]
        # created_at を NOW - 3 days(< 14 days)に設定
        ep_mems = [
            _fake_db_mem(
                id=eid,
                type="episode",
                tier="hot",
                created_at=NOW - 3 * DAY,
            )
            for eid in ep_ids
        ]
        db.all_memories.return_value = ep_mems

        vec = embedder.embed_docs([text])[0]
        db.get_embeddings.return_value = {eid: vec for eid in ep_ids}

        result = engine.consolidation_candidates(now=NOW)
        assert result["clusters"] == []


# ---------------------------------------------------------------------------
# mark_consolidated テスト
# ---------------------------------------------------------------------------

class TestMarkConsolidated:
    def test_mark_consolidated_demotes_to_cold(self, tmp_path):
        """mark_consolidated で episode が cold に降格され derived_from が張られること。"""
        engine, db, store = _build_engine(tmp_path)

        ep_ids = ["EP001", "EP002"]
        new_id = "SUMMARY001"

        ep_records = {eid: _fake_record(id=eid, type="episode") for eid in ep_ids}
        ep_mems = {eid: _fake_db_mem(id=eid, type="episode") for eid in ep_ids}

        db.get_memory.side_effect = lambda i: ep_mems.get(i)
        store.read.side_effect = lambda p: next(
            (r for r in ep_records.values() if str(r.path) == str(p)),
            ep_records[ep_ids[0]],
        )

        result = engine.mark_consolidated(ep_ids, new_id)

        assert result["status"] == "ok"
        assert set(result["consolidated"]) == set(ep_ids)

        # 各 episode が cold に降格される
        tier_calls = [c for c in db.set_tier.call_args_list]
        for eid in ep_ids:
            assert any(c[0] == (eid, "cold") for c in tier_calls)

        # derived_from リンクが張られる
        link_calls = db.add_link.call_args_list
        df_calls = [(c[0][0], c[0][1], c[0][2]) for c in link_calls if c[0][2] == "derived_from"]
        for eid in ep_ids:
            assert any(c[0] == eid and c[1] == new_id for c in df_calls)


# ---------------------------------------------------------------------------
# reindex(手編集検知)テスト
# ---------------------------------------------------------------------------

class TestReindex:
    def test_reindex_detects_new_file(self, tmp_path):
        """DB にない Markdown が added としてカウントされること。"""
        engine, db, store = _build_engine(tmp_path)

        new_record = _fake_record(id="NEW_FILE", content_hash="abc")
        store.scan_all.return_value = iter([new_record])
        db.get_memory.return_value = None  # DB にない
        db.all_memories.return_value = []

        result = engine.reindex()
        assert result["added"] == 1
        assert result["updated"] == 0
        assert result["removed"] == 0
        assert result["unchanged"] == 0
        db.upsert_memory.assert_called_once()

    def test_reindex_detects_edit(self, tmp_path):
        """content_hash の差異が updated としてカウントされること。"""
        engine, db, store = _build_engine(tmp_path)

        record = _fake_record(id="EDIT001", content_hash="new_hash")
        store.scan_all.return_value = iter([record])
        # DB には古いハッシュが記録されている
        db.get_memory.return_value = _fake_db_mem(
            id="EDIT001", content_hash="old_hash"
        )
        db.all_memories.return_value = [_fake_db_mem(id="EDIT001")]

        result = engine.reindex()
        assert result["updated"] == 1
        assert result["added"] == 0
        assert result["unchanged"] == 0

    def test_reindex_removes_orphan(self, tmp_path):
        """ファイルが消えた DB エントリが removed としてカウントされること。"""
        engine, db, store = _build_engine(tmp_path)

        # store にはファイルなし、DB には ORPHAN が残っている
        store.scan_all.return_value = iter([])
        db.all_memories.return_value = [_fake_db_mem(id="ORPHAN")]

        result = engine.reindex()
        assert result["removed"] == 1
        db.delete_memory.assert_called_once_with("ORPHAN")

    def test_reindex_unchanged(self, tmp_path):
        """ハッシュが一致する場合は unchanged としてカウントされること。"""
        engine, db, store = _build_engine(tmp_path)

        record = _fake_record(id="SAME001", content_hash="same_hash")
        store.scan_all.return_value = iter([record])
        db.get_memory.return_value = _fake_db_mem(
            id="SAME001", content_hash="same_hash"
        )
        db.all_memories.return_value = [_fake_db_mem(id="SAME001")]

        result = engine.reindex()
        assert result["unchanged"] == 1
        assert result["added"] == 0
        assert result["updated"] == 0


# ---------------------------------------------------------------------------
# auto_deepened テスト
# ---------------------------------------------------------------------------

class TestAutoDeepened:
    def test_auto_deepened_when_low_score(self, tmp_path):
        """fast のスコアが deep_score_threshold 未満なら auto_deepened=True になること。"""
        engine, db, store = _build_engine(tmp_path)

        # スコアが非常に低くなる状況: 類似度が低い
        mem = _fake_db_mem(id="COLD_MEM", importance=1, created_at=NOW - 365 * DAY)
        db.all_memories.return_value = [mem]
        db.vector_search.return_value = [("COLD_MEM", 0.1)]  # 類似度低
        db.keyword_search.return_value = []
        db.get_events.return_value = {}  # イベントなし → 活性度0
        db.get_links.return_value = []
        db.get_embeddings.return_value = {}
        record = _fake_record(id="COLD_MEM", importance=1, content="関係なさそうな記憶")
        store.read.return_value = record

        result = engine.recall("全く関係ないクエリ", mode="fast", now=NOW)

        # final_score(0.1, 0, 1) = 0.6*0.1 + 0.25*0 + 0.15*(1/10) = 0.075 < 0.35
        assert result["auto_deepened"] is True
        assert result["mode"] == "deep"

    def test_no_auto_deepened_when_high_score(self, tmp_path):
        """fast のスコアが deep_score_threshold 以上なら auto_deepened=False になること。"""
        engine, db, store = _build_engine(tmp_path)

        mem = _fake_db_mem(id="HOT_MEM", importance=9, created_at=NOW - DAY)
        db.all_memories.return_value = [mem]
        db.vector_search.return_value = [("HOT_MEM", 0.98)]  # 高類似度
        db.keyword_search.return_value = [("HOT_MEM", -1.0)]
        # reinforce イベントがある → 高活性度
        db.get_events.return_value = {
            "HOT_MEM": [(NOW - DAY, 3.0), (NOW - 2 * DAY, 2.0)]
        }
        record = _fake_record(id="HOT_MEM", importance=9, content="非常に関連性の高い記憶")
        store.read.return_value = record

        result = engine.recall("関連性の高いクエリ", mode="fast", now=NOW)

        assert result["auto_deepened"] is False
        assert result["mode"] == "fast"


# ---------------------------------------------------------------------------
# deep recall での連想経由ノードテスト
# ---------------------------------------------------------------------------

class TestDeepRecall:
    def test_associative_via_link(self, tmp_path):
        """deep recall でリンク経由のみのノードが via=associative になること。"""
        engine, db, store = _build_engine(tmp_path)

        # DIRECT: ベクトル検索にヒット
        # ASSOC: ベクトル検索には出ないが DIRECT から co_recall リンクで繋がる
        direct_mem = _fake_db_mem(id="DIRECT", importance=5, created_at=NOW - DAY)
        assoc_mem = _fake_db_mem(id="ASSOC", importance=5, created_at=NOW - DAY)

        db.all_memories.return_value = [direct_mem, assoc_mem]
        db.vector_search.return_value = [("DIRECT", 0.85)]
        db.keyword_search.return_value = []
        db.get_events.return_value = {}

        # DIRECT から ASSOC への co_recall リンク
        db.get_links.return_value = [("DIRECT", "ASSOC", "co_recall", 0.9)]

        # ASSOC の埋め込みを返す(relevance 計算用)
        embedder = engine.embedder
        assoc_vec = embedder.embed_docs(["連想記憶"])[0]
        db.get_embeddings.return_value = {"ASSOC": assoc_vec}

        direct_record = _fake_record(id="DIRECT", content="直接ヒットした記憶")
        assoc_record = _fake_record(id="ASSOC", content="連想で到達した記憶")
        store.read.side_effect = lambda p: (
            assoc_record if "ASSOC" in str(p) else direct_record
        )

        result = engine.recall("クエリ", mode="deep", now=NOW)

        hit_map = {h["id"]: h for h in result["hits"]}
        if "ASSOC" in hit_map:
            assert hit_map["ASSOC"]["via"] == "associative"
        if "DIRECT" in hit_map:
            assert hit_map["DIRECT"]["via"] == "direct"


# ---------------------------------------------------------------------------
# correction タグが importance を引き上げるテスト
# ---------------------------------------------------------------------------

class TestCorrectionTag:
    def test_correction_tag_raises_importance(self, tmp_path):
        """tags に correction が含まれる場合 importance が引き上げられること。"""
        engine, db, store = _build_engine(tmp_path)

        record = _fake_record(id="CORR001", importance=7)
        store.create.return_value = record
        db.vector_search.return_value = []

        # importance=3 で correction タグ付き
        engine.remember(
            "訂正情報",
            type="knowledge",
            importance=3,
            tags=["correction"],
            now=NOW,
        )

        # store.create に渡された importance が 7 以上になること
        create_kwargs = store.create.call_args[1]
        assert create_kwargs["importance"] >= 7


# ---------------------------------------------------------------------------
# exhaustive recall(沈んだ記憶の掘り起こし)テスト
# ---------------------------------------------------------------------------


class _FixedQueryEmbedder:
    """embed_query が常に固定ベクトルを返すスタブ(relevance を厳密に制御する)。"""

    def __init__(self, qvec):
        self._q = np.asarray(qvec, dtype=np.float32)
        self.dim = len(self._q)

    def embed_query(self, text):
        return self._q

    def embed_docs(self, texts):
        return [self._q for _ in texts]


class TestExhaustiveRecall:
    def test_exhaustive_ranks_by_relevance_ignoring_activation(self, tmp_path):
        """mode=exhaustive は活性度を無視し関連度順。沈んだ高関連記憶が浮上し、
        関連度が floor 未満の記憶は除外されること。"""
        embedder = _FixedQueryEmbedder([1.0, 0.0, 0.0, 0.0])
        engine, db, store = _build_engine(tmp_path, embedder=embedder)

        # R: 沈んだ(古い1イベントのみ)が関連度最大、G: 新しく高活性だが関連度中、
        # L: 関連度ゼロ(floor 未満で除外される)
        r = _fake_db_mem(id="R", type="knowledge", importance=1,
                         created_at=NOW - 1000 * DAY, path="/tmp/R.md")
        g = _fake_db_mem(id="G", type="preference", importance=9,
                         created_at=NOW - DAY, path="/tmp/G.md")
        ll = _fake_db_mem(id="L", type="knowledge", importance=5,
                          created_at=NOW - DAY, path="/tmp/L.md")
        db.all_memories.return_value = [r, g, ll]
        db.get_embeddings.return_value = {
            "R": np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),   # cos 1.0
            "G": np.asarray([0.6, 0.8, 0.0, 0.0], dtype=np.float32),   # cos 0.6
            "L": np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float32),   # cos 0.0
        }
        db.get_events.return_value = {
            "R": [(NOW - 1000 * DAY, 1.0)],                 # 沈んでいる
            "G": [(NOW - DAY, 3.0), (NOW - 2 * DAY, 2.0)],  # 高活性
            "L": [(NOW - DAY, 1.0)],
        }
        store.read.side_effect = lambda p: _fake_record(
            id=Path(p).stem, content=f"{Path(p).stem} の本文"
        )

        result = engine.recall("ターゲット", mode="exhaustive", now=NOW)

        assert result["mode"] == "exhaustive"
        hits = result["hits"]
        ids = [h["id"] for h in hits]
        # R が最上位(関連度 1.0)、G が続く。L は floor 未満で除外。
        assert ids[0] == "R"
        assert "G" in ids
        assert "L" not in ids
        hit_map = {h["id"]: h for h in hits}
        # 沈んだ R が、より高活性の G より上に来ている(活性度を無視している証拠)
        assert hit_map["R"]["activation"] < hit_map["G"]["activation"]
        assert hit_map["R"]["via"] == "exhaustive"
        assert hit_map["R"]["relevance"] == pytest.approx(1.0, abs=1e-5)

    def test_fast_buries_what_exhaustive_surfaces(self, tmp_path):
        """対比: 同じ沈んだ高関連記憶 R は fast では高活性 G に首位を奪われる。"""
        embedder = _FixedQueryEmbedder([1.0, 0.0, 0.0, 0.0])
        engine, db, store = _build_engine(tmp_path, embedder=embedder)

        r = _fake_db_mem(id="R", type="knowledge", importance=1,
                         created_at=NOW - 1000 * DAY, path="/tmp/R.md")
        g = _fake_db_mem(id="G", type="preference", importance=9,
                         created_at=NOW - DAY, path="/tmp/G.md")
        db.all_memories.return_value = [r, g]
        db.vector_search.return_value = [("R", 1.0), ("G", 0.6)]
        db.keyword_search.return_value = []
        db.get_events.return_value = {
            "R": [(NOW - 1000 * DAY, 1.0)],
            "G": [(NOW - DAY, 3.0), (NOW - 2 * DAY, 2.0)],
        }
        store.read.side_effect = lambda p: _fake_record(
            id=Path(p).stem, content=f"{Path(p).stem} の本文"
        )

        result = engine.recall("ターゲット", mode="fast", now=NOW)

        # fast は活性度を加味するので、関連度最大の R ではなく高活性 G が首位
        assert result["hits"][0]["id"] == "G"

    def test_deep_auto_escalates_to_exhaustive(self, tmp_path):
        """deep でも最高スコアが弱いとき exhaustive へ自動エスカレーションし、
        沈んだ高関連記憶を掘り起こすこと。"""
        embedder = _FixedQueryEmbedder([1.0, 0.0, 0.0, 0.0])
        engine, db, store = _build_engine(tmp_path, embedder=embedder)

        m = _fake_db_mem(id="M", type="knowledge", importance=5,
                         created_at=NOW - 1000 * DAY, path="/tmp/M.md")
        db.all_memories.return_value = [m]
        db.vector_search.return_value = [("M", 0.2)]   # fast/deep では低関連
        db.keyword_search.return_value = []
        db.get_events.return_value = {}                # 活性度ゼロ
        db.get_links.return_value = []
        # exhaustive で再計算される実コサインは高い(本来は関連の高い記憶)
        db.get_embeddings.return_value = {
            "M": np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        }
        store.read.return_value = _fake_record(id="M", content="本来は関連の高い記憶")

        result = engine.recall("ターゲット", mode="fast", now=NOW)

        assert result["auto_deepened"] is True
        assert result["mode"] == "exhaustive"
        assert result["hits"][0]["id"] == "M"
        assert result["hits"][0]["relevance"] == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# 起動時インデックス同期チェック(マルチマシン共有対策)
# ---------------------------------------------------------------------------


class TestIndexFreshness:
    @staticmethod
    def _active(n):
        return [_fake_db_mem(id=f"M{i}") for i in range(n)]

    def test_in_sync_no_reindex(self, tmp_path):
        """raw .md 件数と index 件数が一致 → in_sync・scan も reindex もしない。"""
        engine, db, store = _build_engine(tmp_path)
        store.count_memory_files.return_value = 5
        db.all_memories.return_value = self._active(5)
        engine.reindex = MagicMock()

        res = engine.check_index_freshness(mode="auto")

        assert res["action"] == "in_sync"
        assert res["markdown"] == 5 and res["index"] == 5
        store.scan_all.assert_not_called()
        engine.reindex.assert_not_called()

    def test_phantom_md_counts_as_in_sync(self, tmp_path):
        """raw .md が index より多くても、scan_all の有効件数が一致すれば
        (空/壊れた/非記憶 md による見かけ上のズレ)reindex しない。"""
        engine, db, store = _build_engine(tmp_path)
        store.count_memory_files.return_value = 6      # phantom 1件込み
        db.all_memories.return_value = self._active(5)
        store.scan_all.return_value = [object()] * 5    # 有効な記憶は5件
        engine.reindex = MagicMock()

        res = engine.check_index_freshness(mode="auto")

        assert res["action"] == "in_sync"
        assert res["valid"] == 5
        engine.reindex.assert_not_called()

    def test_auto_reindexes_on_real_drift(self, tmp_path):
        """有効な記憶が index より多い(他マシンの未取り込み)→ auto は reindex。"""
        engine, db, store = _build_engine(tmp_path)
        store.count_memory_files.return_value = 10
        db.all_memories.return_value = self._active(5)
        store.scan_all.return_value = [object()] * 10
        engine.reindex = MagicMock(return_value={
            "added": 5, "updated": 0, "removed": 0, "unchanged": 5})

        res = engine.check_index_freshness(mode="auto")

        assert res["action"] == "reindexed"
        assert res["valid"] == 10 and res["index"] == 5
        engine.reindex.assert_called_once()

    def test_warn_does_not_reindex(self, tmp_path):
        """warn モードは乖離を報告するだけで reindex しない。"""
        engine, db, store = _build_engine(tmp_path)
        store.count_memory_files.return_value = 10
        db.all_memories.return_value = self._active(5)
        store.scan_all.return_value = [object()] * 10
        engine.reindex = MagicMock()

        res = engine.check_index_freshness(mode="warn")

        assert res["action"] == "warn"
        assert res["drift"] == 5
        engine.reindex.assert_not_called()

    def test_off_does_nothing(self, tmp_path):
        """off モードはカウントもせず何もしない。"""
        engine, db, store = _build_engine(tmp_path)
        store.count_memory_files.return_value = 10
        db.all_memories.return_value = self._active(5)
        engine.reindex = MagicMock()

        res = engine.check_index_freshness(mode="off")

        assert res["action"] == "off"
        store.count_memory_files.assert_not_called()
        engine.reindex.assert_not_called()
