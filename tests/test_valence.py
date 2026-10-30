"""Phase 3(書き込みの増量): 感情価(valence)と符号化粒度のテスト。"""

from __future__ import annotations

import json

import frontmatter

from engram.config import Settings, get_settings
from engram.db import IndexDB
from engram.embedder import FakeEmbedder
from engram.engine import MemoryEngine
from engram.store import MarkdownStore
from engram.transcript import build_episodes, estimate_valence

NOW = 1_800_000_000.0


def _make_settings(tmp_path, **overrides) -> Settings:
    return Settings(
        memories_dir=tmp_path / "memories",
        data_dir=tmp_path / "data",
        **overrides,
    )


def _make_engine(settings) -> MemoryEngine:
    embedder = FakeEmbedder(dim=64)
    store = MarkdownStore(settings.memories_dir)
    db = IndexDB(settings.db_path, embedder.dim)
    return MemoryEngine(settings=settings, store=store, db=db,
                        embedder=embedder)


# ---------------------------------------------------------------------------
# 感情価の保存経路
# ---------------------------------------------------------------------------

def test_valence_roundtrip_file_and_db(tmp_path):
    settings = _make_settings(tmp_path)
    engine = _make_engine(settings)
    r = engine.remember(
        "本番障害の原因はドライブ文字の漂流だった", "knowledge", 8,
        valence=-0.7, now=NOW,
    )
    mem = engine.db.get_memory(r["id"])
    assert mem["valence"] == -0.7

    post = frontmatter.load(r["path"])
    assert float(post["valence"]) == -0.7

    # ファイル→レコード読み戻し
    rec = engine.store.read(engine.store.find_by_id(r["id"]).path)
    assert rec.valence == -0.7
    engine.db.close()


def test_valence_clamped_and_default_zero(tmp_path):
    settings = _make_settings(tmp_path)
    engine = _make_engine(settings)
    r1 = engine.remember("極端な値", "knowledge", 5, valence=5.0, now=NOW)
    assert engine.db.get_memory(r1["id"])["valence"] == 1.0
    r2 = engine.remember("既定値", "knowledge", 5, now=NOW)
    assert engine.db.get_memory(r2["id"])["valence"] == 0.0
    engine.db.close()


def test_correct_records_negative_valence(tmp_path):
    """訂正は「避けろ」信号として負の感情価で刻まれる。"""
    settings = _make_settings(tmp_path)
    engine = _make_engine(settings)
    r = engine.remember("会議は毎週月曜と覚えた", "knowledge", 5, now=NOW)
    c = engine.correct(
        r["id"], "会議は毎週火曜が正しい", "ユーザー訂正", source="test"
    )
    new_id = c.get("new_id") or c.get("id")
    mem = engine.db.get_memory(new_id)
    assert mem is not None
    assert mem["valence"] < 0
    engine.db.close()


def test_reindex_preserves_valence(tmp_path):
    settings = _make_settings(tmp_path)
    engine = _make_engine(settings)
    r = engine.remember("感情価つきの記憶", "knowledge", 6,
                        valence=0.4, now=NOW)
    # DB を作り直しても frontmatter から復元される
    engine.db.delete_memory(r["id"])
    engine.reindex()
    assert engine.db.get_memory(r["id"])["valence"] == 0.4
    engine.db.close()


# ---------------------------------------------------------------------------
# 感情価の推定(決定的ヒューリスティック)
# ---------------------------------------------------------------------------

def test_estimate_valence_signs():
    assert estimate_valence(["また間違ってる。訂正して。前も失敗したよね"]) < 0
    assert estimate_valence(["完璧！ありがとう、すごく助かった"]) > 0
    assert estimate_valence(["今日の天気を教えて"]) == 0.0


# ---------------------------------------------------------------------------
# 符号化の粒度(build_episodes)
# ---------------------------------------------------------------------------

def _many_turns(n):
    return {
        "user_texts": [f"発言その{i}: 作業の指示です" for i in range(1, n + 1)],
        "last_assistant": "全部完了しました。",
        "summary": "長いセッション",
        "cwd": "",
    }


def test_build_episodes_single_for_short_session():
    msgs = _many_turns(5)
    eps = build_episodes(msgs, date_str="2026-08-21", segment_user_turns=12)
    assert len(eps) == 1
    assert "パート" not in eps[0]


def test_build_episodes_segments_long_session():
    msgs = _many_turns(30)
    eps = build_episodes(msgs, date_str="2026-08-21", segment_user_turns=12)
    assert len(eps) == 3  # 12 + 12 + 6
    assert "パート1/3" in eps[0]
    assert "パート3/3" in eps[2]
    # 中盤の発言が失われない(従来の1要約では捨てられていた領域)
    assert any("発言その15" in ep for ep in eps)
    # 結末は最終パートのみ
    assert "結末" not in eps[0]
    assert "結末" in eps[2]


def test_build_episodes_zero_disables_segmentation():
    msgs = _many_turns(30)
    eps = build_episodes(msgs, date_str="2026-08-21", segment_user_turns=0)
    assert len(eps) == 1


# ---------------------------------------------------------------------------
# session-end 統合(複数 episode + 感情価)
# ---------------------------------------------------------------------------

def test_session_end_encodes_multiple_episodes(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("ENGRAM_HOME", str(home))
    monkeypatch.delenv("ENGRAM_MEMORIES_DIR", raising=False)
    monkeypatch.delenv("ENGRAM_DATA_DIR", raising=False)
    (home / "config.toml").write_text(
        "encode_segment_user_turns = 4" + chr(10)
        + "assistant_cue_mode = " + chr(39) + "off" + chr(39) + chr(10),
        encoding="utf-8",
    )

    import engram.engine
    from engram.hooks import run_session_end

    def _fake_build_engine(settings=None, *, embedder=None):
        settings = settings or get_settings()
        embedder = embedder or FakeEmbedder(dim=64)
        return MemoryEngine(
            settings=settings,
            store=MarkdownStore(settings.memories_dir),
            db=IndexDB(settings.db_path, embedder.dim),
            embedder=embedder,
        )

    monkeypatch.setattr(engram.engine, "build_engine", _fake_build_engine)

    transcript = tmp_path / "t.jsonl"
    # パート間で語彙が重ならないよう、話題の異なる発言にする(重複検知に
    # 併合されると粒度テストにならない)
    topics = [
        "決算資料の縦計を検算してほしい",
        "サムネイルの配色をピンク基調に変えたい",
        "会議室の予約は前日までに済ませる運用にする",
        "動画の台本に鉄則の章を追加して",
        "予算要求は財務課の様式7で出すこと",
        "リポジトリの単体テストを回して結果を教えて",
        "ドライブ文字の漂流問題を調査したい",
        "スライドの罫線と余白を直してほしい",
        "認証情報の保存場所をローカル側へ移す",
        "朝のブリーフィングを6時半に設定する",
    ]
    objs = []
    for text in topics:
        objs.append({
            "type": "user",
            "message": {"role": "user", "content": text},
        })
    objs.append({
        "type": "assistant",
        "message": {"role": "assistant",
                    "content": [{"type": "text", "text": "完了しました。"}]},
    })
    with transcript.open("w", encoding="utf-8") as f:
        for o in objs:
            f.write(json.dumps(o, ensure_ascii=False) + chr(10))

    stdin = json.dumps({
        "session_id": "sess-p3-1",
        "transcript_path": str(transcript),
        "cwd": "C:/proj/demo",
    })
    assert run_session_end(stdin) == 0

    settings = get_settings()
    db = IndexDB(settings.db_path, 64)
    episodes = db.all_memories(types=["episode"])
    db.close()
    assert len(episodes) == 3  # 10発言 ÷ 4 = 3パート
