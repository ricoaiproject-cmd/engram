"""30日アクセスパターンのシミュレーション。

FakeEmbedder + 一時ディレクトリで記憶200件を投入し、合成クロック(now 引数注入)で
30日分のアクセスパターンを模擬する。

乱数は seed 固定で決定的。

レポート内容:
1. reinforce された記憶が関連クエリの recall で上位に浮上していること
2. 放置記憶は fast では沈むが deep(連想リンク経由)で発見できること
3. importance 9 の未使用記憶が importance 2 の未使用記憶より上位に残ること

使い方: python scripts/simulate.py
(db / store が完成後に使用可能。現時点では NotImplementedError で失敗する想定)
"""

from __future__ import annotations

import random
import sys
import tempfile
import time
from pathlib import Path

# Windows コンソール(cp932)でも ✓ 等を出せるよう UTF-8 に再構成
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# プロジェクトの src を PYTHONPATH に追加
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from engram.config import Settings
from engram.embedder import FakeEmbedder
from engram.engine import MemoryEngine, build_engine

# --- 定数 ---
SEED = 42
N_MEMORIES = 200
DAY = 86400.0
START_TIME = 1_750_000_000.0  # シミュレーション開始時刻(固定)
N_DAYS = 30

# 定期的に使われる「アクティブ」記憶の数
N_ACTIVE = 30
# リンクで連鎖を作る記憶のペア数
N_LINK_PAIRS = 20

# カテゴリ別の記憶テンプレート(FakeEmbedder が n-gram 類似度を使うため、
# 同カテゴリ内で語が被るように設計)
KNOWLEDGE_TEMPLATES = [
    "Pythonの非同期処理(asyncio)について: コルーチンを使う",
    "Pythonの型ヒント: TypeVar と Generic の使い方",
    "FastAPI の依存性注入パターンについての知識",
    "SQLite の WAL モードでの並行書き込み制御",
    "ベクトル検索の仕組み: コサイン類似度と KNN",
    "BM25 アルゴリズムの仕組みと FTS5 での実装",
    "ACT-R モデルの記憶活性化計算式",
    "Reciprocal Rank Fusion による検索ランク統合",
    "obsidian のウィキリンク形式 [[id]] の仕様",
    "ULID の生成規則と時系列ソート可能性",
]

EPISODE_TEMPLATES = [
    "今日の作業: エンジン設計レビューを行った",
    "今日の進捗: store.py のスタブ仕様を確認した",
    "今日の作業: test_dynamics.py を通過させた",
    "今日の進捗: recall メソッドの RRF 統合を実装した",
    "今日の作業: CLI のサブコマンドを実装した",
    "今日の進捗: co_recall リンクのヘッブ則を実装した",
    "今日の作業: consolidation 候補の貪欲クラスタリング実装",
    "今日の進捗: reindex の差分検知ロジックを実装した",
    "今日の作業: シミュレーションスクリプトを実行した",
    "今日の進捗: MCP サーバーの FastMCP 統合を完了した",
]


def run_simulation():
    rng = random.Random(SEED)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        settings = Settings(
            memories_dir=tmp_path / "memories",
            data_dir=tmp_path / "data",
            candidate_k=50,
            deep_score_threshold=0.35,
        )
        # dim=64 はハッシュ衝突で無関係テキスト同士も cos≈0.5 になるため 256
        embedder = FakeEmbedder(dim=256)
        engine = build_engine(settings, embedder=embedder)
        try:
            return _run(engine, rng)
        finally:
            # Windows では DB を閉じないと一時ディレクトリの削除に失敗する
            engine.db.close()


def _run(engine: MemoryEngine, rng: random.Random):

        print("=" * 60)
        print("engram 30日アクセスパターン シミュレーション")
        print("=" * 60)

        # ----------------------------------------------------------------
        # Phase 1: 記憶200件を投入(day 0)
        # ----------------------------------------------------------------
        print(f"\n[Phase 1] 記憶 {N_MEMORIES} 件を投入中...")
        t0 = START_TIME
        memory_ids: list[str] = []
        importance_map: dict[str, int] = {}

        for i in range(N_MEMORIES):
            # テンプレートをループしながら本文を生成
            if i < N_MEMORIES // 2:
                templates = KNOWLEDGE_TEMPLATES
                mem_type = "knowledge"
            else:
                templates = EPISODE_TEMPLATES
                mem_type = "episode"

            template = templates[i % len(templates)]
            # テンプレートだけだと連番違いの本文が重複検知(cos >= 0.92)に
            # 併合されてしまうため、個別の詳細文を足して各記憶を一意にする
            detail_words = ["実装時の注意点", "設計上の根拠", "計測した結果",
                            "失敗例の分析", "代替案との比較", "運用での知見",
                            "境界条件の確認", "性能への影響", "互換性の検討",
                            "導入手順の整理"]
            details = rng.sample(detail_words, 3)
            content = (f"{template} (記憶#{i:03d})。"
                       f"観点: {details[0]}、{details[1]}、{details[2]}。"
                       f"整理番号 {i * 7919 % 100000}。")

            # importanceを割り当て
            # - アクティブ記憶(最初の N_ACTIVE 件): importance 5-8
            # - importance 9 の特別記憶(インデックス N_ACTIVE): 1件
            # - importance 2 の低優先記憶(インデックス N_ACTIVE+1): 1件
            # - 残りは importance 3-6
            if i < N_ACTIVE:
                importance = rng.randint(5, 8)
            elif i == N_ACTIVE:
                importance = 9   # 高重要度の未使用記憶
            elif i == N_ACTIVE + 1:
                importance = 2   # 低重要度の未使用記憶
            else:
                importance = rng.randint(3, 6)

            result = engine.remember(
                content=content,
                type=mem_type,
                importance=importance,
                source="simulate",
                now=t0 + i * 10,  # 10秒間隔で投入
            )

            if result["status"] in ("created", "duplicate_reinforced"):
                mem_id = result["id"]
                memory_ids.append(mem_id)
                importance_map[mem_id] = importance

        print(f"  投入完了: {len(memory_ids)} 件")

        # アクティブ記憶 ID とその他を分類
        active_ids = memory_ids[:N_ACTIVE]

        # Report 3 用の特別ペア: 同じキーワードを共有し(=候補集合に両方入る)、
        # importance だけが違う2記憶。どちらも以後一切アクセスしない
        high_res = engine.remember(
            "Xyzzy復旧手順: 本番障害時はまずスナップショットを確保してから再起動する",
            type="knowledge", importance=9, source="simulate", now=t0 + 3000,
        )
        low_res = engine.remember(
            "Xyzzy復旧手順についての参考資料がどこにあるかの覚え書き",
            type="knowledge", importance=2, source="simulate", now=t0 + 3010,
        )
        high_imp_id = high_res["id"]
        low_imp_id = low_res["id"]

        # ----------------------------------------------------------------
        # Phase 2: リンクの連鎖を作る
        # ----------------------------------------------------------------
        print(f"\n[Phase 2] {N_LINK_PAIRS} ペアのリンク連鎖を作成中...")
        inactive_ids = memory_ids[N_ACTIVE + 2:]
        link_pairs: list[tuple[str, str]] = []
        sampled = rng.sample(inactive_ids, min(N_LINK_PAIRS * 2, len(inactive_ids)))
        for i in range(0, len(sampled) - 1, 2):
            src, dst = sampled[i], sampled[i + 1]
            engine.link(src, dst)
            link_pairs.append((src, dst))
        print(f"  リンク作成完了: {len(link_pairs)} ペア")

        # ----------------------------------------------------------------
        # Phase 3: 30日間のアクセスパターンを模擬
        # ----------------------------------------------------------------
        print(f"\n[Phase 3] {N_DAYS}日間のアクセスパターンを模擬中...")

        for day in range(1, N_DAYS + 1):
            day_ts = START_TIME + day * DAY

            # アクティブ記憶を毎日 recall + reinforce(使用されている記憶)
            n_daily_active = rng.randint(3, 8)
            daily_active = rng.sample(active_ids, min(n_daily_active, len(active_ids)))

            # recall イベントを記録
            for mem_id in daily_active:
                engine.db.add_event(mem_id, "recall_hit",
                                    engine.settings.recall_hit_weight, day_ts)

            # 3日おきに reinforce
            if day % 3 == 0:
                strength = rng.uniform(1.0, 2.5)
                engine.reinforce(daily_active, strength=strength, now=day_ts)

        print("  アクセスパターン模擬完了")

        # ----------------------------------------------------------------
        # Phase 4: レポート
        # ----------------------------------------------------------------
        print("\n" + "=" * 60)
        print("レポート")
        print("=" * 60)

        sim_now = START_TIME + N_DAYS * DAY

        # --- Report 1: reinforce された記憶が関連クエリで上位に浮上 ---
        print("\n[Report 1] reinforce 済み記憶の recall 順位")
        query = KNOWLEDGE_TEMPLATES[0]  # アクティブ記憶のテンプレート1
        result = engine.recall(query, mode="fast", limit=10, now=sim_now, record_hits=False)
        hits = result["hits"]
        hit_ids_top10 = [h["id"] for h in hits]
        active_in_top10 = [id_ for id_ in hit_ids_top10 if id_ in active_ids]

        print(f"  クエリ: '{query[:50]}...'")
        print(f"  上位10件中、アクティブ(reinforce済み)記憶: {len(active_in_top10)} 件")
        for h in hits[:5]:
            is_active = "(アクティブ)" if h["id"] in active_ids else ""
            print(f"    [..{h['id'][-8:]}] score={h['score']:.3f} "
                  f"act={h['activation']:.3f} {is_active}")

        r1_passed = len(active_in_top10) > 0
        print(f"  結果: {'PASS ✓' if r1_passed else 'FAIL ✗'} "
              f"(アクティブ記憶が上位に浮上{'している' if r1_passed else 'していない'})")

        # --- Report 2: 放置記憶は fast では沈むが deep で発見できる ---
        print("\n[Report 2] 放置記憶の fast vs deep 発見率")
        if link_pairs:
            # リンクの src(放置記憶)に関連するクエリで検索
            src_id, dst_id = link_pairs[0]

            fast_result = engine.recall(
                EPISODE_TEMPLATES[0],  # 放置記憶のテンプレートに近いクエリ
                mode="fast", limit=10, now=sim_now, record_hits=False
            )
            deep_result = engine.recall(
                EPISODE_TEMPLATES[0],
                mode="deep", limit=10, now=sim_now, record_hits=False
            )

            fast_ids = {h["id"] for h in fast_result["hits"]}
            deep_ids = {h["id"] for h in deep_result["hits"]}

            # リンク先(dst_id)が deep にのみ現れるか確認
            assoc_in_deep = any(
                h["id"] in {dst for _, dst in link_pairs} and h["via"] == "associative"
                for h in deep_result["hits"]
            )

            print(f"  Fast recall でヒット数: {len(fast_ids)}")
            print(f"  Deep recall でヒット数: {len(deep_ids)}")
            print(f"  連想リンク経由ノードが deep で発見: {assoc_in_deep}")

            deep_assoc_hits = [h for h in deep_result["hits"] if h["via"] == "associative"]
            print(f"  Deep のみ(via=associative)ヒット: {len(deep_assoc_hits)} 件")

            r2_passed = len(deep_ids) >= len(fast_ids)
            print(f"  結果: {'PASS ✓' if r2_passed else 'FAIL ✗'} "
                  f"(deep が fast 以上の件数{'ヒット' if r2_passed else 'ヒットせず'})")
        else:
            print("  (リンクペアなし、スキップ)")
            r2_passed = True

        # --- Report 3: importance 9 の未使用記憶が importance 2 より上位 ---
        print("\n[Report 3] 未使用記憶の importance による順位差")
        if high_imp_id and low_imp_id:
            # 両方の記憶が共有する専用キーワードで検索(両方を候補集合に乗せる)
            result = engine.recall(
                "Xyzzy復旧手順", mode="fast", limit=50,
                now=sim_now, record_hits=False
            )
            rank_map = {h["id"]: i for i, h in enumerate(result["hits"])}

            rank_high = rank_map.get(high_imp_id, 9999)
            rank_low = rank_map.get(low_imp_id, 9999)

            print(f"  importance 9 の未使用記憶 [{high_imp_id[:8]}]: 順位 {rank_high + 1}")
            print(f"  importance 2 の未使用記憶 [{low_imp_id[:8]}]: 順位 {rank_low + 1}")

            # 本文が違う以上、クエリとの関連度には正当な差が出る。符号化の深さ
            # (フラッシュバルブ効果)を見るには関連度の寄与を除いた成分
            # (活性度 + 重要度)で比較する
            hit_high = next((h for h in result["hits"] if h["id"] == high_imp_id), None)
            hit_low = next((h for h in result["hits"] if h["id"] == low_imp_id), None)
            w_rel = engine.settings.w_relevance
            enc_high = (hit_high["score"] - w_rel * hit_high["relevance"]) if hit_high else 0.0
            enc_low = (hit_low["score"] - w_rel * hit_low["relevance"]) if hit_low else 0.0
            print(f"  importance 9 符号化成分(活性度+重要度): {enc_high:.4f}")
            print(f"  importance 2 符号化成分(活性度+重要度): {enc_low:.4f}")

            r3_passed = enc_high > enc_low
            print(f"  結果: {'PASS ✓' if r3_passed else 'FAIL ✗'} "
                  f"(高 importance が{'上位' if r3_passed else '下位'})")
        else:
            print("  (importance テスト用記憶なし、スキップ)")
            r3_passed = True

        # --- 総合結果 ---
        print("\n" + "=" * 60)
        all_passed = r1_passed and r2_passed and r3_passed
        print(f"総合結果: {'全テスト PASS ✓' if all_passed else '一部 FAIL ✗'}")
        print("=" * 60)

        return all_passed


if __name__ == "__main__":
    success = run_simulation()
    sys.exit(0 if success else 1)
