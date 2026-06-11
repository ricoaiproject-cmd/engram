"""連想想起の標的検証。

「クエリと意味的に遠い(ベクトル検索では出ない)が、関連記憶とリンクで
繋がっている記憶」が deep recall で via="associative" として発見されることを
確認する。構想の核心(辿れば必ず引っ張ってこれる)の直接テスト。
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC))

from engram.config import Settings
from engram.embedder import FakeEmbedder
from engram.engine import build_engine

DAY = 86400.0
T0 = 1_750_000_000.0

with tempfile.TemporaryDirectory() as tmpdir:
    tmp = Path(tmpdir)
    # candidate_k をコーパスより小さくし、ベクトル検索の網に
    # 「意味的に遠い記憶」が入らない状況を作る(実運用の縮図)
    settings = Settings(memories_dir=tmp / "memories", data_dir=tmp / "data",
                        candidate_k=10)
    # dim=64 はハッシュ衝突で無関係テキスト同士も cos≈0.5 になりノイズが大きい
    engine = build_engine(settings, embedder=FakeEmbedder(dim=256))
    try:
        # ノイズ: クエリと無関係な記憶を多数(重複検知を避けるため文面を変える)
        topics = ["予算配分", "採用面接", "週次定例", "顧客訪問", "障害対応訓練",
                  "棚卸し作業", "契約更新", "備品発注", "勉強会準備", "評価面談"]
        for i in range(60):
            engine.remember(
                f"メモ{i}: {topics[i % 10]}に関する記録その{i}。詳細は別紙参照。",
                type="knowledge", importance=4, now=T0,
            )

        # ハブ: クエリと意味的に近い記憶
        hub = engine.remember(
            "SQLiteのWALモードでは書き込みと読み取りが並行できる",
            type="knowledge", importance=6, now=T0,
        )

        # 孤立記憶: クエリと意味的に遠い(語彙の重なりなし)が、ハブとリンク
        iso = engine.remember(
            "圧力鍋で豚の角煮を作るときは下茹でを30分する",
            type="knowledge", importance=5, now=T0,
        )
        engine.link(hub["id"], iso["id"])

        # 孤立記憶を cold に落とす(古い未使用記憶を模擬)
        rec = engine.store.find_by_id(iso["id"])
        engine.store.set_tier(rec, "cold")
        engine.db.set_tier(iso["id"], "cold")

        query = "SQLite WALモード 並行書き込み"
        fast = engine.recall(query, mode="fast", limit=10, now=T0 + 30 * DAY,
                             record_hits=False)
        deep = engine.recall(query, mode="deep", limit=10, now=T0 + 30 * DAY,
                             record_hits=False)

        fast_ids = [h["id"] for h in fast["hits"]]
        deep_hits = {h["id"]: h for h in deep["hits"]}

        print(f"fast に孤立記憶が出る(出ないのが正): {iso['id'] in fast_ids}")
        in_deep = iso["id"] in deep_hits
        print(f"deep に孤立記憶が出る(出るのが正): {in_deep}")
        if in_deep:
            h = deep_hits[iso["id"]]
            print(f"  via={h['via']} score={h['score']:.3f} "
                  f"relevance={h['relevance']:.3f} tier={h['tier']}")
        ok = (iso["id"] not in fast_ids) and in_deep \
            and deep_hits[iso["id"]]["via"] == "associative"
        print(f"\n結果: {'PASS' if ok else 'FAIL'}")
        sys.exit(0 if ok else 1)
    finally:
        engine.db.close()
