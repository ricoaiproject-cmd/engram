"""実 db/store/engine を貫通する統合回帰テスト。

特に「意味的に遠い記憶でも、リンクを辿れば deep recall で必ず引っ張ってこれる」
という構想の核心(連想想起)を守る。
"""

import pytest

from engram.config import Settings
from engram.embedder import FakeEmbedder
from engram.engine import build_engine

DAY = 86400.0
T0 = 1_750_000_000.0


@pytest.fixture
def engine(tmp_path):
    settings = Settings(
        memories_dir=tmp_path / "memories",
        data_dir=tmp_path / "data",
        candidate_k=10,
    )
    # dim=64 はハッシュ衝突ノイズが大きく無関係テキストも cos≈0.5 になるため 256
    eng = build_engine(settings, embedder=FakeEmbedder(dim=256))
    yield eng
    eng.db.close()


def test_associative_recall_reaches_semantically_distant_memory(engine):
    """ベクトル検索では届かない記憶が、連想リンク経由で deep recall に浮上する。"""
    topics = ["予算配分", "採用面接", "週次定例", "顧客訪問", "障害対応訓練",
              "棚卸し作業", "契約更新", "備品発注", "勉強会準備", "評価面談"]
    for i in range(60):
        engine.remember(
            f"メモ{i}: {topics[i % 10]}に関する記録その{i}。詳細は別紙参照。",
            type="knowledge", importance=4, now=T0,
        )

    hub = engine.remember(
        "SQLiteのWALモードでは書き込みと読み取りが並行できる",
        type="knowledge", importance=6, now=T0,
    )
    # クエリと語彙の重なりが無い記憶を、ハブにリンクして cold に沈める
    iso = engine.remember(
        "圧力鍋で豚の角煮を作るときは下茹でを30分する",
        type="knowledge", importance=5, now=T0,
    )
    engine.link(hub["id"], iso["id"])
    rec = engine.store.find_by_id(iso["id"])
    engine.store.set_tier(rec, "cold")
    engine.db.set_tier(iso["id"], "cold")

    later = T0 + 30 * DAY
    fast = engine.recall("SQLite WALモード 並行書き込み", mode="fast",
                         limit=10, now=later, record_hits=False)
    deep = engine.recall("SQLite WALモード 並行書き込み", mode="deep",
                         limit=10, now=later, record_hits=False)

    fast_ids = [h["id"] for h in fast["hits"]]
    deep_map = {h["id"]: h for h in deep["hits"]}

    assert iso["id"] not in fast_ids, "fast(hot のみ)に cold の孤立記憶が出てはいけない"
    assert iso["id"] in deep_map, "deep はリンクを辿って孤立記憶に到達できるべき"
    assert deep_map[iso["id"]]["via"] == "associative"


def test_reinforce_lifts_memory_above_rival(engine):
    """同程度に関連する2記憶のうち、reinforce された方が上位に来る。"""
    # 文面が似すぎると重複検知(cos >= 0.92)で同一記憶に併合されるため、
    # 同じ主題で十分に異なる2文にする
    a = engine.remember("Pythonの型ヒントでは 3.9 以降 List ではなく list を書く",
                        type="knowledge", importance=5, now=T0)
    b = engine.remember("Pythonの型ヒントで構造的部分型を表すなら typing.Protocol",
                        type="knowledge", importance=5, now=T0 - 1)
    # b だけを繰り返し使う
    for day in range(1, 6):
        engine.reinforce([b["id"]], now=T0 + day * DAY)

    res = engine.recall("Python 型ヒント typing", limit=2,
                        now=T0 + 6 * DAY, record_hits=False)
    ids = [h["id"] for h in res["hits"]]
    assert ids.index(b["id"]) < ids.index(a["id"])


def test_correct_flow_end_to_end(engine):
    """訂正フロー: 旧記憶は fast から消え、deep では訂正済み注記付きで辿れる。"""
    old = engine.remember("設定ファイルは config.ini を使う",
                          type="knowledge", importance=5, now=T0)
    res = engine.correct(old["id"], "設定ファイルは settings.toml を使う",
                         "config.ini は v2 で廃止された", now=T0 + DAY)
    assert res["status"] == "corrected"

    fast = engine.recall("設定ファイル はどれを使う", limit=10,
                         now=T0 + 2 * DAY, record_hits=False)
    fast_ids = [h["id"] for h in fast["hits"]]
    assert old["id"] not in fast_ids
    assert res["new_id"] in fast_ids

    deep = engine.recall("設定ファイル はどれを使う", mode="deep", limit=10,
                         now=T0 + 2 * DAY, record_hits=False)
    deep_map = {h["id"]: h for h in deep["hits"]}
    if old["id"] in deep_map:
        assert res["new_id"] in deep_map[old["id"]]["note"]
