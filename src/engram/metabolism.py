"""代謝の計測(「代謝する脳」プラン Phase 0)。

賢さは蔵書量ではなく代謝ループ(書く→眠って圧縮・忘却→知恵→連想)の回転量
から生まれる、という設計方針(R8.8.21 合意)の物差し。4指標を返す:

1. 浮上命中率  — surface で浮上した記憶が一定時間内に reinforce された割合。
   ※ reinforce は「実際に役立った」報告の代理指標(報告忘れがあるため下限値)。
   recall 由来の reinforce と区別できない点も含め、厳密な因果ではなく
   同一の物差しで推移を見るための指標である。
2. 書き込み量  — create イベント数/日(符号化の流量)。
3. hot 在庫    — tier 別件数と hot 内の episode/非episode 内訳(在庫の淀み)。
4. 統合の滞留  — consolidation_state.json のクラスタ数と、統合対象年齢に
   達した hot episode の数(眠りの未消化量)。

surface.py と同じ軽量経路(素の sqlite3・埋め込み不使用)で動く。
夜間の眠りジョブ(sleep.py)のレポートもこの関数を使う。
"""

from __future__ import annotations

import json
import sqlite3
import time
from bisect import bisect_right
from pathlib import Path

from .config import Settings


# ---------------------------------------------------------------------------
# 入力の読み取り
# ---------------------------------------------------------------------------

def iter_surface_log(settings: Settings):
    """surface ログ(ローテーション済み .old → 現行)を古い順に yield する。"""
    d = settings.data_dir / "surface"
    for name in ("surface_log.jsonl.old", "surface_log.jsonl"):
        path = d / name
        if not path.is_file():
            continue
        try:
            with path.open(encoding="utf-8", errors="replace") as f:
                for line in f:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(row, dict):
                        yield row
        except OSError:
            continue


def surfaced_instances(settings: Settings) -> list[tuple[float, str, bool]]:
    """浮上インスタンス [(ts, memory_id, 意味検索あり), ...] を時刻順で返す。

    3要素目は、その浮上の際に意味の網(埋め込みデーモン)が効いていたか。
    semantic 欄は v0.21.3 以降のログにのみ存在し、無い行は字句のみ(False)。
    """
    out: list[tuple[float, str, bool]] = []
    for row in iter_surface_log(settings):
        ts = row.get("ts")
        if not isinstance(ts, (int, float)):
            continue
        sem = bool(row.get("semantic"))
        for mid in row.get("surfaced", []) or []:
            out.append((float(ts), str(mid), sem))
    out.sort()
    return out


def _connect_ro(db_path: Path) -> sqlite3.Connection | None:
    if not Path(db_path).is_file():
        return None
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=2000")
    return conn


def reinforce_events(db_path: Path) -> dict[str, list[float]]:
    """{memory_id: [reinforce の ts(昇順), ...]} を返す。DB が無ければ空。"""
    conn = _connect_ro(db_path)
    if conn is None:
        return {}
    try:
        result: dict[str, list[float]] = {}
        for row in conn.execute(
            "SELECT memory_id, ts FROM access_events "
            "WHERE kind='reinforce' ORDER BY ts"
        ):
            result.setdefault(row["memory_id"], []).append(float(row["ts"]))
        return result
    except sqlite3.Error:
        return {}
    finally:
        conn.close()


def usage_events(db_path: Path) -> dict[str, list[float]]:
    """{memory_id: [使用の ts(昇順)]} を返す。reinforce(明示の申告)に加えて
    implicit_use(浮上した記憶の内容が応答に現れた形跡=申告なしの使用)も含む。

    reinforce は申告制で漏れるため、これだけを分子にすると使用率を過小に映す
    (実測 R8.9.26: reinforce のみ 4.0% / implicit_use 込み 13.7%)。
    """
    conn = _connect_ro(db_path)
    if conn is None:
        return {}
    try:
        result: dict[str, list[float]] = {}
        for row in conn.execute(
            "SELECT memory_id, ts FROM access_events "
            "WHERE kind IN ('reinforce', 'implicit_use') ORDER BY ts"
        ):
            result.setdefault(row["memory_id"], []).append(float(row["ts"]))
        return result
    except sqlite3.Error:
        return {}
    finally:
        conn.close()


def reinforced_within(
    reinf: dict[str, list[float]], memory_id: str, ts: float, window: float
) -> bool:
    """浮上時刻 ts から window 秒以内に reinforce があったか。"""
    lst = reinf.get(memory_id)
    if not lst:
        return False
    i = bisect_right(lst, ts)
    return i < len(lst) and lst[i] - ts <= window


# ---------------------------------------------------------------------------
# 4指標の計算
# ---------------------------------------------------------------------------

def compute_metrics(
    settings: Settings,
    *,
    now: float | None = None,
    window_days: float = 7.0,
) -> dict:
    """4指標を dict で返す。DB・ログが無い項目は 0/None で埋める(落とさない)。"""
    ts_now = now if now is not None else time.time()
    window = window_days * 86400.0
    hit_window = settings.sleep_hit_window_hours * 3600.0

    # --- 1. 浮上命中率 ---
    instances = surfaced_instances(settings)
    reinf = reinforce_events(settings.db_path)
    hits_total = sum(
        1 for t, mid, _s in instances
        if reinforced_within(reinf, mid, t, hit_window)
    )
    recent = [(t, mid, s) for t, mid, s in instances if ts_now - t <= window]
    hits_recent = sum(
        1 for t, mid, _s in recent
        if reinforced_within(reinf, mid, t, hit_window)
    )
    # 実使用率: 明示の申告(reinforce)に暗黙の使用(implicit_use)を加えた分子。
    # 申告漏れの影響を受けない、実態に近い物差し
    used = usage_events(settings.db_path)
    use_recent = sum(
        1 for t, mid, _s in recent
        if reinforced_within(used, mid, t, hit_window)
    )

    def rate(h: int, n: int) -> float | None:
        return round(h / n, 4) if n else None

    # 層別(意味検索あり/なし): 2週間判定の中心の問い——意味あり浮上の
    # 命中率は字句のみを超えるか——に直接答えるための切り口
    sem_recent = [(t, mid) for t, mid, s in recent if s]
    lex_recent = [(t, mid) for t, mid, s in recent if not s]
    sem_hits = sum(1 for t, mid in sem_recent
                   if reinforced_within(reinf, mid, t, hit_window))
    lex_hits = sum(1 for t, mid in lex_recent
                   if reinforced_within(reinf, mid, t, hit_window))

    surface_metrics = {
        "instances_total": len(instances),
        "hits_total": hits_total,
        "hit_rate_total": rate(hits_total, len(instances)),
        "instances_window": len(recent),
        "hits_window": hits_recent,
        "hit_rate_window": rate(hits_recent, len(recent)),
        "use_window": {
            "hits": use_recent,
            "use_rate": rate(use_recent, len(recent)),
        },
        "semantic_window": {
            "instances": len(sem_recent),
            "hits": sem_hits,
            "hit_rate": rate(sem_hits, len(sem_recent)),
        },
        "lexical_window": {
            "instances": len(lex_recent),
            "hits": lex_hits,
            "hit_rate": rate(lex_hits, len(lex_recent)),
        },
    }

    # --- 2〜3. 書き込み量・在庫(DB) ---
    writes: dict = {
        "creates_window": 0, "per_day_window": None, "total_memories": 0,
    }
    stock: dict = {
        "by_tier": {}, "hot_total": 0, "hot_episode": 0, "hot_other": 0,
    }
    old_hot_episodes = 0
    conn = _connect_ro(settings.db_path)
    if conn is not None:
        try:
            n_creates = conn.execute(
                "SELECT COUNT(*) AS c FROM access_events "
                "WHERE kind='create' AND ts >= ?",
                (ts_now - window,),
            ).fetchone()["c"]
            writes["creates_window"] = n_creates
            writes["per_day_window"] = round(n_creates / window_days, 2)
            writes["total_memories"] = conn.execute(
                "SELECT COUNT(*) AS c FROM memories"
            ).fetchone()["c"]

            for row in conn.execute(
                "SELECT tier, COUNT(*) AS c FROM memories GROUP BY tier"
            ):
                stock["by_tier"][row["tier"]] = row["c"]
            for row in conn.execute(
                "SELECT type, COUNT(*) AS c FROM memories "
                "WHERE tier='hot' GROUP BY type"
            ):
                stock["hot_total"] += row["c"]
                if row["type"] == "episode":
                    stock["hot_episode"] += row["c"]
                else:
                    stock["hot_other"] += row["c"]

            min_age = settings.consolidate_min_age_days * 86400.0
            old_hot_episodes = conn.execute(
                "SELECT COUNT(*) AS c FROM memories "
                "WHERE tier='hot' AND type='episode' AND created_at <= ?",
                (ts_now - min_age,),
            ).fetchone()["c"]
        except sqlite3.Error:
            pass
        finally:
            conn.close()

    # --- 4. 統合の滞留 ---
    consolidation = {
        "clusters": 0,
        "skill_clusters": 0,
        "checked_at": None,
        "old_hot_episodes": old_hot_episodes,
    }
    try:
        state = json.loads(
            (settings.data_dir / "consolidation_state.json").read_text(
                encoding="utf-8"
            )
        )
        consolidation["clusters"] = int(state.get("clusters", 0) or 0)
        consolidation["skill_clusters"] = int(
            state.get("skill_clusters", 0) or 0
        )
        consolidation["checked_at"] = state.get("checked_at")
    except Exception:
        pass

    # --- 参考: 常連抑制の現況(Phase 1 の sleep が生成するファイル) ---
    suppression_active = 0
    try:
        data = json.loads(
            (settings.data_dir / "surface" / "suppression.json").read_text(
                encoding="utf-8"
            )
        )
        suppression_active = sum(
            1
            for item in (data.get("suppressed") or {}).values()
            if float(item.get("until", 0)) > ts_now
        )
    except Exception:
        pass

    return {
        "computed_at": ts_now,
        "window_days": window_days,
        "surface": surface_metrics,
        "writes": writes,
        "stock": stock,
        "consolidation": consolidation,
        "suppression": {"active": suppression_active},
    }


def format_report(metrics: dict) -> str:
    """人間可読(日本語)の短いレポート文字列を組み立てる。"""
    s = metrics["surface"]
    w = metrics["writes"]
    st = metrics["stock"]
    c = metrics["consolidation"]

    def pct(v: float | None) -> str:
        return f"{v * 100:.1f}%" if v is not None else "-"

    lines = [
        f"[engram 代謝レポート] 窓={metrics['window_days']:g}日",
        (
            f"1. 浮上命中率: 全期間 {pct(s['hit_rate_total'])} "
            f"({s['hits_total']}/{s['instances_total']}) / "
            f"直近 {pct(s['hit_rate_window'])} "
            f"({s['hits_window']}/{s['instances_window']})"
        ),
        (
            f"   実使用率(直近・申告+暗黙): {pct(s['use_window']['use_rate'])} "
            f"({s['use_window']['hits']}/{s['instances_window']})"
        ),
        (
            f"   層別(直近): 意味あり {pct(s['semantic_window']['hit_rate'])} "
            f"({s['semantic_window']['hits']}/{s['semantic_window']['instances']})"
            f" / 字句のみ {pct(s['lexical_window']['hit_rate'])} "
            f"({s['lexical_window']['hits']}/{s['lexical_window']['instances']})"
        ),
        (
            f"2. 書き込み量: 直近 {w['creates_window']}件 "
            f"({w['per_day_window'] if w['per_day_window'] is not None else '-'}"
            f"件/日) / 総記憶 {w['total_memories']}件"
        ),
        (
            f"3. hot在庫: {st['hot_total']}件 "
            f"(episode {st['hot_episode']} / その他 {st['hot_other']}) "
            f"tier内訳 {st['by_tier']}"
        ),
        (
            f"4. 統合の滞留: 候補{c['clusters']}クラスタ・"
            f"スキル化候補{c['skill_clusters']}クラスタ・"
            f"統合適齢のhot episode {c['old_hot_episodes']}件"
        ),
        f"(参考) 常連抑制中: {metrics['suppression']['active']}件",
    ]
    return "\n".join(lines)
