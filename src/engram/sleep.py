"""眠りの配管(「代謝する脳」プラン Phase 1)。

夜間に走る記憶の保守。人間の睡眠中の記憶整理に相当する3つの仕事を、
決定的な純 Python(LLM 不要・モデル非依存)で行う:

1. 沈黙(silence)   — 使われないまま一定期間過ぎた hot 記憶を cold へ降格する。
   削除ではない: deep recall では今後も到達でき、正本 Markdown は内容不変
   (frontmatter の tier のみ書き換え)。人間の忘却の実態(痕跡の消滅ではなく
   アクセスの喪失=サイレント・エングラム)と同型。
   ユーザーの訂正・流儀(feedback / preference / user)と高 importance は対象外。
2. 常連抑制(suppress) — surface で浮上したのに reinforce されないことが続く
   記憶を、一定期間 surface から黙らせる(suppression.json)。surface 側は
   このファイルを読むだけなので、毎プロンプトのコストは増えない。
3. 棚卸し(report)  — 統合・スキル化候補の滞留と代謝指標(metabolism)を
   「眠りレポート」として書き出す。翌朝のエージェント(どのモデルでも)が読む。

要約を伴う統合(consolidation)は LLM にしかできないため、ここでは行わない。
夜間ジョブのエージェント呼び出し(モデル差し替え可能)またはナッジ経由で
エージェントが実行する。

既定は dry-run。commit=True のときだけ DB・ファイル・suppression.json を書く。
"""

from __future__ import annotations

import datetime
import json
import sqlite3
import time
from pathlib import Path

from .config import Settings
from .metabolism import (
    compute_metrics,
    format_report,
    reinforced_within,
    surfaced_instances,
    usage_events,
)

_PREVIEW_CHARS = 60


# ---------------------------------------------------------------------------
# 1. 沈黙(hot → cold 降格)
# ---------------------------------------------------------------------------

def silence_candidates(settings: Settings, *, now: float | None = None) -> list[dict]:
    """降格候補を返す。

    共通の保護: tier=hot / type が保護対象外(訂正・流儀)。
    そのうえで判定方式が2つある(settings.sleep_mode):
    - "age"      : 最終アクセスからの経過日数が閾値を超えたもの(既定・従来)。
      importance > sleep_demote_max_importance は降格しない。
    - "pressure" : hot 在庫が目標を超えたぶんだけ、放置が長い順に沈める。
      人間の忘却は経過時間ではなく干渉(新しい記憶との競合)で進むため、
      書き込み量が増えれば自動的に忘却も増えるこちらの方が同型。
      保護は別格(importance >= sleep_pressure_shield_importance)のみで、
      生後 sleep_pressure_min_age_days 未満の新参には猶予を与える。
      importance 8 まで恒久保護にすると降格可能な記憶が尽き、圧力の全量が
      新参にかかる硬直が起きる(実測 R8.9.27: hot 940件中934件が不動、
      生後1日未満の記憶だけが毎晩沈んでいた)。
    """
    ts_now = now if now is not None else time.time()
    threshold = ts_now - settings.sleep_demote_days * 86400.0
    exempt = {t.casefold() for t in settings.sleep_exempt_types}

    if not Path(settings.db_path).is_file():
        return []
    conn = sqlite3.connect(str(settings.db_path))
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA busy_timeout=5000")
        rows = conn.execute(
            "SELECT m.id, m.type, m.importance, m.room, m.path, m.created_at, "
            "       COALESCE(MAX(e.ts), m.created_at) AS last_ts "
            "FROM memories m LEFT JOIN access_events e ON e.memory_id = m.id "
            "WHERE m.tier = 'hot' GROUP BY m.id"
        ).fetchall()
    except sqlite3.Error:
        return []
    finally:
        conn.close()

    pressure = (settings.sleep_mode or "age").casefold() == "pressure"
    # 在庫圧力モードでは、保護を通った全件を放置の長い順に並べ、
    # hot 在庫が目標を超えたぶんだけ沈める(超過が無ければ何も沈めない)
    over = 0
    if pressure:
        hot_total = len(rows)
        over = max(0, hot_total - settings.sleep_hot_target)
        over = min(over, settings.sleep_pressure_max_demote)

    grace = settings.sleep_pressure_min_age_days * 86400.0
    out = []
    for r in rows:
        if (r["type"] or "").casefold() in exempt:
            continue
        imp = int(r["importance"] or 0)
        if pressure:
            if imp >= settings.sleep_pressure_shield_importance:
                continue
            if ts_now - float(r["created_at"] or 0.0) < grace:
                continue
        elif imp > settings.sleep_demote_max_importance:
            continue
        last = float(r["last_ts"] or 0.0)
        if not pressure and last > threshold:
            continue
        out.append(
            {
                "id": r["id"],
                "type": r["type"],
                "importance": r["importance"],
                "room": r["room"],
                "path": r["path"],
                "last_access": last,
                "idle_days": round((ts_now - last) / 86400.0, 1),
            }
        )
    out.sort(key=lambda c: c["idle_days"], reverse=True)
    if pressure:
        out = out[:over]
    return out


def apply_silence(settings: Settings, candidates: list[dict]) -> dict:
    """候補を cold へ降格する(DB と Markdown 正本の両方)。

    mark_consolidated と同じ2点セット(db.set_tier + store.set_tier)を、
    sqlite-vec 拡張なしの軽量経路で行う。ファイル側の失敗は集めて返す
    (次回の reindex が DB を正とし self-heal できるよう、DB 側を正で更新する)。
    """
    from .store import MarkdownStore

    demoted: list[str] = []
    file_errors: list[str] = []
    if not candidates:
        return {"demoted": demoted, "file_errors": file_errors}

    store = MarkdownStore(settings.memories_dir)
    conn = sqlite3.connect(str(settings.db_path))
    try:
        conn.execute("PRAGMA busy_timeout=5000")
        for c in candidates:
            with conn:
                conn.execute(
                    "UPDATE memories SET tier='cold' WHERE id=? AND tier='hot'",
                    (c["id"],),
                )
            demoted.append(c["id"])
            try:
                rec = store.read(Path(c["path"]))
                store.set_tier(rec, "cold")
            except Exception as exc:
                file_errors.append(f"{c['id']}: {exc!r}")
    finally:
        conn.close()
    return {"demoted": demoted, "file_errors": file_errors}


# ---------------------------------------------------------------------------
# 2. 常連抑制(suppression.json の生成)
# ---------------------------------------------------------------------------

def suppression_path(settings: Settings) -> Path:
    return settings.data_dir / "surface" / "suppression.json"


def compute_suppression(settings: Settings, *, now: float | None = None) -> dict:
    """「使われないまま連続 N 回浮上」した記憶の抑制表を作る。

    surface ログ(直近 window 日)を記憶ごとに時系列で見て、末尾の連続
    ミス(浮上後 hit_window 時間以内に使用なし)が閾値以上なら、
    最終浮上時刻 + 抑制日数まで黙らせる。使用があった時点でミスは
    リセットされる(実測: この規則でノイズ-15%・失う当たり78件中6件)。

    「使用」は reinforce(明示の申告)に加えて implicit_use(応答に内容が
    現れた形跡)を含む。reinforce のみを見ると、申告漏れの記憶が実際は
    使われているのに黙らされる(実測 R8.9.26: 抑制316件中88件=28%が誤爆)。
    """
    ts_now = now if now is not None else time.time()
    window_start = ts_now - settings.sleep_suppress_window_days * 86400.0
    hit_window = settings.sleep_hit_window_hours * 3600.0

    per_memory: dict[str, list[float]] = {}
    for t, mid, _sem in surfaced_instances(settings):
        if t >= window_start:
            per_memory.setdefault(mid, []).append(t)

    reinf = usage_events(settings.db_path)

    suppressed: dict[str, dict] = {}
    for mid, ts_list in per_memory.items():
        streak = 0
        last_surface = ts_list[-1]
        for t in ts_list:
            # hit_window がまだ満了していない直近の浮上は判定保留(ミス扱いに
            # しない)。眠りは夜に走るので、その日の浮上を早計に罰しない
            if ts_now - t < hit_window:
                continue
            if reinforced_within(reinf, mid, t, hit_window):
                streak = 0
            else:
                streak += 1
        if streak < settings.sleep_suppress_misses:
            continue
        days = settings.sleep_suppress_days
        rounds = 1
        if settings.sleep_suppress_escalate:
            # 間隔効果と同型: 外し続ける記憶ほど長く黙らせる(14→28→56日)。
            # 閾値を超えたミスの回数ぶんだけ倍にする
            extra = streak - settings.sleep_suppress_misses
            rounds = 1 + extra
            days = min(days * (2 ** extra), settings.sleep_suppress_max_days)
        until = last_surface + days * 86400.0
        if until <= ts_now:
            continue  # とうに期限切れ
        suppressed[mid] = {
            "until": until,
            "misses": streak,
            "days": days,
            "rounds": rounds,
            "last_surface": last_surface,
        }
    return {"computed_at": ts_now, "suppressed": suppressed}


def write_suppression(settings: Settings, data: dict) -> Path:
    path = suppression_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return path


# ---------------------------------------------------------------------------
# 3. 眠り本体
# ---------------------------------------------------------------------------

def _content_preview(path_str: str | None) -> str:
    """レポート用に記憶本文の先頭を安全に覗く(失敗は空文字)。"""
    if not path_str:
        return ""
    try:
        import frontmatter

        post = frontmatter.load(path_str)
        return " ".join(post.content.split())[:_PREVIEW_CHARS]
    except Exception:
        return ""


def run_sleep(
    settings: Settings, *, now: float | None = None, commit: bool = False
) -> dict:
    """眠りを1回実行する。commit=False(既定)は何も書かない dry-run。

    返り値: {"dry_run", "silence": {...}, "suppression": {...},
             "metrics": {...}, "report_path": str | None}
    """
    ts_now = now if now is not None else time.time()

    candidates = silence_candidates(settings, now=ts_now)
    suppression = compute_suppression(settings, now=ts_now)

    result: dict = {
        "dry_run": not commit,
        "silence": {
            "candidates": len(candidates),
            "demoted": [],
            "file_errors": [],
            "sample": [
                {k: c[k] for k in ("id", "type", "importance", "idle_days")}
                for c in candidates[:10]
            ],
        },
        "suppression": {
            "count": len(suppression["suppressed"]),
            "ids": sorted(suppression["suppressed"]),
        },
        "report_path": None,
    }

    if commit:
        applied = apply_silence(settings, candidates)
        result["silence"]["demoted"] = applied["demoted"]
        result["silence"]["file_errors"] = applied["file_errors"]
        write_suppression(settings, suppression)

    # 指標は降格を反映した後に測る(commit 時)
    metrics = compute_metrics(settings, now=ts_now)
    result["metrics"] = metrics

    if commit:
        result["report_path"] = str(_write_report(settings, ts_now, result,
                                                  candidates))
    return result


def _write_report(
    settings: Settings, ts_now: float, result: dict, candidates: list[dict]
) -> Path:
    """眠りレポート(Markdown)を data_dir/sleep/ へ書く。"""
    d = settings.data_dir / "sleep"
    d.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.fromtimestamp(ts_now)
    path = d / f"report-{stamp.strftime('%Y%m%d-%H%M%S')}.md"

    sil = result["silence"]
    lines = [
        f"# engram 眠りレポート {stamp.strftime('%Y-%m-%d %H:%M')}",
        "",
        "## 沈黙(hot→cold 降格)",
        (
            f"- 降格 {len(sil['demoted'])}件"
            f"(候補 {sil['candidates']}件、ファイル側エラー "
            f"{len(sil['file_errors'])}件)"
        ),
    ]
    for c in candidates[:20]:
        preview = _content_preview(c.get("path"))
        lines.append(
            f"  - {c['id']} [{c['type']}/imp{c['importance']}] "
            f"{c['idle_days']}日未使用 {preview}"
        )
    if len(candidates) > 20:
        lines.append(f"  - …ほか {len(candidates) - 20} 件(全リストは --json で)")
    if sil["file_errors"]:
        lines.append("- ファイル側エラー(DB は cold 済み・reindex で自己修復可):")
        lines.extend(f"  - {e}" for e in sil["file_errors"][:10])

    sup = result["suppression"]
    lines += [
        "",
        "## 常連抑制(使われないまま浮上を繰り返す記憶。使用=申告+暗黙)",
        f"- 抑制 {sup['count']}件: " + (", ".join(sup["ids"]) if sup["ids"] else "なし"),
        "",
        "## 代謝指標",
        format_report(result["metrics"]),
        "",
        "## 翌朝のエージェントへ",
        (
            "- 統合候補が溜まっていれば consolidation_candidates → 要約を "
            "remember(type=knowledge) → mark_consolidated の順で消化すること"
            "(これは LLM にしかできない仕事)。"
        ),
        "- スキル化候補はユーザーへの提案まで(勝手に作らない)。",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
