"""動作確認用 CLI(担当: Agent B)。

argparse のサブコマンドで engine を直接叩く。出力は人間可読(recall はスコア
内訳を表形式)+ --json でそのまま JSON。

    engram remember "本文" --type knowledge --importance 7 --tags a,b
    engram recall "クエリ" [--deep] [--limit 5] [--type knowledge] [--no-record]
    engram reinforce ID [ID...] [--strength 2.0]
    engram correct ID --content "正しい内容" --reason "理由"
    engram forget ID / engram link SRC DST
    engram stats / engram reindex
    engram consolidation-candidates
    engram mark-consolidated NEW_ID --episodes ID1,ID2,...  統合完了を記録する
    engram skill-candidates              スキル化候補の episode クラスタを表示する
    engram surface "発話" [--room X]     自発的想起の手動確認(何も書き込まない)
    engram daemon [--status|--stop]      埋め込み常駐デーモン(意味の網を毎発話で使う)
    engram metabolism [--days 7]         代謝レポート(命中率・書込量・在庫・滞留)
    engram sleep [--commit]              眠り(沈黙降格・常連抑制・棚卸し)。既定は dry-run
    engram hook session-end|user-prompt  エージェントのフック用入口(stdin JSON)
    engram export-onnx [--force]         埋め込みモデルの ONNX 化(起動高速化)

--fake-embedder フラグで FakeEmbedder を使う(モデル未導入環境での試験用)。
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def _build_engine(fake_embedder: bool = False):
    """エンジンを構築する。--fake-embedder フラグに従い埋め込みを選択。"""
    from .config import get_settings
    from .engine import build_engine
    from .embedder import FakeEmbedder

    settings = get_settings()
    embedder = FakeEmbedder() if fake_embedder else None
    return build_engine(settings, embedder=embedder)


class _LazyEngine:
    """初めて使われたときにエンジンを組む(起きない DMN ティックで埋め込みモデルを起こさない)。"""

    def __init__(self, factory):
        self._factory = factory
        self._engine = None

    def __getattr__(self, name):
        if self._engine is None:
            self._engine = self._factory()
        return getattr(self._engine, name)


def _print_recall(result: dict, as_json: bool = False) -> None:
    """recall 結果を整形して表示。"""
    if as_json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    mode = result.get("mode", "?")
    auto_deepened = result.get("auto_deepened", False)
    hits = result.get("hits", [])

    header = f"[recall: mode={mode}"
    if auto_deepened:
        header += ", auto_deepened=True"
    header += f", {len(hits)} hits]"
    print(header)
    print()

    if not hits:
        print("  (結果なし)")
        return

    # カラム幅の計算
    col_id = 26
    col_score = 7
    col_rel = 7
    col_act = 7
    col_imp = 5
    col_via = 12
    col_tier = 12

    # ヘッダー行
    print(
        f"{'ID':<{col_id}} "
        f"{'score':>{col_score}} "
        f"{'relevance':>{col_rel}} "
        f"{'activation':>{col_act}} "
        f"{'importance':>{col_imp}} "
        f"{'via':<{col_via}} "
        f"{'tier':<{col_tier}}"
    )
    print("-" * (col_id + col_score + col_rel + col_act + col_imp + col_via + col_tier + 6))

    for hit in hits:
        print(
            f"{hit['id']:<{col_id}} "
            f"{hit['score']:>{col_score}.4f} "
            f"{hit['relevance']:>{col_rel}.4f} "
            f"{hit['activation']:>{col_act}.4f} "
            f"{hit['importance']:>{col_imp}.2f} "
            f"{hit['via']:<{col_via}} "
            f"{hit['tier']:<{col_tier}}"
        )
        # content のプレビュー(先頭80文字)
        content_preview = hit.get("content", "").replace("\n", " ")[:80]
        if content_preview:
            print(f"  > {content_preview}")
        # note がある場合
        if hit.get("note"):
            print(f"  [note] {hit['note']}")
        print()


def _print_result(result: dict, as_json: bool = False) -> None:
    """汎用 dict 出力。"""
    if as_json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        for k, v in result.items():
            print(f"  {k}: {v}")


def main() -> None:
    """CLI エントリーポイント。"""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        prog="engram",
        description="engram 記憶エンジン CLI",
    )
    parser.add_argument(
        "--fake-embedder",
        action="store_true",
        help="FakeEmbedder を使う(モデル未導入環境での試験用)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="JSON 出力",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- remember ---
    p_remember = subparsers.add_parser("remember", help="記憶を保存する")
    p_remember.add_argument("content", help="記憶の本文")
    p_remember.add_argument(
        "--type",
        default="knowledge",
        choices=["knowledge", "preference", "project", "episode"],
        help="記憶のタイプ(デフォルト: knowledge)",
    )
    p_remember.add_argument(
        "--importance",
        type=int,
        default=5,
        help="重要度 1-10(デフォルト: 5)",
    )
    p_remember.add_argument(
        "--tags",
        default="",
        help="カンマ区切りのタグ",
    )
    p_remember.add_argument(
        "--source",
        default="cli",
        help="記憶の出所",
    )
    p_remember.add_argument(
        "--related",
        dest="related_ids",
        default="",
        help="関連記憶 ID のカンマ区切り",
    )
    p_remember.add_argument(
        "--room",
        default=None,
        help="記憶の部屋(省略時は現在のディレクトリから自動判定)",
    )
    p_remember.add_argument(
        "--valence",
        type=float,
        default=0.0,
        help="感情価 -1..1(負=避けろ/失敗、正=近づけ/成功。省略時 0)",
    )

    # --- recall ---
    p_recall = subparsers.add_parser("recall", help="記憶を検索する")
    p_recall.add_argument("query", help="検索クエリ")
    p_recall.add_argument(
        "--deep",
        action="store_true",
        help="deep モードで検索(cold/superseded/episode も含む)",
    )
    p_recall.add_argument(
        "--limit",
        type=int,
        default=5,
        help="最大件数(デフォルト: 5)",
    )
    p_recall.add_argument(
        "--type",
        default=None,
        help="タイプでフィルタ",
    )
    p_recall.add_argument(
        "--no-record",
        action="store_true",
        help="recall_hit イベントを記録しない",
    )
    p_recall.add_argument(
        "--room",
        default=None,
        help='検索する部屋(省略時は現在のディレクトリから自動判定。"*" で全部屋)',
    )

    # --- reinforce ---
    p_read = subparsers.add_parser("read", help="id で記憶の本文を読む(縁の想起から引く)")
    p_read.add_argument("ids", nargs="+", help="記憶 ID のリスト")
    p_read.add_argument("--no-record", action="store_true", help="使用記録を付けない")

    p_reinforce = subparsers.add_parser("reinforce", help="記憶の使用を報告する")
    p_reinforce.add_argument("ids", nargs="+", help="強化する記憶 ID のリスト")
    p_reinforce.add_argument(
        "--strength",
        type=float,
        default=1.0,
        help="強化の強さ 0.1-3.0(デフォルト: 1.0)",
    )

    # --- correct ---
    p_correct = subparsers.add_parser("correct", help="記憶を訂正する")
    p_correct.add_argument("id", help="訂正する記憶 ID")
    p_correct.add_argument(
        "--content",
        required=True,
        help="正しい内容",
    )
    p_correct.add_argument(
        "--reason",
        required=True,
        help="訂正の理由",
    )
    p_correct.add_argument(
        "--source",
        default="cli",
        help="訂正の出所",
    )

    # --- forget ---
    p_forget = subparsers.add_parser("forget", help="記憶をゴミ箱へ移動する")
    p_forget.add_argument("id", help="削除する記憶 ID")

    # --- link ---
    p_link = subparsers.add_parser("link", help="2つの記憶をリンクする")
    p_link.add_argument("src", help="リンク元の記憶 ID")
    p_link.add_argument("dst", help="リンク先の記憶 ID")

    # --- stats ---
    subparsers.add_parser("stats", help="統計情報を表示する")

    # --- reindex ---
    subparsers.add_parser("reindex", help="DB を Markdown から再構築する")

    # --- consolidation-candidates ---
    subparsers.add_parser(
        "consolidation-candidates",
        help="統合候補の episode クラスタを表示する",
    )

    # --- mark-consolidated ---
    p_mark_consolidated = subparsers.add_parser(
        "mark-consolidated",
        help="統合完了を記録する(episode→new_memory_id のリンク+cold降格)",
    )
    p_mark_consolidated.add_argument(
        "new_memory_id", help="統合先(新規作成済み)の記憶 ID"
    )
    p_mark_consolidated.add_argument(
        "--episodes",
        required=True,
        help="統合元 episode ID のカンマ区切り",
    )

    # --- skill-candidates ---
    subparsers.add_parser(
        "skill-candidates",
        help="スキル化候補の episode クラスタを表示する",
    )

    # --- setup ---
    p_setup = subparsers.add_parser("setup", help="セットアップウィザードを実行する")
    p_setup.add_argument(
        "--non-interactive",
        action="store_true",
        help="質問せず既定値でセットアップ",
    )
    p_setup.add_argument(
        "--memories-dir",
        default=None,
        help="記憶フォルダのパス(省略時は ~/.engram/memories)",
    )
    p_setup.add_argument(
        "--agents",
        default=None,
        metavar="AGENTS",
        help="登録するエージェントをカンマ区切りで指定(例: claude,codex)。省略時は検出された全部",
    )
    p_setup.add_argument(
        "--no-schedule",
        action="store_true",
        help="夜の整理(毎晩)と DMN(15分毎)の自動実行を登録しない(設定に schedule_night/schedule_dmn=false を残す)",
    )

    # --- doctor ---
    subparsers.add_parser("doctor", help="環境診断を表示する")

    # --- export-onnx ---
    p_export = subparsers.add_parser(
        "export-onnx",
        help="埋め込みモデルを ONNX 化して起動を高速化する(一度だけ実行)",
    )
    p_export.add_argument(
        "--force",
        action="store_true",
        help="既存の ONNX モデルを上書きする",
    )

    # --- surface ---
    p_surface = subparsers.add_parser(
        "surface",
        help="自発的想起の手動確認(ログ・状態は書き込まない)",
    )
    p_surface.add_argument("query", help="発話(プロンプト)に相当するテキスト")
    p_surface.add_argument(
        "--room",
        default=None,
        help="部屋(省略時は現在のディレクトリから自動判定)",
    )

    # --- daemon ---
    p_daemon = subparsers.add_parser(
        "daemon",
        help="埋め込み常駐デーモン(意味の網を毎発話で使えるようにする)",
    )
    p_daemon.add_argument(
        "--status", action="store_true", help="生きているか確認する"
    )
    p_daemon.add_argument(
        "--stop", action="store_true", help="終了させる"
    )

    # --- metabolism ---
    p_meta = subparsers.add_parser(
        "metabolism",
        help="代謝レポート(浮上命中率・書き込み量・在庫・統合滞留)",
    )
    p_meta.add_argument(
        "--days", type=float, default=7.0, help="直近窓の日数(既定7)"
    )

    # --- sleep ---
    p_sleep = subparsers.add_parser(
        "sleep",
        help="眠り(夜間保守): 沈黙降格・常連抑制・棚卸し。既定は dry-run",
    )
    p_sleep.add_argument(
        "--commit",
        action="store_true",
        help="実際に書き込む(省略時は何も変更しない dry-run)",
    )

    # --- night(眠り一式) ---
    p_night = subparsers.add_parser(
        "night",
        help="夜間の眠り一式: 統合(LLM)→沈黙→蒸留→追加処理(night.d)。--install で毎晩の自動実行を登録",
    )
    p_night.add_argument("--dry-run", action="store_true", help="何も書かず段取りだけ表示")
    p_night.add_argument("--install", action="store_true", help="OSスケジューラに毎晩の実行を登録")
    p_night.add_argument("--uninstall", action="store_true", help="登録を解除")
    p_night.add_argument("--time", default="02:30", help="--install の実行時刻 HH:MM(既定 02:30)")
    p_night.add_argument("--home", default=None, help="扱う engram のホーム(スケジューラ登録が付ける)")
    p_night.add_argument("--force-schedule", dest="force_schedule", action="store_true",
                         help="同じ名前の登録が別のホームや別の仕組みのものでも、上書き・削除する")
    p_night.add_argument("--allow", action="append", choices=["claude", "codex"], default=[],
                         help="夜の整理の要約にこの CLI を使うことに同意する(利用枠を使い、記憶の文章を送る)")
    p_night.add_argument("--deny", action="append", choices=["claude", "codex"], default=[],
                         help="夜の整理でこの CLI を使わない")

    # --- dmn(自発思考) ---
    p_dmn = subparsers.add_parser(
        "dmn",
        help="DMN: 入力の無い時間に世界の差分を見て、ふと考えたことを日記と記憶に残す(1ティック)",
    )
    p_dmn.add_argument("--force", action="store_true", help="理由や位相に関係なく起きる")
    p_dmn.add_argument("--dry-run", action="store_true", help="思考を表示するだけで書かない")
    p_dmn.add_argument("--install", action="store_true", help="OSスケジューラに15分毎の実行を登録")
    p_dmn.add_argument("--uninstall", action="store_true", help="登録を解除")
    p_dmn.add_argument("--every", type=int, default=15, help="--install の間隔(分・既定15)")
    p_dmn.add_argument("--home", default=None, help="扱う engram のホーム(スケジューラ登録が付ける)")
    p_dmn.add_argument("--force-schedule", dest="force_schedule", action="store_true",
                       help="同じ名前の登録が別のホームや別の仕組みのものでも、上書き・削除する")

    # --- hyp-run(仮説サイクルを1回回す。毎日の自動実行の登録もここ) ---
    p_hyp = subparsers.add_parser(
        "hyp-run",
        help="仮説サイクルを1回回す(選んだ CLI で。Web で調べものをし、記憶の一部を送る)",
    )
    p_hyp.add_argument("--use", choices=["claude", "codex"], default=None,
                       help="この CLI で毎日回すことに同意し、毎日 7:30 の自動実行を登録する")
    p_hyp.add_argument("--stop", action="store_true", help="回さない(自動実行も外す)")
    p_hyp.add_argument("--home", default=None, help="扱う engram のホーム(スケジューラ登録が付ける)")

    # --- rsi(ループ回転計) ---
    p_rsi = subparsers.add_parser(
        "rsi",
        help="RSIループ回転計: 自律反復/日・提案採用率・DMN朝送り率",
    )
    p_rsi.add_argument("--days", type=float, default=7.0, help="直近窓の日数(既定7)")

    # --- hyp(仮説サイクル: 種選び・書く・一覧・判定) ---
    p_hyp = subparsers.add_parser("hyp", help="仮説サイクル(記憶の組から仮説を立て、確かめ、また覚える)")
    hyp_sub = p_hyp.add_subparsers(dest="hyp_command", required=True)
    ph = hyp_sub.add_parser("seeds", help="仮説の種になる記憶の組を表示")
    ph.add_argument("--n", type=int, default=3)
    ph = hyp_sub.add_parser("add", help="仮説を書く")
    ph.add_argument("claim")
    ph.add_argument("--sources", required=True, help="元にした記憶 ID のカンマ区切り(2件以上)")
    ph.add_argument("--falsify", required=True, help="外れたと分かる条件")
    ph.add_argument("--test", default="", help="確かめ方")
    ph.add_argument("--due", default=None, help="期限 YYYY-MM-DD または日数")
    ph.add_argument("--source", default="cli")
    ph.add_argument("--origin", default="")
    ph = hyp_sub.add_parser("list", help="仮説の一覧")
    ph.add_argument("--status", default="open", choices=["open", "due", "resolved", "all"])
    ph.add_argument("--limit", type=int, default=20)
    ph = hyp_sub.add_parser("resolve", help="仮説に判定を付ける")
    ph.add_argument("id")
    ph.add_argument("verdict", help="当たり / 外れ / 判定不能")
    ph.add_argument("--evidence", required=True)
    ph.add_argument("--urls", default="", help="出典 URL のカンマ区切り")
    ph.add_argument("--evidence-ids", default="")
    ph.add_argument("--note", default="")
    ph.add_argument("--judge", default="model")
    ph.add_argument("--source", default="cli")
    hyp_sub.add_parser("stats", help="答え合わせ率・当たり率")

    # --- media(媒体の棚: 画像・音声・動画・文書。索引係は別 venv) ---
    p_media = subparsers.add_parser("media", help="媒体の棚(画像・音声・動画・文書を言葉で探す)")
    media_sub = p_media.add_subparsers(dest="media_command", required=True)
    media_sub.add_parser("status", help="索引係の稼働・件数・対象フォルダ")
    pm = media_sub.add_parser("find", help="言葉で探す")
    pm.add_argument("query")
    pm.add_argument("--k", type=int, default=5)
    pm.add_argument("--kind", action="append", help="image/video/audio/document(複数可)")
    pm.add_argument("--room", default=None, help='部屋(省略時は現在のディレクトリから。"*" で全部屋)')
    pm.add_argument("--deep", action="store_true", help="cold も含める")
    pm.add_argument("--no-record", action="store_true")
    pm = media_sub.add_parser("index", help="対象フォルダの差分を索引化(索引係に頼む)")
    pm.add_argument("--full", action="store_true")
    pm.add_argument("--wait", action="store_true", help="終わるまで待って結果を表示")
    pm = media_sub.add_parser("remember", help="ファイル1件を今すぐ覚える")
    pm.add_argument("path")
    pm.add_argument("--note", default=None)
    pm.add_argument("--room", default=None)
    pm = media_sub.add_parser("daemon", help="索引係の起動/状態/停止")
    pm.add_argument("--status", action="store_true")
    pm.add_argument("--stop", action="store_true")
    pm = media_sub.add_parser("setup", help="索引係の venv を作り engram[media] を入れる(数分・数GB)")
    pm.add_argument("--source", default="git+https://github.com/ricoaiproject-cmd/engram.git",
                    help="pip が解釈する入手先(ローカルのソースフォルダ可)")
    pm.add_argument("--python", default=None, help="venv の元にする Python(既定: 今の Python)")
    pm.add_argument("--upgrade", action="store_true")
    media_sub.add_parser("install", help="夜間処理(night.d)に差分索引を配置する")
    pm = media_sub.add_parser("uninstall", help="撤去(既定は停止+夜間処理の削除のみ。記憶本体は無変更)")
    pm.add_argument("--purge-index", action="store_true", help="媒体の表4つも落とす")
    pm.add_argument("--purge-env", action="store_true", help="venv・モデル・索引フォルダも消す")
    pm.add_argument("--yes", action="store_true")
    media_sub.add_parser("nightly", help="夜間処理を今すぐ実行(差分索引→cold 降格)")

    # --- hook ---
    p_hook = subparsers.add_parser(
        "hook",
        help="エージェントのフックから呼ばれる入口(stdin の JSON を読む)",
    )
    p_hook.add_argument(
        "event",
        choices=["session-end", "user-prompt"],
        help="フックイベント名",
    )
    p_hook.add_argument(
        "--event-name", default="UserPromptSubmit",
        help="応答の hookEventName(Gemini CLI は BeforeAgent)",
    )

    args = parser.parse_args()

    # スケジューラから起動された night / dmn は環境を引き継がないため、登録時のホームを
    # --home で受け取る(設定の読み込みより先に効かせる)
    if getattr(args, "home", None):
        import os
        os.environ["ENGRAM_HOME"] = args.home

    # setup / doctor はエンジン構築なしで動く
    if args.command == "setup":
        from .setup import parse_agents, setup_main
        memories_dir = None
        if args.memories_dir:
            from pathlib import Path
            memories_dir = Path(args.memories_dir)
        selected_agents = None
        if args.agents is not None:
            try:
                selected_agents = parse_agents(args.agents)
            except ValueError as e:
                print(f"エラー: {e}", file=sys.stderr)
                sys.exit(1)
        setup_main(
            memories_dir=memories_dir,
            non_interactive=args.non_interactive,
            agents=selected_agents,
            schedule=not args.no_schedule,
        )
        return

    if args.command == "doctor":
        from .setup import doctor_main
        doctor_main()
        return

    if args.command == "export-onnx":
        from .config import get_settings
        from .onnx_export import export_onnx

        try:
            report = export_onnx(get_settings(), force=args.force)
        except (ImportError, FileExistsError, RuntimeError) as e:
            print(f"エラー: {e}", file=sys.stderr)
            sys.exit(1)
        if args.as_json:
            print(json.dumps(report, ensure_ascii=False, indent=2))
        else:
            print(f"ONNX モデルを生成しました: {report['target']}")
            print(f"  モデル: {report['model']} (dim={report['dim']}, "
                  f"{report['onnx_size_mb']} MB)")
            print(f"  パリティ: min cosine = {report['min_cosine']:.6f} "
                  f"(torch 経路と一致)")
            print("  以後の起動は ONNX 経路(embed_backend=auto)で軽くなります")
        return

    # hook / surface はエンジン構築なし(高速経路)で動く
    if args.command == "hook":
        from .hooks import run_session_end, run_user_prompt
        if args.event == "session-end":
            sys.exit(run_session_end())
        else:
            sys.exit(run_user_prompt(event_name=args.event_name))

    if args.command == "surface":
        from pathlib import Path

        from .config import get_settings, resolve_room
        from .surface import run_surface

        settings = get_settings()
        room = args.room or resolve_room(Path.cwd(), settings.room_paths)
        result = run_surface(
            args.query, settings=settings, room=room, mode="shadow",
            dry_run=True,
        )
        if args.as_json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(f"[surface: room={room}, "
                  f"threshold={settings.surface_threshold}]")
            print()
            if not result["candidates"]:
                print("  (候補なし)")
            for c in result["candidates"]:
                mark = "◎ 浮上" if c["id"] in result["surfaced"] else "  沈黙"
                print(f"{mark}  {c['id']}  score={c['score']:.4f} "
                      f"(rel={c['relevance']:.4f} act={c['activation']:.4f} "
                      f"imp={c['importance']}) [{c['type']}/{c['room']}]")
                preview = " ".join(c["content"].split())[:80]
                print(f"        > {preview}")
        return

    if args.command == "daemon":
        from .config import get_settings
        from . import daemon as daemon_mod

        settings = get_settings()
        if args.status:
            alive = daemon_mod.is_alive(settings)
            if args.as_json:
                print(json.dumps({"alive": alive}))
            else:
                print("稼働中" if alive else "停止中")
            return
        if args.stop:
            ok = daemon_mod.stop(settings)
            print("停止しました" if ok else "稼働していません")
            return
        sys.exit(daemon_mod.serve(settings))

    if args.command == "metabolism":
        from .config import get_settings
        from .metabolism import compute_metrics, format_report

        metrics = compute_metrics(get_settings(), window_days=args.days)
        if args.as_json:
            print(json.dumps(metrics, ensure_ascii=False, indent=2))
        else:
            print(format_report(metrics))
        return

    if args.command == "sleep":
        from .config import get_settings
        from .metabolism import format_report
        from .sleep import run_sleep

        result = run_sleep(get_settings(), commit=args.commit)
        if args.as_json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            mode = "commit" if args.commit else "dry-run(書き込みなし)"
            sil = result["silence"]
            print(f"[engram sleep: {mode}]")
            line = f"  沈黙候補: {sil['candidates']}件"
            if args.commit:
                line += (f" → 降格 {len(sil['demoted'])}件"
                         f"(ファイル側エラー {len(sil['file_errors'])}件)")
            print(line)
            for c in sil["sample"]:
                print(f"    - {c['id']} [{c['type']}/imp{c['importance']}] "
                      f"{c['idle_days']}日未使用")
            print(f"  常連抑制: {result['suppression']['count']}件")
            if result.get("report_path"):
                print(f"  レポート: {result['report_path']}")
            print()
            print(format_report(result["metrics"]))
        return

    if args.command == "rsi":
        from .config import get_settings
        from .rsi import compute, format_report as fmt_rsi

        m = compute(get_settings(), window_days=args.days)
        print(json.dumps(m, ensure_ascii=False, indent=2) if args.as_json else fmt_rsi(m))
        return

    if args.command == "hyp":
        from . import hypothesis as _h

        def _csv(v):
            return [x.strip() for x in (v or "").split(",") if x.strip()]

        hc = args.hyp_command
        if hc == "stats":
            from .config import get_settings
            res = _h.ledger_stats(get_settings())
        else:
            engine = _build_engine(fake_embedder=args.fake_embedder)
            if hc == "seeds":
                res = _h.hypothesis_seeds(engine, n=args.n)
            elif hc == "add":
                res = _h.hypothesize(engine, args.claim, _csv(args.sources), args.falsify, args.test,
                                     due=args.due, source=args.source, origin=args.origin)
            elif hc == "list":
                res = _h.list_hypotheses(engine, status=args.status, limit=args.limit)
            else:
                res = _h.resolve(engine, args.id, args.verdict, args.evidence,
                                 evidence_urls=_csv(args.urls), evidence_ids=_csv(args.evidence_ids),
                                 note=args.note, judge=args.judge, source=args.source)
        print(json.dumps(res, ensure_ascii=False, indent=2, default=str))
        ok = res.get("status") not in ("rejected", "not_found")
        sys.exit(0 if ok else 1)

    if args.command == "media":
        from pathlib import Path

        from .config import get_settings, resolve_room
        from . import media as _media

        settings = get_settings()
        mc = args.media_command
        if mc == "status":
            s = _media.status(settings)
            print(json.dumps(s, ensure_ascii=False, indent=2, default=str) if args.as_json else _media.format_status(s))
            return
        if mc == "find":
            room = args.room or resolve_room(Path.cwd(), settings.room_paths)
            rooms = None if room == "*" else sorted({room, "common"})
            res = _media.find(settings, args.query, k=args.k, kinds=args.kind, rooms=rooms,
                              tiers=None if args.deep else ["hot"], record_hits=not args.no_record)
            if args.as_json:
                print(json.dumps(res, ensure_ascii=False, indent=2, default=str))
            elif res.get("error"):
                print(f"エラー: {res['error']}")
                sys.exit(1)
            else:
                for h in res["hits"]:
                    print(f"{h['score']:.3f} cos={h['cosine']:.3f} [{h['kind']}/{h['room']}] {h['path']}"
                          + (f" p.{h['page']}" if h.get("page") else "")
                          + (f"  — {h['caption']}" if h.get("caption") else "")
                          + (f"  ({h['note']})" if h.get("note") else ""))
            return
        if mc == "index":
            res = _media.start_index(settings, full=args.full)
            if res.get("ok") and args.wait:
                res = {"started": True, "progress": _media.wait_index(settings, timeout=settings.night_agent_timeout_sec)}
            print(json.dumps(res, ensure_ascii=False, indent=2, default=str))
            sys.exit(0 if res.get("ok") or res.get("started") else 1)
        if mc == "remember":
            res = _media.index_path(settings, args.path, room=args.room, source="manual",
                                    source_meta={"note": args.note} if args.note else None)
            print(json.dumps(res, ensure_ascii=False, indent=2, default=str))
            sys.exit(0 if res.get("ok") else 1)
        if mc == "daemon":
            if args.stop:
                print("停止" if _media.stop(settings) else "索引係は動いていません")
            elif args.status:
                print("稼働中" if _media.is_alive(settings) else "停止")
            else:
                ok = _media.ensure_running(settings, wait=120.0)
                print("稼働中" if ok else "起動できませんでした(media.log を参照)")
                sys.exit(0 if ok else 1)
            return
        if mc == "setup":
            # 1行でオンにする: 環境づくり → 有効化 → 対象フォルダ(未設定なら既定) → 夜間の差分索引
            res = _media.setup_env(settings, source=args.source, python=args.python, upgrade=args.upgrade)
            if res.get("ok"):
                res["enabled"] = _media.enable(settings)
            print(json.dumps(res, ensure_ascii=False, indent=2))
            sys.exit(0 if res.get("ok") else 1)
        if mc == "install":
            p = _media.install_plugin(settings)
            print(f"夜間処理に配置: {p}(次回の `engram night` から差分索引が走る)")
            return
        if mc == "uninstall":
            if (args.purge_index or args.purge_env) and not args.yes:
                print("索引や環境を消す撤去です。--yes を付けて実行してください(記憶本体は無変更)。")
                sys.exit(2)
            res = _media.uninstall(settings, purge_index=args.purge_index, purge_env=args.purge_env)
            from .config import config_path
            from .setup import set_config_scalar
            set_config_scalar(config_path(), "media_enabled", False)
            print("\n".join(res["done"] + ["設定を media_enabled = false にした"]))
            return
        if mc == "nightly":
            res = _media.run_nightly(settings)
            print(json.dumps(res, ensure_ascii=False, indent=2, default=str))
            sys.exit(0 if res.get("ok") else 1)

    if args.command == "hyp-run":
        from .config import config_path, get_settings
        from . import hyp_run, scheduler
        from .setup import set_config_scalar

        settings = get_settings()
        sfx = settings.schedule_task_suffix
        if args.use or args.stop:
            set_config_scalar(config_path(), "hyp_agent", args.use or "none")
            if args.use:
                ok, msg = scheduler.install("hyp", home=settings.data_dir, suffix=sfx)
                print(f"仮説サイクルを毎日 7:30 に {args.use} で回します" if ok else f"登録に失敗しました: {msg}")
            else:
                ok, msg = scheduler.uninstall("hyp", suffix=sfx, home=settings.data_dir)
                print("仮説サイクルを回さないようにしました" + ("" if ok else f"(自動実行は外せませんでした: {msg})"))
            sys.exit(0 if ok else 1)
        res = hyp_run.run(settings)
        print(json.dumps(res, ensure_ascii=False))
        sys.exit(0)

    if args.command in ("night", "dmn"):
        from .config import get_settings
        from . import scheduler

        settings = get_settings()
        if args.command == "night" and (args.allow or args.deny):
            from .config import config_path
            from .night import CLOUD_AGENTS, CLOUD_VENDORS
            from .setup import set_config_scalar

            for key in args.allow:
                set_config_scalar(config_path(), f"night_allow_{key}", True)
                print(f"夜の整理の要約に {CLOUD_AGENTS[key]} を使います"
                      f"(利用枠を毎晩少し使い、記憶の文章を {CLOUD_VENDORS[key]} へ送ります)")
            for key in args.deny:
                set_config_scalar(config_path(), f"night_allow_{key}", False)
                print(f"夜の整理で {CLOUD_AGENTS[key]} を使いません")
            return
        if args.install or args.uninstall:
            from .config import config_path
            from .setup import set_config_scalar

            sfx = settings.schedule_task_suffix
            # 別のホームや別の仕組みの登録は、--force が無ければ上書きも削除もしない
            if args.uninstall:
                ok, msg = scheduler.uninstall(args.command, suffix=sfx, home=settings.data_dir,
                                              force=args.force_schedule)
            elif args.command == "night":
                ok, msg = scheduler.install("night", night_time=args.time, home=settings.data_dir,
                                            suffix=sfx, force=args.force_schedule)
            else:
                ok, msg = scheduler.install("dmn", dmn_minutes=args.every, home=settings.data_dir,
                                            suffix=sfx, force=args.force_schedule)
            # 止めた/戻した意思を設定に残す(setup の再実行で勝手に戻さないため)
            if ok:
                set_config_scalar(config_path(), f"schedule_{args.command}", bool(args.install))
            print(("登録" if args.install else "解除") + ("完了: " if ok else "失敗: ") + msg)
            sys.exit(0 if ok else 1)

        # エンジンは実際に必要になったときだけ組む(dry-run や「起きない」ティックで
        # 埋め込みモデルを起こさない)
        lazy = _LazyEngine(lambda: _build_engine(fake_embedder=args.fake_embedder))
        if args.command == "night":
            from .night import format_night, run_night
            result = run_night(settings, engine=lazy, dry_run=args.dry_run)
            print(json.dumps(result, ensure_ascii=False, indent=2) if args.as_json else format_night(result))
            return
        from .dmn import run_dmn
        result = run_dmn(settings, engine=lazy, force=args.force, dry_run=args.dry_run)
        if args.as_json or result.get("result") in ("dry-run", "wrote"):
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(f"[engram dmn] fired={result['fired']} why={result['why']} reasons={result['reasons']}")
        sys.exit(result.get("exit", 0))

    # エンジン構築
    engine = _build_engine(fake_embedder=args.fake_embedder)

    if args.command == "remember":
        tags = [t.strip() for t in args.tags.split(",") if t.strip()] if args.tags else []
        related_ids = (
            [r.strip() for r in args.related_ids.split(",") if r.strip()]
            if args.related_ids
            else None
        )
        from pathlib import Path

        from .config import resolve_room
        room = args.room or resolve_room(
            Path.cwd(), engine.settings.room_paths
        )
        result = engine.remember(
            content=args.content,
            type=args.type,
            importance=args.importance,
            tags=tags,
            source=args.source,
            related_ids=related_ids,
            room=room,
            valence=args.valence,
        )
        _print_result(result, as_json=args.as_json)

    elif args.command == "recall":
        mode = "deep" if args.deep else "fast"
        from pathlib import Path

        from .config import resolve_room
        room = args.room or resolve_room(
            Path.cwd(), engine.settings.room_paths
        )
        result = engine.recall(
            query=args.query,
            mode=mode,
            limit=args.limit,
            type=args.type,
            record_hits=not args.no_record,
            room=room,
        )
        _print_recall(result, as_json=args.as_json)

    elif args.command == "read":
        result = engine.read(ids=args.ids, record=not args.no_record)
        if args.as_json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            for m in result["memories"]:
                print(f"[{m['id']}] ({m['type']}/{m['tier']}/{m['room']})")
                print(m["content"])
                print()
            if result["missing"]:
                print("見つからない id:", ", ".join(result["missing"]))

    elif args.command == "reinforce":
        result = engine.reinforce(ids=args.ids, strength=args.strength)
        _print_result(result, as_json=args.as_json)

    elif args.command == "correct":
        result = engine.correct(
            id=args.id,
            corrected_content=args.content,
            reason=args.reason,
            source=args.source,
        )
        _print_result(result, as_json=args.as_json)

    elif args.command == "forget":
        result = engine.forget(id=args.id)
        _print_result(result, as_json=args.as_json)

    elif args.command == "link":
        result = engine.link(src=args.src, dst=args.dst)
        _print_result(result, as_json=args.as_json)

    elif args.command == "stats":
        result = engine.stats()
        _print_result(result, as_json=args.as_json)

    elif args.command == "reindex":
        result = engine.reindex()
        _print_result(result, as_json=args.as_json)

    elif args.command == "consolidation-candidates":
        result = engine.consolidation_candidates()
        if args.as_json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            clusters = result.get("clusters", [])
            print(f"統合候補クラスタ: {len(clusters)} 件")
            for i, cluster in enumerate(clusters, 1):
                print(f"\nクラスタ {i} ({len(cluster['ids'])} 件):")
                for id_, content in zip(cluster["ids"], cluster.get("contents", [])):
                    preview = content.replace("\n", " ")[:60] if content else "(内容なし)"
                    print(f"  [{id_}] {preview}")

    elif args.command == "mark-consolidated":
        episode_ids = [e.strip() for e in args.episodes.split(",") if e.strip()]
        result = engine.mark_consolidated(episode_ids, args.new_memory_id)
        _print_result(result, as_json=args.as_json)

    elif args.command == "skill-candidates":
        result = engine.skill_candidates()
        if args.as_json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            clusters = result.get("clusters", [])
            print(f"スキル化候補クラスタ: {len(clusters)} 件")
            for i, cluster in enumerate(clusters, 1):
                print(f"\nクラスタ {i} ({len(cluster['ids'])} 件):")
                for id_, content in zip(cluster["ids"], cluster.get("contents", [])):
                    preview = content.replace("\n", " ")[:60] if content else "(内容なし)"
                    print(f"  [{id_}] {preview}")


if __name__ == "__main__":
    main()
