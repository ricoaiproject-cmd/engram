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

    args = parser.parse_args()

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
            sys.exit(run_user_prompt())

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
