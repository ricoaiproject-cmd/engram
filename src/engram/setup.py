"""セットアップウィザード & doctor コマンドの実装。

設計方針:
- 「判定・追記ロジック」を引数でパス注入できる純粋関数群に分離(テスト容易)
- 副作用を持つ関数を小さく分離し、テストしやすい設計にする
- すべての操作が冪等(何度実行しても安全)
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# ---------------------------------------------------------------------------
# config.toml の読み書き(tomllib は読み専用なので書き出しは自前生成)
# ---------------------------------------------------------------------------

def read_config_toml(config_file: Path) -> dict[str, Any]:
    """config.toml を読み込んで dict を返す。ファイルが無い/壊れていたら空 dict。"""
    if not config_file.is_file():
        return {}
    try:
        import tomllib
        with config_file.open("rb") as f:
            return tomllib.load(f)
    except Exception:
        return {}


def write_config_toml(config_file: Path, data: dict[str, Any]) -> None:
    """dict を config.toml に書き出す。文字列値はシングルクォート。他キーは保持。"""
    config_file.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for key, value in data.items():
        if isinstance(value, str):
            lines.append(f"{key} = '{value}'\n")
        elif isinstance(value, bool):
            lines.append(f"{key} = {'true' if value else 'false'}\n")
        elif isinstance(value, (int, float)):
            lines.append(f"{key} = {value}\n")
        else:
            lines.append(f"{key} = '{value}'\n")
    config_file.write_text("".join(lines), encoding="utf-8")


def merge_config_toml(config_file: Path, updates: dict[str, Any]) -> None:
    """既存キーを保持しつつ updates で上書きして config.toml に書き戻す。"""
    existing = read_config_toml(config_file)
    existing.update(updates)
    write_config_toml(config_file, existing)


# ---------------------------------------------------------------------------
# テンプレート設置
# ---------------------------------------------------------------------------

def copy_templates(dest_dir: Path) -> None:
    """パッケージ同梱のテンプレートを dest_dir にコピーする(常に上書き)。"""
    import importlib.resources as ir
    dest_dir.mkdir(parents=True, exist_ok=True)
    for name in ("MEMORY_PROTOCOL.md", "ONBOARDING.md"):
        try:
            ref = ir.files("engram.templates").joinpath(name)
            content = ref.read_bytes()
            (dest_dir / name).write_bytes(content)
        except Exception as e:
            print(f"  警告: テンプレート {name} のコピーに失敗しました: {e}")


# ---------------------------------------------------------------------------
# エージェント登録: Claude Code
# ---------------------------------------------------------------------------

def get_engram_mcp_path() -> Path | None:
    """engram-mcp の実行可能ファイルのパスを返す。見つからなければ None。"""
    exe_dir = Path(sys.executable).parent
    for name in ("engram-mcp.exe", "engram-mcp"):
        candidate = exe_dir / name
        if candidate.is_file():
            return candidate
    found = shutil.which("engram-mcp")
    if found:
        return Path(found)
    return None


def _claude_cmd() -> str | None:
    """claude CLI の実体パス。

    npm 経由のインストールでは claude.cmd になっており、subprocess に
    裸の "claude" を渡すと WinError 2 で失敗する(実機で発生した実例)。
    必ず which の解決結果(拡張子付きフルパス)を使う。
    """
    return shutil.which("claude")


def is_claude_mcp_registered() -> bool:
    """claude mcp list に "engram" が含まれるか確認。"""
    cmd = _claude_cmd()
    if cmd is None:
        return False
    try:
        result = subprocess.run(
            [cmd, "mcp", "list"],
            capture_output=True, text=True, timeout=30,
            encoding="utf-8", errors="replace",
        )
        output = result.stdout + result.stderr
        return "engram:" in output
    except Exception:
        return False


def _claude_registered_path(cmd: str) -> str | None:
    """claude mcp list の出力から engram の登録先パスを取り出す(未登録なら None)。"""
    try:
        result = subprocess.run(
            [cmd, "mcp", "list"],
            capture_output=True, text=True, timeout=30,
            encoding="utf-8", errors="replace",
        )
        output = result.stdout + result.stderr
        m = re.search(r"engram:\s*(\S+)", output)
        return m.group(1) if m else None
    except Exception:
        return None


def register_claude_mcp(engram_mcp_path: Path) -> tuple[bool, str]:
    """Claude Code に engram MCP を登録する。成功すれば (True, メッセージ)。"""
    cmd = _claude_cmd()
    if cmd is None:
        return False, "claude コマンドが見つかりません(Claude Code 未インストール)"

    registered = _claude_registered_path(cmd)
    if registered is not None:
        if registered == str(engram_mcp_path):
            return True, "既登録(スキップ)"
        # パスが古い(再インストールで場所が変わった等)場合は登録し直す
        try:
            subprocess.run(
                [cmd, "mcp", "remove", "engram"],
                capture_output=True, text=True, timeout=60,
                encoding="utf-8", errors="replace",
            )
        except Exception:
            pass

    try:
        result = subprocess.run(
            [cmd, "mcp", "add", "--scope", "user", "engram",
             "--", str(engram_mcp_path)],
            capture_output=True, text=True, timeout=90,
            encoding="utf-8", errors="replace",
        )
        if result.returncode == 0:
            return True, "登録完了"
        else:
            return False, f"登録失敗(exit {result.returncode}): {result.stderr.strip()}"
    except Exception as e:
        return False, f"登録失敗: {e}"


def update_claude_md(claude_md_path: Path, protocol_path: Path) -> tuple[bool, str]:
    """~/.claude/CLAUDE.md に記憶プロトコルの @import を追記する。冪等。"""
    try:
        claude_md_path.parent.mkdir(parents=True, exist_ok=True)
        existing = ""
        if claude_md_path.is_file():
            existing = claude_md_path.read_text(encoding="utf-8")
        if "engram" in existing:
            return True, "既追記済み(スキップ)"
        abs_path = str(protocol_path.resolve()).replace("\\", "/")
        addition = f"\n# 記憶プロトコル(engram)\n\n@{abs_path}\n"
        with claude_md_path.open("a", encoding="utf-8") as f:
            f.write(addition)
        return True, "追記完了"
    except Exception as e:
        return False, f"追記失敗: {e}"


# ---------------------------------------------------------------------------
# エージェント登録: Codex
# ---------------------------------------------------------------------------

def register_codex(
    codex_config_path: Path,
    engram_mcp_path: Path,
) -> tuple[bool, str]:
    """~/.codex/config.toml に engram MCP ブロックを追記する。冪等。"""
    try:
        import re

        codex_config_path.parent.mkdir(parents=True, exist_ok=True)
        existing = ""
        if codex_config_path.is_file():
            existing = codex_config_path.read_text(encoding="utf-8")
        mcp_path_str = str(engram_mcp_path).replace("\\", "/")
        if "[mcp_servers.engram]" in existing:
            # 登録済みでもパスが古い(再インストールで場所が変わった等)場合は
            # 更新する。壊れたパスのまま残すと接続不能になる(実機で発生)
            pattern = r"(\[mcp_servers\.engram\]\s*\r?\ncommand = ')([^']*)(')"
            m = re.search(pattern, existing)
            if m and m.group(2) == mcp_path_str:
                return True, "既追記済み(スキップ)"
            if m:
                updated = re.sub(pattern, rf"\g<1>{mcp_path_str}\g<3>", existing)
                codex_config_path.write_text(updated, encoding="utf-8")
                return True, "パスを更新"
            return True, "既追記済み(手動編集を検出したため変更せず)"
        addition = (
            f"\n[mcp_servers.engram]\n"
            f"command = '{mcp_path_str}'\n"
        )
        with codex_config_path.open("a", encoding="utf-8") as f:
            f.write(addition)
        return True, "追記完了"
    except Exception as e:
        return False, f"追記失敗: {e}"


def update_agents_md(
    agents_md_path: Path,
    protocol_path: Path,
) -> tuple[bool, str]:
    """~/.codex/AGENTS.md に記憶プロトコル全文を追記する。冪等。"""
    try:
        agents_md_path.parent.mkdir(parents=True, exist_ok=True)
        existing = ""
        if agents_md_path.is_file():
            existing = agents_md_path.read_text(encoding="utf-8")
        if "engram" in existing:
            return True, "既追記済み(スキップ)"
        protocol_text = ""
        if protocol_path.is_file():
            protocol_text = protocol_path.read_text(encoding="utf-8")
        abs_path = str(protocol_path.resolve()).replace("\\", "/")
        addition = (
            f"\n{protocol_text}\n"
            f"> 正本: {abs_path}\n"
        )
        with agents_md_path.open("a", encoding="utf-8") as f:
            f.write(addition)
        return True, "追記完了"
    except Exception as e:
        return False, f"追記失敗: {e}"


# ---------------------------------------------------------------------------
# エージェント登録: Gemini / Antigravity
# ---------------------------------------------------------------------------

def register_gemini_mcp(
    mcp_config_path: Path,
    engram_mcp_path: Path,
) -> tuple[bool, str]:
    """~/.gemini/config/mcp_config.json に engram エントリを追加する。冪等。"""
    try:
        mcp_config_path.parent.mkdir(parents=True, exist_ok=True)
        if mcp_config_path.is_file():
            raw = mcp_config_path.read_text(encoding="utf-8")
            if raw.strip() == "":
                # 空ファイルは「設定なし」として扱ってよい(実機で発生した実例)
                data = {"mcpServers": {}}
            else:
                try:
                    data = json.loads(raw)
                except Exception:
                    # 壊れた既存設定を空で上書きすると他サーバーの登録が消える。
                    # 触らずに中断して手動修復を促す(破壊的変更をしない)
                    return False, "既存の mcp_config.json が解析できないため変更しません(手動で修復してください)"
        else:
            data = {"mcpServers": {}}

        if "mcpServers" not in data:
            data["mcpServers"] = {}

        mcp_path_str = str(engram_mcp_path).replace("\\", "/")
        if "engram" in data["mcpServers"]:
            current = data["mcpServers"]["engram"].get("command", "")
            if current == mcp_path_str:
                return True, "既登録(スキップ)"
            # パスが古い場合は更新(壊れたパスのまま残さない)
            data["mcpServers"]["engram"]["command"] = mcp_path_str
            mcp_config_path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return True, "パスを更新"

        data["mcpServers"]["engram"] = {"command": mcp_path_str}
        mcp_config_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return True, "登録完了"
    except Exception as e:
        return False, f"登録失敗: {e}"


def update_gemini_md(
    gemini_md_path: Path,
    protocol_path: Path,
) -> tuple[bool, str]:
    """~/.gemini/GEMINI.md に記憶プロトコルの要約を追記する。冪等。"""
    try:
        gemini_md_path.parent.mkdir(parents=True, exist_ok=True)
        existing = ""
        if gemini_md_path.is_file():
            existing = gemini_md_path.read_text(encoding="utf-8")
        if "engram" in existing:
            return True, "既追記済み(スキップ)"
        abs_path = str(protocol_path.resolve()).replace("\\", "/")
        addition = (
            f"\n# 記憶プロトコル(engram)\n\n"
            f"タスク開始時は必ず `recall` を呼び、タスク完了時に `reinforce` を呼ぶこと。"
            f"重要な知見は `remember`、誤りは `forget` でなく `correct` を使うこと。"
            f"詳細は {abs_path} を参照。\n"
        )
        with gemini_md_path.open("a", encoding="utf-8") as f:
            f.write(addition)
        return True, "追記完了"
    except Exception as e:
        return False, f"追記失敗: {e}"


# ---------------------------------------------------------------------------
# セットアップウィザード本体
# ---------------------------------------------------------------------------

def setup_main(
    memories_dir: Path | None = None,
    non_interactive: bool = False,
    *,
    # パス注入(テスト・カスタマイズ用)
    engram_home: Path | None = None,
    config_file: Path | None = None,
    claude_md_path: Path | None = None,
    codex_dir: Path | None = None,
    gemini_dir: Path | None = None,
) -> None:
    """セットアップウィザードのメイン処理。冪等。"""
    from .config import config_path as _config_path
    from .config import _engram_home as _get_engram_home
    from .config import get_settings
    from .store import MarkdownStore

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    home = engram_home if engram_home is not None else _get_engram_home()
    cfg_path = config_file if config_file is not None else _config_path()

    results: list[tuple[str, bool, str]] = []

    print("=" * 60)
    print("engram セットアップウィザード")
    print("=" * 60)
    print()

    # ------------------------------------------------------------------
    # Step 1: memories_dir を決定して config.toml に書き込む
    # ------------------------------------------------------------------
    print("[1/6] 設定ファイルの作成")

    existing_cfg = read_config_toml(cfg_path)

    if memories_dir is not None:
        chosen_dir = Path(memories_dir)
        print(f"  memories_dir: {chosen_dir} (引数で指定)")
    elif "memories_dir" in existing_cfg:
        chosen_dir = Path(existing_cfg["memories_dir"])
        print(f"  既存の設定ファイルを尊重します: memories_dir={chosen_dir}")
        print(f"  設定ファイル: {cfg_path}")
        print()
        results.append(("設定ファイル", True, f"既存を尊重: {cfg_path}"))
    elif non_interactive:
        chosen_dir = home / "memories"
        print(f"  memories_dir: {chosen_dir} (既定値)")
    else:
        default = home / "memories"
        print()
        print("  記憶を保存するディレクトリを指定してください。")
        print("  Google Drive や OneDrive の同期フォルダを指定すると記憶がバックアップ")
        print("  されます。検索用DBは常にローカルに置かれるので安全です。")
        print()
        answer = input(f"  memories_dir [{default}]: ").strip()
        chosen_dir = Path(answer) if answer else default

    if "memories_dir" not in existing_cfg or memories_dir is not None:
        chosen_dir = chosen_dir.expanduser().resolve()
        merge_config_toml(cfg_path, {"memories_dir": str(chosen_dir)})
        print(f"  設定ファイルを作成しました: {cfg_path}")
        print(f"  memories_dir: {chosen_dir}")
        results.append(("設定ファイル作成", True, str(cfg_path)))

    chosen_dir = Path(existing_cfg.get("memories_dir", chosen_dir)).expanduser().resolve() \
        if "memories_dir" in existing_cfg and memories_dir is None else chosen_dir.expanduser().resolve()

    # ------------------------------------------------------------------
    # Step 2: 記憶フォルダの初期化
    # ------------------------------------------------------------------
    if "memories_dir" not in existing_cfg or memories_dir is not None:
        print()
    print("[2/6] 記憶フォルダの初期化")
    try:
        MarkdownStore(chosen_dir)
        print(f"  記憶フォルダを初期化しました: {chosen_dir}")
        results.append(("記憶フォルダ初期化", True, str(chosen_dir)))
    except Exception as e:
        results.append(("記憶フォルダ初期化", False, str(e)))
        print(f"  エラー: {e}")
    print()

    # ------------------------------------------------------------------
    # Step 3: テンプレート設置
    # ------------------------------------------------------------------
    print("[3/6] テンプレートのコピー")
    try:
        copy_templates(home)
        print(f"  MEMORY_PROTOCOL.md -> {home / 'MEMORY_PROTOCOL.md'}")
        print(f"  ONBOARDING.md      -> {home / 'ONBOARDING.md'}")
        results.append(("テンプレート設置", True, "完了"))
    except Exception as e:
        results.append(("テンプレート設置", False, str(e)))
        print(f"  エラー: {e}")
    print()

    # ------------------------------------------------------------------
    # Step 4: 埋め込みモデルの取得
    # ------------------------------------------------------------------
    print("[4/6] 埋め込みモデルの取得")
    print("  初回は約500MBをダウンロードします(既にキャッシュがあればスキップ)...")
    embedder = None
    try:
        from .embedder import RuriEmbedder
        embedder = RuriEmbedder()
        _ = embedder.dim  # メインスレッドでロード(ワーカースレッドだと激遅になる既知問題)
        results.append(("埋め込みモデル取得", True, "完了"))
        print("  埋め込みモデルの準備が完了しました")
    except Exception as e:
        results.append(("埋め込みモデル取得", False, str(e)))
        print(f"  警告: モデル取得に失敗しました({e})")
        print("  サーバー初回起動時に再試行されます。")
    print()

    # ------------------------------------------------------------------
    # Step 5: エージェント登録
    # ------------------------------------------------------------------
    print("[5/6] エージェントへの登録")

    engram_mcp = get_engram_mcp_path()
    if engram_mcp is None:
        print("  警告: engram-mcp が見つかりません。インストールを確認してください。")
        results.append(("engram-mcp の検索", False, "見つかりません"))
    else:
        print(f"  engram-mcp: {engram_mcp}")

    # --- Claude Code ---
    _claude_md = claude_md_path if claude_md_path is not None else Path.home() / ".claude" / "CLAUDE.md"
    if shutil.which("claude") is not None and engram_mcp is not None:
        ok, msg = register_claude_mcp(engram_mcp)
        results.append(("Claude Code MCP 登録", ok, msg))
        print(f"  Claude Code MCP 登録: {msg}")

        ok2, msg2 = update_claude_md(_claude_md, home / "MEMORY_PROTOCOL.md")
        results.append(("CLAUDE.md 更新", ok2, msg2))
        print(f"  CLAUDE.md 更新: {msg2}")
    else:
        print("  Claude Code: 未検出(インストールされていれば後で `engram setup` を再実行)")

    # --- Codex ---
    _codex_dir = codex_dir if codex_dir is not None else Path.home() / ".codex"
    if _codex_dir.is_dir() and engram_mcp is not None:
        ok, msg = register_codex(_codex_dir / "config.toml", engram_mcp)
        results.append(("Codex config.toml 更新", ok, msg))
        print(f"  Codex config.toml 更新: {msg}")

        ok2, msg2 = update_agents_md(_codex_dir / "AGENTS.md", home / "MEMORY_PROTOCOL.md")
        results.append(("Codex AGENTS.md 更新", ok2, msg2))
        print(f"  Codex AGENTS.md 更新: {msg2}")
    else:
        print("  Codex: 未検出(インストールされていれば後で `engram setup` を再実行)")

    # --- Gemini / Antigravity ---
    _gemini_dir = gemini_dir if gemini_dir is not None else Path.home() / ".gemini"
    if _gemini_dir.is_dir() and engram_mcp is not None:
        ok, msg = register_gemini_mcp(
            _gemini_dir / "config" / "mcp_config.json",
            engram_mcp,
        )
        results.append(("Gemini mcp_config.json 更新", ok, msg))
        print(f"  Gemini mcp_config.json 更新: {msg}")

        ok2, msg2 = update_gemini_md(_gemini_dir / "GEMINI.md", home / "MEMORY_PROTOCOL.md")
        results.append(("Gemini GEMINI.md 更新", ok2, msg2))
        print(f"  Gemini GEMINI.md 更新: {msg2}")
    else:
        print("  Antigravity/Gemini: 未検出(インストールされていれば後で `engram setup` を再実行)")

    print()

    # ------------------------------------------------------------------
    # Step 6: 動作確認 + stats
    # ------------------------------------------------------------------
    print("[6/6] 動作確認")
    if embedder is None:
        results.append(("動作確認", False, "モデル未取得のためスキップ"))
        print("  モデル未取得のためスキップ(サーバー初回起動時に再試行されます)")
    else:
        try:
            from .config import get_settings
            from .engine import build_engine
            settings = get_settings()
            # Step 4 でロード済みの本物の埋め込みを使い回す(テスト用の
            # FakeEmbedder だと既存DBと次元が合わず開けない)
            engine = build_engine(settings, embedder=embedder)
            engine.recall("セットアップ確認", mode="fast", limit=1,
                          record_hits=False)
            stats = engine.stats()
            engine.db.close()
            n = stats.get("total_memories", 0)
            print(f"  動作確認 [OK]: 記憶 {n} 件")
            results.append(("動作確認", True, f"記憶 {n} 件"))
        except Exception as e:
            results.append(("動作確認", False, str(e)))
            print(f"  警告: {e}")
    print()

    # ------------------------------------------------------------------
    # 完了案内
    # ------------------------------------------------------------------
    print("=" * 60)
    print("セットアップ結果")
    print("=" * 60)
    max_step = max((len(s) for s, _, _ in results), default=20)
    for step, ok, msg in results:
        mark = "[OK]" if ok else "[NG]"
        print(f"  {mark} {step:<{max_step}}  {msg}")

    onboarding_path = home / "ONBOARDING.md"
    print()
    print("次のステップ:")
    print("  1. エージェントを再起動(新しいセッションを開始)してください")
    print(f"  2. 最初に ONBOARDING.md のインタビューを受けると効果的です")
    print(f"     エージェントに「{onboarding_path} を読んでインタビューして」と依頼してください")
    print()


# ---------------------------------------------------------------------------
# doctor コマンド本体
# ---------------------------------------------------------------------------

def doctor_main(
    *,
    engram_home: Path | None = None,
    config_file: Path | None = None,
) -> None:
    """環境診断を表形式で表示する(エンジン構築なし・モデルロードなし)。"""
    import platform
    from .config import config_path as _config_path
    from .config import _engram_home as _get_engram_home
    from .config import get_settings

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    home = engram_home if engram_home is not None else _get_engram_home()
    cfg_file = config_file if config_file is not None else _config_path()

    print("=" * 60)
    print("engram doctor -- 環境診断")
    print("=" * 60)
    print()

    W = 38  # ラベル列幅

    def row(label: str, status: str, detail: str = "") -> None:
        detail_str = f"  {detail}" if detail else ""
        print(f"  {label:<{W}} {status}{detail_str}")

    # Python バージョン
    py_ver = platform.python_version()
    major, minor, *_ = py_ver.split(".")
    py_ok = int(major) >= 3 and int(minor) >= 12
    row("Python バージョン", "[OK]" if py_ok else "[NG]", py_ver)
    print()

    # config.toml
    cfg_exists = cfg_file.is_file()
    cfg_parseable = False
    cfg_data: dict = {}
    if cfg_exists:
        cfg_data = read_config_toml(cfg_file)
        cfg_parseable = bool(cfg_data) or cfg_file.stat().st_size < 10  # 空ファイルも許容
        try:
            import tomllib
            with cfg_file.open("rb") as f:
                tomllib.load(f)
            cfg_parseable = True
        except Exception:
            cfg_parseable = False
    row("config.toml", "[OK]" if (cfg_exists and cfg_parseable) else ("[--]" if not cfg_exists else "[NG]"),
        str(cfg_file))

    # memories_dir
    try:
        settings = get_settings()
        memories_dir = settings.memories_dir
    except Exception:
        memories_dir = home / "memories"

    memories_accessible = memories_dir.is_dir()
    md_count = 0
    if memories_accessible:
        trash_dir = memories_dir / "_trash"
        for f in memories_dir.rglob("*.md"):
            try:
                f.relative_to(trash_dir)
            except ValueError:
                md_count += 1
    row("memories_dir アクセス", "[OK]" if memories_accessible else "[--]",
        f"{memories_dir}  ({md_count} 件)" if memories_accessible else str(memories_dir))

    # index.db
    db_path = home / "index.db"
    db_exists = db_path.is_file()
    db_size = db_path.stat().st_size if db_exists else 0
    row("index.db", "[OK]" if db_exists else "[--]",
        f"{db_size:,} bytes" if db_exists else "未作成")

    print()

    # 埋め込みモデルキャッシュ
    hf_home = os.environ.get("HF_HOME") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    if hf_home:
        hf_cache = Path(hf_home) / "hub"
    else:
        hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    embed_cache = False
    if hf_cache.is_dir():
        embed_cache = any(
            d.name.startswith("models--cl-nagoya--")
            for d in hf_cache.iterdir()
            if d.is_dir()
        )
    row("埋め込みモデルキャッシュ", "[OK]" if embed_cache else "[--]",
        "あり" if embed_cache else f"未ダウンロード ({hf_cache})")

    print()

    # engram-mcp
    mcp_path = get_engram_mcp_path()
    row("engram-mcp", "[OK]" if mcp_path else "[NG]",
        str(mcp_path) if mcp_path else "見つかりません")

    print()

    # Claude Code
    claude_installed = shutil.which("claude") is not None
    if not claude_installed:
        row("Claude Code CLI", "[--]", "未インストール")
        row("Claude Code MCP 登録", "[--]", "Claude Code 未インストール")
    else:
        row("Claude Code CLI", "[OK]", shutil.which("claude") or "")
        registered = is_claude_mcp_registered()
        row("Claude Code MCP 登録", "[OK]" if registered else "[NG]",
            "登録済み" if registered else "未登録(`engram setup` を実行してください)")

    # CLAUDE.md
    claude_md = Path.home() / ".claude" / "CLAUDE.md"
    if claude_md.is_file():
        has_engram = "engram" in claude_md.read_text(encoding="utf-8", errors="replace")
        row("CLAUDE.md engram 組み込み", "[OK]" if has_engram else "[NG]",
            str(claude_md) if has_engram else f"未追記: {claude_md}")
    else:
        row("CLAUDE.md engram 組み込み", "[--]", f"ファイルなし: {claude_md}")

    print()

    # Codex
    codex_cfg = Path.home() / ".codex" / "config.toml"
    if codex_cfg.is_file():
        try:
            text = codex_cfg.read_text(encoding="utf-8")
            registered = "[mcp_servers.engram]" in text
            row("Codex MCP 登録", "[OK]" if registered else "[NG]",
                "登録済み" if registered else "未登録")
        except Exception:
            row("Codex MCP 登録", "[NG]", "読み取りエラー")
    else:
        row("Codex MCP 登録", "[--]", "Codex 未インストール")

    # AGENTS.md
    agents_md = Path.home() / ".codex" / "AGENTS.md"
    if agents_md.is_file():
        has_engram = "engram" in agents_md.read_text(encoding="utf-8", errors="replace")
        row("AGENTS.md engram 組み込み", "[OK]" if has_engram else "[NG]",
            "組み込み済み" if has_engram else "未追記")
    else:
        row("AGENTS.md engram 組み込み", "[--]", "ファイルなし")

    print()

    # Antigravity / Gemini
    gemini_cfg = Path.home() / ".gemini" / "config" / "mcp_config.json"
    if gemini_cfg.is_file():
        try:
            data = json.loads(gemini_cfg.read_text(encoding="utf-8"))
            servers = data.get("mcpServers", {})
            registered = "engram" in servers
            row("Antigravity MCP 登録", "[OK]" if registered else "[NG]",
                "登録済み" if registered else "未登録")
        except Exception:
            row("Antigravity MCP 登録", "[NG]", "JSON 読み取りエラー")
    else:
        row("Antigravity MCP 登録", "[--]", "Antigravity 未インストール")

    # GEMINI.md
    gemini_md = Path.home() / ".gemini" / "GEMINI.md"
    if gemini_md.is_file():
        has_engram = "engram" in gemini_md.read_text(encoding="utf-8", errors="replace")
        row("GEMINI.md engram 組み込み", "[OK]" if has_engram else "[NG]",
            "組み込み済み" if has_engram else "未追記")
    else:
        row("GEMINI.md engram 組み込み", "[--]", "ファイルなし")

    print()
