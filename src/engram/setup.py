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


def _toml_scalar(value: Any) -> str:
    """スカラー値の TOML 表現。文字列はシングルクォート(エスケープ不要が利点)。"""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return f"'{value}'"


def write_config_toml(config_file: Path, data: dict[str, Any]) -> None:
    """dict を config.toml に書き出す。

    スカラー値はトップレベルに、dict 値は [セクション] として書き出す
    (例: room_paths)。セクションのキーはパス等を含むためクォートする。
    """
    config_file.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    sections: list[str] = []
    for key, value in data.items():
        if isinstance(value, dict):
            sections.append(f"\n[{key}]\n")
            for k, v in value.items():
                sections.append(f"'{k}' = {_toml_scalar(v)}\n")
        else:
            lines.append(f"{key} = {_toml_scalar(value)}\n")
    config_file.write_text("".join(lines + sections), encoding="utf-8")


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


def get_engram_cli_path() -> Path | None:
    """engram CLI 本体のパスを返す(フック登録に使う)。見つからなければ None。"""
    exe_dir = Path(sys.executable).parent
    for name in ("engram.exe", "engram"):
        candidate = exe_dir / name
        if candidate.is_file():
            return candidate
    found = shutil.which("engram")
    if found:
        return Path(found)
    return None


#: Claude Code に登録するフック: (イベント名, engram hook の引数, タイムアウト秒)
_CLAUDE_HOOK_EVENTS: list[tuple[str, str, int]] = [
    ("SessionEnd", "session-end", 180),      # モデルロード+要約があるため長め
    ("UserPromptSubmit", "user-prompt", 15),  # 軽量経路。プロンプトを待たせない
]


def _is_engram_hook_cmd(command: str) -> bool:
    """既存のフックコマンドが engram のものか判定する(更新・冪等化用)。"""
    return "engram" in command and " hook " in f"{command} "


def register_claude_hooks(
    settings_path: Path,
    engram_cli: Path,
) -> tuple[bool, str]:
    """~/.claude/settings.json に自動符号化・自発的想起のフックを登録する。冪等。

    - 既に同じコマンドが登録済みならスキップ
    - engram のフックだがパスが古い場合は更新
    - engram 以外のフックには触れない
    - 壊れた JSON は変更しない(破壊的変更をしない)
    """
    try:
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        if settings_path.is_file():
            raw = settings_path.read_text(encoding="utf-8")
            if raw.strip() == "":
                data: dict = {}
            else:
                try:
                    data = json.loads(raw)
                except Exception:
                    return False, "既存の settings.json が解析できないため変更しません(手動で修復してください)"
        else:
            data = {}

        hooks = data.setdefault("hooks", {})
        changed = []
        for event, hook_arg, timeout in _CLAUDE_HOOK_EVENTS:
            command = f'"{engram_cli}" hook {hook_arg}'
            entries = hooks.setdefault(event, [])
            found = False
            for entry in entries:
                for h in entry.get("hooks", []):
                    if _is_engram_hook_cmd(h.get("command", "")):
                        found = True
                        if h.get("command") != command:
                            h["command"] = command
                            changed.append(f"{event}(パス更新)")
                        h.setdefault("timeout", timeout)
            if not found:
                entries.append({
                    "hooks": [{
                        "type": "command",
                        "command": command,
                        "timeout": timeout,
                    }]
                })
                changed.append(event)

        if not changed:
            return True, "既登録(スキップ)"
        settings_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return True, f"登録完了({', '.join(changed)})"
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
# エージェント選択ユーティリティ
# ---------------------------------------------------------------------------

#: 内部キー → 表示名
_AGENT_DISPLAY: dict[str, str] = {
    "claude": "Claude Code",
    "codex": "Codex",
    "gemini": "Antigravity (Gemini)",
}

#: 入力エイリアス → 内部キー (小文字正規化済み)
_AGENT_ALIASES: dict[str, str] = {
    "claude": "claude",
    "codex": "codex",
    "gemini": "gemini",
    "antigravity": "gemini",
}

_VALID_AGENT_KEYS = frozenset(_AGENT_DISPLAY.keys())


def parse_agents(value: str) -> set[str]:
    """カンマ区切りのエージェント名を内部キーの集合に変換する。

    エイリアス: "antigravity" -> "gemini"(大文字小文字無視)。
    不正な名前が含まれる場合は ValueError(有効名の一覧をメッセージに含める)。
    """
    valid_aliases = sorted(_AGENT_ALIASES.keys())
    result: set[str] = set()
    for raw in value.split(","):
        token = raw.strip().lower()
        if not token:
            continue
        if token not in _AGENT_ALIASES:
            raise ValueError(
                f"不明なエージェント名: '{raw.strip()}'\n"
                f"有効な名前: {', '.join(valid_aliases)}"
            )
        result.add(_AGENT_ALIASES[token])
    return result


def _detect_agents(
    codex_dir: Path | None = None,
    gemini_dir: Path | None = None,
) -> list[str]:
    """現在の環境で検出できるエージェントの内部キーリストを返す。"""
    _codex_dir = codex_dir if codex_dir is not None else Path.home() / ".codex"
    _gemini_dir = gemini_dir if gemini_dir is not None else Path.home() / ".gemini"
    detected: list[str] = []
    if shutil.which("claude") is not None:
        detected.append("claude")
    if _codex_dir.is_dir():
        detected.append("codex")
    if _gemini_dir.is_dir():
        detected.append("gemini")
    return detected


# ---------------------------------------------------------------------------
# セットアップウィザード本体
# ---------------------------------------------------------------------------

def setup_main(
    memories_dir: Path | None = None,
    non_interactive: bool = False,
    *,
    agents: set[str] | None = None,
    # パス注入(テスト・カスタマイズ用)
    engram_home: Path | None = None,
    config_file: Path | None = None,
    claude_md_path: Path | None = None,
    codex_dir: Path | None = None,
    gemini_dir: Path | None = None,
) -> None:
    """セットアップウィザードのメイン処理。冪等。

    agents: 登録対象エージェントの内部キー集合("claude"/"codex"/"gemini")。
            None = 従来どおり検出された全部。
    """
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

    # 自発的想起のモードを明示しておく(既定: shadow=ログのみで様子見)
    if "surface_mode" not in read_config_toml(cfg_path):
        merge_config_toml(cfg_path, {"surface_mode": "shadow"})
        print("  surface_mode: shadow(自発的想起はまずログのみで観察)")

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

    # --- エージェント選択 ---
    _codex_dir = codex_dir if codex_dir is not None else Path.home() / ".codex"
    _gemini_dir = gemini_dir if gemini_dir is not None else Path.home() / ".gemini"

    # --agents で明示指定された場合は検出不問でそのまま使う
    # 未指定 + 対話あり → 2件以上検出時に選択プロンプトを出す
    # 未指定 + 非対話   → 検出された全部(従来どおり)
    if agents is not None:
        # --agents で明示: 指定集合をそのまま使用
        selected_agents: set[str] = agents
        detected_agents: list[str] = _detect_agents(_codex_dir, _gemini_dir)
    else:
        detected_agents = _detect_agents(_codex_dir, _gemini_dir)
        if non_interactive or len(detected_agents) <= 1:
            # 非対話 or 選択肢が1つ以下 → 全部
            selected_agents = set(detected_agents)
        else:
            # 対話モード: 選択肢を提示
            print()
            print("  検出されたエージェント:")
            for idx, key in enumerate(detected_agents, 1):
                print(f"    [{idx}] {_AGENT_DISPLAY[key]}")
            print("  登録先を選んでください(Enter=すべて / 番号をカンマ区切り 例: 1,3):")
            try:
                answer = input("  > ").strip()
            except EOFError:
                answer = ""
            if not answer:
                selected_agents = set(detected_agents)
            else:
                chosen: set[str] = set()
                valid = True
                for part in answer.split(","):
                    part = part.strip()
                    if not part.isdigit():
                        print(f"  無効な入力: '{part}' — すべてのエージェントを登録します")
                        valid = False
                        break
                    n = int(part)
                    if 1 <= n <= len(detected_agents):
                        chosen.add(detected_agents[n - 1])
                    else:
                        print(f"  番号が範囲外: {n} — すべてのエージェントを登録します")
                        valid = False
                        break
                selected_agents = chosen if valid else set(detected_agents)
            print()

    # --- Claude Code ---
    _claude_md = claude_md_path if claude_md_path is not None else Path.home() / ".claude" / "CLAUDE.md"
    if "claude" in selected_agents:
        if shutil.which("claude") is not None and engram_mcp is not None:
            ok, msg = register_claude_mcp(engram_mcp)
            results.append(("Claude Code MCP 登録", ok, msg))
            print(f"  Claude Code MCP 登録: {msg}")

            ok2, msg2 = update_claude_md(_claude_md, home / "MEMORY_PROTOCOL.md")
            results.append(("CLAUDE.md 更新", ok2, msg2))
            print(f"  CLAUDE.md 更新: {msg2}")

            # フック登録(自動符号化 + 自発的想起)
            engram_cli = get_engram_cli_path()
            if engram_cli is not None:
                ok3, msg3 = register_claude_hooks(
                    _claude_md.parent / "settings.json", engram_cli
                )
                results.append(("Claude Code フック登録", ok3, msg3))
                print(f"  Claude Code フック登録: {msg3}")
                print("    - セッション終了時の自動記憶(SessionEnd)")
                print("    - 関連記憶の自発的想起(UserPromptSubmit、まずは影モード)")
            else:
                results.append(("Claude Code フック登録", False, "engram CLI が見つかりません"))
                print("  Claude Code フック登録: engram CLI が見つかりません")
        else:
            # --agents で明示指定されたが未検出
            print("  Claude Code: 未検出(インストールされていれば後で `engram setup` を再実行)")
    else:
        # selected_agents に含まれない理由を判定して表示
        if "claude" in detected_agents:
            # 検出されたが選択されなかった(対話/--agents で除外)
            print(f"  {_AGENT_DISPLAY['claude']}: 選択外(スキップ)")
        else:
            # そもそも未検出(agents=None で従来どおり全部が対象だが存在しない)
            print("  Claude Code: 未検出(インストールされていれば後で `engram setup` を再実行)")

    # --- Codex ---
    if "codex" in selected_agents:
        if _codex_dir.is_dir() and engram_mcp is not None:
            ok, msg = register_codex(_codex_dir / "config.toml", engram_mcp)
            results.append(("Codex config.toml 更新", ok, msg))
            print(f"  Codex config.toml 更新: {msg}")

            ok2, msg2 = update_agents_md(_codex_dir / "AGENTS.md", home / "MEMORY_PROTOCOL.md")
            results.append(("Codex AGENTS.md 更新", ok2, msg2))
            print(f"  Codex AGENTS.md 更新: {msg2}")
        else:
            print("  Codex: 未検出(インストールされていれば後で `engram setup` を再実行)")
    else:
        if "codex" in detected_agents:
            print(f"  {_AGENT_DISPLAY['codex']}: 選択外(スキップ)")
        else:
            print("  Codex: 未検出(インストールされていれば後で `engram setup` を再実行)")

    # --- Gemini / Antigravity ---
    if "gemini" in selected_agents:
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
    else:
        if "gemini" in detected_agents:
            print(f"  {_AGENT_DISPLAY['gemini']}: 選択外(スキップ)")
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

def check_embed_backend(settings) -> tuple[str, str]:
    """埋め込み実行系の診断。(status, detail) を返す(doctor の行に載せる)。

    ONNX モデルが生成済みなら [OK] と dim・パリティを、未生成なら [--] と
    export-onnx の案内を返す。embed_backend=torch の場合はその旨を返す。
    """
    backend = getattr(settings, "embed_backend", "auto")
    onnx_dir = settings.onnx_model_dir
    meta_path = onnx_dir / "meta.json"

    if backend == "torch":
        return "[--]", "torch を強制中(起動が重い。auto に戻すと ONNX を使う)"

    if meta_path.is_file():
        try:
            with meta_path.open(encoding="utf-8") as f:
                meta = json.load(f)
            parity = meta.get("parity", {}).get("min_cosine")
            parity_str = f", parity={parity:.6f}" if parity is not None else ""
            return "[OK]", (
                f"ONNX (dim={meta.get('dim')}{parity_str})  {onnx_dir}"
            )
        except Exception:
            return "[NG]", f"meta.json が壊れています: {meta_path}"
    return "[--]", (
        "ONNX 未生成(torch フォールバックで起動が重い)。"
        "`engram export-onnx` を実行してください"
    )


def find_install_remnants(purelib: Path | None = None) -> list[str]:
    """site-packages に残った pip 再インストール失敗の残骸を探す。

    pip は更新時にパッケージを「~ngram」等へ一時リネームする。その最中に
    プロセスが落ちる(実例: 実行中の engram がファイルをロックしていた)と
    残骸だけが残り、`import engram` が不能になる。対処: engram プロセスを
    止める → 残骸を削除 → 再インストール。
    """
    if purelib is None:
        import sysconfig
        purelib = Path(sysconfig.get_paths()["purelib"])
    if not purelib.is_dir():
        return []
    return sorted(p.name for p in purelib.glob("~*gram*"))


def check_fts5() -> tuple[str, str]:
    """FTS5(trigram トークナイザ)の可否を診断する。(status, detail) を返す。

    インメモリ DB で `tokenize='trigram'` の FTS5 仮想テーブルを実際に作って
    確認する(SQLite>=3.34 が必要)。失敗時は db.py の keyword_search が
    OperationalError を握りつぶしてベクトル検索のみに黙って縮退する旨を注記する
    (無音の劣化なのでここで顕在化させる)。
    """
    import sqlite3

    sqlite_version = sqlite3.sqlite_version
    try:
        conn = sqlite3.connect(":memory:")
        try:
            conn.execute(
                "CREATE VIRTUAL TABLE t USING fts5"
                "(content, tokenize='trigram')"
            )
        finally:
            conn.close()
        return "[OK]", f"sqlite3 {sqlite_version}"
    except sqlite3.OperationalError:
        return "[NG]", (
            f"sqlite3 {sqlite_version}(trigram 非対応)。"
            "keyword_search が無音でベクトル検索のみに縮退します"
        )


def summarize_perf(perf_log_path: Path, max_lines: int = 500) -> tuple[str, str]:
    """perf_log.jsonl の直近エントリから recall p50 / preload の要約を作る。

    (status, detail) を返す。ファイルが無ければ [--] と未記録の案内。壊れた行
    (JSON でない・キー欠落)はスキップする。
    """
    if not perf_log_path.is_file():
        return "[--]", "未記録(初回のツール呼び出し後に生成)"

    try:
        with perf_log_path.open("r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except OSError:
        return "[NG]", f"読み取りエラー: {perf_log_path}"

    tail = lines[-max_lines:]
    recall_ms: list[float] = []
    last_preload_ms: float | None = None
    n_parsed = 0

    for line in tail:
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
            kind = entry["kind"]
            ms = float(entry["ms"])
        except Exception:
            continue
        n_parsed += 1
        if kind == "tool" and entry.get("name") == "recall":
            recall_ms.append(ms)
        elif kind == "preload":
            last_preload_ms = ms

    if n_parsed == 0:
        return "[--]", "有効な記録なし"

    detail_parts = []
    if recall_ms:
        recall_ms.sort()
        mid = len(recall_ms) // 2
        if len(recall_ms) % 2 == 0:
            p50 = (recall_ms[mid - 1] + recall_ms[mid]) / 2
        else:
            p50 = recall_ms[mid]
        detail_parts.append(f"recall p50 {p50:.0f}ms")
    if last_preload_ms is not None:
        detail_parts.append(f"preload {last_preload_ms:.0f}ms")

    if not detail_parts:
        return "[--]", f"recall/preload の記録なし(直近{n_parsed}件)"

    detail = " / ".join(detail_parts) + f"(直近{n_parsed}件)"
    return "[OK]", detail


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

    # SQLite 拡張ロード対応(sqlite-vec に必須。macOS 標準 Python は非対応)
    import sqlite3 as _sqlite3
    _conn = _sqlite3.connect(":memory:")
    ext_ok = hasattr(_conn, "enable_load_extension")
    _conn.close()
    row("SQLite 拡張ロード対応", "[OK]" if ext_ok else "[NG]",
        "" if ext_ok else "uv 管理の Python で入れ直してください(README 参照)")

    # FTS5(trigram)対応。非対応だと keyword_search が無音でベクトル検索のみに縮退する
    fts5_status, fts5_detail = check_fts5()
    row("FTS5(trigram)対応", fts5_status, fts5_detail)

    # pip 再インストール失敗の残骸(~ngram 等)があると import 自体が壊れる
    remnants = find_install_remnants()
    row("インストール健全性", "[OK]" if not remnants else "[NG]",
        "" if not remnants else (
            f"site-packages に残骸: {', '.join(remnants)} "
            "(engram プロセスを止めて残骸を削除し、再インストールしてください)"
        ))
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

    # 埋め込み実行系(ONNX が既定。未生成なら torch フォールバックで重い)
    try:
        _settings_for_backend = get_settings()
        backend_status, backend_detail = check_embed_backend(_settings_for_backend)
    except Exception as e:
        backend_status, backend_detail = "[NG]", f"設定読み込み失敗: {e}"
    row("埋め込み実行系", backend_status, backend_detail)

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

    # フック登録(自動符号化・自発的想起)
    claude_settings = Path.home() / ".claude" / "settings.json"
    if claude_settings.is_file():
        try:
            sdata = json.loads(
                claude_settings.read_text(encoding="utf-8", errors="replace")
            )
            hk = sdata.get("hooks", {})

            def _has_engram_hook(event: str) -> bool:
                for entry in hk.get(event, []):
                    for h in entry.get("hooks", []):
                        if _is_engram_hook_cmd(h.get("command", "")):
                            return True
                return False

            se = _has_engram_hook("SessionEnd")
            up = _has_engram_hook("UserPromptSubmit")
            row("自動符号化フック(SessionEnd)", "[OK]" if se else "[NG]",
                "登録済み" if se else "未登録(`engram setup` を実行してください)")
            row("自発的想起フック(UserPromptSubmit)", "[OK]" if up else "[NG]",
                "登録済み" if up else "未登録(`engram setup` を実行してください)")
        except Exception:
            row("フック登録", "[NG]", "settings.json 読み取りエラー")
    else:
        row("フック登録", "[--]", f"ファイルなし: {claude_settings}")

    # 自発的想起の動作モードとログ
    try:
        _settings = get_settings()
        mode = _settings.surface_mode
        surface_log = _settings.data_dir / "surface" / "surface_log.jsonl"
        if surface_log.is_file():
            with surface_log.open("r", encoding="utf-8", errors="replace") as f:
                n_log = sum(1 for _ in f)
            detail = f"ログ {n_log} 件"
        else:
            detail = "ログなし"
        row("surface_mode", "[OK]", f"{mode}({detail})")
    except Exception:
        row("surface_mode", "[NG]", "設定読み取りエラー")

    # perf ログ要約(recall p50 / preload 直近値)。ツール呼び出しの体感速度を診断する
    try:
        _settings_perf = get_settings()
        perf_log_path = _settings_perf.data_dir / "perf" / "perf_log.jsonl"
        if not _settings_perf.perf_log:
            row("perf ログ", "[--]", "無効化されています(settings.perf_log=false)")
        else:
            perf_status, perf_detail = summarize_perf(perf_log_path)
            row("perf ログ", perf_status, perf_detail)
    except Exception:
        row("perf ログ", "[NG]", "設定読み取りエラー")

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
