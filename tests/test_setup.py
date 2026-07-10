"""setup.py の純粋ロジックをテストする。

実際のエージェント登録コマンドは実行しない。
ENGRAM_HOME を tmp_path に monkeypatch して純粋ロジックをテストする。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from engram.setup import (
    copy_templates,
    merge_config_toml,
    parse_agents,
    read_config_toml,
    register_codex,
    register_gemini_mcp,
    setup_main,
    update_agents_md,
    update_claude_md,
    update_gemini_md,
    write_config_toml,
)


# ---------------------------------------------------------------------------
# config.toml の生成と冪等性
# ---------------------------------------------------------------------------

class TestConfigToml:
    def test_write_and_read(self, tmp_path):
        cfg = tmp_path / "config.toml"
        write_config_toml(cfg, {"memories_dir": str(tmp_path / "mem"), "dup_threshold": 0.9})
        data = read_config_toml(cfg)
        assert data["memories_dir"] == str(tmp_path / "mem")
        assert data["dup_threshold"] == 0.9

    def test_read_missing_returns_empty(self, tmp_path):
        result = read_config_toml(tmp_path / "nonexistent.toml")
        assert result == {}

    def test_read_broken_returns_empty(self, tmp_path):
        cfg = tmp_path / "config.toml"
        cfg.write_text("this is [not valid toml!!!", encoding="utf-8")
        result = read_config_toml(cfg)
        assert result == {}

    def test_merge_preserves_existing_keys(self, tmp_path):
        cfg = tmp_path / "config.toml"
        write_config_toml(cfg, {"existing_key": "existing_value", "dup_threshold": 0.95})
        merge_config_toml(cfg, {"memories_dir": "/new/path"})
        data = read_config_toml(cfg)
        assert data["existing_key"] == "existing_value"
        assert data["memories_dir"] == "/new/path"
        assert data["dup_threshold"] == 0.95

    def test_new_config_created(self, tmp_path):
        """新規の場合は config.toml が作成される。"""
        cfg = tmp_path / "sub" / "config.toml"
        write_config_toml(cfg, {"memories_dir": "/some/path"})
        assert cfg.is_file()
        data = read_config_toml(cfg)
        assert data["memories_dir"] == "/some/path"

    def test_existing_config_respected_on_merge(self, tmp_path):
        """既存 config.toml があれば上書きされず、merge のみ。"""
        cfg = tmp_path / "config.toml"
        cfg.write_text("memories_dir = '/original/path'\n", encoding="utf-8")
        merge_config_toml(cfg, {"new_key": "new_value"})
        data = read_config_toml(cfg)
        assert data["memories_dir"] == "/original/path"
        assert data["new_key"] == "new_value"


# ---------------------------------------------------------------------------
# CLAUDE.md への追記ロジック
# ---------------------------------------------------------------------------

class TestClaudeMd:
    def test_append_when_missing(self, tmp_path):
        claude_md = tmp_path / ".claude" / "CLAUDE.md"
        protocol = tmp_path / "MEMORY_PROTOCOL.md"
        protocol.write_text("# プロトコル", encoding="utf-8")

        ok, msg = update_claude_md(claude_md, protocol)
        assert ok
        content = claude_md.read_text(encoding="utf-8")
        assert "engram" in content
        assert "MEMORY_PROTOCOL.md" in content

    def test_idempotent_if_engram_present(self, tmp_path):
        claude_md = tmp_path / ".claude" / "CLAUDE.md"
        claude_md.parent.mkdir(parents=True, exist_ok=True)
        claude_md.write_text("# 記憶プロトコル(engram)\n@some/path\n", encoding="utf-8")
        protocol = tmp_path / "MEMORY_PROTOCOL.md"
        protocol.write_text("# プロトコル", encoding="utf-8")

        ok, msg = update_claude_md(claude_md, protocol)
        assert ok
        assert "スキップ" in msg
        # 内容が重複追加されていないこと
        content = claude_md.read_text(encoding="utf-8")
        assert content.count("engram") == 1

    def test_creates_file_if_not_exists(self, tmp_path):
        claude_md = tmp_path / ".claude" / "CLAUDE.md"
        protocol = tmp_path / "MEMORY_PROTOCOL.md"
        protocol.write_text("# プロトコル", encoding="utf-8")

        assert not claude_md.exists()
        ok, msg = update_claude_md(claude_md, protocol)
        assert ok
        assert claude_md.is_file()

    def test_existing_content_preserved(self, tmp_path):
        claude_md = tmp_path / ".claude" / "CLAUDE.md"
        claude_md.parent.mkdir(parents=True, exist_ok=True)
        original = "# 既存の内容\n\nここに元の文章があります。\n"
        claude_md.write_text(original, encoding="utf-8")
        protocol = tmp_path / "MEMORY_PROTOCOL.md"
        protocol.write_text("# プロトコル", encoding="utf-8")

        update_claude_md(claude_md, protocol)
        content = claude_md.read_text(encoding="utf-8")
        assert "既存の内容" in content
        assert "engram" in content


# ---------------------------------------------------------------------------
# AGENTS.md への追記ロジック
# ---------------------------------------------------------------------------

class TestAgentsMd:
    def test_append_when_missing(self, tmp_path):
        agents_md = tmp_path / ".codex" / "AGENTS.md"
        protocol = tmp_path / "MEMORY_PROTOCOL.md"
        protocol.write_text("# 記憶プロトコル(engram)\nプロトコル本文。", encoding="utf-8")

        ok, msg = update_agents_md(agents_md, protocol)
        assert ok
        content = agents_md.read_text(encoding="utf-8")
        assert "engram" in content

    def test_idempotent_if_engram_present(self, tmp_path):
        agents_md = tmp_path / ".codex" / "AGENTS.md"
        agents_md.parent.mkdir(parents=True, exist_ok=True)
        agents_md.write_text("# engram 記憶プロトコル\n既存の内容。\n", encoding="utf-8")
        protocol = tmp_path / "MEMORY_PROTOCOL.md"
        protocol.write_text("# プロトコル", encoding="utf-8")

        ok, msg = update_agents_md(agents_md, protocol)
        assert ok
        assert "スキップ" in msg

    def test_creates_file_if_not_exists(self, tmp_path):
        agents_md = tmp_path / ".codex" / "AGENTS.md"
        protocol = tmp_path / "MEMORY_PROTOCOL.md"
        protocol.write_text("# プロトコル本文", encoding="utf-8")

        assert not agents_md.exists()
        ok, _ = update_agents_md(agents_md, protocol)
        assert ok
        assert agents_md.is_file()

    def test_existing_content_preserved(self, tmp_path):
        agents_md = tmp_path / ".codex" / "AGENTS.md"
        agents_md.parent.mkdir(parents=True, exist_ok=True)
        original = "# 既存のAGENTS設定\n\nここに元の文章。\n"
        agents_md.write_text(original, encoding="utf-8")
        protocol = tmp_path / "MEMORY_PROTOCOL.md"
        # プロトコル本文には engram が含まれることを前提とする
        protocol.write_text("# engram 記憶プロトコル\n詳細手順。\n", encoding="utf-8")

        update_agents_md(agents_md, protocol)
        content = agents_md.read_text(encoding="utf-8")
        assert "既存のAGENTS設定" in content
        assert "engram" in content


# ---------------------------------------------------------------------------
# GEMINI.md への追記ロジック
# ---------------------------------------------------------------------------

class TestGeminiMd:
    def test_append_when_missing(self, tmp_path):
        gemini_md = tmp_path / ".gemini" / "GEMINI.md"
        protocol = tmp_path / "MEMORY_PROTOCOL.md"
        protocol.write_text("# プロトコル", encoding="utf-8")

        ok, msg = update_gemini_md(gemini_md, protocol)
        assert ok
        content = gemini_md.read_text(encoding="utf-8")
        assert "engram" in content

    def test_idempotent_if_engram_present(self, tmp_path):
        gemini_md = tmp_path / ".gemini" / "GEMINI.md"
        gemini_md.parent.mkdir(parents=True, exist_ok=True)
        gemini_md.write_text("# 記憶プロトコル(engram)\n既存の内容。\n", encoding="utf-8")
        protocol = tmp_path / "MEMORY_PROTOCOL.md"
        protocol.write_text("# プロトコル", encoding="utf-8")

        ok, msg = update_gemini_md(gemini_md, protocol)
        assert ok
        assert "スキップ" in msg
        content = gemini_md.read_text(encoding="utf-8")
        assert content.count("engram") == 1

    def test_creates_file_if_not_exists(self, tmp_path):
        gemini_md = tmp_path / ".gemini" / "GEMINI.md"
        protocol = tmp_path / "MEMORY_PROTOCOL.md"
        protocol.write_text("# プロトコル", encoding="utf-8")

        assert not gemini_md.exists()
        ok, _ = update_gemini_md(gemini_md, protocol)
        assert ok
        assert gemini_md.is_file()


# ---------------------------------------------------------------------------
# mcp_config.json への JSON 挿入ロジック
# ---------------------------------------------------------------------------

class TestGeminiMcpJson:
    def test_insert_new_server(self, tmp_path):
        mcp_json = tmp_path / ".gemini" / "config" / "mcp_config.json"
        fake_mcp = tmp_path / "engram-mcp.exe"
        fake_mcp.write_bytes(b"")

        ok, msg = register_gemini_mcp(mcp_json, fake_mcp)
        assert ok
        data = json.loads(mcp_json.read_text(encoding="utf-8"))
        assert "engram" in data["mcpServers"]

    def test_idempotent_if_already_registered(self, tmp_path):
        """同じパスで登録済みならスキップ。"""
        fake_mcp = tmp_path / "engram-mcp.exe"
        fake_mcp.write_bytes(b"")
        same_path = str(fake_mcp).replace("\\", "/")
        mcp_json = tmp_path / ".gemini" / "config" / "mcp_config.json"
        mcp_json.parent.mkdir(parents=True, exist_ok=True)
        mcp_json.write_text(
            json.dumps({"mcpServers": {"engram": {"command": same_path}}}),
            encoding="utf-8",
        )

        ok, msg = register_gemini_mcp(mcp_json, fake_mcp)
        assert ok
        assert "スキップ" in msg
        # 重複追加されていないこと
        data = json.loads(mcp_json.read_text(encoding="utf-8"))
        assert len(data["mcpServers"]) == 1

    def test_stale_path_is_updated(self, tmp_path):
        """登録済みでもパスが古ければ更新する(壊れたパスを残さない)。"""
        mcp_json = tmp_path / ".gemini" / "config" / "mcp_config.json"
        mcp_json.parent.mkdir(parents=True, exist_ok=True)
        mcp_json.write_text(
            json.dumps({"mcpServers": {
                "engram": {"command": "//old-server/broken/engram-mcp.exe"},
                "other": {"command": "/path/to/other"},
            }}),
            encoding="utf-8",
        )
        fake_mcp = tmp_path / "engram-mcp.exe"
        fake_mcp.write_bytes(b"")

        ok, msg = register_gemini_mcp(mcp_json, fake_mcp)
        assert ok
        assert "更新" in msg
        data = json.loads(mcp_json.read_text(encoding="utf-8"))
        assert data["mcpServers"]["engram"]["command"] == str(fake_mcp).replace("\\", "/")
        assert "other" in data["mcpServers"]  # 他のサーバーは無傷

    def test_existing_servers_preserved(self, tmp_path):
        mcp_json = tmp_path / ".gemini" / "config" / "mcp_config.json"
        mcp_json.parent.mkdir(parents=True, exist_ok=True)
        existing = {
            "mcpServers": {
                "other_tool": {"command": "/path/to/other"},
                "another": {"command": "/path/to/another"},
            }
        }
        mcp_json.write_text(json.dumps(existing), encoding="utf-8")
        fake_mcp = tmp_path / "engram-mcp.exe"
        fake_mcp.write_bytes(b"")

        ok, _ = register_gemini_mcp(mcp_json, fake_mcp)
        assert ok
        data = json.loads(mcp_json.read_text(encoding="utf-8"))
        assert "other_tool" in data["mcpServers"]
        assert "another" in data["mcpServers"]
        assert "engram" in data["mcpServers"]

    def test_json_indent_2(self, tmp_path):
        """インデント2で書き戻されることを確認。"""
        mcp_json = tmp_path / ".gemini" / "config" / "mcp_config.json"
        fake_mcp = tmp_path / "engram-mcp.exe"
        fake_mcp.write_bytes(b"")

        register_gemini_mcp(mcp_json, fake_mcp)
        text = mcp_json.read_text(encoding="utf-8")
        # インデント2なら "  " で始まる行がある
        assert any(line.startswith("  ") for line in text.splitlines())


# ---------------------------------------------------------------------------
# Codex config.toml へのテキスト追記の冪等性
# ---------------------------------------------------------------------------

class TestCodexConfig:
    def test_append_new_entry(self, tmp_path):
        codex_cfg = tmp_path / ".codex" / "config.toml"
        fake_mcp = tmp_path / "engram-mcp.exe"
        fake_mcp.write_bytes(b"")

        ok, msg = register_codex(codex_cfg, fake_mcp)
        assert ok
        content = codex_cfg.read_text(encoding="utf-8")
        assert "[mcp_servers.engram]" in content
        # 既定30秒では起動が間に合わないことがあるため必ず明示する
        assert "startup_timeout_sec = 120.0" in content

    def test_idempotent_if_already_registered(self, tmp_path):
        """同じパスで登録済み(起動待ち時間も設定済み)ならスキップ。"""
        fake_mcp = tmp_path / "engram-mcp.exe"
        fake_mcp.write_bytes(b"")
        same_path = str(fake_mcp).replace("\\", "/")
        codex_cfg = tmp_path / ".codex" / "config.toml"
        codex_cfg.parent.mkdir(parents=True, exist_ok=True)
        codex_cfg.write_text(
            f"[mcp_servers.engram]\ncommand = '{same_path}'\n"
            f"startup_timeout_sec = 120.0\n",
            encoding="utf-8",
        )

        ok, msg = register_codex(codex_cfg, fake_mcp)
        assert ok
        assert "スキップ" in msg
        content = codex_cfg.read_text(encoding="utf-8")
        # 重複追加されていないこと
        assert content.count("[mcp_servers.engram]") == 1
        assert content.count("startup_timeout_sec") == 1

    def test_timeout_added_to_legacy_entry(self, tmp_path):
        """旧バージョンで登録された(起動待ち時間なし)ブロックには追記する。
        実機で発生: Codex 既定の初期接続30秒では engram の起動
        (モデル読込+記憶フォルダ確認)が間に合わず接続タイムアウトした。"""
        fake_mcp = tmp_path / "engram-mcp.exe"
        fake_mcp.write_bytes(b"")
        same_path = str(fake_mcp).replace("\\", "/")
        codex_cfg = tmp_path / ".codex" / "config.toml"
        codex_cfg.parent.mkdir(parents=True, exist_ok=True)
        codex_cfg.write_text(
            f"[mcp_servers.engram]\ncommand = '{same_path}'\n"
            f"\n[mcp_servers.engram.tools.recall]\napproval_mode = 'auto'\n",
            encoding="utf-8",
        )

        ok, msg = register_codex(codex_cfg, fake_mcp)
        assert ok
        assert "起動待ち時間" in msg
        content = codex_cfg.read_text(encoding="utf-8")
        assert "startup_timeout_sec = 120.0" in content
        # command 行の直後(ブロック内)に入っていること
        assert re.search(
            r"\[mcp_servers\.engram\]\ncommand = '[^']*'\nstartup_timeout_sec = 120\.0\n",
            content,
        )
        # tools サブセクション等の他の設定は無傷
        assert "[mcp_servers.engram.tools.recall]" in content
        assert content.count("startup_timeout_sec") == 1

    def test_timeout_added_to_double_quoted_entry(self, tmp_path):
        """Codex 本体が config.toml を書き直すと引用符が二重引用符に変わる
        ことがある(実機で確認)。その形式のブロックにも追記できる。"""
        fake_mcp = tmp_path / "engram-mcp.exe"
        fake_mcp.write_bytes(b"")
        same_path = str(fake_mcp).replace("\\", "/")
        codex_cfg = tmp_path / ".codex" / "config.toml"
        codex_cfg.parent.mkdir(parents=True, exist_ok=True)
        codex_cfg.write_text(
            f'[mcp_servers.engram]\ncommand = "{same_path}"\n', encoding="utf-8"
        )

        ok, msg = register_codex(codex_cfg, fake_mcp)
        assert ok
        assert "起動待ち時間" in msg
        content = codex_cfg.read_text(encoding="utf-8")
        assert "startup_timeout_sec = 120.0" in content
        assert f'command = "{same_path}"' in content  # 引用符スタイルは維持
        assert content.count("[mcp_servers.engram]") == 1

    def test_stale_path_is_updated(self, tmp_path):
        """登録済みでもパスが古ければ更新する(壊れたパスを残さない)。
        職場PC実機で発生: 1回目の失敗時のネットワークパスが残り Codex が接続不能になった。"""
        codex_cfg = tmp_path / ".codex" / "config.toml"
        codex_cfg.parent.mkdir(parents=True, exist_ok=True)
        codex_cfg.write_text(
            "[other]\nkey = 'v'\n\n[mcp_servers.engram]\ncommand = '//old-server/broken/engram-mcp.exe'\n",
            encoding="utf-8",
        )
        fake_mcp = tmp_path / "engram-mcp.exe"
        fake_mcp.write_bytes(b"")

        ok, msg = register_codex(codex_cfg, fake_mcp)
        assert ok
        assert "更新" in msg
        content = codex_cfg.read_text(encoding="utf-8")
        assert str(fake_mcp).replace("\\", "/") in content
        assert "//old-server/broken" not in content
        assert "[other]" in content  # 他の設定は無傷
        assert content.count("[mcp_servers.engram]") == 1

    def test_existing_content_preserved(self, tmp_path):
        codex_cfg = tmp_path / ".codex" / "config.toml"
        codex_cfg.parent.mkdir(parents=True, exist_ok=True)
        original = "[some_other_setting]\nkey = 'value'\n"
        codex_cfg.write_text(original, encoding="utf-8")
        fake_mcp = tmp_path / "engram-mcp.exe"
        fake_mcp.write_bytes(b"")

        register_codex(codex_cfg, fake_mcp)
        content = codex_cfg.read_text(encoding="utf-8")
        assert "[some_other_setting]" in content
        assert "[mcp_servers.engram]" in content

    def test_creates_file_if_not_exists(self, tmp_path):
        codex_cfg = tmp_path / ".codex" / "config.toml"
        fake_mcp = tmp_path / "engram-mcp.exe"
        fake_mcp.write_bytes(b"")

        assert not codex_cfg.exists()
        ok, _ = register_codex(codex_cfg, fake_mcp)
        assert ok
        assert codex_cfg.is_file()


# ---------------------------------------------------------------------------
# テンプレートコピー
# ---------------------------------------------------------------------------

class TestCopyTemplates:
    def test_copies_both_templates(self, tmp_path):
        dest = tmp_path / "engram"
        copy_templates(dest)
        assert (dest / "MEMORY_PROTOCOL.md").is_file()
        assert (dest / "ONBOARDING.md").is_file()

    def test_templates_have_content(self, tmp_path):
        dest = tmp_path / "engram"
        copy_templates(dest)
        protocol_text = (dest / "MEMORY_PROTOCOL.md").read_text(encoding="utf-8")
        onboarding_text = (dest / "ONBOARDING.md").read_text(encoding="utf-8")
        assert len(protocol_text) > 100
        assert len(onboarding_text) > 100

    def test_overwrite_existing(self, tmp_path):
        """既存ファイルを上書きできること。"""
        dest = tmp_path / "engram"
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "MEMORY_PROTOCOL.md").write_text("old content", encoding="utf-8")
        copy_templates(dest)
        content = (dest / "MEMORY_PROTOCOL.md").read_text(encoding="utf-8")
        assert content != "old content"
        assert len(content) > 100

    def test_creates_dest_dir_if_missing(self, tmp_path):
        dest = tmp_path / "new_dir" / "sub"
        assert not dest.exists()
        copy_templates(dest)
        assert dest.is_dir()


# ---------------------------------------------------------------------------
# parse_agents
# ---------------------------------------------------------------------------

class TestParseAgents:
    # --- 正常系 ---
    def test_single_claude(self):
        assert parse_agents("claude") == {"claude"}

    def test_multiple_claude_codex(self):
        assert parse_agents("claude,codex") == {"claude", "codex"}

    def test_antigravity_alias_normalized_to_gemini(self):
        assert parse_agents("Antigravity") == {"gemini"}

    def test_gemini_key(self):
        assert parse_agents("gemini") == {"gemini"}

    def test_all_three(self):
        assert parse_agents("claude,codex,gemini") == {"claude", "codex", "gemini"}

    def test_whitespace_around_commas(self):
        assert parse_agents("claude , codex") == {"claude", "codex"}

    def test_uppercase_claude(self):
        assert parse_agents("CLAUDE") == {"claude"}

    def test_mixed_case_codex(self):
        assert parse_agents("CoDex") == {"codex"}

    def test_antigravity_mixed_case(self):
        assert parse_agents("ANTIGRAVITY") == {"gemini"}

    # --- 異常系 ---
    def test_invalid_name_raises_value_error(self):
        with pytest.raises(ValueError) as exc_info:
            parse_agents("unknown")
        msg = str(exc_info.value)
        assert "unknown" in msg
        # エラーメッセージに有効名が含まれること
        assert "claude" in msg
        assert "codex" in msg

    def test_partially_invalid_raises_value_error(self):
        with pytest.raises(ValueError) as exc_info:
            parse_agents("claude,bad_agent")
        msg = str(exc_info.value)
        assert "bad_agent" in msg
        assert "claude" in msg  # 有効名の一覧が含まれる

    def test_valid_names_listed_in_error(self):
        """エラーメッセージにすべての有効エイリアスが含まれること。"""
        with pytest.raises(ValueError) as exc_info:
            parse_agents("nope")
        msg = str(exc_info.value)
        for valid in ("claude", "codex", "gemini", "antigravity"):
            assert valid in msg


# ---------------------------------------------------------------------------
# setup_main の agents フィルタ
# ---------------------------------------------------------------------------

class TestSetupMainAgentsFilter:
    """実コマンドを実行せず、パス注入でファイルへの副作用だけを検証する。

    shutil.which を None 固定にして claude は常に「未検出」にする。
    codex / gemini は tmp_path 配下のディレクトリ有無で制御する。
    """

    @pytest.fixture()
    def patch_which_none(self, monkeypatch):
        """shutil.which を常に None を返すようにパッチ。"""
        import shutil as _shutil
        import engram.setup as setup_mod
        monkeypatch.setattr(setup_mod.shutil, "which", lambda *a, **kw: None)

    @pytest.fixture()
    def fake_engram_mcp(self, tmp_path):
        """偽の engram-mcp 実行可能ファイルを作成し、get_engram_mcp_path をパッチ。"""
        import engram.setup as setup_mod
        mcp = tmp_path / "engram-mcp.exe"
        mcp.write_bytes(b"")
        return mcp

    def _run_setup(self, tmp_path, monkeypatch, fake_mcp, agents, non_interactive=True):
        """setup_main を最小限の引数で呼び出す共通ヘルパー。"""
        import engram.setup as setup_mod

        # engram_home / config 周りを tmp_path に向ける
        engram_home = tmp_path / "engram_home"
        engram_home.mkdir(parents=True, exist_ok=True)
        config_file = engram_home / "config.toml"
        claude_md = tmp_path / "claude_md" / "CLAUDE.md"
        codex_dir = tmp_path / "codex"
        gemini_dir = tmp_path / "gemini"

        # codex / gemini ディレクトリを作る(存在=「検出済み」扱い)
        codex_dir.mkdir()
        (gemini_dir / "config").mkdir(parents=True)

        # engram-mcp の検索をパッチ
        monkeypatch.setattr(setup_mod, "get_engram_mcp_path", lambda: fake_mcp)

        # MarkdownStore / embedder / build_engine はスキップさせるため
        # RuriEmbedder をパッチして即 raise させる
        import engram.embedder as emb_mod
        monkeypatch.setattr(emb_mod, "RuriEmbedder", lambda: (_ for _ in ()).throw(RuntimeError("skip")))

        setup_main(
            memories_dir=tmp_path / "memories",
            non_interactive=non_interactive,
            agents=agents,
            engram_home=engram_home,
            config_file=config_file,
            claude_md_path=claude_md,
            codex_dir=codex_dir,
            gemini_dir=gemini_dir,
        )
        return codex_dir, gemini_dir

    def test_agents_codex_only_writes_codex_not_gemini(
        self, tmp_path, monkeypatch, patch_which_none, fake_engram_mcp
    ):
        """agents={"codex"} のとき codex だけ書かれ gemini は変更なし。"""
        codex_dir, gemini_dir = self._run_setup(
            tmp_path, monkeypatch, fake_engram_mcp, agents={"codex"}
        )
        # codex の config.toml が作成されている
        codex_cfg = codex_dir / "config.toml"
        assert codex_cfg.is_file()
        assert "[mcp_servers.engram]" in codex_cfg.read_text(encoding="utf-8")

        # gemini の mcp_config.json は変更されていない(存在しない)
        gemini_cfg = gemini_dir / "config" / "mcp_config.json"
        assert not gemini_cfg.exists()

    def test_agents_none_writes_all_detected(
        self, tmp_path, monkeypatch, patch_which_none, fake_engram_mcp
    ):
        """agents=None(デフォルト)のとき検出された全エージェントに書き込む。
        claude は which=None なのでスキップ。codex / gemini は書かれる。"""
        codex_dir, gemini_dir = self._run_setup(
            tmp_path, monkeypatch, fake_engram_mcp, agents=None
        )
        # codex が書かれている
        codex_cfg = codex_dir / "config.toml"
        assert codex_cfg.is_file()
        assert "[mcp_servers.engram]" in codex_cfg.read_text(encoding="utf-8")

        # gemini が書かれている
        gemini_cfg = gemini_dir / "config" / "mcp_config.json"
        assert gemini_cfg.is_file()
        data = json.loads(gemini_cfg.read_text(encoding="utf-8"))
        assert "engram" in data["mcpServers"]

    def test_agents_gemini_only_writes_gemini_not_codex(
        self, tmp_path, monkeypatch, patch_which_none, fake_engram_mcp
    ):
        """agents={"gemini"} のとき gemini だけ書かれ codex は変更なし。"""
        codex_dir, gemini_dir = self._run_setup(
            tmp_path, monkeypatch, fake_engram_mcp, agents={"gemini"}
        )
        gemini_cfg = gemini_dir / "config" / "mcp_config.json"
        assert gemini_cfg.is_file()

        codex_cfg = codex_dir / "config.toml"
        # codex は書かれていないか、engram セクションがない
        if codex_cfg.exists():
            assert "[mcp_servers.engram]" not in codex_cfg.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# doctor の診断ヘルパー(埋め込み実行系・インストール健全性)


class TestCheckEmbedBackend:
    def _settings(self, tmp_path, **overrides):
        from engram.config import Settings

        return Settings(data_dir=tmp_path, **overrides)

    def _write_onnx_model(self, tmp_path, meta: dict | str) -> Path:
        from engram.config import Settings

        onnx_dir = Settings(data_dir=tmp_path).onnx_model_dir
        onnx_dir.mkdir(parents=True)
        (onnx_dir / "model.onnx").touch()
        (onnx_dir / "tokenizer.json").touch()
        text = meta if isinstance(meta, str) else json.dumps(meta)
        (onnx_dir / "meta.json").write_text(text, encoding="utf-8")
        return onnx_dir

    def test_onnx_present_reports_ok_with_dim_and_parity(self, tmp_path):
        from engram.setup import check_embed_backend

        self._write_onnx_model(
            tmp_path,
            {"dim": 512, "parity": {"min_cosine": 0.999999}},
        )
        status, detail = check_embed_backend(self._settings(tmp_path))
        assert status == "[OK]"
        assert "dim=512" in detail
        assert "parity=0.999999" in detail

    def test_onnx_missing_suggests_export(self, tmp_path):
        from engram.setup import check_embed_backend

        status, detail = check_embed_backend(self._settings(tmp_path))
        assert status == "[--]"
        assert "export-onnx" in detail

    def test_torch_forced_is_reported(self, tmp_path):
        from engram.setup import check_embed_backend

        status, detail = check_embed_backend(
            self._settings(tmp_path, embed_backend="torch")
        )
        assert status == "[--]"
        assert "torch" in detail

    def test_broken_meta_is_ng(self, tmp_path):
        from engram.setup import check_embed_backend

        self._write_onnx_model(tmp_path, "{not json")
        status, detail = check_embed_backend(self._settings(tmp_path))
        assert status == "[NG]"


class TestFindInstallRemnants:
    def test_clean_site_packages_is_empty(self, tmp_path):
        from engram.setup import find_install_remnants

        (tmp_path / "engram").mkdir()
        (tmp_path / "numpy").mkdir()
        assert find_install_remnants(tmp_path) == []

    def test_detects_pip_rename_remnants(self, tmp_path):
        from engram.setup import find_install_remnants

        (tmp_path / "~ngram").mkdir()
        (tmp_path / "~ngram-0.5.0.dist-info").mkdir()
        (tmp_path / "~-gram").mkdir()
        (tmp_path / "engram").mkdir()  # 正常な方は引っかからない
        found = find_install_remnants(tmp_path)
        assert "~ngram" in found
        assert "~ngram-0.5.0.dist-info" in found
        assert "~-gram" in found
        assert "engram" not in found

    def test_missing_dir_is_empty(self, tmp_path):
        from engram.setup import find_install_remnants

        assert find_install_remnants(tmp_path / "nope") == []


# ---------------------------------------------------------------------------
# doctor の診断ヘルパー(FTS5/trigram 対応・perf ログ要約)


class TestCheckFts5:
    def test_ok_on_this_machine(self):
        from engram.setup import check_fts5

        status, detail = check_fts5()
        assert status == "[OK]"
        assert detail  # sqlite バージョン文字列が入っている


class TestSummarizePerf:
    def test_missing_file_reports_not_recorded(self, tmp_path):
        from engram.setup import summarize_perf

        status, detail = summarize_perf(tmp_path / "no_such" / "perf_log.jsonl")
        assert status == "[--]"
        assert "未記録" in detail

    def test_median_and_last_preload_are_reported(self, tmp_path):
        from engram.setup import summarize_perf

        log = tmp_path / "perf_log.jsonl"
        entries = [
            {"ts": 1.0, "kind": "tool", "name": "recall", "ms": 10.0, "ok": True},
            {"ts": 2.0, "kind": "tool", "name": "recall", "ms": 20.0, "ok": True},
            {"ts": 3.0, "kind": "tool", "name": "recall", "ms": 30.0, "ok": True},
            # 別ツールの記録は recall の中央値に影響しない
            {"ts": 4.0, "kind": "tool", "name": "remember", "ms": 999.0, "ok": True},
            {"ts": 5.0, "kind": "preload", "name": "preload", "ms": 1000.0, "ok": True},
            {"ts": 6.0, "kind": "preload", "name": "preload", "ms": 1850.0, "ok": True},
        ]
        log.write_text(
            "\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8"
        )
        status, detail = summarize_perf(log)
        assert status == "[OK]"
        assert "recall p50 20ms" in detail
        # 最新の preload エントリ(直近)が使われる
        assert "preload 1850ms" in detail

    def test_malformed_lines_are_skipped(self, tmp_path):
        from engram.setup import summarize_perf

        log = tmp_path / "perf_log.jsonl"
        good = {"ts": 1.0, "kind": "tool", "name": "recall", "ms": 40.0, "ok": True}
        lines = [
            "not json at all",
            json.dumps({"ts": 2.0, "kind": "tool"}),  # ms 欠落
            json.dumps(good),
            "",  # 空行
        ]
        log.write_text("\n".join(lines) + "\n", encoding="utf-8")
        status, detail = summarize_perf(log)
        assert status == "[OK]"
        assert "recall p50 40ms" in detail
        assert "直近1件" in detail

    def test_no_valid_entries_reports_dashes(self, tmp_path):
        from engram.setup import summarize_perf

        log = tmp_path / "perf_log.jsonl"
        log.write_text("garbage\nmore garbage\n", encoding="utf-8")
        status, detail = summarize_perf(log)
        assert status == "[--]"
