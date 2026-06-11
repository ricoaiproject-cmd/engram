"""setup.py の純粋ロジックをテストする。

実際のエージェント登録コマンドは実行しない。
ENGRAM_HOME を tmp_path に monkeypatch して純粋ロジックをテストする。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from engram.setup import (
    copy_templates,
    merge_config_toml,
    read_config_toml,
    register_codex,
    register_gemini_mcp,
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

    def test_idempotent_if_already_registered(self, tmp_path):
        """同じパスで登録済みならスキップ。"""
        fake_mcp = tmp_path / "engram-mcp.exe"
        fake_mcp.write_bytes(b"")
        same_path = str(fake_mcp).replace("\\", "/")
        codex_cfg = tmp_path / ".codex" / "config.toml"
        codex_cfg.parent.mkdir(parents=True, exist_ok=True)
        codex_cfg.write_text(
            f"[mcp_servers.engram]\ncommand = '{same_path}'\n", encoding="utf-8"
        )

        ok, msg = register_codex(codex_cfg, fake_mcp)
        assert ok
        assert "スキップ" in msg
        content = codex_cfg.read_text(encoding="utf-8")
        # 重複追加されていないこと
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
