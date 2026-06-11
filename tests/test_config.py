"""設定の読み込み優先順位(既定値 < config.toml < 環境変数)のテスト。"""

from pathlib import Path

from engram import config as cfg


def test_defaults_without_config(monkeypatch, tmp_path):
    monkeypatch.setenv("ENGRAM_HOME", str(tmp_path))
    monkeypatch.delenv("ENGRAM_MEMORIES_DIR", raising=False)
    s = cfg.get_settings()
    assert s.memories_dir == tmp_path / "memories"
    assert s.db_path == tmp_path / "index.db"
    assert s.dup_threshold == 0.95


def test_config_file_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("ENGRAM_HOME", str(tmp_path))
    monkeypatch.delenv("ENGRAM_MEMORIES_DIR", raising=False)
    (tmp_path / "config.toml").write_text(
        f"memories_dir = '{tmp_path / 'mem'}'\ndup_threshold = 0.9\n"
        "unknown_key = 'ignored'\n",
        encoding="utf-8",
    )
    s = cfg.get_settings()
    assert s.memories_dir == tmp_path / "mem"
    assert s.dup_threshold == 0.9


def test_env_beats_config_file(monkeypatch, tmp_path):
    monkeypatch.setenv("ENGRAM_HOME", str(tmp_path))
    (tmp_path / "config.toml").write_text(
        f"memories_dir = '{tmp_path / 'from_file'}'\n", encoding="utf-8"
    )
    monkeypatch.setenv("ENGRAM_MEMORIES_DIR", str(tmp_path / "from_env"))
    s = cfg.get_settings()
    assert s.memories_dir == tmp_path / "from_env"


def test_broken_config_file_does_not_crash(monkeypatch, tmp_path):
    monkeypatch.setenv("ENGRAM_HOME", str(tmp_path))
    monkeypatch.delenv("ENGRAM_MEMORIES_DIR", raising=False)
    (tmp_path / "config.toml").write_text("this is [not toml", encoding="utf-8")
    s = cfg.get_settings()
    assert s.memories_dir == tmp_path / "memories"  # 既定値で動く


def test_templates_shipped_in_package():
    import engram

    tdir = Path(engram.__file__).parent / "templates"
    assert (tdir / "MEMORY_PROTOCOL.md").is_file()
    assert (tdir / "ONBOARDING.md").is_file()
