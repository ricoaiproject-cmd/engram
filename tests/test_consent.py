"""夜の整理で入れた人の claude / codex を使うときの、本人の同意(0.25.4)のテスト。"""

from __future__ import annotations

from engram import night as night_mod
from engram import setup as setup_mod
from engram.config import Settings
from engram.night import consent, run_integration, template_agent_lines


def _s(tmp_path, **kw):
    return Settings(data_dir=tmp_path / "d", memories_dir=tmp_path / "m", **kw)


def test_default_is_local_only_without_consent(tmp_path):
    s = _s(tmp_path)
    assert consent("claude", s)[0] is False and "同意がまだ無い" in consent("claude", s)[1]
    assert consent("claude", _s(tmp_path, night_allow_claude=True))[0] is True
    assert consent("codex", _s(tmp_path, night_allow_codex=False))[0] is False


def test_template_copy_is_not_consent_but_user_edit_is(tmp_path):
    s = _s(tmp_path)
    s.data_dir.mkdir(parents=True)
    tpl = "\n".join(template_agent_lines()) + "\n"
    (s.data_dir / "sleep_agents.txt").write_text("# 説明\n" + tpl, encoding="utf-8")
    assert consent("claude", s)[0] is False  # 雛形のまま = 0.25.3 から上げただけの人
    (s.data_dir / "sleep_agents.txt").write_text("claude -p --model x\nlocal-llm\n", encoding="utf-8")
    ok, why = consent("claude", s)
    assert ok and "自分で書き換え" in why
    # 書き換えていても、本人が「使わない」と答えていればそちらが勝つ
    assert consent("claude", _s(tmp_path, night_allow_claude=False))[0] is False


def test_integration_skips_cloud_cli_without_consent(tmp_path):
    from engram.embedder import FakeEmbedder
    from engram.engine import build_engine

    class Down:
        model = "x"

        def available(self):
            return False

    for allow, expect in ((None, []), (True, ["claude"])):
        s = _s(tmp_path / str(allow), night_allow_claude=allow)
        s.data_dir.mkdir(parents=True)
        tpl = "\n".join(template_agent_lines()) + "\n"
        (s.data_dir / "sleep_agents.txt").write_text(tpl, encoding="utf-8")
        eng = build_engine(s, embedder=FakeEmbedder())
        calls = []
        run_integration(eng, s, dry_run=False, llm=Down(),
                        runner=lambda argv: calls.append(argv[0]) or 0)
        assert calls == expect
        eng.db.close()


def _ask(answers):
    seq = iter(answers)

    def ask(prompt):
        assert "[y/N]" in prompt and "送ります" in prompt
        v = next(seq)
        if isinstance(v, BaseException):
            raise v
        return v
    return ask


def test_consent_step_interactive_yes_no_and_keeps_answers(tmp_path):
    cfg = tmp_path / "c.toml"
    which = lambda name: f"C:/bin/{name}.cmd"  # noqa: E731
    setup_mod.consent_step(cfg, tmp_path, interactive=True, ask=_ask(["y", ""]), which=which)
    data = setup_mod.read_config_toml(cfg)
    assert data["night_allow_claude"] is True and data["night_allow_codex"] is False
    # やり直しても聞き直さない(聞いたら StopIteration で落ちる)
    setup_mod.consent_step(cfg, tmp_path, interactive=True, ask=_ask([]), which=which)
    assert setup_mod.read_config_toml(cfg)["night_allow_claude"] is True


def test_consent_step_non_interactive_and_eof_do_not_decide(tmp_path):
    cfg = tmp_path / "c.toml"
    which = lambda name: f"C:/bin/{name}.cmd" if name == "claude" else None  # noqa: E731
    notices = setup_mod.consent_step(cfg, tmp_path, interactive=False, ask=_ask([]), which=which)
    assert any("engram night --allow claude" in n for n in notices) and not cfg.exists()
    notices = setup_mod.consent_step(cfg, tmp_path, interactive=True, ask=_ask([EOFError()]), which=which)
    assert any("engram night --allow claude" in n for n in notices) and not cfg.exists()


def test_consent_step_respects_user_edited_agents(tmp_path):
    (tmp_path / "sleep_agents.txt").write_text("codex exec\n", encoding="utf-8")
    cfg = tmp_path / "c.toml"
    which = lambda name: f"C:/bin/{name}.cmd"  # noqa: E731
    setup_mod.consent_step(cfg, tmp_path, interactive=True, ask=_ask([]), which=which)
    assert not cfg.exists()  # 聞かず、勝手に同意済みとも書かない(その指定に従うだけ)
    assert night_mod.user_edited_agents(Settings(data_dir=tmp_path))
