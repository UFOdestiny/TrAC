import os

from trac import config


def test_load_env_strips_inline_comments(tmp_path, monkeypatch):
    p = tmp_path / ".env"
    p.write_text("# comment\nTRAC_TEST_A=/x/y   # trailing comment\nTRAC_TEST_B='quoted'\n\nnot a pair\n")
    monkeypatch.delenv("TRAC_TEST_A", raising=False)
    monkeypatch.delenv("TRAC_TEST_B", raising=False)
    env = config.load_env(p)
    assert env == {"TRAC_TEST_A": "/x/y", "TRAC_TEST_B": "quoted"}
    assert os.environ["TRAC_TEST_A"] == "/x/y"


def test_environment_overrides_env_file(tmp_path, monkeypatch):
    p = tmp_path / ".env"
    p.write_text("TRAC_TEST_C=from_file\n")
    monkeypatch.setenv("TRAC_TEST_C", "from_env")
    config.load_env(p)
    assert os.environ["TRAC_TEST_C"] == "from_env"
