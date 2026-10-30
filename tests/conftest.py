"""テスト全体の前提。

最重要: **テストが常駐デーモンの実プロセスを起こさないようにする**。
R8.8.27 に、フック経路のテストが本物のデーモンを起動して 40 プロセスまで
積み上がる事故を起こした(テスト実行時間も 11秒→4分37秒へ悪化した)。
環境変数でデーモンの自動起動を全テスト共通で止める。
"""

from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True, scope="session")
def _no_daemon_autostart() -> None:
    """テスト中はデーモンの自動起動を禁止する(実プロセスを作らない)。"""
    os.environ["ENGRAM_DISABLE_DAEMON"] = "1"
