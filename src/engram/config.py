"""設定。優先順位: 環境変数 > ~/.engram/config.toml > 既定値。

配布可能にするため、コードの設置場所には一切依存しない。
記憶の保存先などユーザーごとの値は config.toml に永続化する(engram setup が生成)。
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path


def _engram_home() -> Path:
    # SQLite を Google Drive 等の同期下に置くとロック破損リスクがあるためローカル側。
    # %LOCALAPPDATA% は MSIX サンドボックスで仮想化されることがあるため ~/.engram を使う
    return Path(os.environ.get("ENGRAM_HOME", str(Path.home() / ".engram")))


@dataclass(frozen=True)
class Settings:
    # --- 配置 ---
    memories_dir: Path = field(
        default_factory=lambda: _engram_home() / "memories"
    )
    data_dir: Path = field(default_factory=_engram_home)

    # --- 埋め込み(Ruri-v3 はプレフィックス必須)---
    embed_model: str = "cl-nagoya/ruri-v3-130m"
    query_prefix: str = "検索クエリ: "
    doc_prefix: str = "検索文書: "

    # --- 再ランクの重み(関連度を支配的にして富者益富ループを防ぐ)---
    w_relevance: float = 0.6
    w_activation: float = 0.25
    w_importance: float = 0.15

    # --- アクセスイベントの重み ---
    recall_hit_weight: float = 0.3   # recall で返却されただけ(弱い強化)
    reinforce_weight: float = 1.0    # 「実際に役立った」報告(強い強化)
    reinforce_strength_max: float = 3.0
    create_alpha: float = 2.0        # 初期符号化ブースト: w = 1 + α·(importance/10)

    # --- 減衰(フラッシュバルブ記憶: 高 importance ほど減衰が遅い)---
    decay_base: float = 0.5
    decay_spread: float = 0.2        # d_i = clamp(base − spread·(imp−5)/5, min, max)
    decay_min: float = 0.3
    decay_max: float = 0.6
    min_elapsed_seconds: float = 60.0  # t^-d の発散防止(直後アクセスの経過時間下限)

    # --- 検索 ---
    candidate_k: int = 50            # ベクトル/FTS 各候補数
    rrf_k: int = 60                  # Reciprocal Rank Fusion の定数
    # これ以上の cos 類似は重複とみなす。Ruri-v3 は同一話題の短い日本語文
    # 同士でも cos が高めに出るため、0.92 では別事実が併合された実例があり
    # 0.95 に調整(2026-06-11)
    dup_threshold: float = 0.95
    deep_score_threshold: float = 0.35  # fast の最高スコアがこれ未満なら deep を自動発動
    hop_decay: float = 0.7           # 拡散活性化のホップ毎減衰
    max_hops: int = 2

    # --- ヘッブ結合 ---
    colink_increment: float = 0.1
    colink_max: float = 1.0

    # --- 訂正(ハイパーコレクション効果)---
    correction_min_importance: int = 7

    # --- 統合(睡眠)---
    consolidate_min_age_days: int = 14
    consolidate_cluster_sim: float = 0.75

    @property
    def db_path(self) -> Path:
        return self.data_dir / "index.db"


def config_path() -> Path:
    """ユーザー設定ファイルの場所(engram setup が生成する)。"""
    return _engram_home() / "config.toml"


def _load_config_file() -> dict:
    path = config_path()
    if not path.is_file():
        return {}
    try:
        with path.open("rb") as f:
            return tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError):
        return {}  # 壊れた設定ファイルで起動不能にしない(既定値で動かす)


_PATH_FIELDS = {"memories_dir", "data_dir"}
_ENV_OVERRIDES = {
    "memories_dir": "ENGRAM_MEMORIES_DIR",
    "data_dir": "ENGRAM_DATA_DIR",
    "embed_model": "ENGRAM_EMBED_MODEL",
}


def get_settings() -> Settings:
    """既定値 < config.toml < 環境変数 の順で上書きして Settings を返す。"""
    overrides: dict = {}
    file_conf = _load_config_file()
    valid = {f.name: f.type for f in fields(Settings)}

    for key, value in file_conf.items():
        if key not in valid:
            continue
        overrides[key] = Path(value) if key in _PATH_FIELDS else value

    for key, env_name in _ENV_OVERRIDES.items():
        value = os.environ.get(env_name)
        if value:
            overrides[key] = Path(value) if key in _PATH_FIELDS else value

    return Settings(**overrides)
