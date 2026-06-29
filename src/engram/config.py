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

    # --- 網羅検索(exhaustive: 沈んだ記憶の掘り起こし)---
    # fast/deep は活性度を加味するため、長く使われず沈んだ記憶は関連が高くても
    # 最終スコアで埋もれる(「忘れない記憶」なのに想起できない)。exhaustive は
    # 活性度を無視し関連度のみで全 tier/type を総当たりするので、沈んだ記憶も
    # 意味的に近ければ必ず浮上する。
    exhaustive_min_relevance: float = 0.30    # これ未満の関連度は返さない(ノイズ防止のゲート)
    exhaustive_score_threshold: float = 0.30  # deep の最高スコアがこれ未満なら exhaustive を自動発動

    # --- ヘッブ結合 ---
    colink_increment: float = 0.1
    colink_max: float = 1.0

    # --- 訂正(ハイパーコレクション効果)---
    correction_min_importance: int = 7

    # --- 統合(睡眠)---
    consolidate_min_age_days: int = 14
    consolidate_cluster_sim: float = 0.75

    # --- 自動符号化(セッション終了フック)---
    auto_encode: bool = True             # SessionEnd フックでの自動 episode 保存
    auto_episode_importance: int = 3     # 自動要約 episode の importance(粗い記録なので低め)
    auto_encode_min_chars: int = 16      # ユーザー発言の合計がこれ未満のセッションは記録しない

    # --- 自発的想起(surface)---
    surface_mode: str = "shadow"         # off | shadow(ログのみ) | active(文脈に差し込む)
    surface_threshold: float = 0.45      # このスコア以上の記憶だけ浮上する
    # 関連度の最低ライン(ゲート)。活性度・重要度がいくら高くても、発話との
    # 字句関連がこれ未満なら浮上しない。よく使う重要な記憶が無関係な文脈に
    # 出しゃばる事故を防ぐ(実データで閾値ぎりぎりまで来た実例があった)
    surface_min_relevance: float = 0.25
    surface_max_items: int = 2           # 1プロンプトで差し込む最大件数
    surface_min_prompt_chars: int = 8    # これより短い発言では想起しない

    # --- 記憶の部屋(仕事/個人の文脈分離)---
    # {ディレクトリのプレフィックス: 部屋名}。最長一致で判定、該当なしは "common"
    room_paths: dict = field(default_factory=dict)

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


def resolve_room(cwd: str | Path | None, room_paths: dict) -> str:
    """作業ディレクトリから記憶の部屋を決める。

    room_paths のキー(ディレクトリのプレフィックス)と最長一致で照合する。
    区切り文字(\\ と /)と大文字小文字の差は吸収する。該当なしは "common"。
    """
    if not cwd or not room_paths:
        return "common"
    norm = str(cwd).replace("\\", "/").casefold().rstrip("/")
    best_len = -1
    best_room = "common"
    for prefix, room in room_paths.items():
        p = str(prefix).replace("\\", "/").casefold().rstrip("/")
        if not p:
            continue
        if (norm == p or norm.startswith(p + "/")) and len(p) > best_len:
            best_len = len(p)
            best_room = str(room)
    return best_room
