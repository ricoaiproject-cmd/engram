"""記憶力学のコア — ACT-R 活性化モデル + RRF + 拡散活性化。

設計原則: 埋め込み(意味の位置)は固定し、活性度という別軸で検索順位を変調する。
活性度はアクセスイベント列からクエリ時に都度計算する(バッチ減衰更新が不要)。
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence


def decay_rate(
    importance: int,
    *,
    base: float = 0.5,
    spread: float = 0.2,
    d_min: float = 0.3,
    d_max: float = 0.6,
) -> float:
    """記憶ごとの減衰指数 d_i(フラッシュバルブ記憶の実装)。

    重大な文脈で得た記憶(高 importance)は減衰が遅く、リハーサルなしでも
    長期間想起圏内に残る。些末な記憶は速く沈む。
        d_i = clamp(base − spread·(importance − 5)/5, d_min, d_max)
    """
    d = base - spread * (importance - 5) / 5
    return max(d_min, min(d_max, d))


def create_event_weight(importance: int, *, alpha: float = 2.0) -> float:
    """初期符号化ブースト。importance 10 の記憶は誕生時点で
    「既に複数回使われた記憶」相当の強度(w = 1 + α)で始まる。"""
    return 1.0 + alpha * (importance / 10.0)


def base_strength(
    events: Iterable[tuple[float, float]],
    now: float,
    decay: float,
    *,
    min_elapsed: float = 60.0,
) -> float:
    """ACT-R 基底レベルの内側の和 S = Σ_j w_j · (now − t_j)^(−d)。

    events: (unix秒タイムスタンプ, 重み) の列。
    経過時間は min_elapsed 秒でクランプし、直後アクセスでの発散を防ぐ。
    未来のタイムスタンプ(時計ずれ)も min_elapsed 扱い。
    """
    s = 0.0
    for ts, weight in events:
        elapsed = max(now - ts, min_elapsed)
        s += weight * elapsed ** (-decay)
    return s


def activation(events: Iterable[tuple[float, float]], now: float, decay: float,
               *, min_elapsed: float = 60.0) -> float:
    """ACT-R 基底レベル活性度 B = ln(S)。イベントが無ければ -inf。"""
    s = base_strength(events, now, decay, min_elapsed=min_elapsed)
    return math.log(s) if s > 0 else float("-inf")


def activation_norm(events: Iterable[tuple[float, float]], now: float, decay: float,
                    *, min_elapsed: float = 60.0,
                    center: float = -6.0, scale: float = 1.5) -> float:
    """活性度を 0..1 に正規化した値。sigmoid((ln S − center) / scale)。

    S は秒単位のべき乗減衰なので日スケールでは 1e-4〜1e-2 程度になり、
    素朴な S/(S+1) では差が潰れる。ACT-R 本来の流儀どおり対数空間
    (B = ln S)で比較し、シグモイドの中心・幅を日〜月スケールの差が
    判別域(おおよそ 0.1〜0.9)に乗るよう校正している:
      - 作成直後:               ≈ 0.95(強い新近性)
      - 1日前に1回アクセス:      ≈ 0.55
      - 90日放置(importance 5): ≈ 0.25
      - 90日放置(importance 10):≈ 0.8 (フラッシュバルブ)
    S=0(イベント無し)→ 0。再ランクの加重和に直接使える。
    """
    s = base_strength(events, now, decay, min_elapsed=min_elapsed)
    if s <= 0.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-(math.log(s) - center) / scale))


def final_score(
    relevance: float,
    act_norm: float,
    importance: int,
    *,
    w_relevance: float = 0.6,
    w_activation: float = 0.25,
    w_importance: float = 0.15,
) -> float:
    """検索の最終スコア。関連度を支配的にし、活性度・重要度は変調に留める
    (よく使う記憶が無関係な文脈に出しゃばる富者益富ループの防止)。"""
    return (
        w_relevance * relevance
        + w_activation * act_norm
        + w_importance * (importance / 10.0)
    )


def rrf_merge(rankings: Sequence[Sequence[str]], *, k: int = 60) -> dict[str, float]:
    """Reciprocal Rank Fusion。複数の順位リスト(ベクトル近傍・BM25 等)を統合する。

    score(id) = Σ_r 1/(k + rank_r(id))   (rank は 1 始まり、リストに無ければ寄与 0)
    戻り値はスコア降順を保証しない dict。呼び出し側でソートする。
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, mem_id in enumerate(ranking, start=1):
            scores[mem_id] = scores.get(mem_id, 0.0) + 1.0 / (k + rank)
    return scores


def spread(
    seeds: Mapping[str, float],
    neighbors: Callable[[str], Iterable[tuple[str, float]]],
    *,
    max_hops: int = 2,
    hop_decay: float = 0.7,
) -> dict[str, float]:
    """拡散活性化(deep recall の「辿れば必ず引っ張ってこれる」の実装)。

    seeds の各記憶から連想リンクを max_hops まで広げ、
        propagated = seed_score · Π_hop (link_weight · hop_decay)
    を伝播する。複数経路で到達した場合は最大値を採る。
    neighbors(id) は (隣接id, リンク重み 0..1) を返す関数。
    戻り値は seed 自身を含む全到達ノードのスコア(seed は元の値を保持)。
    """
    best: dict[str, float] = dict(seeds)
    frontier: dict[str, float] = dict(seeds)
    for _ in range(max_hops):
        next_frontier: dict[str, float] = {}
        for node, score in frontier.items():
            for nbr, link_w in neighbors(node):
                w = max(0.0, min(1.0, link_w))
                propagated = score * w * hop_decay
                if propagated > best.get(nbr, 0.0):
                    best[nbr] = propagated
                    next_frontier[nbr] = max(next_frontier.get(nbr, 0.0), propagated)
        if not next_frontier:
            break
        frontier = next_frontier
    return best
