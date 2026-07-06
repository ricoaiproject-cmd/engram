"""記憶力学コアの性質テスト。人間の記憶の振る舞いを満たすことを確認する。"""

import math

from engram import dynamics as dyn

DAY = 86400.0
NOW = 1_750_000_000.0


def ev(days_ago: float, weight: float = 1.0) -> tuple[float, float]:
    return (NOW - days_ago * DAY, weight)


# --- 減衰率(フラッシュバルブ記憶) ---

def test_decay_rate_importance_slows_decay():
    assert dyn.decay_rate(10) < dyn.decay_rate(5) < dyn.decay_rate(1)
    assert dyn.decay_rate(5) == 0.5


def test_decay_rate_clamped():
    assert dyn.decay_rate(10) == 0.3
    assert dyn.decay_rate(1) == 0.6
    assert dyn.decay_rate(-100) == 0.6
    assert dyn.decay_rate(100) == 0.3


# --- 初期符号化ブースト ---

def test_create_weight_scales_with_importance():
    assert dyn.create_event_weight(10) == 3.0
    assert dyn.create_event_weight(5) == 2.0
    assert dyn.create_event_weight(1) < dyn.create_event_weight(10)


# --- 活性度 ---

def test_recent_beats_old():
    recent = dyn.activation_norm([ev(1)], NOW, 0.5)
    old = dyn.activation_norm([ev(300)], NOW, 0.5)
    assert recent > old > 0.0


def test_frequent_beats_rare():
    frequent = dyn.activation_norm([ev(d) for d in (2, 5, 10, 20, 40)], NOW, 0.5)
    rare = dyn.activation_norm([ev(20)], NOW, 0.5)
    assert frequent > rare


def test_old_memory_never_zero():
    """放置された記憶も活性度は 0 にならない = 消えない。"""
    ancient = dyn.activation_norm([ev(365 * 5)], NOW, 0.5)
    assert ancient > 0.0


def test_flashbulb_resists_forgetting():
    """高 importance の未使用記憶は、低 importance の未使用記憶より
    90日後でも大幅に想起しやすい(初期ブースト + 低減衰の複合効果)。"""
    events_hi = [ev(90, dyn.create_event_weight(10))]
    events_lo = [ev(90, dyn.create_event_weight(2))]
    hi = dyn.activation_norm(events_hi, NOW, dyn.decay_rate(10))
    lo = dyn.activation_norm(events_lo, NOW, dyn.decay_rate(2))
    assert hi > lo * 2


def test_min_elapsed_clamp():
    """直後のアクセスで発散しない。未来時刻(時計ずれ)も安全。"""
    s_now = dyn.base_strength([(NOW, 1.0)], NOW, 0.5)
    s_future = dyn.base_strength([(NOW + 999, 1.0)], NOW, 0.5)
    assert s_now == s_future == 60.0 ** -0.5


def test_activation_empty_is_neg_inf_and_norm_zero():
    assert dyn.activation([], NOW, 0.5) == float("-inf")
    assert dyn.activation_norm([], NOW, 0.5) == 0.0


# --- 最終スコア ---

def test_relevance_dominates():
    """関連度が低い高活性記憶は、関連度が高い低活性記憶に勝てない
    (富者益富ループの防止)。"""
    irrelevant_but_hot = dyn.final_score(0.2, 1.0, 10)
    relevant_but_cold = dyn.final_score(0.9, 0.0, 3)
    assert relevant_but_cold > irrelevant_but_hot


# --- RRF ---

def test_rrf_merge_rewards_agreement():
    scores = dyn.rrf_merge([["a", "b", "c"], ["b", "a", "d"]])
    assert scores["a"] > scores["c"]
    assert scores["b"] > scores["d"]
    ranked = sorted(scores, key=scores.get, reverse=True)
    assert set(ranked[:2]) == {"a", "b"}


def test_rrf_single_list_preserves_order():
    scores = dyn.rrf_merge([["x", "y"]])
    assert scores["x"] > scores["y"]


# --- 拡散活性化(deep recall) ---

def _graph(edges: dict[str, list[tuple[str, float]]]):
    return lambda node: edges.get(node, [])


def test_spread_reaches_two_hops():
    nbrs = _graph({"seed": [("mid", 1.0)], "mid": [("far", 1.0)]})
    out = dyn.spread({"seed": 1.0}, nbrs, max_hops=2, hop_decay=0.7)
    assert out["seed"] == 1.0
    assert math.isclose(out["mid"], 0.7)
    assert math.isclose(out["far"], 0.49)


def test_spread_respects_hop_limit():
    nbrs = _graph({"a": [("b", 1.0)], "b": [("c", 1.0)], "c": [("d", 1.0)]})
    out = dyn.spread({"a": 1.0}, nbrs, max_hops=2, hop_decay=0.7)
    assert "d" not in out


def test_spread_takes_best_path():
    nbrs = _graph({"s1": [("t", 0.2)], "s2": [("t", 1.0)]})
    out = dyn.spread({"s1": 1.0, "s2": 1.0}, nbrs, max_hops=1, hop_decay=0.7)
    assert math.isclose(out["t"], 0.7)


def test_spread_weak_links_attenuate():
    nbrs = _graph({"seed": [("weak", 0.1), ("strong", 0.9)]})
    out = dyn.spread({"seed": 1.0}, nbrs, max_hops=1, hop_decay=0.7)
    assert out["strong"] > out["weak"] > 0.0


# --- 関連度の候補内 min-max 正規化(コサイン圧縮対策) ---

def test_normalize_stretches_compressed_band():
    """Ruri 的な圧縮帯(0.8〜0.9)が 0..1 へ引き伸ばされること。"""
    out = dyn.normalize_relevances({"a": 0.80, "b": 0.90, "c": 0.85}, floor=0.10)
    assert math.isclose(out["a"], 0.0)
    assert math.isclose(out["b"], 1.0)
    assert math.isclose(out["c"], 0.5)


def test_normalize_floor_damps_tiny_spread():
    """候補間の差が床未満のとき、微小差を全力増幅しないこと。"""
    out = dyn.normalize_relevances({"a": 0.80, "b": 0.82}, floor=0.10)
    assert math.isclose(out["a"], 0.0)
    assert math.isclose(out["b"], 0.2)  # 0.02/0.10。1.0 まで増幅しない


def test_normalize_preserves_order():
    vals = {"a": 0.81, "b": 0.87, "c": 0.84, "d": 0.79}
    out = dyn.normalize_relevances(vals, floor=0.10)
    order_in = sorted(vals, key=vals.get)
    order_out = sorted(out, key=out.get)
    assert order_in == order_out


def test_normalize_edge_cases():
    assert dyn.normalize_relevances({}) == {}
    single = dyn.normalize_relevances({"only": 0.85}, floor=0.10)
    assert math.isclose(single["only"], 0.0)  # 1候補は相対差ゼロ扱い
