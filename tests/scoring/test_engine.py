"""Unit tests for scoring/engine.py (docs/FRD.md, Section 6 / Task 11)."""

import pytest

from scoring.engine import assign_research_priority, compute_composite, load_weights_config

WEIGHTS = {
    "market_demand": 0.20,
    "competitive_intensity": 0.15,
    "margin_signal": 0.20,
    "cost_to_start": 0.15,
    "complexity": 0.10,
    "regulatory_friction": 0.10,
    "trend_momentum": 0.10,
}


def _score(value: float, confidence: float) -> dict:
    return {"value": value, "confidence": confidence}


def test_config_weights_yaml_loads_and_matches_frd_section_6():
    config = load_weights_config()
    assert config["weights"] == WEIGHTS
    assert config["research_priority_thresholds"] == {
        "research_this_week_percentile": 0.85,
        "revisit_later_percentile": 0.30,
    }


def test_composite_with_only_cheap_pass_dimensions_is_preliminary_and_hand_computed():
    scores = {
        "market_demand": _score(8, 0.9),
        "competitive_intensity": _score(4, 0.7),
        "margin_signal": _score(6, 0.5),
        "trend_momentum": _score(7, 0.8),
    }

    result = compute_composite(scores, WEIGHTS)

    # present weights: .20 + .15 + .20 + .10 = .65
    # composite = (8*.20 + 4*.15 + 6*.20 + 7*.10) / .65 = 4.1 / .65
    assert result["composite"] == pytest.approx(4.1 / 0.65)
    # confidence = (.9*.20 + .7*.15 + .5*.20 + .8*.10) / .65 = 0.465 / .65
    assert result["confidence"] == pytest.approx(0.465 / 0.65)
    assert result["status"] == "preliminary"


def test_composite_with_all_7_dimensions_is_final_and_hand_computed():
    scores = {
        "market_demand": _score(8, 0.9),
        "competitive_intensity": _score(4, 0.7),
        "margin_signal": _score(6, 0.5),
        "cost_to_start": _score(5, 0.6),
        "complexity": _score(3, 1.0),
        "regulatory_friction": _score(9, 0.4),
        "trend_momentum": _score(7, 0.8),
    }

    result = compute_composite(scores, WEIGHTS)

    # weights already sum to 1.0, so no renormalization changes anything
    assert result["composite"] == pytest.approx(6.05)
    assert result["confidence"] == pytest.approx(0.695)
    assert result["status"] == "final"


def test_single_dimension_renormalizes_to_its_own_raw_value():
    scores = {"market_demand": _score(7, 0.6)}

    result = compute_composite(scores, WEIGHTS)

    assert result["composite"] == pytest.approx(7.0)
    assert result["confidence"] == pytest.approx(0.6)
    assert result["status"] == "preliminary"


def test_unknown_dimension_raises_value_error():
    scores = {"margin": _score(6, 0.5)}  # wrong key: not "margin_signal"

    with pytest.raises(ValueError, match="margin"):
        compute_composite(scores, WEIGHTS)


def test_empty_scores_raises_value_error():
    with pytest.raises(ValueError):
        compute_composite({}, WEIGHTS)


THRESHOLDS = {"research_this_week_percentile": 0.85, "revisit_later_percentile": 0.30}


def test_priority_tiering_over_10_distinct_scores_matches_hand_computed_percentiles():
    ideas = [{"id": i, "composite": float(i)} for i in range(1, 11)]  # composites 1..10

    result = assign_research_priority(ideas, THRESHOLDS)

    labels = {idea["id"]: idea["research_priority"] for idea in result}
    # percentile(i) = i/10 -> 9,10 are >= .85; 3..8 are >= .30 and < .85; 1,2 are < .30
    assert labels[10] == "research_this_week"
    assert labels[9] == "research_this_week"
    for i in range(3, 9):
        assert labels[i] == "revisit_later"
    assert labels[2] == "deprioritize"
    assert labels[1] == "deprioritize"


def test_tied_composites_receive_the_same_label():
    ideas = [
        {"id": "a", "composite": 8.0},
        {"id": "b", "composite": 8.0},
        {"id": "c", "composite": 5.0},
        {"id": "d", "composite": 3.0},
        {"id": "e", "composite": 1.0},
    ]

    result = assign_research_priority(ideas, THRESHOLDS)

    labels = {idea["id"]: idea["research_priority"] for idea in result}
    assert labels["a"] == "research_this_week"
    assert labels["b"] == "research_this_week"
    assert labels["c"] == "revisit_later"
    assert labels["d"] == "revisit_later"
    assert labels["e"] == "deprioritize"


def test_priority_never_emits_a_verdict_label():
    ideas = [{"id": i, "composite": float(i)} for i in range(1, 6)]

    result = assign_research_priority(ideas, THRESHOLDS)

    allowed = {"research_this_week", "revisit_later", "deprioritize"}
    assert all(idea["research_priority"] in allowed for idea in result)


def test_does_not_mutate_input_ideas():
    ideas = [{"id": 1, "composite": 5.0}]

    result = assign_research_priority(ideas, THRESHOLDS)

    assert "research_priority" not in ideas[0]
    assert result[0]["research_priority"] == "research_this_week"
    assert result is not ideas


def test_empty_ideas_list_returns_empty_list():
    assert assign_research_priority([], THRESHOLDS) == []
