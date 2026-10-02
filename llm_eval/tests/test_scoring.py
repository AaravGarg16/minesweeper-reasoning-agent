"""The metrics, and the two fixed points that give them meaning.

A Brier score on its own is unreadable -- whether 0.14 is good depends
entirely on how hard the positions were. The skill score fixes that by
anchoring to the base rate, so these tests pin both ends: answering exactly
scores 1, and answering the base rate everywhere scores 0.
"""

import pytest

from llm_eval import models, runner, scoring, task


def _pred(predicted, truth, base_rate=0.2, kind="risky", error=None):
    return scoring.Prediction(
        item_id="x", kind=kind, predicted=predicted, truth=truth,
        base_rate=base_rate, error=error,
    )


def test_exact_answers_score_zero_brier():
    p = _pred({"a": 0.25, "b": 1.0}, {"a": 0.25, "b": 1.0})
    assert scoring.brier(scoring._pairs([p])) == pytest.approx(0.0)


def test_skill_is_one_for_exact_and_zero_for_the_base_rate():
    truth = {"a": 0.0, "b": 0.5}
    exact = scoring.score("m", [_pred(dict(truth), truth, base_rate=0.2)])
    assert exact.skill == pytest.approx(1.0)

    flat = scoring.score(
        "m", [_pred({"a": 0.2, "b": 0.2}, truth, base_rate=0.2)]
    )
    assert flat.skill == pytest.approx(0.0)


def test_skill_goes_negative_when_worse_than_the_base_rate():
    truth = {"a": 0.0}
    report = scoring.score("m", [_pred({"a": 1.0}, truth, base_rate=0.2)])
    assert report.skill < 0


def test_overconfidence_sign_points_the_right_way():
    truth = {"a": 0.2}
    high = scoring.score("m", [_pred({"a": 0.9}, truth)])
    low = scoring.score("m", [_pred({"a": 0.0}, truth)])
    assert high.overconfidence > 0, "calling a cell more dangerous reads positive"
    assert low.overconfidence < 0


def test_safe_recall_counts_only_provably_safe_cells():
    truth = {"a": 0.0, "b": 0.0, "c": 0.4}
    report = scoring.score("m", [_pred({"a": 0.0, "b": 0.3, "c": 0.0}, truth)])
    assert report.safe_recall == pytest.approx(0.5)      # found a, missed b
    assert report.safe_precision == pytest.approx(0.5)   # called a and c safe


def test_failed_predictions_are_counted_but_not_scored():
    truth = {"a": 0.5}
    report = scoring.score(
        "m",
        [_pred({}, truth, error="malformed JSON"), _pred({"a": 0.5}, truth)],
    )
    assert report.items == 2
    assert report.failed == 1
    assert report.cells == 1, "the failed item contributes no cells"
    assert report.brier == pytest.approx(0.0)


def test_calibration_error_is_zero_when_stated_matches_actual():
    preds = [
        _pred({"a": 0.5}, {"a": 0.5}),
        _pred({"a": 0.1}, {"a": 0.1}),
    ]
    assert scoring.expected_calibration_error(scoring._pairs(preds)) == pytest.approx(0.0)


def test_reliability_curve_buckets_by_stated_probability():
    preds = [_pred({"a": 0.05}, {"a": 0.0}), _pred({"a": 0.95}, {"a": 1.0})]
    rows = scoring.reliability(preds)
    assert [r["bin"] for r in rows] == ["0.0-0.1", "0.9-1.0"]
    assert rows[0]["actual"] == pytest.approx(0.0)
    assert rows[1]["actual"] == pytest.approx(1.0)


def test_the_oracle_responder_scores_perfectly_through_the_whole_pipeline():
    items = runner.load(limit=12)
    _, _, report = runner.run(items, models.oracle_responder, model="oracle")
    assert report.failed == 0
    assert report.brier == pytest.approx(0.0)
    assert report.skill == pytest.approx(1.0)
    assert report.safe_recall == 1.0


def test_the_base_rate_responder_scores_exactly_zero_skill():
    items = runner.load(limit=40)
    _, _, report = runner.run(items, models.base_rate_responder, model="base-rate")
    assert report.failed == 0
    # The control rounds its answers to six places so the JSON stays readable,
    # which leaves skill a hair off zero rather than exactly zero.
    assert report.skill == pytest.approx(0.0, abs=1e-6)


def test_a_garbled_reply_is_recorded_as_a_failure_not_a_bad_score():
    items = runner.load(limit=4)
    responder = models.scripted_responder({items[0]["id"]: "sorry, I can't"})
    _, preds, report = runner.run(items, responder, model="broken")
    assert report.failed == len(items)
    assert all(p.error for p in preds)
    assert report.cells == 0


def test_cost_estimate_halves_under_the_batch_api():
    questions = [task.build(i) for i in runner.load(limit=20)]
    live = models.estimate_cost(questions, "claude-sonnet-5", batch=False)
    batched = models.estimate_cost(questions, "claude-sonnet-5", batch=True)
    assert live > 0
    assert batched == pytest.approx(live / 2)


def test_thinking_is_turned_off_explicitly_where_it_defaults_on():
    """Omitting the field is not the same as disabling it on every model."""
    assert models.thinking_param("claude-opus-5", False) == {"type": "disabled"}
    assert models.thinking_param("claude-sonnet-5", False) == {"type": "disabled"}
    # Haiku has no adaptive mode and is off unless given a budget, so the
    # field is left off the request entirely.
    assert models.thinking_param("claude-haiku-4-5", False) is None


def test_thinking_on_uses_the_mode_each_model_actually_accepts():
    assert models.thinking_param("claude-opus-5", True) == {"type": "adaptive"}
    haiku = models.thinking_param("claude-haiku-4-5", True)
    assert haiku["type"] == "enabled"
    assert haiku["budget_tokens"] >= 1024
