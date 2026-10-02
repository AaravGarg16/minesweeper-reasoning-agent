"""Scoring a set of predicted probabilities against exact ones.

Three things are worth separating, because a model can be good at one and
useless at another:

  resolution    on decidable positions, does it find the cells that are
                provably safe? This is deduction, and it has a right answer.

  calibration   when it says 0.3, are three in ten of those cells mines?
                Measured by Brier score and by expected calibration error.

  skill         is it beating the trivial predictor that ignores the board
                and answers `mines / covered` everywhere? Brier score alone
                cannot tell you this, because an easy set of positions
                gives a low score to anything. The skill score does: 0 means
                no better than the base rate, 1 means exact.
"""

from dataclasses import dataclass, field

BINS = 10


@dataclass
class Prediction:
    item_id: str
    kind: str
    predicted: dict[str, float]
    truth: dict[str, float]
    base_rate: float
    error: str | None = None


@dataclass
class Report:
    model: str
    items: int = 0
    parsed: int = 0
    failed: int = 0
    cells: int = 0
    brier: float = 0.0
    base_brier: float = 0.0
    skill: float = 0.0
    ece: float = 0.0
    mean_absolute: float = 0.0
    overconfidence: float = 0.0
    safe_recall: float | None = None
    safe_precision: float | None = None
    mine_recall: float | None = None
    by_kind: dict[str, dict] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


def _pairs(preds: list[Prediction]) -> list[tuple[float, float, float]]:
    """(predicted, truth, base_rate) for every scored cell."""
    out = []
    for p in preds:
        if p.error is not None:
            continue
        for cell, truth in p.truth.items():
            out.append((p.predicted[cell], truth, p.base_rate))
    return out


def brier(pairs) -> float:
    """Mean squared error against the exact probability.

    Scored against the true probability rather than the realised outcome.
    The outcome is a coin flip the oracle already integrated out, so using
    it would add variance that says nothing about the model.
    """
    return sum((p - t) ** 2 for p, t, _ in pairs) / len(pairs) if pairs else 0.0


def expected_calibration_error(pairs, bins: int = BINS) -> float:
    """Average gap between stated probability and truth, bucketed by stated."""
    if not pairs:
        return 0.0
    buckets: list[list[tuple[float, float]]] = [[] for _ in range(bins)]
    for p, t, _ in pairs:
        index = min(bins - 1, int(p * bins))
        buckets[index].append((p, t))
    total = 0.0
    for bucket in buckets:
        if not bucket:
            continue
        mean_p = sum(p for p, _ in bucket) / len(bucket)
        mean_t = sum(t for _, t in bucket) / len(bucket)
        total += len(bucket) / len(pairs) * abs(mean_p - mean_t)
    return total


def _resolution(preds: list[Prediction]) -> tuple[float | None, float | None, float | None]:
    """How reliably provable cells are recognised as provable."""
    called_safe = truly_safe = hit_safe = 0
    truly_mine = hit_mine = 0
    for p in preds:
        if p.error is not None:
            continue
        for cell, truth in p.truth.items():
            said = p.predicted[cell]
            if said <= 0.005:
                called_safe += 1
                if truth == 0.0:
                    hit_safe += 1
            if truth == 0.0:
                truly_safe += 1
            if truth == 1.0:
                truly_mine += 1
                if said >= 0.995:
                    hit_mine += 1
    return (
        hit_safe / truly_safe if truly_safe else None,
        hit_safe / called_safe if called_safe else None,
        hit_mine / truly_mine if truly_mine else None,
    )


def score(model: str, preds: list[Prediction]) -> Report:
    pairs = _pairs(preds)
    b = brier(pairs)
    base = sum((r - t) ** 2 for _, t, r in pairs) / len(pairs) if pairs else 0.0
    safe_recall, safe_precision, mine_recall = _resolution(preds)

    report = Report(
        model=model,
        items=len(preds),
        parsed=sum(1 for p in preds if p.error is None),
        failed=sum(1 for p in preds if p.error is not None),
        cells=len(pairs),
        brier=b,
        base_brier=base,
        skill=(1 - b / base) if base > 0 else 0.0,
        ece=expected_calibration_error(pairs),
        mean_absolute=(
            sum(abs(p - t) for p, t, _ in pairs) / len(pairs) if pairs else 0.0
        ),
        # Positive means the model calls cells more dangerous than they are.
        overconfidence=(
            sum(p - t for p, t, _ in pairs) / len(pairs) if pairs else 0.0
        ),
        safe_recall=safe_recall,
        safe_precision=safe_precision,
        mine_recall=mine_recall,
    )

    for kind in sorted({p.kind for p in preds}):
        group = [p for p in preds if p.kind == kind]
        sub = _pairs(group)
        sub_base = (
            sum((r - t) ** 2 for _, t, r in sub) / len(sub) if sub else 0.0
        )
        sub_brier = brier(sub)
        report.by_kind[kind] = {
            "items": len(group),
            "failed": sum(1 for p in group if p.error is not None),
            "cells": len(sub),
            "brier": sub_brier,
            "base_brier": sub_base,
            "skill": (1 - sub_brier / sub_base) if sub_base > 0 else 0.0,
            "ece": expected_calibration_error(sub),
        }
    return report


def reliability(preds: list[Prediction], bins: int = BINS) -> list[dict]:
    """Per-bucket stated-vs-actual, which is the calibration curve."""
    pairs = _pairs(preds)
    buckets: list[list[tuple[float, float]]] = [[] for _ in range(bins)]
    for p, t, _ in pairs:
        buckets[min(bins - 1, int(p * bins))].append((p, t))
    rows = []
    for i, bucket in enumerate(buckets):
        if not bucket:
            continue
        rows.append(
            {
                "bin": f"{i / bins:.1f}-{(i + 1) / bins:.1f}",
                "count": len(bucket),
                "stated": sum(p for p, _ in bucket) / len(bucket),
                "actual": sum(t for _, t in bucket) / len(bucket),
            }
        )
    return rows
