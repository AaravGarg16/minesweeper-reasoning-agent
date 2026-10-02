"""Load the benchmark, ask a model, score what comes back."""

import json
from pathlib import Path

from . import models, scoring, task

DATASET = Path(__file__).resolve().parent / "data" / "minebench-v1.json"


def load(path: Path = DATASET, *, kind: str | None = None, limit: int | None = None):
    with open(path) as fh:
        payload = json.load(fh)
    items = payload["items"]
    if kind:
        items = [i for i in items if i["kind"] == kind]
    if limit is not None:
        items = items[:limit]
    return items


def run(items: list[dict], responder, *, model: str, cells: int = task.QUERY_CELLS):
    questions = [task.build(item, cells) for item in items]
    replies = responder(questions)

    preds = []
    for q in questions:
        text = replies.get(q.item_id)
        if text is None:
            preds.append(_failed(q, "no response"))
            continue
        if text.startswith("__error__:"):
            preds.append(_failed(q, text.removeprefix("__error__:")))
            continue
        try:
            predicted = task.parse(text, q.query_cells)
        except task.ParseError as exc:
            preds.append(_failed(q, str(exc)))
            continue
        preds.append(
            scoring.Prediction(
                item_id=q.item_id,
                kind=q.kind,
                predicted=predicted,
                truth=q.truth,
                base_rate=q.base_rate,
            )
        )
    return questions, preds, scoring.score(model, preds)


def _failed(q: task.Question, reason: str) -> scoring.Prediction:
    return scoring.Prediction(
        item_id=q.item_id,
        kind=q.kind,
        predicted={},
        truth=q.truth,
        base_rate=q.base_rate,
        error=reason,
    )


def write(path: Path, report: scoring.Report, preds, curve) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "report": report.as_dict(),
        "reliability": curve,
        "predictions": [
            {
                "id": p.item_id,
                "kind": p.kind,
                "error": p.error,
                "predicted": p.predicted,
                "truth": p.truth,
            }
            for p in preds
        ],
    }
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
