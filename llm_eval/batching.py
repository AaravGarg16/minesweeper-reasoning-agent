"""Submitting a batch and collecting it later, as two separate acts.

A batch runs on Anthropic's side for up to twenty-four hours and its results
stay available for twenty-nine days. A process that submits and then blocks
until the results arrive throws that away: close the laptop and the work is
lost, not because the batch stopped but because nothing was left to ask for
it. Worse, a run of several models submits the second only after the first
returns, so a sleeping machine means later models are never queued at all.

So submission writes a small ticket to disk and returns. Collection reads
the ticket whenever, which may be hours later, on a machine that has been
asleep in between.
"""

import json
import time
from pathlib import Path

from .models import Usage, _client
from .task import SYSTEM, Question, schema

TICKETS = Path(__file__).resolve().parent / "runs"


def submit(
    questions: list[Question],
    model: str,
    *,
    thinking: bool = True,
    max_tokens: int = 32000,
    effort: str | None = None,
) -> dict:
    """Queue every question as one batch and return a ticket describing it.

    `effort` bounds how long the model reasons. Left unset it runs at the
    default, which on this task means it will use very nearly whatever
    ceiling it is given -- 17k tokens per position at a 32k cap, up from 12k
    at 16k. That is not just expensive: the replies that run out are the ones
    scored as failures, so the positions that survive are the ones the model
    found easy, and the average flatters itself. A bounded effort keeps the
    sample honest as well as affordable.
    """
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    from .models import thinking_param

    client = _client()
    requests = []
    for q in questions:
        params = {
            "model": model,
            "max_tokens": max_tokens,
            "system": SYSTEM,
            "messages": [{"role": "user", "content": q.prompt}],
            "output_config": {"format": schema()},
        }
        field = thinking_param(model, thinking)
        if field is not None:
            params["thinking"] = field
        # Haiku has no effort control and rejects the field; its thinking is
        # already bounded by an explicit token budget instead.
        if effort and model.startswith(("claude-opus-", "claude-sonnet-", "claude-fable-")):
            params["output_config"]["effort"] = effort
        requests.append(
            Request(
                custom_id=q.item_id,
                params=MessageCreateParamsNonStreaming(**params),
            )
        )

    batch = client.messages.batches.create(requests=requests)
    return {
        "batch_id": batch.id,
        "model": model,
        "thinking": thinking,
        "items": [q.item_id for q in questions],
        "cells": {q.item_id: q.query_cells for q in questions},
        "submitted": time.time(),
    }


def status(batch_id: str) -> dict:
    client = _client()
    batch = client.messages.batches.retrieve(batch_id)
    counts = batch.request_counts
    return {
        "state": batch.processing_status,
        "processing": counts.processing,
        "succeeded": counts.succeeded,
        "errored": counts.errored,
        "canceled": counts.canceled,
        "expired": counts.expired,
    }


def collect(batch_id: str) -> tuple[dict[str, str], Usage]:
    """Fetch a finished batch. Raises if it is still running."""
    client = _client()
    batch = client.messages.batches.retrieve(batch_id)
    if batch.processing_status != "ended":
        raise RuntimeError(
            f"batch {batch_id} is {batch.processing_status}, not ready to collect"
        )

    usage = Usage()
    out: dict[str, str] = {}
    for result in client.messages.batches.results(batch_id):
        if result.result.type == "succeeded":
            usage.add(result.result.message.usage)
            out[result.custom_id] = "".join(
                b.text for b in result.result.message.content if b.type == "text"
            )
        else:
            detail = getattr(result.result, "error", None)
            reason = getattr(detail, "message", None) or result.result.type
            out[result.custom_id] = f"__error__:{reason}"
    return out, usage


def write_ticket(ticket: dict, path: Path | None = None) -> Path:
    path = path or TICKETS / f"ticket-{ticket['model']}-{ticket['batch_id'][-8:]}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(ticket, fh, indent=2)
    return path


def read_ticket(path: Path) -> dict:
    with open(path) as fh:
        return json.load(fh)


def tickets(directory: Path | None = None) -> list[Path]:
    directory = directory or TICKETS
    return sorted(directory.glob("ticket-*.json")) if directory.exists() else []
