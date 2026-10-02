"""Getting answers out of a model, or out of something standing in for one.

Everything here returns the same shape -- a mapping from item id to raw
response text -- so the scoring code never learns which provider produced
it, and the whole pipeline can be exercised without a key or a bill.
"""

import json
import os
import time
from collections.abc import Callable

from .task import Question, schema

# Per million tokens, Anthropic list price. The Batch API halves both.
PRICING = {
    "claude-opus-5": (5.0, 25.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-haiku-4-5": (1.0, 5.0),
}

DEFAULT_MODEL = "claude-sonnet-5"

Responder = Callable[[list[Question]], dict[str, str]]


class Usage:
    """What a run actually consumed, as reported by the API.

    The cost estimate has to guess how much a model will think, and that
    guess drives the whole bill. This records what really happened, so the
    next run can be priced from a measurement instead of a guess.
    """

    def __init__(self):
        self.input_tokens = 0
        self.output_tokens = 0
        self.responses = 0

    def add(self, usage) -> None:
        self.input_tokens += getattr(usage, "input_tokens", 0) or 0
        self.output_tokens += getattr(usage, "output_tokens", 0) or 0
        self.responses += 1

    def cost(self, model: str, *, batch: bool) -> float:
        if model not in PRICING:
            return 0.0
        input_rate, output_rate = PRICING[model]
        total = (
            self.input_tokens * input_rate + self.output_tokens * output_rate
        ) / 1_000_000
        return total / 2 if batch else total

    def summary(self, model: str, *, batch: bool) -> str:
        if not self.responses:
            return "no usage reported"
        return (
            f"{self.input_tokens:,} in / {self.output_tokens:,} out over "
            f"{self.responses} responses "
            f"({self.output_tokens / self.responses:,.0f} output tokens each, "
            f"thinking included) = ${self.cost(model, batch=batch):.2f}"
        )


def base_rate_responder(questions: list[Question]) -> dict[str, str]:
    """The null model: answers `mines / covered` everywhere, ignoring the board.

    Not a mock. This is the control the skill score is measured against, and
    running it through the same path as a real model is what makes that
    comparison honest.
    """
    return {
        q.item_id: json.dumps({c: round(q.base_rate, 6) for c in q.query_cells})
        for q in questions
    }


def oracle_responder(questions: list[Question]) -> dict[str, str]:
    """Answers exactly. Confirms a perfect score is reachable through the harness."""
    return {
        q.item_id: json.dumps({c: q.truth[c] for c in q.query_cells})
        for q in questions
    }


def scripted_responder(replies: dict[str, str]) -> Responder:
    """Returns canned text per item id. For tests."""

    def respond(questions: list[Question]) -> dict[str, str]:
        return {q.item_id: replies.get(q.item_id, "{}") for q in questions}

    return respond


def thinking_param(model: str, thinking: bool) -> dict | None:
    """The `thinking` field for a model, or None to leave it off the request.

    Whatever this returns, `max_tokens` has to leave room for it. Thinking is
    counted against the same ceiling as the answer, so a budget that fits the
    answer alone produces a reply that is all reasoning and no JSON: the run
    scores as a parse failure and looks like the model could not answer.

    Omitting the field does not mean the same thing everywhere, which is the
    trap. On Opus 5 and Sonnet 5 thinking is on by default, so a request that
    simply leaves it out still reasons and still bills for it at the output
    rate. Turning it off there has to be said explicitly. Haiku 4.5 is the
    other way round: it has no adaptive mode, it wants an explicit token
    budget, and omitting the field is what turns it off.
    """
    adaptive = model.startswith(("claude-opus-", "claude-sonnet-", "claude-fable-"))
    if thinking:
        if adaptive:
            return {"type": "adaptive"}
        return {"type": "enabled", "budget_tokens": 2048}
    return {"type": "disabled"} if adaptive else None


def _client():
    try:
        import anthropic
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise RuntimeError(
            "the anthropic package is not installed -- pip install anthropic, "
            "or run with --model base-rate to exercise the harness without it"
        ) from exc
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Create a key at console.anthropic.com, "
            "or run with --model base-rate to exercise the harness without one."
        )
    # A key created at the organisation level rather than inside a workspace
    # is rejected unless the workspace is named on every request.
    workspace = os.environ.get("ANTHROPIC_WORKSPACE_ID")
    if workspace:
        return anthropic.Anthropic(
            default_headers={"anthropic-workspace-id": workspace}
        )
    return anthropic.Anthropic()


def estimate_cost(
    questions: list[Question],
    model: str,
    *,
    batch: bool,
    thinking_tokens: int = 0,
) -> float:
    """Rough dollar cost, so a run can be priced before it is paid for.

    `thinking_tokens` is the allowance per position, and on this task it
    dominates everything else. The answer itself is under a hundred tokens,
    but working out eight probabilities from overlapping constraints is not
    something a model does in a hundred tokens of reasoning. Thinking is
    billed at the output rate, so leaving it out of the estimate understates
    a run by roughly an order of magnitude. Pass the budget you intend to
    allow, and treat the result as a floor rather than a quote.
    """
    if model not in PRICING:
        return 0.0
    input_rate, output_rate = PRICING[model]
    # ~4 characters per token is close enough to size a bill.
    prompt_tokens = sum(len(q.prompt) for q in questions) / 4
    answer_tokens = sum(len(q.query_cells) for q in questions) * 12
    reply_tokens = answer_tokens + thinking_tokens * len(questions)
    cost = (prompt_tokens * input_rate + reply_tokens * output_rate) / 1_000_000
    return cost / 2 if batch else cost


def anthropic_responder(
    model: str = DEFAULT_MODEL,
    *,
    thinking: bool = True,
    max_tokens: int = 32000,
) -> Responder:
    """One request per position, answered live."""

    usage = Usage()

    def respond(questions: list[Question]) -> dict[str, str]:
        from .task import SYSTEM

        client = _client()
        out = {}
        for q in questions:
            kwargs = {
                "model": model,
                "max_tokens": max_tokens,
                "system": SYSTEM,
                "messages": [{"role": "user", "content": q.prompt}],
                "output_config": {"format": schema(q.query_cells)},
            }
            field = thinking_param(model, thinking)
            if field is not None:
                kwargs["thinking"] = field
            message = client.messages.create(**kwargs)
            usage.add(message.usage)
            out[q.item_id] = "".join(
                b.text for b in message.content if b.type == "text"
            )
        return out

    respond.usage = usage
    return respond


def anthropic_batch_responder(
    model: str = DEFAULT_MODEL,
    *,
    thinking: bool = True,
    max_tokens: int = 32000,
    poll_seconds: int = 30,
    on_status=None,
) -> Responder:
    """All positions in one batch, at half price.

    The work is not latency sensitive -- nobody is waiting on a benchmark --
    so this is the right default for anything past a smoke test.
    """

    usage = Usage()

    def respond(questions: list[Question]) -> dict[str, str]:
        from anthropic.types.message_create_params import (
            MessageCreateParamsNonStreaming,
        )
        from anthropic.types.messages.batch_create_params import Request

        from .task import SYSTEM

        client = _client()
        requests = []
        for q in questions:
            params = {
                "model": model,
                "max_tokens": max_tokens,
                "system": SYSTEM,
                "messages": [{"role": "user", "content": q.prompt}],
                "output_config": {"format": schema(q.query_cells)},
            }
            field = thinking_param(model, thinking)
            if field is not None:
                params["thinking"] = field
            requests.append(
                Request(
                    custom_id=q.item_id,
                    params=MessageCreateParamsNonStreaming(**params),
                )
            )

        batch = client.messages.batches.create(requests=requests)
        while True:
            current = client.messages.batches.retrieve(batch.id)
            if current.processing_status == "ended":
                break
            if on_status:
                on_status(current)
            time.sleep(poll_seconds)

        out = {}
        # Results come back in arbitrary order, so key by custom_id.
        for result in client.messages.batches.results(batch.id):
            if result.result.type == "succeeded":
                usage.add(result.result.message.usage)
                out[result.custom_id] = "".join(
                    b.text for b in result.result.message.content if b.type == "text"
                )
            else:
                out[result.custom_id] = f"__error__:{result.result.type}"
        return out

    respond.usage = usage
    return respond


def get(name: str, *, batch: bool = False, thinking: bool = True) -> Responder:
    if name == "base-rate":
        return base_rate_responder
    if name == "oracle":
        return oracle_responder
    if batch:
        return anthropic_batch_responder(name, thinking=thinking)
    return anthropic_responder(name, thinking=thinking)
