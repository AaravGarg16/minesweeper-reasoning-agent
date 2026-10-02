"""Queue batches now, score them whenever.

    python -m llm_eval.submit --model claude-opus-5 --no-thinking
    python -m llm_eval.submit --status
    python -m llm_eval.submit --collect

Submitting returns as soon as the work is queued, so the machine can sleep.
Collecting reads the tickets left behind and scores whatever has finished.
"""

import argparse
import sys
from pathlib import Path

from . import batching, runner, scoring, task
from .__main__ import _print


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="python -m llm_eval.submit")
    p.add_argument("--model", action="append", help="repeatable; a Claude model id")
    p.add_argument("--data", type=Path, default=runner.DATASET)
    p.add_argument("--kind", choices=["decidable", "risky"])
    p.add_argument("--limit", type=int)
    p.add_argument("--cells", type=int, default=8)
    p.add_argument("--no-thinking", action="store_true")
    p.add_argument(
        "--effort",
        choices=["low", "medium", "high", "xhigh", "max"],
        help="bound how long the model reasons (default: the model's own)",
    )
    p.add_argument("--status", action="store_true", help="report on queued batches")
    p.add_argument("--collect", action="store_true", help="score finished batches")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    if args.status:
        found = batching.tickets()
        if not found:
            print("no tickets queued")
            return 0
        for path in found:
            ticket = batching.read_ticket(path)
            s = batching.status(ticket["batch_id"])
            think = "thinking" if ticket["thinking"] else "no thinking"
            print(
                f"{ticket['model']:<20} {think:<12} {s['state']:<12} "
                f"ok={s['succeeded']:<4} err={s['errored']:<4} "
                f"processing={s['processing']}"
            )
        return 0

    if args.collect:
        found = batching.tickets()
        if not found:
            print("no tickets to collect")
            return 0
        for path in found:
            ticket = batching.read_ticket(path)
            try:
                replies, usage = batching.collect(ticket["batch_id"])
            except RuntimeError as exc:
                print(f"{ticket['model']}: {exc}")
                continue

            items = runner.load(args.data)
            wanted = set(ticket["items"])
            items = [i for i in items if i["id"] in wanted]
            label = ticket["model"] + ("" if ticket["thinking"] else " (no thinking)")

            _, preds, report = runner.run(
                items, lambda qs: replies, model=label, cells=args.cells
            )
            print(f"\n{'=' * 60}\n{label}\n{'=' * 60}")
            _print(report)
            print(f"\nusage: {usage.summary(ticket['model'], batch=True)}")

            out = batching.TICKETS / f"{ticket['model']}-{'think' if ticket['thinking'] else 'nothink'}.json"
            runner.write(out, report, preds, scoring.reliability(preds))
            print(f"wrote {out}")
            path.rename(path.with_suffix(".done"))
        return 0

    if not args.model:
        print("error: give at least one --model, or use --status / --collect",
              file=sys.stderr)
        return 2

    items = runner.load(args.data, kind=args.kind, limit=args.limit)
    questions = [task.build(i, args.cells) for i in items]
    for model in args.model:
        ticket = batching.submit(
            questions, model, thinking=not args.no_thinking, effort=args.effort
        )
        path = batching.write_ticket(ticket)
        print(
            f"queued {len(questions)} positions on {model} "
            f"({'no thinking' if args.no_thinking else 'thinking'}) "
            f"-> {ticket['batch_id']}"
        )
        print(f"  ticket: {path}")
    print("\nSafe to close the laptop. Collect with:")
    print("  python -m llm_eval.submit --collect")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
