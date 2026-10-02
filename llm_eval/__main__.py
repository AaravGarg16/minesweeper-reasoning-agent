"""Command line entry point: python -m llm_eval --model claude-sonnet-5

Scores a language model's mine probabilities against exact ones. Start with
`--model base-rate`, which needs no API key and shows what a score looks
like for something that does no reasoning at all.

Real runs read ANTHROPIC_API_KEY. A key issued at the organisation level
rather than inside a workspace also needs ANTHROPIC_WORKSPACE_ID, or every
request comes back 400.
"""

import argparse
import sys
import time
from pathlib import Path

from . import models, runner, scoring


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="python -m llm_eval",
        description="Score a model's mine probabilities against exact ones.",
    )
    p.add_argument(
        "--model",
        default="base-rate",
        help="a Claude model id, or 'base-rate' / 'oracle' to run without a "
        "key (default: %(default)s)",
    )
    p.add_argument("--data", type=Path, default=runner.DATASET, help="dataset path")
    p.add_argument(
        "--kind",
        choices=["decidable", "risky"],
        help="score only one flavour of position (default: both)",
    )
    p.add_argument("--limit", type=int, help="first N positions only")
    p.add_argument(
        "--cells", type=int, default=8, help="query cells per position (default: %(default)s)"
    )
    p.add_argument("--batch", action="store_true", help="use the Batch API (half price)")
    p.add_argument(
        "--no-thinking", action="store_true", help="disable extended thinking"
    )
    p.add_argument(
        "--thinking-tokens",
        type=int,
        default=2000,
        help="thinking tokens per position, used only to price the run "
        "before it starts (default: %(default)s)",
    )
    p.add_argument("--out", type=Path, help="write the full result as JSON")
    p.add_argument(
        "--yes", action="store_true", help="skip the cost confirmation prompt"
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    items = runner.load(args.data, kind=args.kind, limit=args.limit)
    if not items:
        print("error: no positions matched", file=sys.stderr)
        return 2

    from . import task

    questions = [task.build(i, args.cells) for i in items]
    free = args.model in ("base-rate", "oracle")

    if not free:
        allowance = 0 if args.no_thinking else args.thinking_tokens
        cost = models.estimate_cost(
            questions, args.model, batch=args.batch, thinking_tokens=allowance
        )
        print(
            f"{args.model}: {len(items)} positions x {args.cells} cells"
            f"{' via Batch API' if args.batch else ''}"
        )
        print(
            f"estimated cost: ${cost:.2f}"
            f" (assuming {allowance} thinking tokens per position)"
        )
        if allowance:
            print("thinking is billed as output, so this is a floor, not a quote")
        if not args.yes and sys.stdin.isatty():
            if input("proceed? [y/N] ").strip().lower() not in ("y", "yes"):
                print("cancelled")
                return 1
    else:
        print(f"{args.model}: {len(items)} positions x {args.cells} cells (no API calls)")

    responder = models.get(
        args.model, batch=args.batch, thinking=not args.no_thinking
    )

    started = time.perf_counter()
    _, preds, report = runner.run(items, responder, model=args.model, cells=args.cells)
    wall = time.perf_counter() - started

    _print(report)
    curve = scoring.reliability(preds)
    if curve:
        print("\ncalibration curve")
        print(f"  {'stated':>12}{'actual':>10}{'cells':>8}")
        for row in curve:
            print(
                f"  {row['bin']:>12}{row['actual']:>10.3f}{row['count']:>8}"
                f"   (said {row['stated']:.3f})"
            )

    print(f"\n{wall:.1f}s wall clock")
    usage = getattr(responder, "usage", None)
    if usage is not None and usage.responses:
        print(f"actual usage: {usage.summary(args.model, batch=args.batch)}")
        per_position = usage.output_tokens / usage.responses
        print(
            f"re-price the next run with --thinking-tokens {round(per_position):d}"
        )
    if args.out:
        runner.write(args.out, report, preds, curve)
        print(f"wrote {args.out}")
    return 0


def _print(report) -> None:
    print()
    print(f"{'positions':<22}{report.items}")
    print(f"{'scored':<22}{report.parsed}")
    if report.failed:
        print(f"{'unparseable':<22}{report.failed}")
    print(f"{'cells':<22}{report.cells}")
    print()
    print(f"{'Brier (lower better)':<22}{report.brier:.4f}")
    print(f"{'base-rate Brier':<22}{report.base_brier:.4f}")
    print(f"{'skill vs base rate':<22}{report.skill:+.3f}   (0 = no better, 1 = exact)")
    print(f"{'calibration error':<22}{report.ece:.4f}")
    print(f"{'mean abs error':<22}{report.mean_absolute:.4f}")
    print(
        f"{'over/under confident':<22}{report.overconfidence:+.4f}"
        "   (+ = calls cells more dangerous than they are)"
    )
    if report.safe_recall is not None:
        print(f"{'provably-safe recall':<22}{report.safe_recall:.1%}")
    if report.safe_precision is not None:
        print(f"{'  of those, correct':<22}{report.safe_precision:.1%}")
    if report.mine_recall is not None:
        print(f"{'provably-mine recall':<22}{report.mine_recall:.1%}")

    if report.by_kind:
        print()
        header = f"{'kind':<14}{'items':>7}{'cells':>8}{'Brier':>10}{'skill':>9}{'ECE':>9}"
        print(header)
        print("-" * len(header))
        for kind, s in report.by_kind.items():
            print(
                f"{kind:<14}{s['items']:>7}{s['cells']:>8}{s['brier']:>10.4f}"
                f"{s['skill']:>+9.3f}{s['ece']:>9.4f}"
            )


if __name__ == "__main__":
    raise SystemExit(main())
