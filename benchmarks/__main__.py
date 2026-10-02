"""Command line entry point: python -m benchmarks --solver single-point"""

import argparse
import os
import sys
import time
from pathlib import Path

from . import registry
from .harness import PRESETS, run_suite, summarize, write_results


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="python -m benchmarks",
        description="Run a Minesweeper solver over many generated boards.",
    )
    p.add_argument(
        "--solver",
        default="single-point",
        help=f"one of: {', '.join(registry.FACTORIES)} (default: %(default)s)",
    )
    p.add_argument(
        "--presets",
        nargs="+",
        default=list(PRESETS),
        choices=list(PRESETS),
        help="difficulties to run (default: all)",
    )
    p.add_argument("--boards", type=int, default=1000, help="boards per preset")
    p.add_argument("--seed", type=int, default=0, help="first board seed")
    p.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 1,
        help="parallel processes (default: %(default)s)",
    )
    p.add_argument("--out", type=Path, help="write full results as JSON")
    p.add_argument(
        "--list", action="store_true", help="show which solvers this checkout can run"
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    if args.list:
        runnable = set(registry.available())
        for name in registry.FACTORIES:
            mark = "available" if name in runnable else "not in this repository"
            print(f"  {name:<14} {mark}")
        return 0

    try:
        registry.check(args.solver)
    except registry.SolverUnavailable as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    presets = [PRESETS[name] for name in args.presets]
    total = args.boards * len(presets)
    print(
        f"{args.solver}: {args.boards} boards x {len(presets)} presets "
        f"= {total} games on {args.workers} workers"
    )

    started = time.perf_counter()
    outcomes = run_suite(
        args.solver, presets, args.boards, first_seed=args.seed, workers=args.workers
    )
    wall = time.perf_counter() - started
    summary = summarize(outcomes)

    print()
    header = f"{'preset':<14}{'boards':>7}{'solved':>9}{'rate':>9}{'progress':>10}{'mean ms':>10}{'max ms':>9}"
    print(header)
    print("-" * len(header))
    for name in args.presets:
        s = summary["by_preset"][name]
        print(
            f"{name:<14}{s['boards']:>7}{s['wins']:>9}{s['solve_rate']:>8.1%}"
            f"{s['mean_progress']:>10.1%}{s['mean_seconds'] * 1000:>10.1f}"
            f"{s['max_seconds'] * 1000:>9.1f}"
        )
    print("-" * len(header))
    print(
        f"{'overall':<14}{summary['boards']:>7}{summary['wins']:>9}"
        f"{summary['solve_rate']:>8.1%}"
    )
    print(f"\n{wall:.1f}s wall clock")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        write_results(args.out, outcomes, summary)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
