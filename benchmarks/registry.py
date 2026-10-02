"""Names the benchmark can be pointed at.

The full solver lives in a separate private package. When it is not
installed the baselines still run, so this suite is useful on its own.
"""

import random
from collections.abc import Callable

from engine import Game, Solver

from .baselines import RandomAgent, SinglePointAgent

Factory = Callable[[Game, random.Random], Solver]


class SolverUnavailable(RuntimeError):
    pass


def _random(game: Game, rng: random.Random) -> Solver:
    return RandomAgent(rng)


def _single_point(game: Game, rng: random.Random) -> Solver:
    return SinglePointAgent(rng)


def _full(game: Game, rng: random.Random) -> Solver:
    try:
        from solver import LegacyAgent
    except ImportError as exc:
        raise SolverUnavailable(
            "the full solver is not part of this repository -- it stays "
            "private while the course that set it still runs. Use --solver "
            "single-point or --solver random, both of which are here."
        ) from exc
    return LegacyAgent(game)


def _exact(game: Game, rng: random.Random) -> Solver:
    try:
        from solver import ExactAgent
    except ImportError as exc:
        raise SolverUnavailable(
            "the full solver is not part of this repository -- it stays "
            "private while the course that set it still runs. Use --solver "
            "single-point or --solver random, both of which are here."
        ) from exc
    return ExactAgent(rng)


def _exact_uniform_guess(game: Game, rng: random.Random) -> Solver:
    """Full deduction, but no preference about what to risk. Ablation only."""
    try:
        from solver import ExactAgent
    except ImportError as exc:
        raise SolverUnavailable("the full solver is not part of this repository") from exc
    return ExactAgent(rng, guess_policy="uniform")


def _deduction_only(game: Game, rng: random.Random) -> Solver:
    """Local rules only, no enumeration. Ablation only."""
    try:
        from solver import ExactAgent
    except ImportError as exc:
        raise SolverUnavailable("the full solver is not part of this repository") from exc
    return ExactAgent(rng, max_component=0)


FACTORIES: dict[str, Factory] = {
    "random": _random,
    "single-point": _single_point,
    "full": _full,
    "exact": _exact,
    "exact-uniform-guess": _exact_uniform_guess,
    "deduction-only": _deduction_only,
}


def get(name: str) -> Factory:
    try:
        return FACTORIES[name]
    except KeyError:
        raise SolverUnavailable(
            f"unknown solver {name!r}; choose from {', '.join(FACTORIES)}"
        ) from None


def check(name: str) -> None:
    """Raise SolverUnavailable unless `name` can be constructed in this checkout."""
    get(name)(_probe_game(), random.Random(0))


def available() -> list[str]:
    """Names that can actually be constructed in this checkout."""
    out = []
    for name in FACTORIES:
        try:
            check(name)
        except SolverUnavailable:
            continue
        out.append(name)
    return out


def _probe_game() -> Game:
    from engine import new_game

    return new_game(4, 4, 1, seed=0)
