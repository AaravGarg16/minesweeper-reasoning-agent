"""Runs a solver over many generated boards and summarises the outcome."""

import json
import random
from dataclasses import asdict, dataclass
from multiprocessing import Pool

from engine import Cell, new_game, run_game

from . import registry


@dataclass(frozen=True)
class Preset:
    name: str
    width: int
    height: int
    mines: int

    @property
    def density(self) -> float:
        return self.mines / (self.width * self.height)

    @property
    def label(self) -> str:
        return f"{self.width}x{self.height}/{self.mines}"


PRESETS: dict[str, Preset] = {
    "beginner": Preset("beginner", 8, 8, 10),
    "intermediate": Preset("intermediate", 16, 16, 40),
    "expert": Preset("expert", 30, 16, 99),
}


@dataclass(frozen=True)
class Outcome:
    solver: str
    preset: str
    seed: int
    won: bool
    moves: int
    uncovered: int
    safe_cells: int
    elapsed: float
    hit_move_limit: bool
    detonated_at: Cell | None

    @property
    def progress(self) -> float:
        return self.uncovered / self.safe_cells if self.safe_cells else 1.0


def play(solver_name: str, preset: Preset, seed: int) -> Outcome:
    """One board, fully deterministic in `seed`."""
    factory = registry.get(solver_name)
    game = new_game(preset.width, preset.height, preset.mines, seed=seed)
    result = run_game(game, factory(game, random.Random(seed)))
    return Outcome(
        solver=solver_name,
        preset=preset.name,
        seed=seed,
        won=result.won,
        moves=result.moves,
        uncovered=result.uncovered,
        safe_cells=result.safe_cells,
        elapsed=result.elapsed,
        hit_move_limit=result.hit_move_limit,
        detonated_at=result.detonated_at,
    )


def _play_star(args) -> Outcome:
    return play(*args)


def run_suite(
    solver_name: str,
    presets: list[Preset],
    boards: int,
    *,
    first_seed: int = 0,
    workers: int = 1,
) -> list[Outcome]:
    """`boards` boards per preset. Seeds are per-board, so results do not
    depend on worker count or completion order."""
    jobs = [
        (solver_name, preset, first_seed + i)
        for preset in presets
        for i in range(boards)
    ]
    if workers <= 1:
        return [_play_star(job) for job in jobs]
    with Pool(workers) as pool:
        return list(pool.imap_unordered(_play_star, jobs, chunksize=8))


def summarize(outcomes: list[Outcome]) -> dict:
    by_preset: dict[str, list[Outcome]] = {}
    for o in outcomes:
        by_preset.setdefault(o.preset, []).append(o)

    presets = {}
    for name, group in by_preset.items():
        elapsed = sorted(o.elapsed for o in group)
        presets[name] = {
            "boards": len(group),
            "wins": sum(o.won for o in group),
            "solve_rate": sum(o.won for o in group) / len(group),
            "mean_progress": sum(o.progress for o in group) / len(group),
            "mean_moves": sum(o.moves for o in group) / len(group),
            "mean_seconds": sum(elapsed) / len(elapsed),
            "p99_seconds": elapsed[min(len(elapsed) - 1, int(0.99 * len(elapsed)))],
            "max_seconds": elapsed[-1],
            "hit_move_limit": sum(o.hit_move_limit for o in group),
        }

    return {
        "solver": outcomes[0].solver if outcomes else None,
        "boards": len(outcomes),
        "wins": sum(o.won for o in outcomes),
        "solve_rate": sum(o.won for o in outcomes) / len(outcomes) if outcomes else 0.0,
        "by_preset": presets,
    }


def write_results(path, outcomes: list[Outcome], summary: dict) -> None:
    payload = {"summary": summary, "outcomes": [asdict(o) for o in outcomes]}
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
