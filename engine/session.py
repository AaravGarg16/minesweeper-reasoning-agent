"""Setting up and running a single game."""

import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from .board import Board
from .core import Action, Cell, GameState
from .game import BoardView, Game
from .core import Move

Solver = Callable[[BoardView], Move]
StartPolicy = Literal["zero", "safe"]


def pick_start(board: Board, rng: random.Random, policy: StartPolicy = "zero") -> Cell | None:
    """Choose the opening cell, uniformly among those the policy allows.

    "zero" restricts the opening to a cell with no adjacent mines, so the
    first move always yields a foothold to reason from rather than an
    isolated number. "safe" only guarantees the opening is not a mine.
    """
    if policy == "zero":
        candidates = [
            c for c in board.cells()
            if not board.is_mine(c) and board.adjacent_mines(c) == 0
        ]
    elif policy == "safe":
        candidates = [c for c in board.cells() if not board.is_mine(c)]
    else:
        raise ValueError(f"unknown start policy {policy!r}")
    return rng.choice(candidates) if candidates else None


def new_game(
    width: int,
    height: int,
    mine_count: int,
    *,
    seed: int | None = None,
    rng: random.Random | None = None,
    start_policy: StartPolicy = "zero",
    max_attempts: int = 200,
) -> Game:
    """Generate a board and open it at a cell the start policy allows.

    Mines are placed first and the opening is drawn from what the layout
    happens to offer, rather than clearing a region around a pre-chosen
    opening. The two are not the same distribution, and solve rates are only
    comparable between runs that used the same one.
    """
    if rng is None:
        rng = random.Random(seed)
    for _ in range(max_attempts):
        board = Board.generate(width, height, mine_count, rng=rng)
        start = pick_start(board, rng, start_policy)
        if start is not None:
            return Game(board, start=start)
    raise RuntimeError(
        f"no {width}x{height} board with {mine_count} mines produced a "
        f"'{start_policy}' opening in {max_attempts} attempts"
    )


@dataclass(frozen=True)
class GameResult:
    won: bool
    state: GameState
    moves: int
    uncovered: int
    safe_cells: int
    elapsed: float
    detonated_at: Cell | None = None
    hit_move_limit: bool = False

    @property
    def progress(self) -> float:
        """Fraction of safe cells uncovered, so partial runs stay comparable."""
        return self.uncovered / self.safe_cells if self.safe_cells else 1.0


def run_game(game: Game, solver: Solver, *, move_limit: int | None = None) -> GameResult:
    """Drive `solver` until the game ends or the move limit is reached."""
    if move_limit is None:
        move_limit = 2 * game.board.width * game.board.height

    detonated_at: Cell | None = None
    hit_limit = False
    started = time.perf_counter()

    while not game.over:
        if game.moves >= move_limit:
            hit_limit = True
            game.leave()
            break
        move = solver(game.view)
        if move.action is Action.UNCOVER:
            if game.uncover(move.cell) is None:
                detonated_at = move.cell
        else:
            game.apply(move)

    return GameResult(
        won=game.state is GameState.WON,
        state=game.state,
        moves=game.moves,
        uncovered=game.uncovered,
        safe_cells=game.board.safe_count,
        elapsed=time.perf_counter() - started,
        detonated_at=detonated_at,
        hit_move_limit=hit_limit,
    )
