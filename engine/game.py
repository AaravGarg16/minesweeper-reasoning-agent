"""Game state and the restricted view a solver is allowed to see."""

from dataclasses import dataclass

from .board import Board
from .core import COVERED, Action, Cell, GameState, Move, neighbors


@dataclass(frozen=True)
class BoardView:
    """Everything a solver may look at, and nothing more.

    `grid` holds COVERED for cells that have not been uncovered and the
    adjacent mine count (0-8) for cells that have.
    """

    width: int
    height: int
    total_mines: int
    grid: tuple[tuple[int, ...], ...]
    flags: frozenset[Cell]

    def value(self, cell: Cell) -> int:
        x, y = cell
        return self.grid[x][y]

    def is_covered(self, cell: Cell) -> bool:
        return self.value(cell) == COVERED

    def covered_cells(self) -> list[Cell]:
        return [
            (x, y)
            for x in range(self.width)
            for y in range(self.height)
            if self.grid[x][y] == COVERED
        ]

    def numbered_cells(self) -> list[Cell]:
        return [
            (x, y)
            for x in range(self.width)
            for y in range(self.height)
            if self.grid[x][y] >= 0
        ]

    def neighbors(self, cell: Cell) -> list[Cell]:
        return neighbors(cell, self.width, self.height)

    def render(self) -> str:
        """Plain-text board, handy in tests and for prompting a language model."""
        rows = []
        for y in range(self.height):
            row = []
            for x in range(self.width):
                if (x, y) in self.flags:
                    row.append("F")
                elif self.grid[x][y] == COVERED:
                    row.append(".")
                elif self.grid[x][y] == 0:
                    row.append("_")
                else:
                    row.append(str(self.grid[x][y]))
            rows.append(" ".join(row))
        return "\n".join(rows)


class Game:
    """Applies moves to a Board and tracks win/loss.

    Uncovering reveals exactly one cell. There is deliberately no flood fill
    on a zero: the agent acts on one percept per move, so a cascade would
    hand it information it never asked for and make solve rates
    incomparable with a one-cell-per-turn run.
    """

    def __init__(self, board: Board, start: Cell | None = None):
        self.board = board
        self.state = GameState.PLAYING
        self.moves = 0
        self.uncovered = 0
        self.start = start
        self._visible = [
            [COVERED] * board.height for _ in range(board.width)
        ]
        self._flags: set[Cell] = set()
        if start is not None:
            self.uncover(start)

    @property
    def view(self) -> BoardView:
        return BoardView(
            width=self.board.width,
            height=self.board.height,
            total_mines=self.board.mine_count,
            grid=tuple(tuple(col) for col in self._visible),
            flags=frozenset(self._flags),
        )

    @property
    def over(self) -> bool:
        return self.state is not GameState.PLAYING

    def apply(self, move: Move) -> int | None:
        """Run one move. Returns the uncovered number, or None otherwise."""
        if move.action is Action.UNCOVER:
            return self.uncover(move.cell)
        if move.action is Action.FLAG:
            self.flag(move.cell)
        elif move.action is Action.UNFLAG:
            self.unflag(move.cell)
        elif move.action is Action.LEAVE:
            self.leave()
        return None

    def uncover(self, cell: Cell) -> int | None:
        """Uncover one cell. Returns its adjacent mine count, or None if it was a mine."""
        self._check(cell)
        x, y = cell
        if self._visible[x][y] != COVERED:
            return self._visible[x][y]

        self.moves += 1
        self._flags.discard(cell)

        if self.board.is_mine(cell):
            self.state = GameState.LOST
            return None

        count = self.board.adjacent_mines(cell)
        self._visible[x][y] = count
        self.uncovered += 1
        if self.uncovered == self.board.safe_count:
            self.state = GameState.WON
        return count

    def flag(self, cell: Cell) -> None:
        self._check(cell)
        x, y = cell
        if self._visible[x][y] != COVERED:
            raise ValueError(f"cannot flag {cell}, it is already uncovered")
        self.moves += 1
        self._flags.add(cell)

    def unflag(self, cell: Cell) -> None:
        self._check(cell)
        self.moves += 1
        self._flags.discard(cell)

    def leave(self) -> None:
        """Agent concedes. Counts as a win only if the board is already cleared."""
        if self.state is GameState.PLAYING:
            self.state = (
                GameState.WON
                if self.uncovered == self.board.safe_count
                else GameState.LOST
            )

    def _check(self, cell: Cell) -> None:
        if self.over:
            raise RuntimeError(f"game is already {self.state.value}")
        if not self.board.in_bounds(cell):
            raise ValueError(f"{cell} is off the board")
