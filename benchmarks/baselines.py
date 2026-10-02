"""Reference agents that anyone can run.

`RandomAgent` is the floor. `SinglePointAgent` applies only the two local
rules every Minesweeper player works out on their own, and is the baseline
the full solver is measured against.
"""

import random

from engine import Action, BoardView, Cell, Move


class RandomAgent:
    """Uncovers a covered cell at random. Exists to anchor the bottom of the scale."""

    def __init__(self, rng: random.Random | None = None):
        self.rng = rng or random.Random()

    def __call__(self, view: BoardView) -> Move:
        covered = [c for c in view.covered_cells() if c not in view.flags]
        if not covered:
            return Move(Action.LEAVE)
        return Move(Action.UNCOVER, self.rng.choice(covered))


class SinglePointAgent:
    """Deduces from one numbered cell at a time, never from two together.

    For a number n with f flagged and u unflagged covered neighbours:
      - n == f          -> every cell in u is safe
      - n - f == len(u) -> every cell in u is a mine

    That is the whole of it. When neither rule fires anywhere on the board
    it guesses uniformly, which is what caps this approach well below a
    solver that reasons over overlapping constraints.
    """

    def __init__(self, rng: random.Random | None = None):
        self.rng = rng or random.Random()

    def __call__(self, view: BoardView) -> Move:
        safe, mines = self.deduce(view)

        for cell in mines:
            if cell not in view.flags:
                return Move(Action.FLAG, cell)
        for cell in safe:
            if view.is_covered(cell) and cell not in view.flags:
                return Move(Action.UNCOVER, cell)

        covered = [c for c in view.covered_cells() if c not in view.flags]
        if not covered:
            return Move(Action.LEAVE)
        return Move(Action.UNCOVER, self.rng.choice(covered))

    @staticmethod
    def deduce(
        view: BoardView, known_mines: set[Cell] | None = None
    ) -> tuple[set[Cell], set[Cell]]:
        """Safe cells and mine cells the two local rules can prove.

        `known_mines` stands in for the flags when an agent tracks mines
        internally instead of spending a move to flag each one.
        """
        marked = view.flags if known_mines is None else known_mines
        safe: set[Cell] = set()
        mines: set[Cell] = set()
        for cell in view.numbered_cells():
            number = view.value(cell)
            covered = [c for c in view.neighbors(cell) if view.is_covered(c)]
            flagged = [c for c in covered if c in marked]
            unknown = [c for c in covered if c not in marked]
            if not unknown:
                continue
            if number == len(flagged):
                safe.update(unknown)
            elif number - len(flagged) == len(unknown):
                mines.update(unknown)
        return safe, mines
