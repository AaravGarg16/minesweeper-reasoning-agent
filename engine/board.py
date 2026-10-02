"""Mine layout and adjacency counts.

This is ground truth. A solver is never handed a Board -- it only ever sees
the BoardView produced by Game.
"""

import random
from collections.abc import Iterable

from .core import Cell, neighbors


class Board:
    def __init__(self, width: int, height: int, mine_cells: Iterable[Cell]):
        if width < 1 or height < 1:
            raise ValueError("board must be at least 1x1")
        self.width = width
        self.height = height
        self.mines = frozenset(mine_cells)
        for cell in self.mines:
            if not self.in_bounds(cell):
                raise ValueError(f"mine at {cell} is off the board")
        self._counts = self._count_adjacent()

    @classmethod
    def generate(
        cls,
        width: int,
        height: int,
        mine_count: int,
        *,
        seed: int | None = None,
        rng: random.Random | None = None,
        exclude: Iterable[Cell] = (),
    ) -> "Board":
        """Place `mine_count` mines uniformly at random.

        Cells in `exclude` are left clear. Pass either `seed` or an existing
        `rng`; passing an rng lets a caller draw several boards from one
        reproducible stream.
        """
        if rng is None:
            rng = random.Random(seed)
        forbidden = set(exclude)
        candidates = [
            (x, y)
            for x in range(width)
            for y in range(height)
            if (x, y) not in forbidden
        ]
        if not 0 <= mine_count <= len(candidates):
            raise ValueError(
                f"cannot place {mine_count} mines in {len(candidates)} available cells"
            )
        return cls(width, height, rng.sample(candidates, mine_count))

    def in_bounds(self, cell: Cell) -> bool:
        x, y = cell
        return 0 <= x < self.width and 0 <= y < self.height

    def is_mine(self, cell: Cell) -> bool:
        return cell in self.mines

    def adjacent_mines(self, cell: Cell) -> int:
        """Number of mines touching `cell`. Raises if `cell` is itself a mine."""
        return self._counts[cell]

    @property
    def mine_count(self) -> int:
        return len(self.mines)

    @property
    def safe_count(self) -> int:
        return self.width * self.height - len(self.mines)

    def cells(self) -> list[Cell]:
        return [(x, y) for x in range(self.width) for y in range(self.height)]

    def _count_adjacent(self) -> dict[Cell, int]:
        counts = {}
        for cell in self.cells():
            if cell in self.mines:
                continue
            counts[cell] = sum(
                1
                for n in neighbors(cell, self.width, self.height)
                if n in self.mines
            )
        return counts

    def __repr__(self) -> str:
        return f"Board({self.width}x{self.height}, {len(self.mines)} mines)"
