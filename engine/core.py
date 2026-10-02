"""Shared types for the Minesweeper engine.

Coordinates are (x, y) with x indexing the column and y the row, both
zero-based from the top-left corner.
"""

from dataclasses import dataclass
from enum import Enum

Cell = tuple[int, int]

# Stored in a visible grid for any cell that has not been uncovered.
COVERED = -1


class Action(Enum):
    UNCOVER = "uncover"
    FLAG = "flag"
    UNFLAG = "unflag"
    LEAVE = "leave"


class GameState(Enum):
    PLAYING = "playing"
    WON = "won"
    LOST = "lost"


@dataclass(frozen=True)
class Move:
    action: Action
    cell: Cell | None = None

    def __post_init__(self):
        if self.action is not Action.LEAVE and self.cell is None:
            raise ValueError(f"{self.action.value} requires a cell")


def neighbors(cell: Cell, width: int, height: int) -> list[Cell]:
    """The up-to-eight in-bounds cells touching `cell`."""
    x, y = cell
    out = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height:
                out.append((nx, ny))
    return out
