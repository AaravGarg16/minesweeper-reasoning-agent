"""Turning a stored position into a question, and an answer back into numbers.

A position has too many covered cells to ask about all of them -- an expert
board can have three hundred -- so each item is reduced to a handful of
query cells. They are drawn from the frontier, because cells with no
uncovered neighbour are indistinguishable from one another and asking about
them measures nothing except whether the model can divide.

The question put to the model is deliberately the same one the oracle
answers: for each query cell, the probability that it holds a mine. That
makes the comparison direct. Asking instead for a move would confound two
things -- whether the model knows the odds, and whether it picks well given
them -- and only the first has an exact ground truth to check against.
"""

import json
import random
from dataclasses import dataclass

QUERY_CELLS = 8

SYSTEM = """You are analysing Minesweeper positions.

The board is given as a grid of characters, one row per line, with columns
separated by spaces. Coordinates are (x, y): x is the column index and y is
the row index, both zero-based from the top-left.

  .   a covered cell, contents unknown
  _   an uncovered cell with no adjacent mines
  1-8 an uncovered cell, showing how many of its up-to-eight neighbours are mines

Every uncovered number is exactly the count of mines among that cell's
neighbours, and the total number of mines on the board is given. Together
these constrain which arrangements of mines are possible.

For each cell you are asked about, report the probability that it holds a
mine: the fraction of all mine arrangements consistent with the visible
numbers and the mine total in which that cell is a mine. Report 0.0 if the
cell is provably safe in every consistent arrangement, and 1.0 if it is
provably a mine in every one. Otherwise give the true fraction.

Answer with a JSON object holding a "cells" list, one entry per cell you
were asked about, each giving the cell exactly as it was written and the
probability as a number between 0 and 1:

  {"cells": [{"cell": "3,4", "probability": 0.25}, ...]}

Answer only with that JSON."""


@dataclass(frozen=True)
class Question:
    item_id: str
    kind: str
    prompt: str
    query_cells: list[str]
    truth: dict[str, float]
    base_rate: float


def _neighbors(x: int, y: int, width: int, height: int):
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx or dy:
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height:
                    yield nx, ny


def _frontier(item: dict) -> list[str]:
    """Covered cells that touch at least one uncovered number."""
    rows = [line.split() for line in item["board"].splitlines()]
    width, height = item["width"], item["height"]
    out = []
    for cell in item["covered"]:
        x, y = cell
        for nx, ny in _neighbors(x, y, width, height):
            glyph = rows[ny][nx]
            if glyph.isdigit() or glyph == "_":
                out.append(f"{x},{y}")
                break
    return out


def choose_cells(item: dict, count: int = QUERY_CELLS) -> list[str]:
    """Pick the cells to ask about, deterministically for a given item.

    A decidable item is guaranteed at least one provably safe cell in the
    selection, otherwise the question would not test the deduction the item
    was selected to test.
    """
    rng = random.Random(item["id"])
    pool = _frontier(item) or [f"{x},{y}" for x, y in item["covered"]]
    picked: list[str] = []

    safe = [f"{x},{y}" for x, y in item["safe"] if f"{x},{y}" in pool]
    if safe:
        picked.append(rng.choice(safe))

    rest = [c for c in pool if c not in picked]
    rng.shuffle(rest)
    picked.extend(rest[: max(0, count - len(picked))])
    return sorted(picked, key=lambda c: tuple(int(v) for v in c.split(",")))


def build(item: dict, count: int = QUERY_CELLS) -> Question:
    cells = choose_cells(item, count)
    covered = len(item["covered"])
    known_mines = len(item["certain_mines"])
    base_rate = item["total_mines"] / covered if covered else 0.0

    prompt = (
        f"Board: {item['width']} wide, {item['height']} tall, "
        f"{item['total_mines']} mines in total.\n\n"
        f"{item['board']}\n\n"
        f"{covered} cells are still covered.\n\n"
        "For each of these cells, give the probability that it holds a mine.\n"
        "Use the cell exactly as written here:\n"
        + "\n".join(f"  {c}" for c in cells)
    )
    return Question(
        item_id=item["id"],
        kind=item["kind"],
        prompt=prompt,
        query_cells=cells,
        truth={c: item["probabilities"][c] for c in cells},
        base_rate=base_rate,
    )


def schema(cells: list[str] | None = None) -> dict:
    """The response shape, identical for every position in the benchmark.

    Keying the properties by cell coordinate seemed natural and was a
    mistake: it made each request carry its own schema, every schema needs a
    grammar compiled server-side, and an organisation may only compile
    twenty a minute. Submitting six hundred at once failed 435 of them.

    So the cells are named in the payload rather than in the schema. One
    grammar covers the whole run, and the model still says which cell each
    number belongs to, which an array of bare numbers would not.

    The range is not expressed here because structured outputs reject
    `minimum` and `maximum` on a number. `parse` enforces it instead, so an
    answer of 1.5 is recorded as a failure rather than silently clamped.
    """
    return {
        "type": "json_schema",
        "schema": {
            "type": "object",
            "properties": {
                "cells": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "cell": {"type": "string"},
                            "probability": {"type": "number"},
                        },
                        "required": ["cell", "probability"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["cells"],
            "additionalProperties": False,
        },
    }


class ParseError(ValueError):
    pass


def parse(text: str, cells: list[str]) -> dict[str, float]:
    """Pull one probability per query cell out of a model's reply."""
    try:
        raw = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ParseError("no JSON object in response") from None
        try:
            raw = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ParseError(f"malformed JSON: {exc}") from None

    if not isinstance(raw, dict):
        raise ParseError(f"expected an object, got {type(raw).__name__}")

    # The schema asks for a list of {cell, probability}. A model that answers
    # with a plain {cell: probability} map instead has still said exactly what
    # was asked, so accept it rather than scoring a formatting habit.
    if isinstance(raw.get("cells"), list):
        stated = {}
        for entry in raw["cells"]:
            if not isinstance(entry, dict) or "cell" not in entry:
                raise ParseError(f"malformed entry: {entry!r}")
            stated[str(entry["cell"]).strip().strip("()")] = entry.get("probability")
    else:
        stated = raw

    out = {}
    for cell in cells:
        if cell not in stated:
            raise ParseError(f"missing cell {cell}")
        value = stated[cell]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ParseError(f"cell {cell} is not a number: {value!r}")
        if not 0.0 <= float(value) <= 1.0:
            raise ParseError(f"cell {cell} out of range: {value}")
        out[cell] = float(value)
    return out
