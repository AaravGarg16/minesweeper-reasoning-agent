"""Question construction and response parsing.

Parsing is where a harness quietly lies to you: a model that answered well
but wrapped its JSON in prose should not be scored as if it reasoned badly,
and a model that returned nonsense should not be silently given a default.
Both directions are pinned here.
"""

import pytest

from llm_eval import task


@pytest.fixture
def item():
    return {
        "id": "fixture-1",
        "kind": "decidable",
        "width": 4,
        "height": 3,
        "total_mines": 2,
        "board": "\n".join(["_ 1 . .", "_ 1 . .", "_ 1 . ."]),
        "covered": [[2, 0], [3, 0], [2, 1], [3, 1], [2, 2], [3, 2]],
        "probabilities": {
            "2,0": 0.0,
            "3,0": 0.5,
            "2,1": 1.0,
            "3,1": 0.25,
            "2,2": 0.0,
            "3,2": 0.25,
        },
        "safe": [[2, 0], [2, 2]],
        "certain_mines": [[2, 1]],
        "min_probability": 0.0,
    }


def test_query_cells_come_from_the_frontier(item):
    cells = task.choose_cells(item)
    # Column 3 does not touch any number, so it is not frontier.
    assert all(c.startswith("2,") for c in cells), cells


def test_a_decidable_item_always_offers_a_provably_safe_cell(item):
    cells = task.choose_cells(item)
    assert any(item["probabilities"][c] == 0.0 for c in cells)


def test_selection_is_deterministic(item):
    assert task.choose_cells(item) == task.choose_cells(item)


def test_prompt_carries_the_board_and_the_mine_total(item):
    q = task.build(item)
    assert item["board"] in q.prompt
    assert "2 mines in total" in q.prompt
    assert q.truth == {c: item["probabilities"][c] for c in q.query_cells}


def test_base_rate_is_mines_over_covered(item):
    q = task.build(item)
    assert q.base_rate == pytest.approx(2 / 6)


def test_parses_a_clean_object():
    assert task.parse('{"1,1": 0.25, "2,2": 1}', ["1,1", "2,2"]) == {
        "1,1": 0.25,
        "2,2": 1.0,
    }


def test_parses_json_buried_in_prose():
    text = 'Looking at the constraints:\n```json\n{"1,1": 0.5}\n```\nDone.'
    assert task.parse(text, ["1,1"]) == {"1,1": 0.5}


@pytest.mark.parametrize(
    "text, cells",
    [
        ("not json at all", ["1,1"]),
        ('{"1,1": 0.5}', ["1,1", "2,2"]),          # missing a cell
        ('{"1,1": 1.5}', ["1,1"]),                  # out of range
        ('{"1,1": "low"}', ["1,1"]),                # not a number
        ('{"1,1": true}', ["1,1"]),                 # bool is not a probability
        ("[0.5]", ["1,1"]),                         # not an object
    ],
)
def test_bad_responses_raise_rather_than_defaulting(text, cells):
    with pytest.raises(task.ParseError):
        task.parse(text, cells)


def test_one_schema_serves_every_position():
    """Per-position schemas exhaust the server-side grammar compilation limit."""
    assert task.schema(["0,0", "1,1"]) == task.schema(["7,7"] * 8)
    assert task.schema()["schema"]["required"] == ["cells"]


def test_schema_leaves_the_range_to_the_parser():
    """Structured outputs reject minimum/maximum on a number, so parse checks."""
    item = task.schema()["schema"]["properties"]["cells"]["items"]
    assert "minimum" not in item["properties"]["probability"]
    with pytest.raises(task.ParseError):
        task.parse('{"cells": [{"cell": "0,0", "probability": 1.5}]}', ["0,0"])


def test_parses_the_cells_list():
    text = '{"cells": [{"cell": "1,1", "probability": 0.25}, {"cell": "2,2", "probability": 1}]}'
    assert task.parse(text, ["1,1", "2,2"]) == {"1,1": 0.25, "2,2": 1.0}


def test_tolerates_a_plain_map_and_parenthesised_cells():
    assert task.parse('{"1,1": 0.5}', ["1,1"]) == {"1,1": 0.5}
    text = '{"cells": [{"cell": "(1,1)", "probability": 0.5}]}'
    assert task.parse(text, ["1,1"]) == {"1,1": 0.5}
