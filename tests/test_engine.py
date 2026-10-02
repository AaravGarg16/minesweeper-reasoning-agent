import random

import pytest

from engine import COVERED, Action, Board, Game, GameState, Move, new_game, run_game
from engine.core import neighbors


def test_neighbors_are_clipped_at_the_edges():
    assert len(neighbors((0, 0), 8, 8)) == 3
    assert len(neighbors((4, 4), 8, 8)) == 8
    assert len(neighbors((7, 7), 8, 8)) == 3


def test_generation_is_reproducible_and_respects_exclusions():
    a = Board.generate(8, 8, 10, seed=7, exclude=[(0, 0)])
    b = Board.generate(8, 8, 10, seed=7, exclude=[(0, 0)])
    assert a.mines == b.mines
    assert (0, 0) not in a.mines
    assert Board.generate(8, 8, 10, seed=8).mines != a.mines


def test_generation_rejects_impossible_requests():
    with pytest.raises(ValueError):
        Board.generate(3, 3, 10)


def test_adjacency_counts_match_a_hand_built_board():
    board = Board(3, 3, [(0, 0), (2, 2)])
    assert board.adjacent_mines((1, 1)) == 2
    assert board.adjacent_mines((0, 1)) == 1
    assert board.adjacent_mines((2, 1)) == 1
    assert board.adjacent_mines((0, 2)) == 0
    assert board.safe_count == 7


def test_zero_start_policy_always_opens_on_an_empty_cell():
    for seed in range(50):
        game = new_game(8, 8, 10, seed=seed)
        assert not game.board.is_mine(game.start)
        assert game.board.adjacent_mines(game.start) == 0


def test_uncovering_reveals_exactly_one_cell():
    game = new_game(16, 16, 40, seed=3)
    before = game.uncovered
    target = next(c for c in game.view.covered_cells() if not game.board.is_mine(c))
    game.uncover(target)
    assert game.uncovered == before + 1


def test_view_never_exposes_the_mine_layout():
    game = new_game(8, 8, 10, seed=1)
    fields = set(game.view.__dataclass_fields__)
    assert fields == {"width", "height", "total_mines", "grid", "flags"}
    assert all(v == COVERED or 0 <= v <= 8 for col in game.view.grid for v in col)


def test_uncovering_a_mine_loses():
    game = new_game(8, 8, 10, seed=5)
    assert game.uncover(next(iter(game.board.mines))) is None
    assert game.state is GameState.LOST


def test_clearing_every_safe_cell_wins():
    game = new_game(5, 5, 3, seed=2)
    for cell in game.board.cells():
        if not game.over and not game.board.is_mine(cell):
            game.uncover(cell)
    assert game.state is GameState.WON


def test_moves_are_rejected_once_the_game_is_over():
    game = new_game(8, 8, 10, seed=4)
    game.uncover(next(iter(game.board.mines)))
    with pytest.raises(RuntimeError):
        game.uncover((0, 0))


def test_off_board_moves_are_rejected():
    game = new_game(8, 8, 10, seed=4)
    with pytest.raises(ValueError):
        game.uncover((99, 99))


def test_move_requires_a_cell_unless_leaving():
    Move(Action.LEAVE)
    with pytest.raises(ValueError):
        Move(Action.UNCOVER)


def test_runner_stops_at_the_move_limit():
    game = new_game(8, 8, 10, seed=6)
    result = run_game(game, lambda v: Move(Action.FLAG, v.covered_cells()[0]), move_limit=20)
    assert result.hit_move_limit
    assert not result.won


def test_runner_reports_where_it_detonated():
    game = new_game(8, 8, 10, seed=9)
    mine = next(iter(game.board.mines))
    result = run_game(game, lambda v: Move(Action.UNCOVER, mine))
    assert result.detonated_at == mine
    assert not result.won
