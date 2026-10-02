import random

from benchmarks.baselines import RandomAgent, SinglePointAgent
from engine import Action, new_game, run_game


def test_single_point_only_claims_cells_it_can_prove():
    """Anything the rule calls safe must really be safe, and likewise for mines."""
    for seed in range(60):
        game = new_game(8, 8, 10, seed=seed)
        agent = SinglePointAgent(random.Random(seed))
        for _ in range(40):
            if game.over:
                break
            view = game.view
            safe, mines = SinglePointAgent.deduce(view, set())
            for cell in safe:
                assert not game.board.is_mine(cell)
            for cell in mines:
                assert game.board.is_mine(cell)
            game.apply(agent(view))


def test_single_point_beats_random_by_a_wide_margin():
    def rate(factory):
        wins = 0
        for seed in range(150):
            game = new_game(8, 8, 10, seed=seed)
            wins += run_game(game, factory(random.Random(seed))).won
        return wins / 150

    assert rate(RandomAgent) < 0.05
    assert rate(SinglePointAgent) > 0.35


def test_agents_only_ever_touch_covered_cells():
    game = new_game(16, 16, 40, seed=11)
    agent = SinglePointAgent(random.Random(0))
    while not game.over:
        view = game.view
        move = agent(view)
        if move.action is Action.LEAVE:
            break
        assert view.is_covered(move.cell)
        game.apply(move)
