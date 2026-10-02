# Minesweeper Reasoning Agent

[![tests](https://github.com/AaravGarg16/minesweeper-reasoning-agent/actions/workflows/tests.yml/badge.svg)](https://github.com/AaravGarg16/minesweeper-reasoning-agent/actions/workflows/tests.yml)

A program that plays Minesweeper by calculating the exact chance that each
hidden square holds a mine, and a test that checks whether AI language models
can work out those same chances.

Minesweeper is unusual in that those chances can be calculated exactly, so
every position has a provably correct answer. That is rare in anything you
would use to measure reasoning, where you normally only have a human's opinion
to compare against.

**[Try the live demo](https://minesweeper-demo-922389797264.us-west1.run.app)**
Press *Watch it play* to see it work through a board a move at a time. Each
covered square is shaded by its chance of holding a mine, and the panel says
why it chose what it chose.

<img width="447" height="796" alt="image" src="https://github.com/user-attachments/assets/28741f2c-644c-443f-9709-97f222a8d9d6" />

---

## 1. It wins 71.6% of games. Ordinary play wins 32.7%.

Both are programs that play complete games by themselves, on the same 12,000
boards.

| | Share of games won |
|---|---|
| Ordinary rule-based play | 32.7% |
| **This program** | **71.6%** |

**Ordinary rule-based play** means looking at one revealed number at a time
and clearing the squares around it that must be safe. It is how most people
play, and it gets stuck as soon as no single number settles the question.

**This program** instead looks at all the numbers together. It works out every
possible arrangement of mines that the whole board allows, counts how many of
those arrangements put a mine in each hidden square, and clears the square
that appears in the fewest. That turns a guess into an exact probability.

By difficulty it wins 89.5% of beginner boards (8x8 with 10 mines), 88.2% of
intermediate (16x16 with 40) and 37.1% of expert (30x16 with 99). Expert is
hard for everyone: with 99 mines there are too many possible arrangements to
count them all in reasonable time, so there the program has to estimate.

## 2. Most of its losses were games nobody could have won

Minesweeper sometimes leaves you with no safe square at all. Every remaining
option might hold a mine, so you have to guess, and you can lose a game
without having played it badly.

Because the exact odds are known, I could check every lost game and ask
whether a safe square existed at the moment it died. **83% of the losses had
none.** They were unwinnable, not mistakes, which means the program was
already playing close to the best that is possible.

The other 17% were real mistakes, and they all came from the same place. On
boards too big to count every arrangement, the program fell back to a rough
estimate, and that estimate ranked the risky squares in the wrong order, so it
would sometimes pick a worse square when a safer one was available. Correcting
how the estimate combined the numbers fixed it: of the losses that remain, 87%
are unwinnable. Nothing in the win rate had pointed at this, because losing to
a guess looks identical to losing to a bad guess.

## 3. Two of three AI models did worse than ignoring the board

Since the correct answers are known, they make a test. **minebench-v1** is 600
Minesweeper positions published with their exact probabilities. For each one, a
model is shown the board and asked for the chance that each of eight specific
squares holds a mine. Its answers are compared against the true values.

| Score | Answering straight away | Allowed to reason first |
|---|---|---|
| Haiku 4.5 | -1.45 | -0.74 |
| Sonnet 5 | -0.23 | +0.10 |
| Opus 5 | +0.29 | **+0.90** |

**How to read the score.** 1.0 means the answers matched the exact
probabilities. 0.0 means they were no better than ignoring the board entirely
and just dividing the mines left by the number of covered squares. A negative
score means worse than that, so looking at the board actively misled the
model.

**What happened.** Answering straight away, Haiku and Sonnet both score below
zero. They are not working the constraints out at all, and Haiku is the clear
case: of the squares it declared safe, only 24% actually were. Letting the
models reason first helps all three, but only Opus gets close to the exact
answer. When Opus says a square is definitely safe, it is right every time.

**Why the two columns cover different amounts.** Given no limit on how long
they may reason, these models use nearly all of whatever budget they are
handed, and a reply that runs out of room part-way through scores as a
failure. Dropping those would quietly discard the hardest positions and
flatter the results. So reasoning is capped at a low setting, and that column
covers the first 80 of the 600 positions to keep the cost sensible. Every
reply in it came back complete. The first column covers all 600.

---

## Try it

Python 3.10 or newer, with nothing to install for the first two commands:

```bash
python -m benchmarks --solver single-point --boards 1000   # play 1,000 games
python -m llm_eval --model base-rate                       # score the control
pytest                                                     # 48 tests
```

The tests need `pytest`. Scoring a live model needs `pip install anthropic`
and an `ANTHROPIC_API_KEY`. CI runs the suite on Python 3.10 through 3.13 on
every push.

## What's in here

```
engine/       board generation, game state, and the view a player may see
benchmarks/   runs whole games and reports win rates
llm_eval/     the 600-position test, its scoring, and the dataset
web/          demo page: board, probability heatmap, reasoning panel
```

Every number above is in `llm_eval/results.json`. The game engine is written
from scratch, so the results do not depend on any course scaffolding.

## Why some parts are private

Some of this work was part of university coursework, so that part is kept
confidential. This repository is a public copy of the rest, including the
engine the results were measured on and the dataset of exact answers.

Happy to walk through the implementation in an interview, or to share access
on request.
