# MCTS implementation and project review

The supplied `PROGRESS.md` describes a simplified, rule-based Monopoly research
project. It was read as project context. Its Step 7 status predates this checkout:
Expectimax and prospect theory were already implemented; only MCTS was empty.
The teammate implementations are retained.

## Context checked against the Python files

Four agents play on 24 squares with 14 deeds in five color sets. Each starts with
$1,500. The engine permits one building per turn, requires even construction,
skips one turn in jail, and eliminates players who cannot pay. It ends at the
last survivor or 150 individual turns. At the cap it compares cash plus printed
deed/building costs. These rules, rather than standard full Monopoly rules,
determine what the planners simulate and how they evaluate spending.

| Python source | Role verified |
| --- | --- |
| `board.py` | Fixed board, groups, rents, construction prices and card contents. |
| `models.py` | Frozen public snapshots, offers, replies, shared trade legality, results. |
| `engine.py` | Dice, movement, landing, payments, bankruptcy, build legality, negotiations and feedback. |
| `valuation.py` | Markov movement model, stationary landing probabilities, purchase/build gains, trade scores and affordable future development. |
| `bots.py` | Separates buying/building from trading through `PolicyBot`; assembles six named bots. |
| `bargaining.py` | Candidate trades, Nash product and tit-for-tat memory. |
| `advanced_bargaining.py` | Zeuthen-style concessions, Bayesian opponent thresholds and minimax regret. |
| `advanced_decisions.py` | Expectimax, new MCTS, prospect theory, immutable action transitions and copied engine forecasts. |
| `experiments.py` | Controlled seed/seat schedules, CSV/JSON logs, summaries, source archives and strict replay. |
| `__main__.py` | Existing demos, full games, pilot and replay command routing. |
| `__init__.py` | Package description; no additional setup. |
| `scripts/benchmark_expectimax.py` | Archived-reference equivalence and timing tool for Expectimax. |

The existing ten test files were also reviewed: engine rules, bot composition,
valuation, development forecasts, bargaining, response feedback, invariants,
experiment saving/replay, Expectimax and prospect theory. New MCTS checks are in
`tests/test_mcts_decisions.py`.

EV means **expected value**: probability-weighted outcomes. The Markov model
estimates long-run landing frequencies, approximating unknown card draws as
uniform. Nash bargaining maximizes the product of positive trade gains;
tit-for-tat changes demands using the previous offer. Bayesian bargaining updates
three guesses about an opponent's acceptance threshold. Minimax regret minimizes
the worst missed payoff across those scenarios. Prospect theory weights losses
more strongly and transforms probabilities. Expectimax enumerates chance outcomes;
MCTS (**Monte Carlo Tree Search**) samples them. These agents are programmed
policies and online planners; there is no offline model training.

The tournament compares complete agents. The ablation, or controlled trading
experiment, must instead hold buying/building and the three Baselines fixed while
changing only the test agent's trading policy.

## What changed

`MCTSDecisions.buy` compares skip versus purchase. `build` compares skip with all
supplied legal upgrades that also satisfy the $250 reserve. Purchases preserve
the $150 reserve. A purchase is followed by the current turn's build opportunity;
a build ends the turn. This matches the engine's call order.

The implementation has the four MCTS stages:

1. **Selection:** choose unvisited actions first, then use UCT (**Upper Confidence
   Bounds applied to Trees**): `mean_reward + exploration * sqrt(log(parent_visits) / action_visits)`.
2. **Expansion:** add at most one new sampled state per iteration. An action's
   statistics aggregate its different chance outcomes. Dice/card outcomes are
   sampled; they are never selected as if the agent controlled chance.
3. **Simulation:** finish the bounded horizon with expected-value buying/building
   for all players. In the tree, the searching player's choices use UCT and
   opponents follow the same expected-value model.
4. **Backpropagation:** update visit counts and reward totals along the selected
   path. Recommend the most visited root action, using mean reward next and
   insertion order last. Exact ties prefer skipping.

Defaults: `MCTSDecisions(iterations=128, depth=8, exploration=sqrt(2), seed=0)`.
Depth counts future individual turns, excluding the current purchase/build.
Even the last sampled turn finishes its actions before evaluation. Jail turns
count; bankrupt seats are skipped. Turn cap, bankruptcy and last-survivor states
stop the search early. Each iteration samples at most `depth` movements and adds
at most one node, bounding runtime and retained tree size without a clock cutoff.
Very small iteration budgets can leave some actions unvisited.

Leaves reuse the existing Expectimax relative-wealth heuristic: owned capital,
cash, stationary future net rent, set bonuses and liquidity versus the strongest
surviving opponent. Terminal cap states use actual net worth differences.
`tanh(score / 1500)` scales rewards to [-1, 1] for UCT; it is not a win probability.

Movement uses `_Forecast`, a private engine copy, so rent, tax, Go payments,
cards and bankruptcy use the real rules. Buy/build transitions create immutable
snapshots. A fresh local seeded RNG makes repeated decisions reproducible.
Search cannot access or advance live dice, decks or trading memory.

`adaptive_bot()` now combines MCTS with the existing `BayesianTrading`.
`adaptive_bot(use_mcts=False)` retains EV, and `experiments.make_game()` explicitly
uses that setting to preserve the controlled comparison. Advanced imports remain
inside factories to avoid the existing circular-import constraint.

## Validation and use

With Python 3.10+ available, run from the project root:

```powershell
python -B -m unittest discover -s tests -v
python -B -m monopoly --step6-game --seed 42
python -B -m scripts.benchmark_mcts
```

The full suite passed **86 tests**, including 12 new MCTS tests. They check all
dice pairs and possible cards from every starting square against the real engine,
including insolvency; jail and eliminated seats; purchase/build ordering and
hotel limits; legal actions and reserves; live state/RNG/deck/memory isolation;
repeatability; tree expansion, backed-up rewards and iteration/depth limits;
terminal ties; and EV retention in all six controlled variants. Existing
per-turn invariant checks run 30 seeds with both search agents in the lineup.

The timing smoke check completed all 15 distinct four-bot lineups at seed 42
using Python 3.12.14. Across 72 MCTS searches, mean search time was 39.0 ms
(maximum 53.7 ms). Mean game time was 0.709 seconds; the slowest game was
4.074 seconds. The six lineups containing both Expectimax and MCTS took
0.483-0.838 seconds each. These are local development measurements, including
other policies and bargaining work, not performance guarantees or final
experimental results. Raw output is saved in `results/mcts-timing.json`.

The existing `run.cmd` expects an older bundled Python path or `py -3`. Neither
exists on this machine. Validation used the installed Codex bundled interpreter
directly; no launcher or Python installation was changed.

## Progress and limitations

Steps 0-6 have implementations and tests in this checkout. However, the supplied
guide's `README.md`, `AUDIT.md`, Step 5/6 result folders and most historical saved
evidence are absent here. Historical win counts cannot be reverified from those
missing files. The existing `results/expectimax-optimization` archive is present.

All three Step 7 buying/building policies are now implemented and wired into
ordinary games. Steps 8-10 remain: the full balanced/resumable 3,000-game
tournament, final 1,200-game controlled evaluation, uncertainty analysis, payoff
matrix/equilibrium, charts and final report. The present runner supports a
controlled pilot and replay, but does not implement the planned tournament or
resume an interrupted batch.

MCTS approximates unknown cards as independent uniform draws instead of the
engine's cycling shuffled decks. Future trades are omitted; Bayesian bargaining
still handles real negotiations outside the planner. Opponents use EV regardless
of their actual class. Beyond the horizon, the shared wealth heuristic does not
forecast future construction or bankruptcies. Finite samples introduce action
noise. Passing tests establishes functionality, not stronger play.

Package changes alter the source fingerprint: replay older results with their
archived source. Do not update old manifests to bypass this check.
