# Expectimax decisions

Expectimax is a planner, not a trained model. There is no dataset, training loop,
GPU requirement, or saved model. It evaluates actions while the game runs.

## Run and verify

From the repository root, using Python 3.10 or newer:

```bash
python3 -m monopoly --step6-game --seed 42
python3 -m unittest discover -s tests -v
```

The first command now runs Hard-bargainer with Expectimax and Zeuthen trading.
Adaptive and Risk-averse still use expected-value decisions; their advanced
policies remain unimplemented.

## Algorithm

At the current decision, compare buying against skipping, or every affordable
legal building against not building. After buying, also consider this turn's
building opportunity, matching the engine's order of actions. Keep the existing
$150 purchase reserve and $250 building reserve. Ties prefer no spending.

For each candidate, expand future individual player turns. Chance nodes average
all 11 dice totals with their exact two-dice probabilities. Card squares branch
uniformly over the six fixed cards. Own future buy/build decisions maximize the
score. Opponents are forecast using expected-value decisions. The real engine's
landing/payment/build-legality helpers operate on private copies of public state.
Search never reads the live RNG, shuffled decks, or opponents' trading memories.

Default settings are `ExpectimaxDecisions(depth=2, max_nodes=1000)`. Depth can be
1, 2, or 3 and counts individual turns, not rounds of the board. Each root buy/skip
or build candidate gets a fresh node budget. The limit counts expanded chance
nodes; all outcomes at an expanded node are still weighted, so this is not a
limit on leaf evaluations or a wall-clock deadline. Once the budget is exhausted,
remaining branches use the leaf heuristic. Fixed traversal makes decisions
repeatable, though budget cutoffs can make branch ordering matter.

The leaf heuristic compares the agent's estimated wealth to its strongest
surviving opponent: cash plus printed property/building values, expected net rent
for the remaining game using shared stationary landing probabilities, the shared
$50 complete-set bonus, and a 10% liquidity preference. Terminal turn-cap states
use actual net-worth differences, without future rent or bonuses. Bankruptcy and
being the last survivor receive large negative/positive scores.

## Scope and limitations

- In a four-player game, two future turns usually examine two opponents, not the
  searching player's next turn. Deeper planning is limited by CPU cost.
- Unknown card draws are approximated as independent uniform draws. The engine
  actually cycles a shuffled deck; this planner does not track card history.
- Future trades are omitted. Zeuthen handles actual trading outside the search.
- Opponents are approximated by expected-value policies, regardless of their
  actual strategy. This is a policy model, not adversarial minimax.
- Beyond the search horizon, expected rents use current ownership/buildings;
  future construction, future trades, and bankruptcy are not forecast in the tail.
- The score is a heuristic, not a calibrated win probability. Passing tests and
  faster execution do not establish superiority; that needs the final tournament.

## Controlled experiments

`hard_bargainer_bot()` uses Expectimax by default. For the trading-only ablation,
`make_game()` explicitly calls `hard_bargainer_bot(use_expectimax=False)` so all
six controlled variants retain identical expected-value buy/build policies.

Changes to package code invalidate strict replay of older source fingerprints.
Use each saved result's `source.zip` to replay its original version. Comparing
current ablation rows/trades against the pre-bot baseline is a separate regression
check; do not edit old manifests to bypass the replay guard.

## Local timing check

On the available Intel Core i7-13700H with Python 3.12.3, ten seeded games
(seeds 0–9, Hard-bargainer in seat 1 against three Baselines) averaged 0.301 s
with default Expectimax, compared with 0.094 s using EV. Across 61 search
buy/build calls, median latency was 35.8 ms, p95 60.3 ms, and maximum 76.3 ms.
These are a small local timing sample, not performance guarantees or evidence of
better play. Different lineups, development-heavy games and deeper search cost
more. No GPU was used. Raw measurements and the benchmark script are under
`results/expectimax/`; rerun with `python3 -m results.expectimax.benchmark` while
the pre-bot baseline folder is present.

Validation: all 64 tests passed (55 restored tests plus 9 new search tests).
All 240 pre-bot controlled experiment rows and accepted-trade logs were reproduced
exactly with the unchanged EV ablation configuration.

## Optimization and sources

The optimized implementation preserves depth, chance probabilities, leaf scoring,
reserves, tie handling, and traversal order. It does not add selective pruning or
make the opponent/card model more approximate.

- A bounded LRU cache (4,096 entries) stores immutable asset values, rent income,
  completed-set counts, and building candidates keyed by ownership, building
  levels, and player count. These quantities do not depend on cash, dice positions,
  or turns. Affordability and the remaining horizon are still evaluated separately.
- A per-decision transposition table (at most 4,096 entries) reuses fully evaluated
  chance subtrees. Keys include every changing public state field, player
  perspective, and search depth; the board is the engine's fixed `BOARD`.
- Cache hits consume the original subtree's logical node cost. A hit is bypassed
  when the remaining budget is insufficient, and budget-truncated subtrees are
  not stored as complete values. This keeps the previous cutoff behaviour.
- Buy/build transitions update only the affected immutable tuples. Leaf evaluation
  and building-option queries no longer allocate a copied game. Movement, cards,
  rent and bankruptcy still use the original engine helpers.

Sources informing these choices:

1. [Berkeley CS188: Expectimax](https://inst.eecs.berkeley.edu/~cs188/textbook/games/expectimax.html)
   explains why ordinary alpha-beta pruning does not apply to chance nodes without
   known finite value bounds. No pruning bounds have been proven for this project's
   wealth heuristic, so ordinary alpha-beta was not added.
2. [Veness and Blair, Effective Use of Transposition Tables in Stochastic Game Tree Search (2007)](https://www.cse.unsw.edu.au/~blair/pubs/2007VenessBlairCIG.pdf)
   motivates reusing search results at chance nodes and distinguishing exact values
   from incomplete bounds. Our cache uses only complete subtrees; it is not an
   implementation of their Star2-TT pruning algorithm.
3. [Python 3.12 profiling](https://docs.python.org/3.12/library/profile.html) and
   [bounded memoization](https://docs.python.org/3.12/library/functools.html#functools.lru_cache)
   support measuring hotspots and caching pure functions with hashable inputs.

### Paired measurements

On the same i7-13700H / Python 3.12.3, compare the archived previous version with
this version using identical seeds and alternating execution order. The workload
uses ten seeds, three repetitions, and seats cycled by seed: 30 paired games and
315 timed decisions per implementation. The asset cache is cleared before every
game. Depth remains 2, with 1,000 logical chance-node expansions per candidate.

| Metric | Before | Optimized |
| --- | ---: | ---: |
| Mean decision | 45.26 ms | 16.01 ms |
| Median decision | 37.66 ms | 14.23 ms |
| 95th percentile decision | 117.41 ms | 36.89 ms |
| Mean complete game | 0.518 s | 0.211 s |

This is 2.83x faster mean decision time and 2.46x faster complete games for this
sample. All chosen actions, game results, and accepted-trade logs match. Another
36 comparisons match exact search values and logical expansion counts for depths
1–3, budgets including tight cutoffs, two/four players, jail, and the turn cap.
These are speed/equivalence checks, not evidence of improved playing strength.
The earlier timing table used a different seat schedule and should not be used
as the baseline for these ratios.

Reproduce the comparison from the project root:

```bash
python3 -m scripts.benchmark_expectimax \
  --reference results/expectimax-optimization/source-before.zip \
  --output results/expectimax-optimization/benchmark-repeat.json
```

The reference ZIP is a locally saved source snapshot; the benchmark executes its
old decision module, so only pass a trusted archive. Raw results and profiles are
in `results/expectimax-optimization/`. Four additional regression tests exercise
cache invalidation, perspective/depth/state separation, logical budget accounting,
and equality to uncached search, including cutoff cases.

Final validation after optimization: all 68 tests pass. The full suite completed
in 21.6 seconds on the local machine.
