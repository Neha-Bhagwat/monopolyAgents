"""Compare current search to a trusted, locally archived previous implementation.

Run from the repository root with python3 -m scripts.benchmark_expectimax.
The reference archive contains executable Python; use only your own snapshot.
"""

import argparse
import hashlib
import json
import platform
import statistics
import types
import zipfile
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

from monopoly.advanced_decisions import ExpectimaxDecisions, _assets
from monopoly.bots import baseline_bot, hard_bargainer_bot
from monopoly.engine import Game
from monopoly.experiments import source_hash


def load_reference(path):
    with zipfile.ZipFile(path) as archive:
        source = archive.read('monopoly/advanced_decisions.py')
    module = types.ModuleType('monopoly._expectimax_benchmark_reference')
    exec(compile(source, str(path), 'exec'), module.__dict__)
    return module.ExpectimaxDecisions, hashlib.sha256(source).hexdigest()


def run_game(policy_class, seed, seat):
    class Timed(policy_class):
        def __init__(self):
            super().__init__()
            self.timings = []
            self.actions = []

        def buy(self, view, player, square):
            start = perf_counter()
            result = super().buy(view, player, square)
            self.timings.append(perf_counter() - start)
            self.actions.append(('buy', view.turns, player, square, result))
            return result

        def build(self, view, player, legal):
            start = perf_counter()
            result = super().build(view, player, legal)
            self.timings.append(perf_counter() - start)
            self.actions.append(('build', view.turns, player, legal, result))
            return result

    bot = hard_bargainer_bot()
    bot.decisions = Timed()
    bots = [baseline_bot() for _ in range(4)]
    bots[seat] = bot
    game = Game(bots, seed=seed)
    _assets.cache_clear()  # Do not time a cache warmed by previous games.
    start = perf_counter()
    result = game.play()
    elapsed = perf_counter() - start
    signature = (asdict(result), [asdict(t) for t in game.trade_history], bot.decisions.actions)
    return elapsed, bot.decisions.timings, signature


def compare_search_values(reference):
    cases = []
    for count in (2, 4):
        game = Game([baseline_bot() for _ in range(count)])
        game.owners[1] = game.owners[3] = 0
        game.owners[10] = game.owners[13] = game.owners[14] = 1
        game.buildings[10] = game.buildings[13] = game.buildings[14] = 1
        game.players[0].cash = 400
        game.players[1].position = 18
        game.turns = 120
        cases.append(game.view())
        game.players[0].jailed = True
        cases.append(game.view())
        game.players[0].jailed = False
        game.turns = 149
        cases.append(game.view())
    comparisons = 0
    for view in cases:
        for depth, limit in ((1, 1000), (2, 1), (2, 5), (2, 1000), (3, 20), (3, 1000)):
            old, new = reference(depth, limit), ExpectimaxDecisions(depth, limit)
            old_budget, new_budget = [0], [0]
            expected = old._search(view, 0, depth, old_budget)
            actual = new._search(view, 0, depth, new_budget)
            if actual != expected or old_budget != new_budget:
                raise AssertionError((depth, limit, expected, actual, old_budget, new_budget))
            comparisons += 1
    return comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path,
                        default=Path('results/expectimax-optimization/source-before.zip'))
    parser.add_argument('--output', type=Path,
                        default=Path('results/expectimax-optimization/benchmark.json'))
    parser.add_argument('--seeds', type=int, default=10)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if args.seeds < 1 or args.repeats < 1:
        parser.error('seeds and repeats must be positive')
    if args.output.exists():
        parser.error('output already exists; choose a new path')
    reference, reference_hash = load_reference(args.reference)
    comparisons = compare_search_values(reference)
    print(f'{comparisons} search values and node budgets match, including tight cutoffs.', flush=True)
    rows = []
    times = {'before': [], 'after': []}
    signatures = {}
    for repeat in range(args.repeats):
        for seed in range(args.seeds):
            # Alternate execution order to reduce warm-up/CPU ordering bias.
            order = [('before', reference), ('after', ExpectimaxDecisions)]
            if (repeat + seed) % 2:
                order.reverse()
            for name, policy in order:
                duration, decisions, signature = run_game(policy, seed, seed % 4)
                if seed in signatures and signature != signatures[seed]:
                    raise AssertionError(f'Game/action/trade mismatch for seed {seed}')
                signatures[seed] = signature
                times[name].extend(decisions)
                rows.append({'version': name, 'repeat': repeat, 'seed': seed,
                             'seat': seed % 4, 'seconds': duration, 'decisions': len(decisions)})
        print(f'Finished paired repeat {repeat + 1}/{args.repeats}.', flush=True)
    summary = {}
    for name in times:
        decisions = times[name]
        summary[name] = {
            'mean_game_seconds': statistics.mean(row['seconds'] for row in rows if row['version'] == name),
            'decision_count': len(decisions),
            'decision_mean_ms': statistics.mean(decisions) * 1000,
            'decision_median_ms': statistics.median(decisions) * 1000,
            'decision_p95_ms': sorted(decisions)[int(.95 * (len(decisions) - 1))] * 1000,
        }
    summary['game_speedup'] = summary['before']['mean_game_seconds'] / summary['after']['mean_game_seconds']
    summary['decision_speedup'] = summary['before']['decision_mean_ms'] / summary['after']['decision_mean_ms']
    cpu = platform.processor()
    cpuinfo = Path('/proc/cpuinfo')
    if cpuinfo.exists():
        cpu = next(line.split(':', 1)[1].strip() for line in cpuinfo.read_text().splitlines()
                   if line.startswith('model name'))
    report = {'python': platform.python_version(), 'cpu': cpu, 'depth': 2, 'max_nodes': 1000,
              'reference_module_sha256': reference_hash, 'current_source_sha256': source_hash(),
              'search_value_comparisons': comparisons, 'unique_game_pairs': args.seeds,
              'paired_runs': args.seeds * args.repeats, 'all_decisions_results_trades_match': True,
              'summary': summary, 'runs': rows,
              'note': 'Paired single-process timing; one search bot against three baselines, seats cycled by seed. No strength claim.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    print('Saved', args.output)


if __name__ == '__main__':
    main()
