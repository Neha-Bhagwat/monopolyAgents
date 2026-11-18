"""Paired E4 analysis: resample entire seeds, retaining seats and variants."""
import csv
import json
from pathlib import Path
from random import Random
from statistics import mean

from .experiments import STRATEGIES
from .research_io import digest, new_output, write_csv, write_json


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def bootstrap(blocks, samples, seed):
    """Same bootstrap seed indices across all variants and control differences."""
    rng = Random(seed)
    count = len(blocks)
    rates = {s: [] for s in STRATEGIES}
    deltas = {s: [] for s in STRATEGIES}
    for _ in range(samples):
        indices = [rng.randrange(count) for _ in range(count)]
        for strategy in STRATEGIES:
            rates[strategy].append(mean(blocks[i][strategy] for i in indices))
            deltas[strategy].append(mean(blocks[i][strategy] - blocks[i]['no_trade'] for i in indices))
    interval = lambda xs: (percentile(xs, .025), percentile(xs, .975))
    return {s: (interval(rates[s]), interval(deltas[s])) for s in STRATEGIES}


def analyze(output):
    from .e4 import load_config, read_journal, require, schedule
    output = Path(output)
    manifest = json.loads((output / 'manifest.json').read_text())
    config = load_config(output / 'config.json')
    require(config == manifest['config'], 'Changed configuration')
    cases = {c['game_id']: c for c in schedule(config)}
    records = read_journal(output, cases)
    require(len(records) == len(cases), 'Analysis requires all scheduled games')
    rows = [records[gid]['row'] for gid in cases]
    with (output / 'games.csv').open(newline='') as stream:
        require(list(csv.DictReader(stream)) == [{k: str(v) for k, v in r.items()} for r in rows],
                'CSV differs from authoritative journal')
    saved_trades = [json.loads(line) for line in (output / 'trades.jsonl').read_text().splitlines()]
    require(saved_trades == [t for gid in cases for t in records[gid]['trades']], 'Trade export differs')
    indexed = {(r['seed'], r['test_seat'], r['strategy']): r for r in rows}
    seeds = list(range(config['seed_start'], config['seed_start'] + config['seed_count']))
    blocks = [{s: mean(indexed[seed, seat, s]['test_won'] for seat in (1, 2, 3, 4))
               for s in STRATEGIES} for seed in seeds]
    intervals = bootstrap(blocks, config['bootstrap_samples'], config['bootstrap_seed'])
    paired = [{'seed': seed, 'test_seat': seat, 'strategy': s,
               'test_won': indexed[seed, seat, s]['test_won'],
               'control_won': indexed[seed, seat, 'no_trade']['test_won'],
               'difference': indexed[seed, seat, s]['test_won'] - indexed[seed, seat, 'no_trade']['test_won']}
              for seed in seeds for seat in (1, 2, 3, 4) for s in STRATEGIES]
    summaries, seat_rows, activity = [], [], []
    for strategy in STRATEGIES:
        group = [r for r in rows if r['strategy'] == strategy]
        rate = mean(r['test_won'] for r in group)
        control = mean(b['no_trade'] for b in blocks)
        (lo, hi), (dlo, dhi) = intervals[strategy]
        summaries.append({'strategy': strategy, 'games': len(group), 'wins': sum(r['test_won'] for r in group),
                          'win_rate': rate, 'ci_95_low': lo, 'ci_95_high': hi,
                          'improvement_pp': 100 * (rate - control),
                          'improvement_ci_95_low_pp': 100 * dlo, 'improvement_ci_95_high_pp': 100 * dhi,
                          'mean_net_worth': mean(r[f'net_worth_{r["test_seat"]}'] for r in group),
                          'bankruptcy_rate': mean(r[f'bankrupt_{r["test_seat"]}'] for r in group),
                          'mean_turns': mean(r['turns'] for r in group)})
        for seat in (1, 2, 3, 4):
            seated = [r for r in group if r['test_seat'] == seat]
            seat_rows.append({'strategy': strategy, 'test_seat': seat, 'games': len(seated),
                              'wins': sum(r['test_won'] for r in seated),
                              'win_rate': mean(r['test_won'] for r in seated)})
        attempts = sum(r['trades_proposed'] for r in group)
        accepted = sum(r['trades_accepted'] for r in group)
        activity.append({'strategy': strategy, 'gamewide_negotiations': attempts,
                         'gamewide_accepted_deals': accepted,
                         'gamewide_acceptance_rate': accepted / attempts if attempts else '',
                         'test_agent_accepted_deals': sum(r['test_trades_accepted'] for r in group),
                         'test_agent_deals_per_game': mean(r['test_trades_accepted'] for r in group)})
    target = output / 'analysis'
    require(not target.exists(), 'Analysis exists; use a new output folder or archive it before regenerating')
    with new_output(target) as bundle:
        export_analysis(bundle, output, config, rows, seeds, cases, summaries, paired, seat_rows, activity)


def export_analysis(target, output, config, rows, seeds, cases, summaries, paired, seat_rows, activity):
    write_csv(target / 'overall_results.csv', summaries)
    write_csv(target / 'trading_effects.csv', summaries)
    write_csv(target / 'paired_differences.csv', paired)
    write_csv(target / 'seat_results.csv', seat_rows)
    write_csv(target / 'trade_activity.csv', activity)
    warnings = []
    if len(seeds) < 2:
        warnings.append('One-seed development run: bootstrap intervals are degenerate and not inferential evidence.')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        warnings.append('PNG figures deferred: install requirements-analysis.txt and regenerate in a new analysis folder.')
    else:
        for name, value, lower, upper, label in (
                ('win_rates', 'win_rate', 'ci_95_low', 'ci_95_high', 'Win rate'),
                ('improvement_over_no_trade', 'improvement_pp', 'improvement_ci_95_low_pp',
                 'improvement_ci_95_high_pp', 'Improvement over no trade (percentage points)')):
            values = [s[value] for s in summaries]
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(range(6), values)
            # Quantile intervals can exclude the point estimate; draw endpoints directly.
            ax.vlines(range(6), [s[lower] for s in summaries], [s[upper] for s in summaries], color='black')
            ax.set_xticks(range(6), STRATEGIES, rotation=20, ha='right')
            ax.set_ylabel(label)
            ax.set_title(f'E4 {config["purpose"]}: {len(rows)} games, {len(seeds)} seed blocks')
            ax.axhline(0, color='gray', linewidth=.5)
            fig.tight_layout()
            fig.savefig(target / (name + '.png'), dpi=160)
            plt.close(fig)
    write_json(target / 'validation.json', {
        'complete': True, 'games': len(rows), 'purpose': config['purpose'], 'seed_blocks': len(seeds),
        'expected_games': len(cases), 'interval_method': 'paired percentile seed-block bootstrap; four seats retained',
        'bootstrap_samples': config['bootstrap_samples'], 'bootstrap_seed': config['bootstrap_seed'],
        'input_sha256': {name: digest(output / name) for name in ('manifest.json', 'games.csv', 'trades.jsonl', 'games.journal.jsonl')},
        'analysis_source_sha256': digest(Path(__file__)),
        'acceptance_definition': 'accepted deals / gamewide negotiation attempts; counters are not separate attempts',
        'limitation': 'Per-agent rejected offer counts are not logged; estimated gains are not realized profit.',
        'warnings': warnings})
