"""Independently verify completed E4 tables, intervals and selected replays."""
import argparse
from collections import Counter
import csv
from hashlib import sha256
import json
from pathlib import Path
from random import Random
import zipfile

from monopoly.e4 import load_config, policy_audit, require, schedule
from monopoly.experiments import run_case, source_hash, STRATEGIES
from monopoly.research_io import digest, write_json


def validate(folder, replay_sample=False):
    import numpy as np

    folder = Path(folder)
    config = load_config(folder / 'config.json')
    manifest = json.loads((folder / 'manifest.json').read_text())
    require(config == manifest['config'], 'Configuration changed')
    source_zip = folder / 'source.zip'
    require(digest(source_zip) == manifest['source_archive_sha256'], 'Source archive changed')
    # Historical reports can be audited after package changes. Replay still
    # requires the exact package that originally produced the saved games.
    fingerprint = sha256()
    with zipfile.ZipFile(source_zip) as archive:
        package_files = sorted(name for name in archive.namelist()
                               if Path(name).parent == Path('monopoly') and name.endswith('.py'))
        require(bool(package_files), 'Archive is missing the simulation package')
        for name in package_files:
            fingerprint.update(Path(name).name.encode())
            fingerprint.update(archive.read(name))
        require(json.loads(archive.read('config.json')) == config, 'Archived configuration differs')
    require(fingerprint.hexdigest() == manifest['source_sha256'], 'Archived package fingerprint differs')
    exact_package_match = source_hash() == manifest['source_sha256']
    require(not replay_sample or exact_package_match, 'Replay requires the package from the saved source.zip')
    audit = policy_audit() if exact_package_match else None
    with (folder / 'games.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    expected = schedule(config)
    require(len(rows) == 1200 and len({r['game_id'] for r in rows}) == 1200, 'Final E4 needs 1200 unique games')
    require([r['game_id'] for r in rows] == [r['game_id'] for r in expected], 'Ordered schedule mismatch')
    for row, case in zip(rows, expected):
        require(all(row[key] == str(value) for key, value in case.items()), 'Game metadata differs from schedule')
        seat = int(row['test_seat'])
        require(int(row['test_won']) == int(int(row['winner']) == seat), 'Test win differs from winner')
        require(all(row[f'seat_{i}'] == (row['strategy'] if i == seat else 'baseline')
                    for i in range(1, 5)), 'Unexpected lineup')
    require(Counter((r['strategy'], int(r['test_seat'])) for r in rows)
            == Counter({(s, seat): 50 for s in STRATEGIES for seat in range(1, 5)}), 'Unbalanced seats')
    journal = [json.loads(line) for line in (folder / 'games.journal.jsonl').read_text().splitlines()]
    require(len(journal) == 1200, 'Journal count mismatch')
    require([{k: str(v) for k, v in record['row'].items()} for record in journal] == rows, 'CSV/journal mismatch')
    trades = [json.loads(line) for line in (folder / 'trades.jsonl').read_text().splitlines()]
    require(trades == [t for record in journal for t in record['trades']], 'Trade journal mismatch')
    require(len(trades) == sum(int(r['trades_accepted']) for r in rows), 'Trade count mismatch')
    for row in rows:
        winner = int(row['winner'])
        require(int(row[f'bankrupt_{winner}']) == 0, 'Bankrupt winner')
        require(int(row[f'net_worth_{winner}']) == max(int(row[f'net_worth_{i}']) for i in range(1, 5)), 'Incorrect winner worth')
    # Independent vectorized bootstrap: reuse random seed/block sampling design,
    # not the analysis implementation or its percentile helper.
    seeds = list(range(config['seed_start'], config['seed_start'] + 50))
    indexed = {(int(r['seed']), int(r['test_seat']), r['strategy']): int(r['test_won']) for r in rows}
    blocks = np.array([[sum(indexed[seed, seat, s] for seat in range(1, 5)) / 4
                        for s in STRATEGIES] for seed in seeds])
    rng = Random(config['bootstrap_seed'])
    draws = np.array([[rng.randrange(50) for _ in range(50)] for _ in range(config['bootstrap_samples'])])
    rates = blocks[draws].mean(axis=1)
    rate_intervals = np.quantile(rates, [.025, .975], axis=0)
    delta_intervals = np.quantile((rates - rates[:, :1]) * 100, [.025, .975], axis=0)
    with (folder / 'analysis/overall_results.csv').open(newline='') as stream:
        results = list(csv.DictReader(stream))
    require([r['strategy'] for r in results] == list(STRATEGIES), 'Missing strategy summary')
    for i, row in enumerate(results):
        require(int(row['games']) == 200, 'Wrong group denominator')
        actual = [float(row[k]) for k in ('win_rate', 'improvement_pp', 'ci_95_low', 'ci_95_high',
                                          'improvement_ci_95_low_pp', 'improvement_ci_95_high_pp')]
        expected_values = [blocks[:, i].mean(), 100 * (blocks[:, i].mean() - blocks[:, 0].mean()),
                           *rate_intervals[:, i], *delta_intervals[:, i]]
        require(np.allclose(actual, expected_values, atol=1e-10, rtol=0), 'Summary/bootstrap mismatch')
    with (folder / 'analysis/paired_differences.csv').open(newline='') as stream:
        pairs = list(csv.DictReader(stream))
    require(len(pairs) == 1200, 'Missing paired rows')
    for pair, case in zip(pairs, expected):
        seed, seat, strategy = case['seed'], case['test_seat'], case['strategy']
        require((int(pair['seed']), int(pair['test_seat']), pair['strategy']) == (seed, seat, strategy), 'Pair key mismatch')
        win = indexed[seed, seat, strategy]
        control_win = indexed[seed, seat, 'no_trade']
        require(int(pair['difference']) == win - control_win, 'Pair difference mismatch')
        require((int(pair['test_won']), int(pair['control_won'])) == (win, control_win), 'Pair win indicators differ')
    control = {r['game_id'] for r in rows if r['strategy'] == 'no_trade'}
    participation = Counter()
    by_id = {r['game_id']: r for r in rows}
    for trade in trades:
        row = by_id[trade['game_id']]
        seat = int(row['test_seat'])
        offer = trade['offer']
        involved = seat in (offer['proposer'], offer['recipient'])
        require(not involved or trade['game_id'] not in control, 'No-trade agent traded')
        participation[trade['game_id']] += involved
    require(all(participation[r['game_id']] == int(r['test_trades_accepted']) for r in rows), 'Participation mismatch')
    replayed = []
    if replay_sample:
        for record in journal:
            row = record['row']
            if row['seed'] in (seeds[0], seeds[-1]) and row['test_seat'] in (1, 4):
                actual_row, actual_trades = run_case(row['strategy'], row['seed'], row['test_seat'] - 1)
                require(actual_row == row, 'Replay result mismatch')
                require(json.loads(json.dumps(actual_trades)) == record['trades'], 'Replay trades mismatch')
                replayed.append(row['game_id'])
    report = {'valid': True, 'games': len(rows), 'variants': 6, 'games_per_variant': 200,
              'games_per_variant_seat': 50, 'seed_blocks': 50, 'seed_start': seeds[0], 'seed_end': seeds[-1],
              'live_policy_audit': audit, 'archived_package_verified': True,
              'live_package_matches_archive': exact_package_match, 'accepted_deals': len(trades),
              'test_agent_deals': sum(participation.values()), 'independent_bootstrap_verified': True,
              'replayed_game_ids': replayed, 'replay_games': len(replayed),
              'validation_script_sha256': digest(Path(__file__)),
              'files_sha256': {p.relative_to(folder).as_posix(): digest(p) for p in sorted(folder.rglob('*'))
                               if p.is_file() and p.name != 'independent_validation.json'},
              'limits': ['Final seed usage checked against supplied local datasets only.',
                         'Intervals are individual percentile seed-block intervals; no multiplicity adjustment.']}
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('results/e4_trading_ablation'))
    parser.add_argument('--replay-sample', action='store_true')
    parser.add_argument('--output', type=Path, help='Write a new JSON report at this path')
    args = parser.parse_args()
    report = validate(args.input, args.replay_sample)
    if args.output is not None:
        require(not args.output.exists(), 'Validation output already exists')
        write_json(args.output, report)
    print(json.dumps(report, indent=2))
