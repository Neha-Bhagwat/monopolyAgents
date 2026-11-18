"""Prepare/run/resume the controlled E4 batch without changing agent logic.

--prepare schedules cases only. --run --resume executes an already prepared batch.
--analyze creates descriptive tables, paired seed-block intervals and PNGs.
"""
import argparse
import csv
from hashlib import sha256
import json
import os
from pathlib import Path
import platform

from .advanced_bargaining import BayesianTrading, MinimaxRegretTrading, ZeuthenTrading
from .bargaining import NashTrading, TitForTatTrading
from .bots import ExpectedValueDecisions, NoTrading, SimpleDecisions, SetCompletionTrading
from .experiments import STRATEGIES, make_game, run_case, source_hash, summarize
from .research_io import archive_source, digest, new_output, write_csv, write_json


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_config(path):
    config = json.loads(Path(path).read_text(encoding='utf-8'))
    require(config['schema_version'] == 1, 'Unsupported schema')
    require(config['strategies'] == list(STRATEGIES), 'All six ordered variants required')
    require(config['test_seats'] == [1, 2, 3, 4], 'All four seats required')
    require(config['turn_limit'] == 150 and config['buy_build_policy'] == 'ExpectedValueDecisions', 'Wrong policies/rules')
    require(config['opponents'] == 'three normal Baselines with set-completion trading', 'Wrong opponents')
    for key in ('seed_start', 'seed_count', 'bootstrap_samples', 'bootstrap_seed'):
        require(type(config[key]) is int and config[key] >= (1 if key in ('seed_count', 'bootstrap_samples') else 0), key)
    require(config['purpose'] in ('final', 'development_smoke'), 'Label the dataset purpose')
    require(config['purpose'] != 'final' or config['seed_count'] == 50, 'Final E4 requires 50 seeds')
    return config


def schedule(config):
    return [{'game_id': f'{strategy}-seed{seed}-seat{seat}', 'strategy': strategy,
             'seed': seed, 'test_seat': seat}
            for seed in range(config['seed_start'], config['seed_start'] + config['seed_count'])
            for seat in config['test_seats'] for strategy in config['strategies']]


def policy_audit():
    trading = (NoTrading, NashTrading, TitForTatTrading, ZeuthenTrading, BayesianTrading, MinimaxRegretTrading)
    for strategy, trade_class in zip(STRATEGIES, trading):
        for seat in range(4):
            game = make_game(strategy, 0, seat)
            require(type(game.bots[seat].decisions) is ExpectedValueDecisions, 'Advanced policy leaked into E4')
            require(type(game.bots[seat].trading) is trade_class, 'Wrong trading policy')
            require(game.turn_limit == 150, 'Wrong turn cap')
            for i, bot in enumerate(game.bots):
                if i != seat:
                    require(type(bot.decisions) is SimpleDecisions and type(bot.trading) is SetCompletionTrading,
                            'Baseline opponent changed')
    return {'checked_configurations': 24, 'buy_build': 'ExpectedValueDecisions',
            'advanced_buy_build_enabled': False, 'baseline_trading_enabled': True}


def prepare(config_path, output):
    config = load_config(config_path)
    require(not output.exists(), 'Output exists: use --run --resume for a prepared batch')
    audit = policy_audit()
    seed_audit = []
    if config['purpose'] == 'final':
        planned_seeds = set(range(config['seed_start'], config['seed_start'] + config['seed_count']))
        root = Path(__file__).resolve().parent.parent / 'results'
        for saved in sorted(root.rglob('games.csv')):
            with saved.open(newline='', encoding='utf-8-sig') as stream:
                used = {int(row['seed']) for row in csv.DictReader(stream)}
            require(not used & planned_seeds, f'Seed overlap with {saved}')
            seed_audit.append({'path': saved.relative_to(root).as_posix(), 'sha256': digest(saved),
                               'unique_seeds': len(used), 'overlap': []})
    planned = schedule(config)
    version = source_hash()
    with new_output(output) as bundle:
        config_copy = bundle / 'config.json'
        write_json(config_copy, config)
        archive_source(bundle / 'source.zip', config_copy)
        require(version == source_hash(), 'Source changed during preparation')
        manifest = {'experiment': 'E4 trading ablation', 'config': config,
                    'games_planned': len(planned), 'source_sha256': version,
                    'source_archive_sha256': digest(bundle / 'source.zip'), 'python': platform.python_version(),
                    'policy_audit': audit, 'trade_seat_convention': '1-based; flat offer records from experiments.run_case',
                    'local_seed_audit': seed_audit}
        write_json(bundle / 'manifest.json', manifest)
        write_csv(bundle / 'schedule.csv', planned)
        write_json(bundle / 'progress.json', {'scheduled': len(planned), 'completed': 0,
                                             'remaining': len(planned), 'status': 'prepared'})
    return manifest


def validate_record(record, case):
    row, trades = record['row'], record['trades']
    require(all(row[k] == case[k] for k in case), 'Unexpected game/seed/seat/strategy')
    seat = row['test_seat']
    require(row['test_won'] == int(row['winner'] == seat), 'Winner mismatch')
    require(type(row['winner']) is int and 1 <= row['winner'] <= 4, 'Invalid winner')
    require(1 <= row['turns'] <= 150 and row[f'bankrupt_{row["winner"]}'] == 0, 'Invalid terminal state')
    require(0 <= row['trades_accepted'] <= row['trades_proposed'], 'Invalid trade counts')
    require(len(trades) == row['trades_accepted'], 'Missing accepted trades')
    participation = 0
    for trade in trades:
        offer = trade['offer']
        require(trade['game_id'] == case['game_id'], 'Unmatched trade')
        require(1 <= offer['proposer'] <= 4 and 1 <= offer['recipient'] <= 4
                and offer['proposer'] != offer['recipient'], 'Invalid trade seats')
        participation += seat in (offer['proposer'], offer['recipient'])
    require(participation == row['test_trades_accepted'], 'Test participation mismatch')
    require(row['strategy'] != 'no_trade' or participation == 0, 'Control participated in trade')
    for i in range(1, 5):
        require(row[f'seat_{i}'] == (row['strategy'] if i == seat else 'baseline'), 'Wrong lineup')
        require(row[f'net_worth_{i}'] >= 0 and row[f'bankrupt_{i}'] in (0, 1), 'Invalid assets')


def read_journal(output, cases, repair=False):
    path = output / 'games.journal.jsonl'
    if not path.exists():
        return {}
    data = path.read_bytes()
    if data and not data.endswith(b'\n'):
        require(repair, 'Partial final journal record; resume to recover')
        # Preserve the interrupted bytes as evidence before truncating the tail.
        boundary = data.rfind(b'\n') + 1
        backup = path.with_name('journal-interrupted-tail.bin')
        # Keep every distinct interrupted tail and make repeat recovery safe.
        if backup.exists() and backup.read_bytes() != data[boundary:]:
            tail_hash = sha256(data[boundary:]).hexdigest()
            backup = path.with_name(f'journal-interrupted-tail-{tail_hash}.bin')
        backup.write_bytes(data[boundary:])
        data = data[:boundary]
        temporary = path.with_suffix('.repair.tmp')
        temporary.write_bytes(data)
        os.replace(temporary, path)
    completed = {}
    for line in data.splitlines():
        record = json.loads(line)
        gid = record['row']['game_id']
        require(gid in cases and gid not in completed, 'Duplicate or unexpected journal game')
        validate_record(record, cases[gid])
        completed[gid] = record
    return completed


def export(output, cases, completed):
    records = [completed[gid] for gid in cases if gid in completed]
    if records:
        write_csv(output / 'games.csv', [r['row'] for r in records])
        write_csv(output / 'summary.csv', summarize([r['row'] for r in records]))
    temp = output / 'trades.jsonl.tmp'
    with temp.open('w', encoding='utf-8') as stream:
        for record in records:
            for trade in record['trades']:
                stream.write(json.dumps(trade, sort_keys=True) + '\n')
    os.replace(temp, output / 'trades.jsonl')
    write_json(output / 'progress.json', {'scheduled': len(cases), 'completed': len(records),
                                         'remaining': len(cases) - len(records),
                                         'status': 'complete' if len(records) == len(cases) else 'incomplete'})


def run(output, max_games=None):
    manifest = json.loads((output / 'manifest.json').read_text())
    config = load_config(output / 'config.json')
    require(config == manifest['config'], 'Configuration changed')
    require(source_hash() == manifest['source_sha256'], 'Source changed: start a new batch')
    require(digest(output / 'source.zip') == manifest['source_archive_sha256'], 'Archive changed')
    policy_audit()
    planned = schedule(config)
    with (output / 'schedule.csv').open(newline='') as stream:
        require(list(csv.DictReader(stream)) == [{k: str(v) for k, v in c.items()} for c in planned], 'Schedule changed')
    cases = {c['game_id']: c for c in planned}
    # Exclusive single-writer lock. A stale lock after a hard crash needs inspection.
    lock = output / '.running'
    with lock.open('x') as stream:
        stream.write(str(os.getpid()))
    try:
        completed = read_journal(output, cases, repair=True)
        count = 0
        try:
            for gid, case in cases.items():
                if gid in completed:
                    continue
                if max_games is not None and count >= max_games:
                    break
                row, trades = run_case(case['strategy'], case['seed'], case['test_seat'] - 1)
                record = {'row': row, 'trades': trades}
                validate_record(record, case)
                require(source_hash() == manifest['source_sha256'], 'Source changed during run')
                with (output / 'games.journal.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(record, sort_keys=True) + '\n')
                    stream.flush()
                    os.fsync(stream.fileno())
                completed[gid] = record
                count += 1
                if count % 24 == 0:
                    export(output, cases, completed)
        finally:
            export(output, cases, completed)
    finally:
        lock.unlink()


def analyze(output):
    from .e4_analysis import analyze as generate
    generate(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/e4_final.json'))
    parser.add_argument('--output', type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--run', action='store_true')
    mode.add_argument('--analyze', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--max-games', type=int)
    args = parser.parse_args()
    if args.prepare:
        prepare(args.config, args.output)
    elif args.run:
        require(args.resume, 'Use --run --resume on an explicitly prepared batch')
        require(args.max_games is None or args.max_games > 0, 'Positive --max-games required')
        run(args.output, args.max_games)
    else:
        analyze(args.output)


if __name__ == '__main__':
    main()
