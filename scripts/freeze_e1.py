"""Audit E1 and copy its complete evidence into a hash-verified snapshot."""
import argparse
from collections import Counter
import csv
import hashlib
from itertools import combinations
import json
from pathlib import Path
import shutil
import tempfile
import zipfile

from monopoly.research_io import digest, write_json


def audit(folder):
    if not __debug__:
        raise RuntimeError('The E1 audit requires assertions; run Python without -O')
    manifest = json.loads((folder / 'manifest.json').read_text())
    agents = manifest['agents']
    assert len(agents) == len(set(agents)) == 6, 'Expected six distinct agents'
    assert manifest['include_mcts'] is True
    assert manifest['games_planned'] == 3000 and manifest['seeds_per_matchup'] == 50
    assert manifest['seat_rotations'] == 4 and manifest['game_turn_limit'] == 150
    with (folder / 'games.csv').open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    expected = {}
    for number, lineup in enumerate(combinations(agents, 4), 1):
        for seed in range(manifest['seed_start'], manifest['seed_start'] + 50):
            for rotation in range(4):
                gid = f'e1-m{number:02d}-s{seed}-r{rotation}'
                expected[gid] = (f'm{number:02d}', seed, rotation,
                                 lineup[rotation:] + lineup[:rotation])
    assert len(rows) == 3000 and {r['game_id'] for r in rows} == set(expected), 'Schedule mismatch'
    by_id = {r['game_id']: r for r in rows}
    for row in rows:
        matchup, seed, rotation, seats = expected[row['game_id']]
        assert (row['matchup_id'], int(row['seed']), int(row['rotation'])) == (matchup, seed, rotation)
        assert tuple(row[f'seat_{s}'] for s in range(1, 5)) == seats
        winner = int(row['winner_seat'])
        assert 1 <= winner <= 4 and row['winner'] == seats[winner - 1]
        assert int(row[f'bankrupt_{winner}']) == 0
        assert 1 <= int(row['turns']) <= 150
        assert row['end_reason'] in ('turn_limit', 'last_player')
        worth = [int(row[f'net_worth_{s}']) for s in range(1, 5)]
        assert min(worth) >= 0 and worth[winner - 1] == max(worth)
        assert 0 <= int(row['trades_accepted']) <= int(row['trades_proposed'])
    journal = [json.loads(line) for line in (folder / 'games.journal.jsonl').read_text().splitlines()]
    assert len(journal) == 3000 and {x['row']['game_id'] for x in journal} == set(expected)
    journal_trades = []
    for record in journal:
        assert {k: str(v) for k, v in record['row'].items()} == by_id[record['row']['game_id']]
        assert len(record['trades']) == int(record['row']['trades_accepted'])
        journal_trades.extend(record['trades'])
    trades = [json.loads(line) for line in (folder / 'trades.jsonl').read_text().splitlines()]
    # Parallel completion order may differ from exported schedule order.
    canonical = lambda records: Counter(json.dumps(r, sort_keys=True) for r in records)
    assert canonical(trades) == canonical(journal_trades), 'Trade journal/export mismatch'
    for trade in trades:
        assert trade['game_id'] in by_id
        offer = trade['accepted_trade']['offer']
        assert 0 <= offer['proposer'] < 4 and 0 <= offer['recipient'] < 4
        assert offer['proposer'] != offer['recipient']
    with zipfile.ZipFile(folder / 'source.zip') as archive:
        h = hashlib.sha256()
        for name in sorted(n for n in archive.namelist() if n.startswith('monopoly/') and n.endswith('.py')):
            h.update(name.encode())
            h.update(b'\0')
            h.update(archive.read(name))
        assert h.hexdigest() == manifest['source_sha256'], 'Archived package fingerprint mismatch'
        source = archive.read('monopoly/tournament.py').decode()
        assert 'return PolicyBot(MCTSDecisions(), BayesianTrading())' in source
        assert 'ExpectimaxDecisions(depth=depth, max_nodes=max_nodes)' in source
        assert 'risk_averse_bot(use_prospect_theory=True)' in source
    return {'valid': True, 'games': len(rows), 'lineups': 15, 'games_per_lineup': 200,
            'appearances_per_agent': dict(Counter(s for r in rows for s in [r[f'seat_{i}'] for i in range(1, 5)])),
            'accepted_trades': len(trades), 'source_fingerprint_verified': True,
            'policy_check': 'Archived tournament factory wiring inspected; no archived code executed.',
            'warnings': ['Existing E1 intervals ignore seed dependence.',
                         'Existing analysis skipped trades, but accepted_trade.offer is present.']}


def freeze(source, output):
    if output.exists():
        raise FileExistsError(output)
    report = audit(source)
    hashes = {p.relative_to(source).as_posix(): digest(p) for p in sorted(source.rglob('*')) if p.is_file()}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix='.e1-freeze-') as temporary:
        bundle = Path(temporary) / 'bundle'
        shutil.copytree(source, bundle)
        assert all(digest(bundle / name) == value == digest(source / name) for name, value in hashes.items())
        write_json(bundle / 'freeze.json', {'schema_version': 1, 'input_directory': str(source.resolve()),
                                          'files_sha256': hashes, 'audit': report,
                                          'note': 'Tamper-evident snapshot; not filesystem write protection.'})
        bundle.rename(output)
    return report


def verify(folder):
    if not __debug__:
        raise RuntimeError('Frozen file verification requires assertions; run Python without -O')
    lock = json.loads((folder / 'freeze.json').read_text())
    actual = {p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file()}
    assert actual == set(lock['files_sha256']) | {'freeze.json'}, 'Snapshot file set changed'
    assert all(digest(folder / name) == value for name, value in lock['files_sha256'].items()), 'Snapshot changed'
    return audit(folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('results/e1_tournament_5'))
    parser.add_argument('--output', type=Path, default=Path('results/e1_frozen'))
    parser.add_argument('--verify', type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.verify) if args.verify else freeze(args.input, args.output), indent=2))
