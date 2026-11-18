"""Small reproducibility helpers shared by research commands."""
import csv
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import tempfile
import zipfile


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def write_csv(path, rows):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def archive_source(path, config_path):
    root = Path(__file__).resolve().parent.parent
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as archive:
        for directory in ('monopoly', 'tests', 'scripts'):
            for source in sorted((root / directory).glob('*.py')):
                archive.write(source, source.relative_to(root).as_posix())
        archive.write(config_path, 'config.json')


@contextmanager
def new_output(target):
    """Publish a new directory only when all of its contents are ready."""
    target = Path(target)
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent, prefix=f'.{target.name}-') as temporary:
        bundle = Path(temporary) / 'bundle'
        bundle.mkdir()
        yield bundle
        if target.exists():
            raise FileExistsError(target)
        bundle.rename(target)
