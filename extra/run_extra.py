"""Portable download/check/merge runner; no shell configuration execution."""
import argparse
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from common import WINDOWS, lock

DATASETS = ('toys4k', 'omniobject3d', 'shapenet')
DOWNLOAD_KEYS = {'ARCHIVE_DIR', 'TOYS4K_ARCHIVE_URL', 'SHAPENETCORE_BASE_URL'}


def load_env(path):
    if not path.is_file():
        return
    addresses = {}
    for number, line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(), 1):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        assignment = re.fullmatch(r'(?:export\s+)?([A-Za-z_][A-Za-z_0-9]*)\s*=\s*(.*)', line)
        if not assignment:
            raise ValueError(f'Configuration line {number} must use KEY=VALUE; shell commands are not executed')
        key, value = assignment.groups()
        if key not in DOWNLOAD_KEYS:
            raise ValueError(f'Configuration line {number}: downloads.env only accepts download URLs and archive paths; remove {key}')
        if value.startswith(('"', "'")):
            end = value.find(value[0], 1)
            if end < 0 or (value[end + 1:].strip() and not value[end + 1:].lstrip().startswith('#')):
                raise ValueError(f'Configuration line {number} has invalid quoting or trailing content')
            value = value[1:end]
        else:
            value = '' if value.startswith('#') else re.split(r'\s+#', value, maxsplit=1)[0].rstrip()
        # Values are literal: preserve Windows backslashes and never run substitutions.
        addresses[key] = value
    os.environ.update(addresses)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Download, verify, and merge extra assets across platforms')
    parser.add_argument('--env', type=Path, default=Path(__file__).with_name('downloads.env'))
    parser.add_argument('--data', type=Path, default=ROOT / 'artifacts')
    parser.add_argument('--stage', choices=('download', 'check', 'merge', 'all'), default='all')
    parser.add_argument('datasets', nargs='*', metavar='DATASET', help='toys4k / omniobject3d / shapenet')
    args = parser.parse_args(argv)
    try:
        load_env(args.env.expanduser().resolve())
        venv = ROOT / '.venv' / ('Scripts/python.exe' if WINDOWS else 'bin/python')
        python = str(venv) if venv.is_file() else sys.executable
        python = os.path.abspath(os.path.expanduser(python))
        if os.path.normcase(python) != os.path.normcase(os.path.abspath(sys.executable)):
            os.execv(python, [python, str(Path(__file__).resolve()), *(sys.argv[1:] if argv is None else argv)])
        data = args.data.expanduser().resolve()
        if not data.is_dir():
            raise ValueError('Data directory not found; download artifacts first or specify its location with --data')
        datasets = args.datasets or DATASETS
        if any(ds not in DATASETS for ds in datasets):
            raise ValueError('Unknown dataset specified')
        from extra import main as run_stage
        os.chdir(ROOT)
        with lock(data, write=True, name='.pipeline.lock'):
            for dataset in datasets:
                for stage in ('download', 'check', 'merge') if args.stage == 'all' else (args.stage,):
                    result = run_stage(['--data', str(data), stage, dataset])
                    if result:
                        return result
        return 0
    except KeyboardInterrupt:
        print('Interrupted; rerun the same command to resume', file=sys.stderr)
        return 130
    except (OSError, ValueError, ImportError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
