"""SQLite records and process coordination; no import/build functionality."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import sqlite3
import sys
import time

WINDOWS = os.name == 'nt'
if WINDOWS:
    import msvcrt
else:
    import fcntl

SOURCES = ('cap3d', 'marvel_40m_plus', 'trellis_500k')
FIELDS = ('uuid', 'source_dataset', 'original_id', 'captions', 'download_urls')


def dumps(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), sort_keys=True)


def format_bytes(value):
    for unit in ('B', 'KiB', 'MiB', 'GiB', 'TiB'):
        if value < 1024 or unit == 'TiB':
            return f'{value:.0f} B' if unit == 'B' else f'{value:.1f} {unit}'
        value /= 1024


def terminal_line(dataset, stage, message, **details):
    parts = [f'[{dataset} / {stage}] {message}']
    if 'completed' in details and ('unit' in details or 'total' in details):
        completed = details.pop('completed')
        total = details.pop('total', None)
        unit = details.pop('unit', '')
        count = format_bytes(completed) if unit == 'bytes' else f'{completed:,}'
        if total is not None:
            count += ' / ' + (format_bytes(total) if unit == 'bytes' else f'{total:,}')
        if unit and unit != 'bytes':
            count += ' ' + unit
        if total:
            count += f' ({100 * completed / total:.1f}%)'
        parts.append(count)
    for key, value in details.items():
        if value is not None:
            label = 'ID' if key == 'original_id' else key.replace('_', ' ').capitalize()
            value = format_bytes(value) if key.endswith('_bytes') else value
            parts.append(f'{label}: {value}')
    line = ' | '.join(parts)
    return ''.join(char if char.isprintable() else repr(char)[1:-1] for char in line)


class Progress:
    def __init__(self, dataset, stage, **context):
        self.dataset, self.stage = dataset, stage
        self.context = context
        self.last = None

    def __call__(self, message, *, force=False, **details):
        now = time.monotonic()
        if force or self.last is None or now - self.last >= 2:
            print(terminal_line(self.dataset, self.stage, message, **{**self.context, **details}),
                  file=sys.stderr, flush=True)
            self.last = now


@contextmanager
def database(path, *, write=False):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f'Published database not found: {path}')
    con = sqlite3.connect(path.as_uri() + ('?mode=rw' if write else '?mode=ro'), uri=True, timeout=30)
    con.row_factory = sqlite3.Row
    con.execute('PRAGMA foreign_keys=ON')
    if write:
        con.execute('PRAGMA synchronous=FULL')
    try:
        yield con
    finally:
        con.close()


@contextmanager
def lock(root, *, write=False, name='.runtime.lock'):
    # ponytail: one lock per release, split locks only if merge blocks queries too long.
    path = Path(root) / name
    with path.open('a+b') as handle:
        if WINDOWS:
            # ponytail: Windows readers are serialized; use LockFileEx if concurrency matters.
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            fcntl.flock(handle, (fcntl.LOCK_EX if write else fcntl.LOCK_SH) | fcntl.LOCK_NB)
        try:
            yield
        finally:
            if WINDOWS:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def sync_directory(path):
    # Windows has no POSIX directory fsync; file fsync and SQLite FULL remain mandatory.
    if not WINDOWS:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def contained(root, relative):
    relative = str(relative)
    if not relative or '\\' in relative or '\x00' in relative or Path(relative).is_absolute():
        raise ValueError('Unsafe relative file path')
    if WINDOWS and any(PureWindowsPath(part).is_reserved() or part.endswith((' ', '.'))
                       or any(char in part for char in '<>:"|?*')
                       for part in PurePosixPath(relative).parts if part not in ('.', '..')):
        raise ValueError('File path contains a Windows reserved name or invalid character')
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if path == root or not path.is_relative_to(root):
        raise ValueError('File path is outside the allowed directory')
    return path


def digest(path, *, progress=None):
    with Path(path).open('rb') as handle:
        if progress is None:
            return hashlib.file_digest(handle, 'sha256').hexdigest()
        checksum = hashlib.sha256()
        completed, total = 0, Path(path).stat().st_size
        progress('Computing SHA256', file=str(path), completed=completed, total=total, unit='bytes')
        while block := handle.read(8 * 1024 * 1024):
            checksum.update(block)
            completed += len(block)
            progress('Computing SHA256', file=str(path), completed=completed, total=total, unit='bytes')
        return checksum.hexdigest()


def record(row):
    result = {field: row[field] for field in FIELDS}
    for field in ('captions', 'download_urls'):
        if isinstance(result[field], str):
            result[field] = json.loads(result[field])
    return result


def values(row):
    return tuple(dumps(row[field]) if field in ('captions', 'download_urls') else row[field] for field in FIELDS)


def caption_key(source, field, text):
    return source, field, ' '.join(text.split())


def union_captions(old, new):
    result = {source: list(old[source]) for source in SOURCES}
    for source in SOURCES:
        seen = {caption_key(source, c['field'], c['text']) for c in result[source]}
        for item in new[source]:
            key = caption_key(source, item['field'], item['text'])
            if key not in seen:
                result[source].append(item)
                seen.add(key)
    return result


def get_state(con, key):
    row = con.execute('SELECT value FROM metadata WHERE key=?', (key,)).fetchone()
    return json.loads(row[0]) if row else None


def set_state(con, key, value):
    con.execute('INSERT INTO metadata VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value',
                (key, dumps(value)))


def error(con, dataset, rid, stage, reason):
    key = dumps([dataset, rid, stage])
    if reason is None:
        con.execute('DELETE FROM operation_logs WHERE key=?', (key,))
    else:
        con.execute('''INSERT INTO operation_logs VALUES (?,?,?,?,?,1) ON CONFLICT(key)
            DO UPDATE SET reason=excluded.reason,attempts=operation_logs.attempts+1''',
                    (key, dataset, rid, stage, reason))
        print(terminal_line(dataset, stage, 'ERROR: ' + reason, original_id=rid),
              file=sys.stderr, flush=True)
