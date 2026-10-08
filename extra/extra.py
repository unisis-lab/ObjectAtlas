"""Install archive files, check immutable staged records, then merge approved assets."""
import gzip
import base64
import hmac
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import struct
import sys
import tarfile
import time
from urllib.parse import urlencode, urlparse
import zipfile

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

from common import (Progress, contained, database, digest, dumps, error, get_state,
                 lock, record, set_state, terminal_line, values)

DATASETS = ('toys4k', 'omniobject3d', 'shapenet')
CORE_URL = 'https://huggingface.co/datasets/ShapeNet/ShapeNetCore'
OPENXLAB_AUTH_HINT = ("Run 'openxlab login --relogin' or configure valid OPENXLAB_AK/OPENXLAB_SK in the process environment. "
                     'OPENXLAB_JWT overrides AK/SK; update or remove an expired JWT. Confirm access to OmniObject3D-New.')


def failure_reason(exc):
    # Network exception text may include credentials or signed download URLs.
    if isinstance(exc, requests.RequestException):
        return type(exc).__name__ + ': Network request failed'
    if isinstance(exc, ValueError):
        return str(exc) or type(exc).__name__
    return type(exc).__name__ + (': ' + str(exc) if str(exc) else '')


def request(session, url, *, stream=False, headers=None, progress=None):
    for attempt in range(5):
        try:
            response = session.get(url, stream=stream, timeout=(30, 90), headers=headers or {})
            if response.status_code not in (408, 429, 500, 502, 503, 504):
                if response.status_code not in (200, 206):
                    code = response.status_code
                    host = urlparse(response.url or url).hostname
                    response.close()
                    if code in (401, 403) and urlparse(url).hostname == 'huggingface.co':
                        if host != 'huggingface.co':
                            raise ValueError(f'Hugging Face download CDN returned HTTP {code}. '
                                             'Repository authentication may have succeeded; rerun for a fresh download link '
                                             'and check proxy caching and system time.')
                        raise ValueError(f'Hugging Face returned HTTP {code}. '
                                         "Run 'hf auth whoami' and 'hf auth login' with the account approved for ShapeNet/ShapeNetCore. "
                                         'HF_TOKEN overrides cached login; the token must have read access to this gated dataset.')
                    if code in (401, 403) and urlparse(url).hostname == 'openxlab.org.cn':
                        raise ValueError(f'OpenXLab returned HTTP {code}. ' + OPENXLAB_AUTH_HINT)
                    raise ValueError(f'Download service returned HTTP {code}')
                return response
            delay = response.headers.get('Retry-After', '')
            response.close()
            delay = min(60, int(delay) if delay.isdigit() else 2 ** attempt)
            if progress:
                progress('Waiting to retry download service', force=True,
                         http_status=response.status_code, attempt=attempt + 1, retry_in_seconds=delay)
            time.sleep(delay)
        except requests.RequestException as exc:
            if attempt == 4:
                raise ValueError(f'Network request failed ({type(exc).__name__}); rerun the command to resume') from None
            if progress:
                progress('Waiting to retry network request', force=True, error_type=type(exc).__name__,
                         attempt=attempt + 1, retry_in_seconds=2 ** attempt)
            time.sleep(2 ** attempt)
    raise ValueError(f'Service remains rate-limited or temporarily unavailable (HTTP {response.status_code})')


def archive_root():
    base = Path(__file__).resolve().parent
    directory = Path(os.environ.get('ARCHIVE_DIR') or 'archives').expanduser()
    return (directory if directory.is_absolute() else base / directory).resolve()


def archive_path(info):
    return contained(archive_root(), f"{info['dataset']}/{info['filename']}")


def cleanup_archive(info):
    path = archive_path(info)
    marker = path.with_name(path.name + '.managed')
    if marker.is_file():
        path.unlink(missing_ok=True)
        marker.unlink()


def package_url(session, info):
    ds = info['dataset']
    local = archive_path(info)
    if local.is_file():
        return str(local), {}
    if ds == 'toys4k':
        value = os.environ.get('TOYS4K_ARCHIVE_URL')
        if not value:
            raise ValueError('Place the official Blender ZIP in ARCHIVE_DIR/toys4k/ or set TOYS4K_ARCHIVE_URL')
        if value.startswith('https://www.dropbox.com/'):
            from urllib.parse import parse_qsl, urlunparse
            parts = urlparse(value)
            query = dict(parse_qsl(parts.query)); query['dl'] = '1'
            value = urlunparse(parts._replace(query=urlencode(query)))
        return value, {}
    if ds == 'shapenet':
        base = os.environ.get('SHAPENETCORE_BASE_URL', CORE_URL).rstrip('/').removesuffix('/resolve')
        if base != CORE_URL or not re.fullmatch('[0-9a-f]{40}', info['revision']):
            raise ValueError('Use the official ShapeNetCore repository with a pinned revision')
        token = os.environ.get('HF_TOKEN', '').strip()
        credential_source = 'HF_TOKEN environment variable' if token else 'Hugging Face login cache'
        if not token:
            try:
                from huggingface_hub import get_token
                token = get_token()
            except ImportError:
                raise ValueError("Install requirements.txt to load Hugging Face login credentials, then run 'hf auth login'") from None
        if not token:
            raise ValueError("No Hugging Face token found. Run 'hf auth login' with the account approved for ShapeNet/ShapeNetCore, "
                             'or set HF_TOKEN in the process environment. Browser login is not used by this script.')
        Progress(ds, 'download')('Using Hugging Face credentials', force=True, credential_source=credential_source)
        return f"{base}/resolve/{info['revision']}/{info['filename']}", {'Authorization': 'Bearer ' + token}
    # Use the user's existing official account configuration; credentials stay in memory.
    token = os.environ.get('OPENXLAB_JWT', '').strip()
    progress = Progress(ds, 'download')
    if token:
        progress('Using OpenXLab credentials', force=True, credential_source='OPENXLAB_JWT environment variable')
    if not token:
        local = Path.home() / '.openxlab/config.json'
        config = json.loads(local.read_text()) if local.is_file() else {}
        ak_env = os.environ.get('OPENXLAB_AK', '').strip()
        sk_env = os.environ.get('OPENXLAB_SK', '').strip()
        ak = ak_env or (config.get('ak') or '').strip()
        sk = sk_env or (config.get('sk') or '').strip()
        if not ak or not sk:
            raise ValueError("No OpenXLab credentials found. Run 'openxlab login' or configure OPENXLAB_AK/OPENXLAB_SK in the process environment; "
                             'alternatively, place official category archives in ARCHIVE_DIR/omniobject3d/')
        progress('Authenticating with OpenXLab', force=True,
                 ak_source='OPENXLAB_AK environment variable' if ak_env else '~/.openxlab/config.json',
                 sk_source='OPENXLAB_SK environment variable' if sk_env else '~/.openxlab/config.json')
        endpoint = 'https://openapi.openxlab.org.cn/api/v1/sso-be/api/v1/open/'
        def auth(action, payload):
            try:
                with session.post(endpoint + action, json=payload, timeout=(30, 90)) as response:
                    if response.status_code != 200:
                        raise ValueError(f'OpenXLab authentication failed (HTTP {response.status_code}, step {action}). ' + OPENXLAB_AUTH_HINT)
                    result = response.json()
                if result.get('msgCode') != '10000' or result.get('data', {}).get('msgCode') != '10000':
                    raise ValueError('OpenXLab rejected AK/SK authentication. ' + OPENXLAB_AUTH_HINT)
                return result['data']['data']
            except requests.RequestException as exc:
                raise ValueError(f'OpenXLab authentication request failed ({type(exc).__name__}); check connectivity and rerun') from None
        challenge = auth('auth', {'ak': ak.strip()})
        algorithm = challenge['algorithm'].removeprefix('Hmac').lower()
        if algorithm not in ('sha256', 'sha512', 'sha1'):
            raise ValueError('OpenXLab returned an unsupported signature algorithm')
        signature = base64.b64encode(hmac.digest(sk.strip().encode(), challenge['nonce'].encode(), algorithm)).decode()
        token = auth('getJwt', {'ak': ak.strip(), 'd': signature})['jwt']
    headers = {'Authorization': token}
    url = ('https://openxlab.org.cn/datasets/api/v2/downloadCheck/' + str(info['dataset_id'])
           + '/main/' + info['source_path'])
    with request(session, url, headers=headers):
        pass
    resolve = f"https://openxlab.org.cn/datasets/resolve/{info['dataset_id']}/main/{info['source_path']}"
    with session.get(resolve, headers=headers, allow_redirects=False, timeout=(30, 90)) as response:
        if response.status_code in (401, 403):
            raise ValueError(f'OpenXLab download URL request returned HTTP {response.status_code}. ' + OPENXLAB_AUTH_HINT)
        value = response.headers.get('Location') if response.status_code == 302 else None
    if not isinstance(value, str) or not value.startswith('https://'):
        raise ValueError('OpenXLab did not provide a valid download URL')
    return value, {}


def fetch(session, info, target):
    progress = Progress(info['dataset'], 'download', package=info.get('package'), file=info['filename'])
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + '.part')
    if target.is_file():
        progress('Using local archive', force=True, path=str(target))
        validate_archive(info, target)
        return
    if partial.is_file() and info.get('size') and partial.stat().st_size >= info['size']:
        try:
            validate_archive(info, partial)
        except ValueError:
            partial.unlink()
        else:
            os.replace(partial, target)
            return
    for attempt in range(5):
        progress('Resolving official download location', force=True, attempt=attempt + 1, path=str(target))
        source, headers = package_url(session, info)
        if not source.startswith(('https://', 'http://')):
            with Path(source).expanduser().open('rb') as src, partial.open('wb') as out:
                completed, total = 0, Path(source).expanduser().stat().st_size
                progress('Copying local archive', force=True, completed=0, total=total, unit='bytes')
                while block := src.read(1024 * 1024):
                    out.write(block)
                    completed += len(block)
                    progress('Copying local archive', completed=completed, total=total, unit='bytes')
                out.flush(); os.fsync(out.fileno())
                progress('Archive copy complete', force=True, completed=completed, total=total, unit='bytes')
            break
        # Only known immutable, checksum-pinned packages can resume byte offsets.
        offset = partial.stat().st_size if partial.exists() and info.get('sha256') else 0
        if offset:
            headers = {**headers, 'Range': f'bytes={offset}-'}
        try:
            progress('Connecting to download service', force=True, resumed_bytes=offset)
            with request(session, source, stream=True, headers=headers, progress=progress) as response:
                if response.status_code == 206:
                    if not offset or not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                        raise ValueError('Download resume response does not match the file offset')
                else:
                    offset = 0
                length = response.headers.get('Content-Length', '')
                total = info.get('size')
                if total is None and length.isdigit():
                    total = offset + int(length)
                completed = offset
                progress('Downloading archive', force=True, completed=completed, total=total, unit='bytes')
                with partial.open('ab' if offset else 'wb') as out:
                    for block in response.iter_content(1024 * 1024):
                        out.write(block)
                        completed += len(block)
                        progress('Downloading archive', completed=completed, total=total, unit='bytes')
                    out.flush(); os.fsync(out.fileno())
                progress('Download complete', force=True, completed=completed, total=total, unit='bytes')
            break
        except requests.RequestException as exc:
            if attempt == 4:
                raise ValueError(f'Download interrupted ({type(exc).__name__}); partial file retained for retry') from None
            progress('Download interrupted; retrying', force=True, attempt=attempt + 1, error_type=type(exc).__name__)
    validate_archive(info, partial)
    os.replace(partial, target)


def validate_archive(info, path):
    progress = Progress(info['dataset'], 'download', file=info['filename'])
    progress('Verifying archive size and checksum', force=True, path=str(path))
    if info.get('size') is not None and path.stat().st_size != info['size']:
        raise ValueError('Archive size does not match')
    if info.get('sha256') and digest(path, progress=progress) != info['sha256']:
        raise ValueError('Archive SHA256 does not match')
    size = path.stat().st_size
    progress('Archive verified', force=True, completed=size, total=size, unit='bytes')


def model_id(ds, package, name):
    parts = PurePosixPath(name).parts
    if ds == 'toys4k' and len(parts) == 4 and parts[0] == 'toys4k_blend_files' and parts[3] == parts[2] + '.blend':
        return parts[2]
    if ds == 'omniobject3d':
        if len(parts) == 4 and parts[0] == package:
            parts = parts[1:]
        if len(parts) == 3 and parts[1:] == ('Scan', 'Scan.obj') and re.fullmatch(re.escape(package) + r'_[0-9]+', parts[0]):
            return parts[0]
    if ds == 'shapenet' and len(parts) == 4 and parts[0] == package and parts[2:] == ('models', 'model_normalized.obj'):
        return parts[0] + '_' + parts[1]


def unpack(con, root, info, archive):
    ds, name = info['dataset'], info['package']
    progress = Progress(ds, 'extract', package=name, archive=info['filename'])
    progress('Opening archive for extraction', force=True, path=str(archive))
    prefix = f'files/{ds}/{name}/'
    with con:
        con.execute("UPDATE packages SET status='extracting' WHERE dataset=? AND name=?", (ds, name))
        con.execute('DELETE FROM checked_assets WHERE source_dataset=?', (ds,))
        con.execute('DELETE FROM checks WHERE uuid IN (SELECT uuid FROM assets WHERE source_dataset=?)', (ds,))
        set_state(con, 'check:' + ds, {'complete': False})
        con.execute('DELETE FROM models WHERE dataset=? AND path LIKE ?', (ds, prefix + '%'))
        con.execute('DELETE FROM files WHERE dataset=? AND package=?', (ds, name))
    seen = set()
    total_files = None
    extracted_bytes = 0
    def save(member, stream):
        nonlocal extracted_bytes
        relative = prefix + member
        if member in seen or '..' in PurePosixPath(member).parts:
            raise ValueError('Archive contains duplicate paths or path traversal')
        seen.add(member)
        path = contained(root, relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + '.extract.tmp')
        progress('Extracting file', force=len(seen) == 1, file=member, completed=len(seen) - 1, total=total_files,
                 unit='files', extracted_bytes=extracted_bytes)
        with temporary.open('wb') as out:
            while block := stream.read(1024 * 1024):
                out.write(block)
                extracted_bytes += len(block)
                progress('Extracting file', file=member, completed=len(seen) - 1, total=total_files,
                         unit='files', extracted_bytes=extracted_bytes)
            out.flush(); os.fsync(out.fileno())
        os.replace(temporary, path)
        progress('Hashing extracted file', file=member)
        con.execute('INSERT INTO files VALUES (?,?,?,?,?)',
                    (relative, ds, name, path.stat().st_size, digest(path, progress=progress)))
        rid = model_id(ds, name, member)
        if rid:
            con.execute('INSERT INTO models VALUES (?,?,?)', (ds, rid, relative))
        if len(seen) % 256 == 0:
            con.commit()
    if info['filename'].endswith('.zip'):
        with zipfile.ZipFile(archive) as source:
            total_files = sum(not member.is_dir() for member in source.infolist())
            for member in source.infolist():
                if member.filename.startswith('/') or '\\' in member.filename or (member.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError('Archive contains an unsafe path or symbolic link')
                if not member.is_dir():
                    with source.open(member) as stream:
                        save(member.filename, stream)
    else:
        with tarfile.open(archive, 'r|gz') as source:
            for member in source:
                if member.isdir():
                    continue
                if not member.isfile():
                    raise ValueError('Archive contains a link or special file')
                with source.extractfile(member) as stream:
                    save(member.name, stream)
    with con:
        con.execute("UPDATE packages SET status='ready' WHERE dataset=? AND name=?", (ds, name))
    progress('Extraction complete', force=True, completed=len(seen), total=total_files,
             unit='files', extracted_bytes=extracted_bytes)


def download(root, dataset):
    archives = archive_root()
    for name in DATASETS:
        (archives / name).mkdir(parents=True, exist_ok=True)
    with lock(root, write=True), lock(archives, write=True, name='.archives.lock'), \
            database(root / 'extra/extra.sqlite', write=True) as con, requests.Session() as session:
        packages = con.execute('SELECT * FROM packages WHERE dataset=? ORDER BY name', (dataset,)).fetchall()
        if not packages:
            raise ValueError('No published official archive manifest exists for this source')
        broken = [r[0] for r in con.execute("SELECT original_id FROM operation_logs WHERE dataset=? AND stage='check'", (dataset,))]
        done = skipped = failed = 0
        progress = Progress(dataset, 'download')
        for number, package in enumerate(packages, 1):
            info = json.loads(package['info'])
            progress('Processing archive', force=True, package=package['name'], file=info['filename'],
                     package_number=number, total_packages=len(packages))
            repair = any(dataset == 'toys4k' or (rid.startswith(package['name'] + '_') if dataset == 'shapenet'
                         else rid.rsplit('_', 1)[0] == package['name']) for rid in broken)
            if package['status'] == 'ready' and not repair:
                cleanup_archive(info)
                skipped += 1
                progress('Skipping previously extracted archive', force=True, package=package['name'])
                continue
            target = archive_path(info)
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.is_file():
                    target.with_name(target.name + '.managed').touch()
                fetch(session, info, target)
                unpack(con, root / 'extra', info, target)
            except (OSError, ValueError, requests.RequestException, zipfile.BadZipFile, tarfile.TarError, EOFError) as exc:
                con.rollback()
                with con:
                    error(con, dataset, package['name'], 'download', failure_reason(exc))
                failed += 1
            else:
                with con:
                    error(con, dataset, package['name'], 'download', None)
                cleanup_archive(info)
                done += 1
            print(terminal_line(dataset, 'download', 'Archive processed', package=package['name'],
                                completed=done, failed=failed), flush=True)
        return {'completed': done, 'skipped': skipped, 'failed': failed}


def verify_asset(con, root, row, relative, cache=None):
    if not relative:
        raise ValueError('No model file matches the original asset identity')
    parts = PurePosixPath(relative).parts
    if len(parts) < 4 or parts[:2] != ('files', row['source_dataset']) or model_id(parts[1], parts[2], '/'.join(parts[3:])) != row['original_id']:
        raise ValueError('Model path does not match the original asset identity')
    model = contained(root, relative)
    dependencies = [model]
    if model.suffix == '.blend':
        with model.open('rb') as handle:
            header = handle.read(12)
        compressed = header[:2] == b'\x1f\x8b'
        if compressed:
            with gzip.open(model, 'rb') as handle:
                header = handle.read(12)
        if not (len(header) == 12 and header.startswith(b'BLENDER') and header[7:8] in (b'_', b'-') and header[8:9] in (b'v', b'V') and header[9:].isdigit()):
            raise ValueError('Invalid Blender file')
        with (gzip.open if compressed else open)(model, 'rb') as handle:
            handle.read(12)
            fmt = ('<' if header[8:9] == b'v' else '>') + ('4sIIII' if header[7:8] == b'_' else '4sIQII')
            dna = False
            while True:
                block = handle.read(struct.calcsize(fmt))
                if len(block) != struct.calcsize(fmt):
                    raise ValueError('Blender block header is truncated')
                code, size, *_ = struct.unpack(fmt, block)
                if code == b'ENDB':
                    if not dna or size:
                        raise ValueError('Blender DNA is missing or the end block is invalid')
                    break
                first = handle.read(min(size, 4))
                if len(first) != min(size, 4):
                    raise ValueError('Blender data block is truncated')
                if code == b'DNA1':
                    dna = first == b'SDNA'
                remaining = size - len(first)
                while remaining:
                    data = handle.read(min(remaining, 1024 * 1024))
                    if not data:
                        raise ValueError('Blender data block is truncated')
                    remaining -= len(data)
    else:
        vertices = faces = False
        materials = []
        with model.open(encoding='utf-8', errors='replace') as lines:
            for line in lines:
                vertices |= line.lstrip().startswith('v ')
                faces |= line.lstrip().startswith('f ')
                if line.lstrip().startswith('mtllib '):
                    for name in shlex.split(line.strip()[7:], comments=True):
                        materials.append(contained(root, (model.parent / name).relative_to(root).as_posix()))
        if not vertices or not faces:
            raise ValueError('OBJ is missing vertices or faces')
        for material in materials:
            dependencies.append(material)
            with material.open(encoding='utf-8', errors='replace') as lines:
                for line in lines:
                    tokens = shlex.split(line, comments=True)
                    if tokens and (tokens[0].lower().startswith('map_') or tokens[0].lower() in ('bump', 'disp', 'decal', 'norm', 'refl')):
                        # MTL options precede the final texture filename; quoted spaces are supported.
                        dependencies.append(contained(root, (material.parent / tokens[-1]).relative_to(root).as_posix()))
    for path in dict.fromkeys(dependencies):
        relative_file = path.relative_to(root).as_posix()
        known = con.execute('SELECT size,sha256 FROM files WHERE path=?', (relative_file,)).fetchone()
        before = path.stat()
        fingerprint = (before.st_size, before.st_mtime_ns, before.st_ctime_ns, before.st_ino)
        if not known or not before.st_size or before.st_size != known['size']:
            raise ValueError('Model or dependency is missing, empty, or has an unexpected size: ' + relative_file)
        if cache is None or cache.get(relative_file) != fingerprint:
            if digest(path) != known['sha256']:
                raise ValueError('Model or dependency checksum verification failed: ' + relative_file)
            after = path.stat()
            if (after.st_size, after.st_mtime_ns, after.st_ctime_ns, after.st_ino) != fingerprint:
                raise ValueError('File changed during verification: ' + relative_file)
            if cache is not None:
                cache[relative_file] = fingerprint
    official = con.execute('SELECT member,sha256 FROM expected WHERE dataset=? AND original_id=?',
                           (row['source_dataset'], row['original_id'])).fetchone()
    if official and official['sha256']:
        known = con.execute('SELECT sha256 FROM files WHERE path=?', (relative,)).fetchone()[0]
        if '/'.join(parts[3:]) != official['member'] or known != official['sha256']:
            raise ValueError('Model does not match the official ID/checksum mapping')
    return model


def check(root, dataset):
    with lock(root, write=True), database(root / 'extra/extra.sqlite', write=True) as con:
        if not con.execute('SELECT 1 FROM packages WHERE dataset=?', (dataset,)).fetchone() or con.execute("SELECT 1 FROM packages WHERE dataset=? AND status!='ready'", (dataset,)).fetchone():
            raise ValueError('All official archives for this source must be downloaded and extracted')
        with con:
            set_state(con, 'check:' + dataset, {'complete': False})
        passed = failed = 0
        total = con.execute('SELECT count(*) FROM assets WHERE source_dataset=?', (dataset,)).fetchone()[0]
        progress = Progress(dataset, 'check')
        progress('Checking local asset files', force=True, completed=0, total=total, unit='assets')
        after = ''
        cache = {}
        while batch := con.execute('''SELECT a.*,m.path FROM assets a LEFT JOIN models m
            ON m.dataset=a.source_dataset AND m.original_id=a.original_id
            WHERE a.source_dataset=? AND a.uuid>? ORDER BY a.uuid LIMIT 256''', (dataset, after)).fetchall():
            with con:
                for item in batch:
                    row = record(item)
                    reason = None
                    progress('Checking asset', force=passed + failed == 0, original_id=row['original_id'], file=item['path'],
                             completed=passed + failed, total=total, unit='assets')
                    try:
                        verify_asset(con, root / 'extra', row, item['path'], cache)
                    except (OSError, ValueError, EOFError) as exc:
                        reason = failure_reason(exc)
                    con.execute('INSERT OR REPLACE INTO checks VALUES (?,?,?,?)',
                                (row['uuid'], dumps(row), 'failed' if reason else 'passed', item['path']))
                    con.execute('DELETE FROM checked_assets WHERE uuid=?', (row['uuid'],))
                    if reason is None:
                        con.execute('INSERT INTO checked_assets SELECT * FROM assets WHERE uuid=?', (row['uuid'],))
                        passed += 1
                    else:
                        failed += 1
                    error(con, dataset, row['original_id'], 'check', reason)
            after = batch[-1]['uuid']
            progress('Asset check progress', original_id=row['original_id'], file=item['path'],
                     completed=passed + failed, total=total, unit='assets', passed=passed, failed=failed)
        with con:
            set_state(con, 'check:' + dataset, {'complete': True, 'passed': passed, 'failed': failed})
        archives = archive_root()
        if archives.is_dir():
            with lock(archives, write=True, name='.archives.lock'):
                for package in con.execute('SELECT info FROM packages WHERE dataset=?', (dataset,)):
                    cleanup_archive(json.loads(package[0]))
        progress('Asset checks complete', force=True, completed=passed + failed, total=total,
                 unit='assets', passed=passed, failed=failed)
        return {'passed': passed, 'failed': failed}


def approved(con, dataset):
    state = get_state(con, 'check:' + dataset)
    if not state or not state.get('complete'):
        raise ValueError('Complete the check stage first')
    after = ''
    while batch := con.execute('''SELECT a.*,c.snapshot,c.status,c.path FROM assets a LEFT JOIN checks c ON c.uuid=a.uuid
        WHERE source_dataset=? AND a.uuid>? ORDER BY a.uuid LIMIT 256''', (dataset, after)).fetchall():
        for item in batch:
            row = record(item)
            if item['status'] not in ('passed', 'failed') or item['snapshot'] != dumps(row):
                raise ValueError('Original record has changed; rerun check')
            checked = con.execute('SELECT * FROM checked_assets WHERE uuid=?', (row['uuid'],)).fetchone()
            if item['status'] == 'passed':
                if checked is None or record(checked) != row:
                    raise ValueError('Checked asset table is inconsistent; rerun check')
                yield row, item['path']
            elif checked is not None:
                raise ValueError('Failed asset is incorrectly present in the checked asset table')
        after = batch[-1]['uuid']


def main(argv=None):
    import argparse
    import signal
    import sqlite3

    if signal.getsignal(signal.SIGINT) == signal.SIG_IGN:
        signal.signal(signal.SIGINT, signal.default_int_handler)
    parser = argparse.ArgumentParser(description='Download, verify, or merge published extra assets')
    parser.add_argument('--data', type=Path, default=Path(__file__).resolve().parents[1] / 'artifacts')
    parser.add_argument('stage', choices=('download', 'check', 'merge'))
    parser.add_argument('dataset', choices=DATASETS)
    args = parser.parse_args(argv)
    progress = Progress(args.dataset, args.stage)
    try:
        progress('Starting stage', force=True, data=str(args.data.resolve()))
        if args.stage == 'merge':
            from merge import merge
            result = merge(args.data.resolve(), args.dataset)
        else:
            result = globals()[args.stage](args.data.resolve(), args.dataset)
        print(terminal_line(args.dataset, args.stage, 'Summary', **result), flush=True)
        progress('Stage complete', force=True, **result)
        return 2 if result.get('conflicts') or (args.stage == 'download' and result.get('failed')) else 0
    except KeyboardInterrupt:
        print('Interrupted; rerun the same command to resume', file=sys.stderr)
        return 130
    except (OSError, ValueError, ImportError, sqlite3.Error) as exc:
        print(terminal_line(args.dataset, args.stage, 'ERROR: ' + failure_reason(exc)),
              file=sys.stderr, flush=True)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
