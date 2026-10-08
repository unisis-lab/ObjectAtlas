"""Append checked extra metadata and pre-encoded vectors in one recoverable commit."""
import os
from pathlib import Path
import sys
import uuid
from urllib.parse import unquote, urlsplit

import faiss
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'extra'))

from common import (WINDOWS, Progress, caption_key, contained, database, dumps, error, get_state, lock,
                 record, set_state, sync_directory, union_captions, values)


def read_index(path, *, mmap=False):
    # Windows FAISS builds do not support mmap-backed inverted lists.
    flags = faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY if mmap and not WINDOWS else 0
    return faiss.read_index(str(path), flags)


def state(con, root, *, load_index=True, mmap=False):
    config = get_state(con, 'vectors')
    if not config or config.get('encoded_count') != config.get('caption_count'):
        raise ValueError('Complete encoded vector files are missing')
    count = con.execute('SELECT count(*) FROM captions').fetchone()[0]
    ends = con.execute('SELECT min(vector_id),max(vector_id) FROM captions').fetchone()
    if count != config['caption_count'] or (count and tuple(ends) != (0, count - 1)):
        raise ValueError('SQLite caption IDs and vector counts do not match')
    offset = 0
    for segment in con.execute('SELECT * FROM vector_segments ORDER BY first_id'):
        path = contained(root, segment['path'])
        if segment['first_id'] != offset or path.stat().st_size != segment['count'] * config['dimension'] * 4:
            raise ValueError('Raw vector segment size or ID range does not match')
        offset += segment['count']
    if offset != count:
        raise ValueError('Vector segments do not cover all caption IDs')
    index = None
    if load_index:
        if not config.get('ready'):
            raise ValueError('FAISS index is incomplete')
        index = read_index(contained(root, config['index_path']), mmap=mmap)
        if index.ntotal != count or index.d != config['dimension']:
            raise ValueError('FAISS and SQLite vector mappings do not match')
    return config, index


def read_vectors(con, root, ids, dimension):
    ids = np.asarray(ids, dtype=np.int64)
    result = np.empty((len(ids), dimension), dtype=np.float32)
    assigned = np.zeros(len(ids), dtype=bool)
    for segment in con.execute('SELECT * FROM vector_segments ORDER BY first_id'):
        mask = (ids >= segment['first_id']) & (ids < segment['first_id'] + segment['count'])
        if mask.any():
            array = np.memmap(contained(root, segment['path']), dtype='<f4', mode='r',
                              shape=(segment['count'], dimension))
            result[mask] = array[ids[mask] - segment['first_id']]
            assigned[mask] = True
            del array
    if not assigned.all():
        raise ValueError('Candidate vector IDs are out of range')
    return result


def cleanup(root, con):
    config = get_state(con, 'vectors') or {}
    referenced = {r[0] for r in con.execute('SELECT path FROM vector_segments')}
    referenced.add(config.get('index_path'))
    for path in (root / 'data/vectors').glob('merge-*'):
        if path.relative_to(root).as_posix() not in referenced and path.is_file():
            path.unlink()


def merge(root, dataset, *, fault=None):
    from extra import approved, verify_asset

    progress = Progress(dataset, 'merge')
    progress('Opening metadata databases', force=True)
    with lock(root, write=True), database(root / 'data/index.sqlite', write=True) as main, \
            database(root / 'extra/extra.sqlite', write=True) as extra:
        cleanup(root, main)
        progress('Loading main FAISS index and validating vector mappings', force=True,
                 file=(get_state(main, 'vectors') or {}).get('index_path'))
        config, index = state(main, root)
        progress('Validating extra vector mappings', force=True)
        extra_config, _ = state(extra, root / 'extra', load_index=False)
        for key in ('model', 'model_revision', 'max_seq_length', 'dimension', 'normalized', 'dtype'):
            if config[key] != extra_config[key]:
                raise ValueError('Extra encoding configuration differs from the main vector library: ' + key)
        # Metadata and the active generation pointer commit in the same SQLite transaction.
        # Immutable vector segments avoid copying the complete raw vector library on merge.
        main.execute('BEGIN IMMEDIATE')
        main.execute('CREATE TEMP TABLE additions (new_id INTEGER PRIMARY KEY,extra_id INTEGER NOT NULL)')
        start = next_id = config['caption_count']
        added_assets = merged_assets = conflicts = 0
        token = uuid.uuid4().hex
        segment_name = f'data/vectors/merge-{token}.f32'
        index_name = f'data/vectors/merge-{token}.faiss'
        created = []
        try:
            cache = {}
            total = extra.execute('SELECT count(*) FROM checked_assets WHERE source_dataset=?', (dataset,)).fetchone()[0]
            processed = 0
            for row, relative in approved(extra, dataset):
                progress('Merging verified asset metadata', force=processed == 0, original_id=row['original_id'],
                         file=relative, completed=processed, total=total, unit='assets')
                processed += 1
                model = verify_asset(extra, root / 'extra', row, relative, cache)
                incoming = {**row, 'download_urls': [model.as_uri()]}
                existing = main.execute('SELECT * FROM assets WHERE source_dataset=? AND original_id=?',
                                        (row['source_dataset'], row['original_id'])).fetchone()
                if existing:
                    current = record(existing)
                    urls = current['download_urls']
                    rebased = len(urls) == 1 and urls[0].startswith('file://') and unquote(urlsplit(urls[0]).path).endswith('/extra/' + relative)
                    if set(urls) != set(incoming['download_urls']) and not rebased:
                        with extra:
                            error(extra, dataset, row['original_id'], 'merge', 'Non-caption field conflict: download_urls')
                        conflicts += 1
                        continue
                    merged = {**current, 'download_urls': incoming['download_urls'],
                              'captions': union_captions(current['captions'], incoming['captions'])}
                    main.execute('UPDATE assets SET captions=?,download_urls=? WHERE uuid=?',
                                 (dumps(merged['captions']), dumps(merged['download_urls']), current['uuid']))
                    merged_assets += 1
                else:
                    if main.execute('SELECT 1 FROM assets WHERE uuid=?', (row['uuid'],)).fetchone():
                        raise ValueError('UUID belongs to a different asset identity; merge stopped')
                    merged = incoming
                    main.execute('INSERT INTO assets VALUES (?,?,?,?,?)', values(merged))
                    added_assets += 1
                known = {caption_key(c['source'], c['field'], c['text']) for c in main.execute(
                    'SELECT source,field,text FROM captions WHERE asset_uuid=?', (merged['uuid'],))}
                encoded = {caption_key(c['source'], c['field'], c['text']): c['vector_id'] for c in extra.execute(
                    'SELECT * FROM captions WHERE asset_uuid=?', (row['uuid'],))}
                for source, descriptions in merged['captions'].items():
                    for description in descriptions:
                        key = caption_key(source, description['field'], description['text'])
                        if key in known:
                            continue
                        if key not in encoded:
                            raise ValueError('Extra is missing the precomputed vector for this caption')
                        main.execute('INSERT INTO captions VALUES (?,?,?,?,?)',
                                     (next_id, merged['uuid'], source, description['field'], description['text']))
                        main.execute('INSERT INTO additions VALUES (?,?)', (next_id, encoded[key]))
                        known.add(key)
                        next_id += 1
                with extra:
                    error(extra, dataset, row['original_id'], 'merge', None)
            progress('Asset metadata prepared', force=True, completed=processed, total=total, unit='assets',
                     new_assets=added_assets, merged_assets=merged_assets, conflicts=conflicts)
            if next_id != start:
                path = contained(root, segment_name)
                created.append(path)
                faiss.omp_set_num_threads(4)
                progress('Appending precomputed vectors', force=True, file=segment_name,
                         completed=0, total=next_id - start, unit='vectors')
                with path.open('wb') as out:
                    after = start - 1
                    while batch := main.execute('SELECT * FROM additions WHERE new_id>? ORDER BY new_id LIMIT 2048', (after,)).fetchall():
                        raw = read_vectors(extra, root / 'extra', [r['extra_id'] for r in batch], config['dimension'])
                        if not np.isfinite(raw).all() or not np.allclose(np.linalg.norm(raw, axis=1), 1, atol=1e-4):
                            raise ValueError('Extra vectors contain nonfinite values or are not normalized')
                        out.write(raw.astype('<f4', copy=False).tobytes())
                        index.add_with_ids(raw, np.asarray([r['new_id'] for r in batch], dtype=np.int64))
                        after = batch[-1]['new_id']
                        progress('Appending precomputed vectors', file=segment_name,
                                 completed=after + 1 - start, total=next_id - start, unit='vectors')
                    out.flush(); os.fsync(out.fileno())
                progress('Vector append complete', force=True, completed=next_id - start,
                         total=next_id - start, unit='vectors')
                index_path = contained(root, index_name)
                created.append(index_path)
                progress('Writing new FAISS index', force=True, file=index_name)
                faiss.write_index(index, str(index_path))
                with index_path.open('r+b') as handle:
                    os.fsync(handle.fileno())
                progress('Validating new FAISS index', force=True, file=index_name)
                if read_index(index_path, mmap=True).ntotal != next_id:
                    raise ValueError('New FAISS index vector count does not match')
                sync_directory(root / 'data/vectors')
                main.execute('INSERT INTO vector_segments VALUES (?,?,?)', (start, next_id - start, segment_name))
                set_state(main, 'vectors', {**config, 'index_path': index_name,
                          'caption_count': next_id, 'encoded_count': next_id, 'indexed_count': next_id})
            if fault:
                fault('before_commit')
            progress('Committing metadata and vector generation', force=True)
            main.commit()
            created.clear()  # Committed generation must survive a later interruption.
            if fault:
                fault('after_commit')
            cleanup(root, main)
            progress('Merge committed', force=True, new_assets=added_assets, merged_assets=merged_assets,
                     new_vectors=next_id - start, conflicts=conflicts)
            return {'new_assets': added_assets, 'merged_assets': merged_assets,
                    'new_vectors': next_id - start, 'conflicts': conflicts}
        except BaseException:
            main.rollback()
            for path in created:
                path.unlink(missing_ok=True)
            progress('Merge interrupted; transaction cleanup complete', force=True)
            raise
