"""Text retrieval example over the published SQLite metadata and FAISS index."""
import argparse
from pathlib import Path
import sqlite3
import sys

import faiss
import numpy as np

from common import database, dumps, lock, record
from merge import read_vectors, state

def query(root, *, text, top_k=10, candidates=500, nprobe=64, device=None, local_only=False):
    if min(top_k, candidates, nprobe) < 1:
        raise ValueError('Search parameters must be greater than zero')
    with lock(root), database(root / 'data/index.sqlite') as con:
        config, index = state(con, root, mmap=True)
        if not text or not text.strip():
            raise ValueError('Query text must not be empty')
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer(config['model'], revision=config['model_revision'],
                                      device=device, local_files_only=local_only)
        encoder.max_seq_length = config['max_seq_length']
        vector = np.asarray(encoder.encode([text], normalize_embeddings=True), dtype=np.float32)
        if hasattr(index, 'nprobe'):
            index.nprobe = min(nprobe, index.nlist)
        faiss.omp_set_num_threads(4)
        _, found = index.search(vector, min(candidates, index.ntotal))
        ids = np.unique(found[0][found[0] >= 0])
        if not len(ids):
            return []
        raw = read_vectors(con, root, ids, config['dimension'])
        scores = raw @ vector[0]
        results = []
        seen = set()
        for position in np.argsort(-scores):
            item = con.execute('SELECT * FROM captions WHERE vector_id=?', (int(ids[position]),)).fetchone()
            if item['asset_uuid'] in seen:
                continue
            row = con.execute('SELECT * FROM assets WHERE uuid=?', (item['asset_uuid'],)).fetchone()
            if row is None:
                raise ValueError('Asset referenced by caption does not exist')
            seen.add(item['asset_uuid'])
            results.append({'asset': record(row), 'score': float(scores[position]),
                            'caption': dict(item)})
            if len(results) == top_k:
                break
        return results



def main(argv=None):
    parser = argparse.ArgumentParser(description='Search individual 3D assets by text')
    parser.add_argument('--data', type=Path, default=Path(__file__).resolve().parents[1] / 'artifacts')
    parser.add_argument('--text', required=True)
    parser.add_argument('--top-k', type=int, default=10)
    parser.add_argument('--candidates', type=int, default=500)
    parser.add_argument('--nprobe', type=int, default=64)
    parser.add_argument('--device')
    parser.add_argument('--local-files-only', action='store_true')
    args = parser.parse_args(argv)
    try:
        print(dumps(query(args.data.resolve(), text=args.text,
                         top_k=args.top_k, candidates=args.candidates, nprobe=args.nprobe,
                         device=args.device, local_only=args.local_files_only)))
        return 0
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError, ImportError, sqlite3.Error) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
