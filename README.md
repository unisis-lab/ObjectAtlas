# ObjectAtlas

English | [简体中文](README.zh-CN.md)

A cross-dataset index and text search library for 3D object assets.

Find individual 3D objects using natural language and retrieve their sources, original IDs, descriptions, and file locations. ObjectAtlas provides a prebuilt SQLite asset index, a FAISS vector index, and a local search example.

Asset descriptions come from Cap3D, MARVEL-40M+, and TRELLIS-500K, covering original asset sources such as Objaverse, Objaverse-XL, ABO, GSO, HSSD, and ShapeNet. Descriptions from multiple annotation sources are stored in a single record for each asset, and search results are returned by asset.

| Collection | Assets | Description vectors |
|---|---:|---:|
| Main database | 1,842,649 | 13,792,908 |
| Extra database | 10,840 | 102,896 |

The main database supports direct searches for assets with individual online download locations. The extra database contains assets distributed in official archives. Once their local files have been installed and verified, these assets can be merged into the main database for searching.

## Prepare the data

Download the prebuilt asset metadata, vector index, and extra data from [ObjectAtlas on Hugging Face](https://huggingface.co/datasets/unisis-lab/ObjectAtlas).

Code and data files are stored separately. Place the downloaded `data/` and `extra/` directories inside `artifacts/` in this project, preserving this layout:

```text
artifacts/
  data/
    index.sqlite
    vectors/
      index.faiss
      vectors.f32
  extra/
    extra.sqlite
    vectors/
      vectors.f32
```

These five files total approximately **54.49 GiB**. The bundle contains asset metadata and vectors; 3D model files can be obtained through the locations returned by search. Extra archives are stored under `extra/archives/` by default, and extracted models are stored under `artifacts/extra/files/`.

You can keep the data on another drive and specify its location with `--data /path/to/artifacts`. This path must point to the directory containing both `data/` and `extra/`.

## Installation

Python 3.11 or later is required. Run the following commands from the directory containing this README.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Download the metadata and vector files from Hugging Face into `artifacts/`:

```bash
hf download Paplet/ObjectAtlas --repo-type dataset --local-dir artifacts
```

## Search

Enter a description of an object:

```bash
python scripts/search.py --text "a wooden dining chair" --top-k 10
```

Results are returned as a JSON array. Each item contains:

- `asset`: the asset's UUID, original source and ID, descriptions grouped by annotation source, and file locations.
- `score`: the cosine similarity score for this query.
- `caption`: the asset's best matching description and its source.

Each asset appears only once. When an asset has multiple descriptions, its highest matching score is used.

The first text query may download the text encoder. Once the model is cached, use `--local-files-only` for offline queries; this option fails if the model is not cached.

| Parameter | Default | Description |
|---|---|---|
| `--text` | Required | Description of the object to search for |
| `--data` | `artifacts/` in the project directory | Data directory containing `data/` and `extra/` |
| `--top-k` | `10` | Maximum number of assets to return |
| `--candidates` | `500` | Number of candidate descriptions to retrieve |
| `--nprobe` | `64` | Number of index partitions to search |
| `--device` | Automatically selected | Text encoding device, such as `cpu` or `cuda` |
| `--local-files-only` | Disabled | Use only the locally cached text model |

Increasing `--candidates` and `--nprobe` usually expands the search and takes more time. If there are too few candidates, the number of returned assets may be smaller than `--top-k`.

## Install extra assets

Extra supports Toys4k, OmniObject3D, and three ShapeNet categories missing from the main database. You can install a single source independently.

First, copy the configuration template:

```bash
cp extra/downloads.env.example extra/downloads.env
```

Only one archive root directory, `ARCHIVE_DIR`, needs to be configured. It defaults to `extra/archives/`. Organize archives into dataset subdirectories and keep the official filenames:

```text
extra/archives/
  toys4k/
    toys4k_blend_files.zip
  omniobject3d/
    apple.tar.gz
    ...
  shapenet/
    02992529.zip
    03085013.zip
    04074963.zip
```

Relative paths are resolved against the project's `extra/` directory. The program first looks for existing official archives in this directory, then uses official download sources for missing files. Provide the Toys4k download URL through `TOYS4K_ARCHIVE_URL`; ShapeNet uses the official repository specified in the template, and OmniObject3D uses the OpenXLab download manifest stored in the database.

Run the complete workflow:

```bash
python extra/run_extra.py
```

Script arguments:

- `DATASET`: optional source names: `toys4k`, `omniobject3d`, or `shapenet`. You can specify multiple sources; omitting them processes all three.
- `--stage`: workflow stage, defaulting to `all`. `download` downloads and extracts archives, `check` verifies local assets, and `merge` adds verified assets and their vectors to the main database.
- `--env`: download configuration file, defaulting to `extra/downloads.env`.
- `--data`: data directory, defaulting to `artifacts/` in the project directory.
- `--help`: display command help.

The workflow downloads, extracts, verifies, and merges assets in order. Verified assets are added to the main database along with their precomputed vectors, and their file locations use actual `file://` paths on the local machine. Original records remain in the extra database so they can be checked again later.

OmniObject3D requires the 216 category archives listed in its manifest. ShapeNet requires three category archives: cellphone (`02992529`), keyboard (`03085013`), and remote control (`04074963`). All archives for each selected source must be fully extracted; only records that pass asset verification are merged.

You can also use `bash extra/run_extra.sh` as a shortcut, with the same arguments as the Python script. Use `--stage` to run individual stages and review verification results before merging into the main database.

After Ctrl+C or an error, fix the issue and rerun the same command to resume. Repeated merges reuse existing asset and vector IDs. After moving the entire data directory, run `check` and `merge` again to update the local file locations of merged assets.

## Inspect assets and errors

Both the main and extra databases are SQLite files and can be opened with a SQLite client. Asset records are stored in the `assets` table:

| Field | Description |
|---|---|
| `uuid` | Stable asset identifier within this index |
| `source_dataset` | Original asset source, such as `objaverse` or `abo` |
| `original_id` | Original model ID, preserving leading zeros and case |
| `captions` | Descriptions grouped under `cap3d`, `marvel_40m_plus`, and `trellis_500k`; each item preserves its `text` and original `field` |
| `download_urls` | Online file locations or local file locations after an extra merge |

## Data sources and licenses

Descriptions come from [Cap3D](https://huggingface.co/datasets/tiange/Cap3D), [MARVEL-40M+](https://huggingface.co/datasets/sankalpsinha77/MARVEL-40M), and [TRELLIS-500K](https://huggingface.co/datasets/JeffreyXiang/TRELLIS-500K).

| Original asset source | Official resource |
|---|---|
| Objaverse / Objaverse-XL | [Objaverse](https://objaverse.allenai.org/), [Objaverse-XL metadata](https://huggingface.co/datasets/allenai/objaverse-xl) |
| ABO | [Amazon Berkeley Objects](https://amazon-berkeley-objects.s3.amazonaws.com/index.html) |
| GSO | [Google Scanned Objects](https://research.google/blog/scanned-objects-by-google-research-a-dataset-of-3d-scanned-common-household-items/) |
| HSSD | [HSSD models](https://huggingface.co/datasets/hssd/hssd-models) |
| ShapeNet | [ShapeNetCore v2](https://huggingface.co/datasets/ShapeNet/ShapeNetCore), [ShapeNetCore GLB](https://huggingface.co/datasets/ShapeNet/shapenetcore-glb) |
| Toys4k | [Official project](https://github.com/rehg-lab/lowshot-shapebias/tree/main/toys4k) |
| OmniObject3D | [Official project](https://github.com/omniobject3d/OmniObject3D), [OpenXLab](https://openxlab.org.cn/datasets/omniobject3d/OmniObject3D-New) |

The runtime code is licensed under MIT. Data, descriptions, models, and vectors remain subject to their publishers' licenses and attribution requirements: Cap3D annotations are listed under ODC-By, MARVEL under CC BY-NC-SA 4.0, and TRELLIS metadata under MIT. Original models remain subject to the terms of their publishers and authors.

ShapeNet requires accepting the official terms and obtaining access approval. Users must obtain the official Toys4k download source themselves. Before using or redistributing the index and vectors, review the applicable source licenses, especially noncommercial restrictions and attribution requirements.
