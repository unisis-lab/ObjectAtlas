# ObjectAtlas

[English](README.md) | 简体中文

跨数据源的 3D 物体资产索引与文本检索库。

通过自然语言查找独立的 3D 物体，获取模型来源、原始 ID、描述和文件地址。本项目提供已构建的 SQLite 资产索引与 FAISS 向量库，以及本地检索示例。

资产描述来自 Cap3D、MARVEL-40M+ 和 TRELLIS-500K，覆盖 Objaverse、Objaverse-XL、ABO、GSO、HSSD、ShapeNet 等原始资产来源。同一资产的多来源描述保存在同一条记录中，检索结果按资产返回。

| 数据 | 资产数 | 描述向量数 |
|---|---:|---:|
| 主库 | 1,842,649 | 13,792,908 |
| extra 库 | 10,840 | 102,896 |

主库用于直接检索有独立在线获取地址的资产。extra 库收录需要先下载官方压缩包的资产，安装并校验本地文件后，可加入主库一起检索。

## 准备数据

请先前往 [ObjectAtlas 的 Hugging Face 数据仓库](https://huggingface.co/datasets/unisis-lab/ObjectAtlas)，下载预构建的资产索引、向量库和 extra 数据文件。

代码和数据文件分开存放。将下载的 `data/` 与 `extra/` 目录放入本项目的 `artifacts/`，保持下面的结构：

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

这五个文件合计约 **54.49 GiB**。数据包包含资产信息和向量；3D 模型文件通过检索结果中的地址获取。extra 的压缩包默认暂存于 `extra/archives/`，解包后的模型保存在 `artifacts/extra/files/` 下。

数据可以放在其他磁盘，运行时通过 `--data /path/to/artifacts` 指定。该路径应指向同时包含 `data/` 和 `extra/` 的目录。

## 安装

需要 Python 3.11 或更新版本。以下命令均在本 README 所在目录运行。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

从 Hugging Face 下载资产索引和向量文件到 `artifacts/`：

```bash
hf download Paplet/ObjectAtlas --repo-type dataset --local-dir artifacts
```

## 开始检索

输入一段物体描述：

```bash
python scripts/search.py --text "a wooden dining chair" --top-k 10
```

结果以 JSON 数组输出，每项包含：

- `asset`：资产的 UUID、原始来源与 ID、各来源描述、文件获取地址。
- `score`：本次查询的余弦相似度分数。
- `caption`：该资产中与查询最匹配的描述及其来源。

每个资产只返回一次。同一资产有多个描述时，采用其中最高的匹配分数。

首次文本查询可能下载文本编码模型。模型缓存后，可使用 `--local-files-only` 离线查询；没有缓存时该选项会报错。

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--text` | 必填 | 查询的物体描述 |
| `--data` | 项目下的 `artifacts/` | 数据目录，包含 `data/` 与 `extra/` |
| `--top-k` | `10` | 最多返回的资产数 |
| `--candidates` | `500` | 召回的描述候选数 |
| `--nprobe` | `64` | 搜索的索引分区数 |
| `--device` | 自动选择 | 文本编码设备，例如 `cpu` 或 `cuda` |
| `--local-files-only` | 关闭 | 仅使用本机缓存的文本模型 |

增加 `--candidates` 和 `--nprobe` 通常会扩大候选范围，也会增加查询时间；候选不足时，返回的资产数可能小于 `--top-k`。

## 安装 extra 资产

extra 支持 Toys4k、OmniObject3D，以及主库未覆盖的三个 ShapeNet 分类。可以只安装其中一个来源。

先复制配置模板：

```bash
cp extra/downloads.env.example extra/downloads.env
```

压缩包只需配置一个外层目录 `ARCHIVE_DIR`，默认位置为 `extra/archives/`。其下按来源分类，使用官方压缩包文件名：

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

相对路径以项目的 `extra/` 目录为基准。程序优先读取目录中已有的官方包，缺少时再使用官方下载源；Toys4k 的下载链接通过 `TOYS4K_ARCHIVE_URL` 提供，ShapeNet 默认使用模板中的官方仓库，OmniObject3D 使用库内的 OpenXLab 下载清单。

随后运行完整流程：

```bash
python extra/run_extra.py
```

脚本参数：

- `DATASET`：可选来源名称 `toys4k`、`omniobject3d`、`shapenet`，支持同时传入多个；不传时处理三个来源。
- `--stage`：执行阶段，默认 `all`；`download` 下载并解包，`check` 校验本地资产，`merge` 将通过校验的资产和向量归并到主库。
- `--env`：下载配置文件路径，默认 `extra/downloads.env`。
- `--data`：数据目录路径，默认项目下的 `artifacts/`。
- `--help`：查看命令帮助。

流程依次完成下载、解包、校验和归并。通过校验的资产与预编码向量一起加入主库，文件地址指向本机实际的 `file://` 路径。extra 的原始资产记录保留，便于以后重新检查。

OmniObject3D 需要安装清单中的 216 个分类包。ShapeNet 需要手机 `02992529`、键盘 `03085013`、遥控器 `04074963` 三个分类包。每个所选来源的包必须完整解包，之后只归并通过资产校验的记录。

也可使用 `bash extra/run_extra.sh` 作为快捷入口，参数与 Python 脚本一致。可通过 `--stage` 分阶段执行，先检查结果再决定何时加入主库。

Ctrl+C 或报错后，修复问题并重跑同一命令即可。重复归并会复用已有资产和向量 ID。移动整个数据目录后，重新执行 `check` 和 `merge` 可更新已归并资产的本地地址。

## 查看资产与错误

主库和 extra 库都是 SQLite 文件，可以用 SQLite 客户端打开。资产保存在 `assets` 表中：

| 字段 | 内容 |
|---|---|
| `uuid` | 资产在本库中的稳定标识 |
| `source_dataset` | 原始资产来源，例如 `objaverse` 或 `abo` |
| `original_id` | 原始模型 ID，保留前导零和大小写 |
| `captions` | 按 `cap3d`、`marvel_40m_plus`、`trellis_500k` 分组的描述，每项保留 `text` 和原始 `field` |
| `download_urls` | 在线获取地址，或 extra 归并后的本地文件地址 |

## 数据来源与许可

描述来源为 [Cap3D](https://huggingface.co/datasets/tiange/Cap3D)、[MARVEL-40M+](https://huggingface.co/datasets/sankalpsinha77/MARVEL-40M)、[TRELLIS-500K](https://huggingface.co/datasets/JeffreyXiang/TRELLIS-500K)。

| 原始资产来源 | 官方入口 |
|---|---|
| Objaverse / Objaverse-XL | [Objaverse](https://objaverse.allenai.org/)、[Objaverse-XL metadata](https://huggingface.co/datasets/allenai/objaverse-xl) |
| ABO | [Amazon Berkeley Objects](https://amazon-berkeley-objects.s3.amazonaws.com/index.html) |
| GSO | [Google Scanned Objects](https://research.google/blog/scanned-objects-by-google-research-a-dataset-of-3d-scanned-common-household-items/) |
| HSSD | [HSSD models](https://huggingface.co/datasets/hssd/hssd-models) |
| ShapeNet | [ShapeNetCore v2](https://huggingface.co/datasets/ShapeNet/ShapeNetCore)、[ShapeNetCore GLB](https://huggingface.co/datasets/ShapeNet/shapenetcore-glb) |
| Toys4k | [官方项目](https://github.com/rehg-lab/lowshot-shapebias/tree/main/toys4k) |
| OmniObject3D | [官方项目](https://github.com/omniobject3d/OmniObject3D)、[OpenXLab](https://openxlab.org.cn/datasets/omniobject3d/OmniObject3D-New) |

运行代码采用 MIT 许可。数据、描述、模型和向量分别遵循原发布者的许可与署名要求：Cap3D 标注为 ODC-By，MARVEL 为 CC BY-NC-SA 4.0，TRELLIS metadata 标注 MIT；原模型仍受各发布者和模型作者条款约束。

ShapeNet 需要接受官方条款并获得访问权限；Toys4k 的官方下载渠道由使用者自行取得。使用或再分发索引与向量前，请核对相应来源的许可，尤其是非商业限制与署名要求。
