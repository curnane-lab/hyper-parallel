# Qwen3.8 VLM 数据管线使用指南

覆盖从原始 parquet（任意 schema、任意分片方式）到训练 batch 的完整路径。

## 1. 架构

```
原始 parquet（任意 schema、单文件或多分片、嵌套目录）
    │  ① schema adapter（hyper_parallel/data/vlm/schema_adapters.py）
    ▼
canonical 记录: {"messages": [...], "images": [data URI | path]}
    │  ② VLMChatTransform（build_data_transform.py，processor 编码 + labels 掩码）
    ▼
模型样本: input_ids / labels / pixel_values / image_grid_thw / mm_token_type_ids
    │  ③ MixVLMDataset（dataset.py，多数据集按 weight 加权混合）
    ▼
训练 batch（VLMCollator 拼接）
```

## 2. 三种数据引用方式

YAML 中 `data_path` 支持三种形式，可按数据集实际情况选择：

| 形式 | 写法 | 适用场景 |
|------|------|----------|
| 单文件 | `/data/ds/train.parquet` | 数据集只有一个 parquet |
| glob 分片 | `/data/ds/train-*.parquet` | HuggingFace 风格的多 shard |
| 目录递归 | `/data/ds` + `recursive: true` | 嵌套深度不一的目录树 |

三种形式可在同一个 mix 里混用。

## 3. Schema Adapter

异构 parquet 的列名各不相同时，用 adapter 把原始记录转成 canonical 格式。内置适配器：

| adapter | 匹配的列 | 逻辑 |
|---------|----------|------|
| `qwen_vl` | `messages` + `images` | 透传（HW_Chinese 格式） |
| `fintabnet_otsl` | `image` + `otsl` + `cells` | 从 cells 重建 OTSL 目标文本 |
| `llava` | `conversations` + `image` | `from` → `role` 映射 |
| `image_text_pair` | `image` + `text`/`caption` | 拼接 user/assistant 对话 |
| `auto` | — | 按列名自动检测（默认值） |

### 注册新 adapter

```python
from hyper_parallel.data.vlm.schema_adapters import register_adapter

@register_adapter("my_dataset")
def _adapter_my_dataset(record):
    return {
        "messages": [
            {"role": "user", "content": "<image>\n描述图片"},
            {"role": "assistant", "content": record["answer"]},
        ],
        "images": [record["image"]],   # bytes / BinaryScalar / data URI 均可
    }
```

注册后在 YAML 的 `schema_adapter: my_dataset` 中引用。如果某个数据集的 schema 属于已有适配器，无需写代码，直接在 manifest 或 YAML 中指定名字即可。

## 4. 单数据集训练配置

```yaml
dataset:
  data_transform:
    _target_: hyper_parallel.data.vlm.build_data_transform.build_vlm_data_transform
    max_seq_len: 1024
    min_pixels: 4096
    max_pixels: 262144
  _target_: hyper_parallel.data.vlm.dataset.build_vlm_dataset
  data_path: /data/ds/train-*.parquet        # 单文件 / glob / 目录均可以
  data_config:
    source_type: parquet
    schema_adapter: auto                     # 或指定名字
    lazy_parquet: true                       # 只读 row-group 元数据，秒级构建索引
    recursive: false                         # 目录引用时是否递归
    eager_filter: false                      # false: 不在启动时全量跑 processor 过滤
```

`lazy_parquet=true` 时，构造只读 row-group 元数据（88441 行 × 13 shard 约 0.01 秒），训练时按逻辑索引定位 `(file, row_group, local_idx)` 后只读对应 row group，超大 parquet 不需要全部载入内存。

`eager_filter=false` 跳过启动阶段的全量 processor 预扫描；训练中遇到解码失败或无有效 labels 的样本会自动跳到下一条。

## 5. 多数据集混合

### 方式 A：YAML 内联声明

```yaml
dataset:
  data_config:
    sources:
      - name: hw_chinese_ocr
        data_path: /data/HW_Chinese/hw_chinese_ocr_train.parquet
        data_config: {source_type: parquet, schema_adapter: qwen_vl, lazy_parquet: true}
        weight: 1.0
      - name: fintabnet_otsl
        data_path: /data/FinTabNet_OTSL/train-*.parquet
        data_config: {source_type: parquet, schema_adapter: fintabnet_otsl, lazy_parquet: true}
        weight: 2.0
      - name: nested_tables
        data_path: /data/NestedTables          # 目录递归
        data_config: {source_type: parquet, schema_adapter: auto, lazy_parquet: true, recursive: true}
        weight: 0.5
    eager_filter: false
```

### 方式 B：manifest 文件（推荐用于 10+ 数据集）

第一步，训练前在 CPU 节点执行一次扫描：

```bash
python -m hyper_parallel.data.vlm.build_manifest \
    --data-root /mnt/sfs-turbo/datasets \
    --output /mnt/sfs-turbo/datasets/dataset_manifest.json
```

`build_manifest` 会：

1. 递归扫描根目录下所有 `**/*.parquet`（跳过 `images/`、`kernel_meta/`、`.cache`、`parts/` 等非数据目录；数据集文件夹内若同时存在原始 parquet 和 repack 后的 `vlm_parquet/`，只收 `vlm_parquet/`）；
2. 按顶层文件夹归组为独立数据集；
3. 对每个数据集按多数投票检测 schema adapter、统计行数；
4. 输出结构化 manifest，文件路径为相对于 `--data-root` 的 POSIX 相对路径。

manifest 是可移植的：加载时相对路径优先按 `data_root` 字段解析，该目录不存在时退回 manifest 所在目录。把 manifest 放在数据集根目录（默认输出位置），整套目录搬到任何机器都能直接用；`run_vlm_train.sh` 的 WEIGHTS 权重覆盖流程也兼容。omnidoc_vlm（18 个文档数据集、448 万行）的现成 manifest 见 `examples/qwen3_8/omnidoc_vlm_manifest.example.json`，复制为数据集根目录下的 `dataset_manifest.json` 即可。

第二步，YAML 只引用 manifest：

```yaml
dataset:
  data_config:
    manifest: /mnt/sfs-turbo/datasets/dataset_manifest.json
    eager_filter: false
```

manifest 示例：

```json
{
  "version": 1,
  "data_root": "/mnt/sfs-turbo/datasets",
  "datasets": [
    {
      "name": "hw_chinese_ocr",
      "schema_adapter": "qwen_vl",
      "weight": 1.0,
      "total_rows": 239650,
      "num_files": 1,
      "files": [{"path": "hw_chinese_ocr/vlm_parquet/hw_chinese_ocr_train.parquet", "rows": 239650, "schema_adapter": "qwen_vl"}]
    },
    {
      "name": "fintabnet_otsl",
      "schema_adapter": "fintabnet_otsl",
      "weight": 2.0,
      "total_rows": 88441,
      "num_files": 13,
      "files": [{"path": "fintabnet_otsl/vlm_parquet/train-00000-of-00013-....parquet", "rows": 6804, "schema_adapter": "fintabnet_otsl"}]
    }
  ]
}
```

调整采样权重只需编辑 manifest 的 `weight` 字段，不需要改训练代码或 YAML。`schema_adapter` 检测错误时同样在 manifest 中手动修正。

## 6. 加权混合采样

`MixVLMDataset` 按 weight 比例把逻辑索引交错分配给各数据集。交错以「run」为粒度——一个 run 对应源数据集的一个 parquet row group，保证相邻逻辑索引落在同一个已解码的 row group 内，配合训练侧的块级 shuffle（见 §7）不会反复解码分片。

## 7. 块级 shuffle 与 IO 局部性

`dataloader_type: single`（VLM 默认）下，DataLoader 会读取数据集的 `shuffle_blocks`（每个 parquet row group 一个块），自动生成「块间 shuffle + 块内 shuffle」的逻辑→物理索引映射（`block_shuffle: true` 默认开启）。效果：

- 每个 row group 只解码一次、整块消费完毕再切下一块，大分片（如单 row group 数百 MB 的分片）不会因随机访问被反复解码；
- 解码缓存是有上限的 LRU（`data_config.parquet_cache_groups`，默认 4 个组），内存占用与数据集总大小无关；
- 同一映射每个 epoch 复用（确定性、支持断点续训）；设为 `block_shuffle: false` 可退回纯顺序读取。

其他新增开关：

| 配置 | 默认 | 说明 |
|---|---|---|
| `data_config.parquet_cache_groups` | 4 | 每个数据源常驻的解码 row group 上限 |
| `data_config.skip_errors` | true | 单条样本解码/变换失败时跳到下一条，不中断训练 |
| `data_transform.drop_image_truncated_sample` | true | 图像 vision token 超过 `max_seq_len` 时丢弃整条样本，而不是在无图情况下训练文本答案 |
| `data_transform.train_all_turns` | true | 多轮对话中每一轮 assistant 回答都参与训练（旧行为只训练最后一轮）；多轮 QA 数据集（如 FineVision 系列）的监督信号不再浪费 |

## 8. 常见问题

**Q: 启动时卡在全量数据处理？**
`eager_filter` 设为 `false`。

**Q: 单文件和多分片数据集混合？**
manifest 和内联 sources 都天然支持——每个数据集独立建索引，文件数不需要一致。

**Q: 新数据集 schema 不在已有 adapter 里？**
在 `schema_adapters.py` 加一个 `@register_adapter` 函数（约 10 行），或者把样本列名给维护者加一个。

**Q: parquet 太大内存不够？**
`lazy_parquet: true`（默认）只保留 row-group 索引和有上限的 LRU 缓存，按需读取。

**Q: 目录里有非 parquet 文件（图片、JSON 等）？**
`file_pattern: "*.parquet"` 过滤；图片目录（如 `images/`）、`.cache`、`parts/` 在扫描时自动跳过。

**Q: 数据集目录里既有原始 parquet 又有 repack 后的 `vlm_parquet/`？**
`build_manifest` 只收 `vlm_parquet/` 下的分片，自动忽略同目录下的原始 parquet，避免 schema 错配或重复训练。
