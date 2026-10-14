# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Build the VLM dataset from JSON, JSONL, or one-or-more parquet shards."""

import base64
import glob
import io
import json
import logging
import os
import re
from collections import OrderedDict
from fnmatch import fnmatch
from collections.abc import Callable, Iterator
from typing import Any, Optional, TypeAlias

import pyarrow.parquet as pq

from torch.utils.data import Dataset

from hyper_parallel.data.constants import IGNORE_INDEX
from hyper_parallel.data.vlm.schema_adapters import adapt_record

logger = logging.getLogger(__name__)

SampleTransform: TypeAlias = Callable[[Any], Any]

_EXCLUDE_SCAN_DIRS = {"images", "images_decoded", "kernel_meta", "__pycache__", ".ipynb_checkpoints"}


def _expand_data_paths(
        data_path: str | list[str],
        *,
        recursive: bool = False,
        file_pattern: str = "*.parquet",
) -> list[str]:
    """Expand one path (or list) into an ordered list of data files.

    Accepts explicit file paths, glob patterns, and directories.  With
    ``recursive=True`` directories are walked to arbitrary depth (MA platform
    datasets nest parquet shards unevenly), filtered by ``file_pattern``.
    Results are sorted so every rank sees the same order.
    """
    if isinstance(data_path, str):
        patterns = [data_path]
    else:
        patterns = list(data_path)
    files: list[str] = []
    for pattern in patterns:
        matched = sorted(glob.glob(os.fspath(pattern), recursive=True))
        matched_files = [
            path for path in matched
            if os.path.isfile(path) and (not recursive or fnmatch(os.path.basename(path), file_pattern))
        ]
        if matched_files:
            files.extend(matched_files)
            continue
        if os.path.isdir(pattern):
            if recursive:
                for dirpath, dirnames, filenames in os.walk(pattern):
                    dirnames[:] = sorted(d for d in dirnames if d not in _EXCLUDE_SCAN_DIRS)
                    files.extend(
                        os.path.join(dirpath, name)
                        for name in sorted(filenames)
                        if fnmatch(name, file_pattern)
                    )
            else:
                files.extend(sorted(
                    entry.path
                    for entry in os.scandir(pattern)
                    if entry.is_file() and entry.name.endswith(".parquet")
                ))
        elif os.path.isfile(pattern):
            files.append(os.fspath(pattern))
        else:
            raise FileNotFoundError(f"VLM data_path does not match any file or directory: {pattern!r}")
    if not files:
        raise FileNotFoundError(f"VLM data_path resolved to zero files: {data_path!r}")
    return files


class _ParquetShardIndex:
    """Memory-light index over one or more parquet files.

    Only row-group metadata is read at construction time; records are decoded
    lazily from the owning row group on first access. Row-group sizes vary per
    file, so a global prefix-sum maps each logical index to ``(file, group,
    local_index)``.

    Decoded row groups are kept in a bounded LRU cache (``max_cached_groups``)
    so sequentially consumed groups stay hot while the resident memory never
    exceeds a few groups, however large the underlying shard set is.
    """

    def __init__(
            self,
            files: list[str],
            columns: list[str] | None = None,
            max_cached_groups: int = 4,
    ) -> None:
        self.files = files
        self.columns = columns
        self._max_cached_groups = max(1, int(max_cached_groups))
        self._entries: list[tuple[int, int, int, int]] = []
        self._open_files: dict[int, pq.ParquetFile] = {}
        self._group_cache: OrderedDict[tuple[int, int], list[dict[str, Any]]] = OrderedDict()
        total = 0
        for file_index, path in enumerate(files):
            parquet_file = pq.ParquetFile(path)
            available = set(parquet_file.schema_arrow.names)
            wanted = [column for column in (columns or []) if column in available]
            if columns and not wanted:
                raise ValueError(
                    f"parquet file {path!r} has none of the requested columns {columns}; "
                    f"available: {sorted(available)}"
                )
            for group_index in range(parquet_file.num_row_groups):
                group_rows = parquet_file.metadata.row_group(group_index).num_rows
                self._entries.append((file_index, group_index, total, group_rows))
                total += group_rows
            self._open_files[file_index] = parquet_file
            self._file_columns = self._file_columns if hasattr(self, "_file_columns") else {}
            self._file_columns[file_index] = wanted
        self.total_rows = total

    def __len__(self) -> int:
        return self.total_rows

    @property
    def blocks(self) -> list[tuple[int, int]]:
        """Return ``(start, length)`` logical runs that share one row group.

        Consuming samples inside one block only touches a single decoded row
        group, so block-aligned access patterns stay IO-local.
        """
        return [(start, rows) for _file, _group, start, rows in self._entries]

    def _read_group(self, file_index: int, group_index: int) -> list[dict[str, Any]]:
        cache_key = (file_index, group_index)
        if cache_key in self._group_cache:
            self._group_cache.move_to_end(cache_key)
            return self._group_cache[cache_key]
        parquet_file = self._open_files[file_index]
        columns = self._file_columns[file_index]
        table = parquet_file.read_row_group(group_index, columns=columns or None)
        records = table.to_pylist()
        self._group_cache[cache_key] = records
        while len(self._group_cache) > self._max_cached_groups:
            self._group_cache.popitem(last=False)
        return records

    def __getitem__(self, index: int) -> dict[str, Any]:
        for file_index, group_index, start, rows in self._entries:
            if start <= index < start + rows:
                group = self._read_group(file_index, group_index)
                return group[index - start]
        raise IndexError(f"parquet index {index} out of range [0, {self.total_rows})")

    def iter_records(self) -> Iterator[dict[str, Any]]:
        """Yield every record in logical order without double-reading groups."""
        for file_index, group_index, _start, _rows in self._entries:
            yield from self._read_group(file_index, group_index)

    def close(self) -> None:
        self._open_files.clear()
        self._group_cache.clear()


class VLMDataset(Dataset):
    """Load multimodal conversations from JSON, JSONL, or parquet shards.

    Parquet support covers:

    * single files, glob patterns (``/data/shard-*.parquet``), directories, and
      ordered path lists;
    * lazy row-group reads so arbitrarily large datasets do not need to fit in
      memory;
    * pluggable schema adapters so heterogeneous raw schemas can share one
      training path (see ``schema_adapters.py``).
    """

    def __init__(
            self,
            data_path: str | list[str],
            source_type: str = "online",
            schema_adapter: str = "auto",
            lazy_parquet: bool = True,
            recursive: bool = False,
            file_pattern: str = "*.parquet",
            parquet_cache_groups: int = 4,
            **dataset_options: Any,
    ) -> None:
        """Load records or build the lazy parquet index."""
        self.source_type = source_type
        self.schema_adapter = schema_adapter
        self.parquet_index = None
        if source_type == "online":
            with open(data_path, "r", encoding="utf-8") as handle:
                self.records: list[dict[str, Any]] = json.load(handle)
        elif source_type == "jsonl":
            self.records = []
            with open(data_path, "r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    record["messages"] = record.get("messages") or record.pop("conversations", [])
                    if record.get("messages"):
                        self.records.append(record)
        elif source_type == "parquet":
            self.files = _expand_data_paths(data_path, recursive=recursive, file_pattern=file_pattern)
            if lazy_parquet:
                self.parquet_index = _ParquetShardIndex(
                    self.files,
                    columns=dataset_options.get("columns"),
                    max_cached_groups=parquet_cache_groups,
                )
                self.records = []
            else:
                self.parquet_index = None
                self.records = []
                index = _ParquetShardIndex(self.files)
                for record in index.iter_records():
                    self.records.append(self._finalize_parquet_record(record))
                index.close()
        else:
            raise ValueError(f"Unsupported VLM source type: {source_type!r}")
        if source_type != "parquet":
            root = os.path.dirname(os.path.abspath(data_path if isinstance(data_path, str) else data_path[0]))
            for record in self.records:
                self._resolve_record_paths(record, root)

    def _finalize_parquet_record(self, record: dict[str, Any]) -> dict[str, Any]:
        """Adapt one raw parquet record into the canonical VLM format."""
        canonical = adapt_record(record, self.schema_adapter)
        images = canonical.get("images") or []
        messages = canonical.get("messages") or []
        media: list[str] = []
        for image in images:
            if isinstance(image, dict):
                raw = image.get("bytes")
                if hasattr(raw, "as_py"):
                    raw = raw.as_py()
                if isinstance(raw, (bytes, bytearray)):
                    media.append("data:image/png;base64," + base64.b64encode(bytes(raw)).decode())
                else:
                    media.append(str(image.get("path") or image))
            elif isinstance(image, str):
                media.append(image)
            else:
                media.append(str(image))
        return {"messages": messages, "images": media}

    def _resolve_record_paths(self, record: dict[str, Any], root: str) -> None:
        """Resolve media paths in one record against the JSON directory."""
        for key in ("images", "videos"):
            if key in record:
                record[key] = [self._resolve(root, path) for path in record[key]]
        for message in record.get("messages", []):
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for item in content:
                if not isinstance(item, dict):
                    continue
                for key in ("url", "image", "video"):
                    if key in item and isinstance(item[key], str):
                        item[key] = self._resolve(root, item[key])

    @staticmethod
    def _resolve(root: str, path: str) -> str:
        """Resolve a relative media path against the JSON directory."""
        if isinstance(path, str) and not path.startswith(("http://", "https://", "/", "data:")):
            return os.path.join(root, path)
        return path

    def __len__(self) -> int:
        if self.parquet_index is not None:
            return len(self.parquet_index)
        return len(self.records)

    @property
    def shuffle_blocks(self) -> list[tuple[int, int]]:
        """Return logical ``(start, length)`` runs that share IO locality.

        The training DataLoader may shuffle at this block granularity so one
        decoded parquet row group is fully consumed before moving to the next.
        In-memory sources expose a single block covering the whole dataset.
        """
        if self.parquet_index is not None:
            return self.parquet_index.blocks
        return [(0, len(self.records))]

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self.parquet_index is not None:
            record = self.parquet_index[index]
            return self._finalize_parquet_record(record)
        record = self.records[index]
        if self.source_type == "parquet":
            return self._finalize_parquet_record(record)
        return record

    def iter_records(self) -> Iterator[dict[str, Any]]:
        """Yield canonical records in logical order (used by eager filtering)."""
        if self.parquet_index is not None:
            for record in self.parquet_index.iter_records():
                yield self._finalize_parquet_record(record)
        else:
            yield from (self[i] for i in range(len(self.records)))


class _TransformDataset(Dataset):
    """Apply one Trainer-built transform after source-specific IO.

    The trainable-label index is built lazily on the first full iteration
    (epoch 0) and cached afterwards, so startup does not run the vision
    processor across the whole corpus when ``eager_filter=False``.

    With ``eager_filter=False`` and ``skip_errors=True`` (the default),
    records that fail the transform (unreadable images, truncated media,
    schema errors) are skipped by advancing to the next source record,
    so one bad sample cannot abort a multi-day training run.
    """

    def __init__(
            self,
            source: Dataset,
            transform: Optional[SampleTransform],
            *,
            eager_filter: bool = True,
            skip_errors: bool = True,
            max_error_skip: int = 64,
    ) -> None:
        self.source = source
        self.transform = transform
        self.eager_filter = eager_filter
        self.skip_errors = skip_errors
        self.max_error_skip = max(1, int(max_error_skip))
        self.indices: Optional[list[int]] = None
        if eager_filter:
            self._build_index()

    def _build_index(self) -> None:
        """Scan the source once and cache which records train successfully."""
        self.indices = []
        for index, record in enumerate(self.source):
            try:
                sample = self.transform(record) if self.transform is not None else record
            except Exception:
                continue
            labels = sample.get("labels")
            if labels is None or self._has_trainable(labels):
                self.indices.append(index)
        if not self.indices:
            raise ValueError("VLM dataset contains no samples with trainable labels")

    @staticmethod
    def _has_trainable(labels: Any) -> bool:
        if hasattr(labels, "ne"):
            return bool(labels.ne(IGNORE_INDEX).any())
        return any(value != IGNORE_INDEX for value in labels)

    def __len__(self) -> int:
        if self.indices is not None:
            return len(self.indices)
        return len(self.source)

    @property
    def shuffle_blocks(self) -> Optional[list[tuple[int, int]]]:
        """Forward the source's IO-local blocks when no filtered index exists."""
        if self.indices is not None:
            return None
        return getattr(self.source, "shuffle_blocks", None)

    def __getitem__(self, index: int) -> Any:
        if self.indices is not None:
            record = self.source[self.indices[index]]
            return self.transform(record) if self.transform is not None else record
        if not self.skip_errors:
            record = self.source[index]
            return self.transform(record) if self.transform is not None else record
        source_length = len(self.source)
        if source_length == 0:
            raise IndexError("VLM source dataset is empty")
        last_error: Optional[Exception] = None
        for offset in range(self.max_error_skip):
            candidate = (index + offset) % source_length
            try:
                record = self.source[candidate]
                return self.transform(record) if self.transform is not None else record
            except Exception as exc:
                last_error = exc
                logger.warning("skipping unusable VLM record at source index %d: %s", candidate, exc)
        raise RuntimeError(
            f"failed to load {self.max_error_skip} consecutive VLM records; "
            f"the dataset is likely unreadable"
        ) from last_error


class MixVLMDataset(Dataset):
    """Weighted mixture of multiple named VLM sources.

    Each source gets its own ``data_config`` (source_type, schema_adapter, ...)
    and a sampling weight. Logical indices are assigned proportionally to the
    weights and interleaved so every global batch sees a stable mix.
    """

    @staticmethod
    def _load_manifest(manifest_path: str) -> list[dict[str, Any]]:
        """Expand a build_manifest output into per-dataset source entries."""
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        sources = []
        for dataset in manifest.get("datasets", []):
            if dataset.get("schema_adapter") in (None, "unknown"):
                logger.warning("manifest dataset %s has no schema adapter; skipping", dataset.get("name"))
                continue
            sources.append({
                "name": dataset["name"],
                "data_path": [entry["path"] for entry in dataset["files"]],
                "data_config": {
                    "source_type": "parquet",
                    "schema_adapter": dataset["schema_adapter"],
                    "lazy_parquet": True,
                },
                "weight": float(dataset.get("weight", 1.0)),
            })
        if not sources:
            raise ValueError(f"manifest {manifest_path!r} contains no usable datasets")
        return sources

    def __init__(
            self,
            sources: list[dict[str, Any]] | str,
            transform: Optional[SampleTransform] = None,
            *,
            eager_filter: bool = True,
            parquet_cache_groups: int = 4,
            skip_errors: bool = True,
    ) -> None:
        """Build one sub-dataset per source and the proportional index map.

        ``sources`` accepts the inline YAML list or a path to a
        ``dataset_manifest.json`` produced by ``build_manifest``.
        """
        if isinstance(sources, str):
            sources = self._load_manifest(sources)
        if not sources:
            raise ValueError("MixVLMDataset requires at least one source")
        self.sub_datasets: list[_TransformDataset] = []
        self.weights: list[float] = []
        self.names: list[str] = []
        for entry in sources:
            name = entry.get("name", f"source_{len(self.sub_datasets)}")
            data_config = dict(entry.get("data_config", {}))
            data_config.setdefault("source_type", "parquet")
            data_config.setdefault("schema_adapter", "auto")
            data_config.setdefault("lazy_parquet", True)
            source = VLMDataset(
                entry["data_path"],
                source_type=data_config["source_type"],
                schema_adapter=data_config["schema_adapter"],
                lazy_parquet=data_config["lazy_parquet"],
                recursive=bool(data_config.get("recursive", False)),
                file_pattern=data_config.get("file_pattern", "*.parquet"),
                parquet_cache_groups=int(data_config.get("parquet_cache_groups", parquet_cache_groups)),
            )
            self.sub_datasets.append(_TransformDataset(
                source,
                transform,
                eager_filter=eager_filter,
                skip_errors=skip_errors,
            ))
            self.weights.append(float(entry.get("weight", 1.0)))
            self.names.append(name)
        total_weight = sum(self.weights)
        self._probabilities = [weight / total_weight for weight in self.weights]
        self._index_map: list[tuple[int, int]] = []
        self._shuffle_blocks: list[tuple[int, int]] = []
        self._build_index_map()

    def _build_index_map(self) -> None:
        """Interleave sub-dataset runs proportionally to their weights.

        Runs align with each source's IO-local blocks (one parquet row group
        per run for lazy parquet sources), so consecutive logical indices stay
        inside a single decoded row group. This keeps block-shuffled training
        IO-efficient even when shards hold one giant row group each.
        """
        runs_per_source: list[list[tuple[int, int]]] = []
        for dataset in self.sub_datasets:
            blocks = dataset.shuffle_blocks or [(0, len(dataset))]
            runs_per_source.append([(start, length) for start, length in blocks if length > 0])
        cursors = [0] * len(self.sub_datasets)
        appended = [0] * len(self.sub_datasets)
        total = sum(len(dataset) for dataset in self.sub_datasets)
        while len(self._index_map) < total:
            best_ratio = -1.0
            best_source = -1
            for source_index, runs in enumerate(runs_per_source):
                if cursors[source_index] >= len(runs):
                    continue
                run_length = runs[cursors[source_index]][1]
                target = self._probabilities[source_index] * total
                ratio = (appended[source_index] + run_length) / target if target else 0.0
                if best_source < 0 or ratio < best_ratio:
                    best_ratio = ratio
                    best_source = source_index
            if best_source < 0:
                break
            run_start, run_length = runs_per_source[best_source][cursors[best_source]]
            self._shuffle_blocks.append((len(self._index_map), run_length))
            self._index_map.extend(
                (best_source, local) for local in range(run_start, run_start + run_length)
            )
            appended[best_source] += run_length
            cursors[best_source] += 1

    @property
    def shuffle_blocks(self) -> list[tuple[int, int]]:
        """Return logical runs aligned with the sub-datasets' row groups."""
        return self._shuffle_blocks

    def __len__(self) -> int:
        return len(self._index_map)

    def __getitem__(self, index: int) -> Any:
        source_index, local_index = self._index_map[index]
        return self.sub_datasets[source_index][local_index]


def build_vlm_dataset(
        *,
        data_config: dict[str, Any],
        data_path: Optional[str | list[str]] = None,
        transform: Optional[SampleTransform] = None,
        tokenizer: Any = None,
        mesh_context: Any = None,
        training_config: Any = None,
        **dataset_options: Any,
) -> Any:
    """Build a transform-wrapped VLM dataset from one source or a mix."""
    del tokenizer, mesh_context, training_config
    sources = data_config.get("sources")
    manifest = data_config.get("manifest")
    parquet_cache_groups = int(data_config.get("parquet_cache_groups", 4))
    skip_errors = bool(data_config.get("skip_errors", True))
    if sources is not None or manifest is not None:
        mix_sources = sources if sources is not None else manifest
        if not mix_sources or not isinstance(mix_sources, (list, str)):
            raise ValueError("data_config.sources/manifest must be a list or a manifest path")
        eager_filter = bool(data_config.get("eager_filter", True))
        return MixVLMDataset(
            mix_sources,
            transform,
            eager_filter=eager_filter,
            parquet_cache_groups=parquet_cache_groups,
            skip_errors=skip_errors,
        )

    source_type = data_config.get("source_type", "online")
    if source_type not in {"online", "jsonl", "parquet"}:
        raise ValueError(f"Unsupported VLM source type: {source_type!r}")
    if data_path is None:
        raise ValueError("VLM dataset requires data_path")
    eager_filter = bool(data_config.get("eager_filter", True))
    lazy_parquet = bool(data_config.get("lazy_parquet", True))
    schema_adapter = data_config.get("schema_adapter", "auto")
    source = VLMDataset(
        data_path,
        source_type=source_type,
        schema_adapter=schema_adapter,
        lazy_parquet=lazy_parquet,
        recursive=bool(data_config.get("recursive", False)),
        file_pattern=data_config.get("file_pattern", "*.parquet"),
        parquet_cache_groups=parquet_cache_groups,
    )
    return _TransformDataset(source, transform, eager_filter=eager_filter, skip_errors=skip_errors)


__all__ = ["VLMDataset", "MixVLMDataset", "build_vlm_dataset"]
