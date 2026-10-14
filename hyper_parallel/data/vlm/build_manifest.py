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
"""Scan a nested dataset root and build one dataset_manifest.json.

Run once before training (CPU only, ~seconds for 20 datasets)::

    python -m hyper_parallel.data.vlm.build_manifest \
        --data-root /mnt/sfs-turbo/datasets \
        --output /mnt/sfs-turbo/datasets/dataset_manifest.json

The manifest records, per dataset folder, the parquet file list, row counts,
detected schema, and a sampling weight.  Training reads only this manifest.
"""

import argparse
import json
import logging
import os
from collections import Counter
from typing import Any
from pathlib import Path

import pyarrow.parquet as pq

from hyper_parallel.data.vlm.schema_adapters import detect_adapter

logger = logging.getLogger(__name__)

_EXCLUDE_DIRS = {
    "images",
    "images_decoded",
    "kernel_meta",
    "__pycache__",
    ".ipynb_checkpoints",
    ".cache",
    "parts",
}

# Directory name produced by the repack pipeline; when a dataset folder has
# both raw parquet dumps and repacked shards, only the repacked ones train.
_REPACK_DIR_NAME = "vlm_parquet"


def _detect_schema(parquet_path: str) -> tuple[str, int, list[str]]:
    """Detect the schema adapter from one parquet file's columns."""
    parquet_file = pq.ParquetFile(parquet_path)
    columns = list(parquet_file.schema_arrow.names)
    probe = {column: None for column in columns}
    try:
        adapter = detect_adapter(probe)
    except ValueError:
        adapter = "unknown"
    return adapter, parquet_file.metadata.num_rows, columns


def scan_dataset_root(
        data_root: str,
        *,
        pattern: str = "**/*.parquet",
        exclude_dirs: set[str] | None = None,
        weight: float = 1.0,
) -> dict[str, Any]:
    """Walk ``data_root`` and group parquet files by top-level dataset folder."""
    exclude = exclude_dirs if exclude_dirs is not None else _EXCLUDE_DIRS
    root = Path(data_root).resolve()
    groups: dict[str, list[dict[str, Any]]] = {}

    for path in sorted(root.glob(pattern)):
        relative = path.relative_to(root)
        parts = relative.parts
        if not parts:
            continue
        if any(part in exclude for part in parts[:-1]):
            continue
        dataset_name = parts[0]
        groups.setdefault(dataset_name, []).append({
            "path": str(path),
            "rows": None,
            "schema_adapter": None,
        })

    manifest_datasets = []
    for name, files in sorted(groups.items()):
        repacked = [entry for entry in files if _REPACK_DIR_NAME in Path(entry["path"]).parts]
        if repacked and len(repacked) < len(files):
            logger.info(
                "dataset %s: using %d %s shards, ignoring %d raw parquet files",
                name, len(repacked), _REPACK_DIR_NAME, len(files) - len(repacked),
            )
            files = repacked
        total_rows = 0
        schemas: set[str] = set()
        resolved_files = []
        for entry in files:
            try:
                adapter, rows, _columns = _detect_schema(entry["path"])
            except Exception as exc:
                logger.warning("skip unreadable parquet %s: %s", entry["path"], exc)
                continue
            entry["rows"] = rows
            entry["schema_adapter"] = adapter
            total_rows += rows
            schemas.add(adapter)
            resolved_files.append(entry)
        if not resolved_files:
            logger.warning("dataset %s has no readable parquet files; skipping", name)
            continue
        if len(schemas) > 1:
            logger.warning("dataset %s mixes schemas %s; using the majority adapter", name, sorted(schemas))
        adapter_counts = Counter(entry["schema_adapter"] for entry in resolved_files)
        majority_adapter = adapter_counts.most_common(1)[0][0]
        if majority_adapter == "unknown" and len(adapter_counts) > 1:
            known = [(adapter, count) for adapter, count in adapter_counts.most_common() if adapter != "unknown"]
            if known:
                majority_adapter = known[0][0]
        manifest_datasets.append({
            "name": name,
            "schema_adapter": majority_adapter,
            "weight": weight,
            "total_rows": total_rows,
            "num_files": len(resolved_files),
            "files": resolved_files,
        })
    return {
        "version": 1,
        "data_root": str(root),
        "datasets": manifest_datasets,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, help="Root folder containing per-dataset subfolders")
    parser.add_argument("--output", default=None, help="Manifest path; defaults to <data-root>/dataset_manifest.json")
    parser.add_argument("--pattern", default="**/*.parquet", help="Glob pattern relative to data-root")
    parser.add_argument("--weight", type=float, default=1.0, help="Default sampling weight for every dataset")
    parser.add_argument("--exclude-dir", action="append", default=None, help="Directory name to skip (repeatable)")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    exclude = set(args.exclude_dir) if args.exclude_dir else None
    manifest = scan_dataset_root(args.data_root, pattern=args.pattern, exclude_dirs=exclude, weight=args.weight)
    output = args.output or str(Path(args.data_root) / "dataset_manifest.json")
    Path(output).write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    total_rows = sum(dataset["total_rows"] for dataset in manifest["datasets"])
    logger.info(
        "manifest: %s | datasets=%d files=%d rows=%d",
        output,
        len(manifest["datasets"]),
        sum(dataset["num_files"] for dataset in manifest["datasets"]),
        total_rows,
    )


if __name__ == "__main__":
    main()
