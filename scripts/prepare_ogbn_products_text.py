#!/usr/bin/env python3
"""Download and align Amazon-3M raw text with OGBN-products nodes.

The OGBN-products graph ships an ASIN mapping but no raw title/description
text. This script downloads the Amazon-3M raw JSONL files from the LIBSVM
dataset mirror and aligns records to OGB node indices through nodeidx2asin.
"""

from __future__ import annotations

import argparse
import bz2
import csv
import gzip
import hashlib
import json
import os
import shutil
import ssl
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


SOURCES = {
    "train": {
        "url": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel/Amazon-3M_raw_data_train.json.bz2",
        "filename": "Amazon-3M_raw_data_train.json.bz2",
    },
    "test": {
        "url": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel/Amazon-3M_raw_data_test.json.bz2",
        "filename": "Amazon-3M_raw_data_test.json.bz2",
    },
}

UNVERIFIED_SSL_CONTEXT = ssl._create_unverified_context()


def open_url(request: Request, timeout: int):
    try:
        return urlopen(request, timeout=timeout)
    except URLError as exc:
        reason = getattr(exc, "reason", None)
        if isinstance(reason, ssl.SSLError):
            print(f"[ssl] certificate verification failed for {request.full_url}; retrying without verification")
            return urlopen(request, timeout=timeout, context=UNVERIFIED_SSL_CONTEXT)
        raise


def head(url: str) -> Dict[str, str]:
    request = Request(url, method="HEAD", headers={"User-Agent": "LLM-SGNN-data-prep/1.0"})
    with open_url(request, timeout=60) as response:
        return {k.lower(): v for k, v in response.headers.items()}


def download_range(url: str, output: Path, size: int, workers: int, chunk_size: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and output.stat().st_size == size:
        print(f"[skip] {output.name}: already downloaded ({size} bytes)")
        return

    part_dir = output.with_suffix(output.suffix + ".parts")
    part_dir.mkdir(parents=True, exist_ok=True)
    ranges: List[Tuple[int, int, Path]] = []
    for start in range(0, size, chunk_size):
        end = min(start + chunk_size - 1, size - 1)
        ranges.append((start, end, part_dir / f"{start:012d}-{end:012d}.part"))

    def fetch(start: int, end: int, part_path: Path) -> Tuple[int, int]:
        expected = end - start + 1
        if part_path.exists() and part_path.stat().st_size == expected:
            return start, expected
        tmp = part_path.with_suffix(".tmp")
        for attempt in range(1, 8):
            try:
                request = Request(
                    url,
                    headers={
                        "Range": f"bytes={start}-{end}",
                        "User-Agent": "LLM-SGNN-data-prep/1.0",
                    },
                )
                with open_url(request, timeout=120) as response, tmp.open("wb") as out:
                    shutil.copyfileobj(response, out, length=1024 * 1024)
                got = tmp.stat().st_size
                if got != expected:
                    raise IOError(f"range {start}-{end} expected {expected} bytes, got {got}")
                os.replace(tmp, part_path)
                return start, got
            except (HTTPError, URLError, TimeoutError, IOError) as exc:
                if tmp.exists():
                    tmp.unlink()
                if attempt == 7:
                    raise
                sleep_for = min(60, 2**attempt)
                print(f"[retry] {output.name} {start}-{end}: {exc}; sleep {sleep_for}s")
                time.sleep(sleep_for)
        raise RuntimeError("unreachable")

    done_bytes = sum(p.stat().st_size for _, _, p in ranges if p.exists())
    print(f"[download] {output.name}: {done_bytes}/{size} bytes already present")
    started = time.time()
    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch, start, end, part_path) for start, end, part_path in ranges]
        for future in as_completed(futures):
            _, got = future.result()
            completed += got
            downloaded = sum(p.stat().st_size for _, _, p in ranges if p.exists())
            elapsed = max(time.time() - started, 1e-6)
            mbps = (completed / 1024 / 1024) / elapsed
            print(f"[progress] {output.name}: {downloaded}/{size} bytes ({mbps:.2f} MiB/s active)")

    tmp_output = output.with_suffix(output.suffix + ".tmp")
    with tmp_output.open("wb") as out:
        for _, _, part_path in ranges:
            with part_path.open("rb") as part:
                shutil.copyfileobj(part, out, length=1024 * 1024)
    if tmp_output.stat().st_size != size:
        raise IOError(f"{output.name} combined size mismatch")
    if output.exists():
        output.unlink()
    os.replace(tmp_output, output)
    shutil.rmtree(part_dir)
    print(f"[ok] {output.name}: downloaded {size} bytes")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_ogb_mapping(path: Path) -> Tuple[Dict[str, int], int]:
    asin_to_node: Dict[str, int] = {}
    with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            asin_to_node[row["asin"]] = int(row["node idx"])
    return asin_to_node, len(asin_to_node)


def iter_jsonl_bz2(path: Path) -> Iterable[dict]:
    with bz2.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc


def clean_text(value: object) -> str:
    if value is None:
        return ""
    return " ".join(str(value).replace("\r", " ").replace("\n", " ").split()).strip()


def build_aligned_text(
    source_files: Dict[str, Path],
    mapping_path: Path,
    output_dir: Path,
    rebuild: bool,
) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    sqlite_path = output_dir / "nodeidx2text.sqlite"
    if rebuild and sqlite_path.exists():
        sqlite_path.unlink()

    asin_to_node, num_nodes = load_ogb_mapping(mapping_path)
    print(f"[mapping] loaded {num_nodes} OGBN-products ASIN mappings")

    conn = sqlite3.connect(sqlite_path)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS records ("
        "node_idx INTEGER PRIMARY KEY, "
        "asin TEXT NOT NULL, "
        "source_split TEXT NOT NULL, "
        "title TEXT NOT NULL, "
        "content TEXT NOT NULL)"
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_records_asin ON records(asin)")

    stats = {
        "num_ogb_nodes": num_nodes,
        "raw_records": {},
        "matched_records": {},
        "matched_with_title": 0,
        "matched_with_content": 0,
        "matched_with_title_or_content": 0,
        "duplicates": 0,
    }

    existing = conn.execute("SELECT COUNT(*) FROM records").fetchone()[0]
    if existing and not rebuild:
        print(f"[skip] sqlite already has {existing} records; use --rebuild to parse again")
    else:
        rows = []
        seen_nodes = set()
        for split, path in source_files.items():
            raw_count = 0
            matched_count = 0
            for obj in iter_jsonl_bz2(path):
                raw_count += 1
                asin = clean_text(obj.get("uid"))
                node_idx = asin_to_node.get(asin)
                if node_idx is None:
                    continue
                title = clean_text(obj.get("title"))
                content = clean_text(obj.get("content"))
                matched_count += 1
                if node_idx in seen_nodes:
                    stats["duplicates"] += 1
                    continue
                seen_nodes.add(node_idx)
                rows.append((node_idx, asin, split, title, content))
                if len(rows) >= 10000:
                    conn.executemany("INSERT OR REPLACE INTO records VALUES (?, ?, ?, ?, ?)", rows)
                    conn.commit()
                    rows.clear()
            if rows:
                conn.executemany("INSERT OR REPLACE INTO records VALUES (?, ?, ?, ?, ?)", rows)
                conn.commit()
                rows.clear()
            stats["raw_records"][split] = raw_count
            stats["matched_records"][split] = matched_count
            print(f"[parse] {split}: raw={raw_count}, matched={matched_count}")

    count = conn.execute("SELECT COUNT(*) FROM records").fetchone()[0]
    stats["matched_unique_nodes"] = count

    jsonl_path = output_dir / "nodeidx2text.jsonl.gz"
    covered_path = output_dir / "covered_nodeidx.txt.gz"
    missing_path = output_dir / "missing_nodeidx.txt.gz"

    with gzip.open(jsonl_path, "wt", encoding="utf-8", newline="\n") as jout, gzip.open(
        covered_path, "wt", encoding="utf-8", newline="\n"
    ) as covered, gzip.open(missing_path, "wt", encoding="utf-8", newline="\n") as missing:
        cursor = conn.execute(
            "SELECT node_idx, asin, source_split, title, content FROM records ORDER BY node_idx"
        )
        next_row = cursor.fetchone()
        for node_idx in range(num_nodes):
            if next_row and next_row[0] == node_idx:
                _, asin, source_split, title, content = next_row
                parts = [p for p in (title, content) if p]
                text = "\n\n".join(parts)
                record = {
                    "node_idx": node_idx,
                    "asin": asin,
                    "source_split": source_split,
                    "title": title,
                    "content": content,
                    "text": text,
                }
                jout.write(json.dumps(record, ensure_ascii=False) + "\n")
                covered.write(f"{node_idx}\n")
                stats["matched_with_title"] += int(bool(title))
                stats["matched_with_content"] += int(bool(content))
                stats["matched_with_title_or_content"] += int(bool(title or content))
                next_row = cursor.fetchone()
            else:
                missing.write(f"{node_idx}\n")

    stats["missing_nodes"] = num_nodes - stats["matched_unique_nodes"]
    stats["coverage"] = stats["matched_unique_nodes"] / num_nodes
    stats["title_coverage"] = stats["matched_with_title"] / num_nodes
    stats["content_coverage"] = stats["matched_with_content"] / num_nodes
    stats["title_or_content_coverage"] = stats["matched_with_title_or_content"] / num_nodes
    stats["outputs"] = {
        "sqlite": str(sqlite_path),
        "jsonl_gz": str(jsonl_path),
        "covered_nodeidx": str(covered_path),
        "missing_nodeidx": str(missing_path),
    }
    conn.close()
    return stats


def write_metadata(
    output_dir: Path,
    archive_dir: Path,
    source_headers: Dict[str, Dict[str, str]],
    checksums: Dict[str, str],
    stats: Dict[str, object],
) -> None:
    manifest = {
        "dataset": "ogbn-products + Amazon-3M raw real text",
        "graph_source": "OGB ogbn-products local directory",
        "text_sources": {
            split: {
                "url": SOURCES[split]["url"],
                "local_file": str(archive_dir / SOURCES[split]["filename"]),
                "http_headers": source_headers.get(split, {}),
                "sha256": checksums.get(split, ""),
            }
            for split in SOURCES
        },
        "alignment_key": "Amazon ASIN: OGB mapping/nodeidx2asin.csv.gz <-> Amazon-3M JSON uid",
        "stats": stats,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    readme = f"""# OGBN-products with Real Text

This directory is a companion text package for the local OGBN-products graph at
`../ogbn_products`. OGB provides the graph, labels, splits, PCA features, and
`mapping/nodeidx2asin.csv.gz`; it does not ship raw product title/description
text. The text here is aligned from Amazon-3M raw JSON records by ASIN.

## Sources

- Train text: {SOURCES['train']['url']}
- Test text: {SOURCES['test']['url']}
- Raw files are preserved in `{archive_dir}`.
- Alignment key: `nodeidx2asin.csv.gz` ASIN equals Amazon-3M JSON `uid`.

## Outputs

- `nodeidx2text.jsonl.gz`: one JSON object per covered OGB node, sorted by
  `node_idx`, with `asin`, `title`, `content`, and combined `text`.
- `covered_nodeidx.txt.gz`: covered OGB node indices.
- `missing_nodeidx.txt.gz`: OGB node indices without matched Amazon-3M text.
- `nodeidx2text.sqlite`: same aligned records in SQLite for random access.
- `manifest.json`: source URLs, HTTP metadata, SHA256 checksums, and stats.

## Coverage

- OGB nodes: {stats['num_ogb_nodes']}
- Matched unique nodes: {stats['matched_unique_nodes']}
- Missing nodes: {stats['missing_nodes']}
- Coverage: {stats['coverage']:.6%}
- Title coverage: {stats['title_coverage']:.6%}
- Content coverage: {stats['content_coverage']:.6%}
- Title-or-content coverage: {stats['title_or_content_coverage']:.6%}

No node text is synthesized or edited beyond whitespace normalization and
joining `title` plus `content` into the convenience `text` field.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    default_dataset_root = Path(__file__).resolve().parents[2] / "dataset"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=default_dataset_root)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--chunk-size-mib", type=int, default=16)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--rebuild", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    ogb_dir = dataset_root / "ogbn_products"
    archive_dir = dataset_root / "source_archives" / "ogbn_products_text"
    output_dir = dataset_root / "ogbn_products_text"
    mapping_path = ogb_dir / "mapping" / "nodeidx2asin.csv.gz"

    if not mapping_path.exists():
        print(f"missing OGB mapping: {mapping_path}", file=sys.stderr)
        return 2

    source_headers: Dict[str, Dict[str, str]] = {}
    source_paths: Dict[str, Path] = {}
    for split, source in SOURCES.items():
        url = source["url"]
        local_path = archive_dir / source["filename"]
        source_paths[split] = local_path
        headers = head(url)
        source_headers[split] = headers
        size = int(headers.get("content-length", "0"))
        if not size:
            raise RuntimeError(f"could not determine content length for {url}")
        if args.skip_download:
            if not local_path.exists():
                raise FileNotFoundError(local_path)
            print(f"[skip] download disabled for {local_path}")
        else:
            download_range(
                url=url,
                output=local_path,
                size=size,
                workers=args.workers,
                chunk_size=args.chunk_size_mib * 1024 * 1024,
            )

    checksums = {}
    for split, path in source_paths.items():
        print(f"[sha256] {path.name}")
        checksums[split] = sha256_file(path)

    stats = build_aligned_text(source_paths, mapping_path, output_dir, rebuild=args.rebuild)
    write_metadata(output_dir, archive_dir, source_headers, checksums, stats)

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
