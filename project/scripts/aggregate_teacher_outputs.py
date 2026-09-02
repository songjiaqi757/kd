#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = Path("/home/wy/sjq/kd/dataset/cmu_mosei")
OUTPUT_ROOT = Path("/home/wy/sjq/kd/outputs/teacher_benchmark")
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rdid_mosei.benchmark import parse_sentiment_score, weighted_mean  # noqa: E402
from rdid_mosei.subsets import SUBSETS  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate teacher window scores to utterances")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DATASET_ROOT / "manifests/benchmark500_windowed.jsonl",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=OUTPUT_ROOT / "teacher_benchmark500_windowed.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_ROOT / "teacher_benchmark500_aggregated.jsonl",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_rows = [
        json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line
    ]
    metadata = {row["sample_id"]: row for row in manifest_rows}
    groups: dict[tuple[str, str], list[tuple[float, float, str]]] = defaultdict(list)
    parse_errors: list[dict[str, str]] = []

    for line_number, line in enumerate(args.input.read_text(encoding="utf-8").splitlines(), start=1):
        if not line:
            continue
        record = json.loads(line)
        if record.get("status") != "ok":
            continue
        sample_id = str(record["sample_id"])
        row = metadata.get(sample_id)
        if row is None:
            raise ValueError(f"benchmark result sample not in manifest: {sample_id}")
        try:
            output = record["output_text"]
            text = output[0] if isinstance(output, list) else str(output)
            score = parse_sentiment_score(text)
        except (KeyError, ValueError) as exc:
            parse_errors.append({"line": str(line_number), "sample_id": sample_id, "error": str(exc)})
            continue
        key = (str(row["parent_sample_id"]), str(record["subset"]))
        groups[key].append((score, float(row["aggregation_weight"]), sample_id))

    parent_rows = {str(row["parent_sample_id"]): row for row in manifest_rows}
    output_rows = []
    missing = []
    for parent_id, row in sorted(parent_rows.items()):
        for subset in SUBSETS:
            values = groups.get((parent_id, subset), [])
            if len(values) != int(row["window_count"]):
                missing.append(
                    {
                        "parent_sample_id": parent_id,
                        "subset": subset,
                        "expected_windows": int(row["window_count"]),
                        "available_windows": len(values),
                    }
                )
                continue
            score = weighted_mean((value, weight) for value, weight, _ in values)
            output_rows.append(
                {
                    "schema_version": "teacher-aggregate-v1",
                    "parent_sample_id": parent_id,
                    "video_id": row["video_id"],
                    "split": row["split"],
                    "subset": subset,
                    "window_count": len(values),
                    "window_sample_ids": [sample_id for _, _, sample_id in values],
                    "teacher_score": score,
                    "target_sentiment": row["sentiment"],
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(args.output)
    summary = {
        "aggregated_records": len(output_rows),
        "expected_records": len(parent_rows) * len(SUBSETS),
        "parse_errors": parse_errors,
        "missing": missing,
        "output": str(args.output),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if not parse_errors and not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
