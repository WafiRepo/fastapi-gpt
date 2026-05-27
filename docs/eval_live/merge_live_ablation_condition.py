#!/usr/bin/env python3
"""Replace one condition in live_ablation.jsonl and optionally merge eval results."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def merge_jsonl(main: Path, patch: Path, condition: str) -> int:
    patch_by_id = {}
    with patch.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                patch_by_id[r["id"]] = r
    out = []
    replaced = 0
    with main.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("prompt_type") == condition and r["id"] in patch_by_id:
                out.append(patch_by_id[r["id"]])
                replaced += 1
            else:
                out.append(r)
    with main.open("w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return replaced


def merge_results_csv(main: Path, patch: Path, condition: str) -> int:
    patch_rows = {}
    with patch.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                patch_rows[r["id"]] = r
    rows = []
    replaced = 0
    with main.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        for r in reader:
            if r.get("prompt_type") == condition and r["id"] in patch_rows:
                pr = patch_rows[r["id"]]
                for k, v in pr.items():
                    if k in fields:
                        r[k] = v if not isinstance(v, (dict, list)) else json.dumps(v)
                replaced += 1
            rows.append(r)
    with main.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return replaced


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--condition", default="few_shot_cot_recognition")
    p.add_argument("--patch-jsonl", required=True)
    p.add_argument("--main-jsonl", default="docs/eval_live/live_ablation.jsonl")
    p.add_argument("--results-csv", default="")
    p.add_argument("--patch-results-jsonl", default="")
    args = p.parse_args()

    n = merge_jsonl(Path(args.main_jsonl), Path(args.patch_jsonl), args.condition)
    print(f"Replaced {n} rows in {args.main_jsonl}", file=sys.stderr)

    if args.results_csv and args.patch_results_jsonl:
        m = merge_results_csv(Path(args.results_csv), Path(args.patch_results_jsonl), args.condition)
        print(f"Replaced {m} rows in {args.results_csv}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
