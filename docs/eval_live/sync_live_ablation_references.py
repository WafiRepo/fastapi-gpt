#!/usr/bin/env python3
"""Salin field `reference` dari live_ablation_cases.json ke live_ablation.jsonl (semua kondisi)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

SCRIPT_DIR = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=SCRIPT_DIR / "live_ablation_cases.json")
    parser.add_argument("--jsonl", type=Path, default=SCRIPT_DIR / "live_ablation.jsonl")
    args = parser.parse_args()

    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    ref_by_id: Dict[str, List[str]] = {}
    for item in cases.get("items") or []:
        refs = item.get("references") or []
        if refs:
            ref_by_id[str(item["id"])] = [str(r) for r in refs if r]

    if not ref_by_id:
        print("ERROR: no references in cases file", file=sys.stderr)
        return 1

    lines_out: List[str] = []
    updated = 0
    with args.jsonl.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row: Dict[str, Any] = json.loads(line)
            gid = str(row.get("item_group_id", ""))
            refs = ref_by_id.get(gid)
            if refs:
                row["reference"] = refs
                updated += 1
            lines_out.append(json.dumps(row, ensure_ascii=False))

    args.jsonl.write_text("\n".join(lines_out) + "\n", encoding="utf-8")
    print(f"Updated references on {updated} row(s) -> {args.jsonl}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
