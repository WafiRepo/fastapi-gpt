#!/usr/bin/env python3
"""Export Firestore user id -> email/name for joining Dashboard CSV with inquiry_logs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv  # type: ignore

    env_path = Path(__file__).resolve().parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass

from export_inquiry_logs_for_eval import _init_firestore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Export user id/email mapping from Firestore.")
    parser.add_argument(
        "-o",
        "--output",
        default="docs/eval_live/firestore_users.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--csv",
        default="docs/eval_live/firestore_users.csv",
        help="Optional CSV output.",
    )
    args = parser.parse_args()

    try:
        db = _init_firestore()
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    users: list[dict] = []
    for coll in ("user", "users", "customers"):
        try:
            for snap in db.collection(coll).stream():
                d = snap.to_dict() or {}
                uid = (
                    (d.get("id") or d.get("uid") or d.get("userId") or snap.id)
                )
                users.append({
                    "id": str(uid).strip(),
                    "doc_id": snap.id,
                    "email": (d.get("email") or d.get("userEmail") or d.get("mail") or "").strip(),
                    "name": (d.get("name") or d.get("displayName") or d.get("fullName") or d.get("nama") or "").strip(),
                    "collection": coll,
                })
        except Exception as exc:
            print(f"WARN: collection {coll}: {exc}", file=sys.stderr)

    dedup: dict[str, dict] = {}
    for u in users:
        if not u["id"]:
            continue
        if u["id"] not in dedup or u["collection"] == "user":
            dedup[u["id"]] = u
    out_list = sorted(dedup.values(), key=lambda x: (x.get("email") or "").lower())

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_list, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.csv:
        csv_path = Path(args.csv)
        with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["id", "email", "name", "collection", "doc_id"])
            w.writeheader()
            w.writerows(out_list)

    print(f"Wrote {len(out_list)} user(s) -> {out_path}", file=sys.stderr)
    if args.csv:
        print(f"Wrote CSV -> {args.csv}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
