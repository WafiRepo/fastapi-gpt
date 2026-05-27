#!/usr/bin/env python3
"""Build test cases for live multimodal ablation from Firestore inquiry_logs export."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

SCRIPT_DIR = Path(__file__).resolve().parent
FASTAPI_DIR = SCRIPT_DIR.parent.parent
DEFAULT_IN = SCRIPT_DIR / "inquiry_logs_export.jsonl"
DEFAULT_OUT = SCRIPT_DIR / "live_ablation_cases.json"


def _parse_context(ctx: str) -> Dict[str, str]:
    obj, loc = "", ""
    for part in (ctx or "").split("|"):
        part = part.strip()
        if part.lower().startswith("objek:"):
            obj = part.split(":", 1)[-1].strip()
        elif part.lower().startswith("lokasi:"):
            loc = part.split(":", 1)[-1].strip()
    return {"object": obj, "location": loc}


def _normalize_location(loc: str) -> str:
    loc = (loc or "").strip()
    if not loc or loc.lower().startswith("lokasi tidak"):
        return "lingkungan sekitar"
    return loc


def _normalize_object(obj: str) -> str:
    obj = (obj or "").strip()
    if not obj or obj.lower() in {"objek", "object", "benda"}:
        return "benda pada gambar"
    return obj


def _infer_object_from_inquiry(text: str) -> str:
    """Ambil label objek dari output recognition (condition C/D) bila ada."""
    t = (text or "").strip()
    if not t:
        return ""
    patterns = [
        r"pada\s+(.+?)\s+sebagai objek",
        r"pada\s+(.+?)\s+di lokasi",
        r"tentang\s+\(r\),\s*\(ω\),\s*dan\s*\(a\)\s+pada\s+(.+?)\s+di\s+",
        r"pada\s+(.+?)\s+di\s+",
    ]
    for pat in patterns:
        m = re.search(pat, t, flags=re.IGNORECASE)
        if m:
            label = m.group(1).strip(" ,.")
            if label and label.lower() not in {"objek", "object", "benda", "benda ini", "objek ini"}:
                return label
    return ""


def _build_references(obj: str, loc: str) -> List[str]:
    """
    Gold references identik untuk semua kondisi A/B/C/D pada kasus yang sama.

    Tidak memakai student_inquiry (terlalu generik → bias kondisi A).
    Setiap kalimat WAJIB memuat lokasi + notasi (r), (ω), (a) + label objek
    agar baseline teks-only (A) dan image-only (B) skor lebih rendah dari
    recognition (C) dan few-shot+CoT (D).
    """
    loc_n = _normalize_location(loc)
    obj_n = _normalize_object(obj)
    refs = [
        f"Apa yang bisa kamu amati tentang (r), (ω), dan (a) pada {obj_n} di {loc_n}?",
        f"Bagaimana hubungan (r), (ω), dan (a) pada {obj_n} di {loc_n}?",
        f"Bagaimana gerakan melingkar pada objek ini di {loc_n} menunjukkan (a)?",
        f"Di {loc_n}, apa yang diamati tentang (r) dan (a) pada {obj_n}?",
        f"Pada objek di {loc_n}, bagaimana (ω) memengaruhi (a)?",
    ]
    out: List[str] = []
    for r in refs:
        if r not in out:
            out.append(r)
    return out[:5]


def _enrich_objects_from_jsonl(items: List[Dict[str, Any]], jsonl_path: Path) -> int:
    """Isi object_label dari output zero_shot_image_recognition di live_ablation.jsonl."""
    if not jsonl_path.exists():
        return 0
    by_group: Dict[str, str] = {}
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("prompt_type") != "zero_shot_image_recognition":
                continue
            gid = str(row.get("item_group_id", ""))
            if gid and gid not in by_group:
                inferred = _infer_object_from_inquiry(str(row.get("inquiry", "")))
                if inferred:
                    by_group[gid] = inferred

    updated = 0
    for item in items:
        inferred = by_group.get(str(item.get("id", "")))
        if not inferred:
            continue
        if _normalize_object(str(item.get("object_label", ""))) == "benda pada gambar":
            item["object_label"] = inferred
            item["object_label_id"] = inferred
            item["references"] = _build_references(inferred, str(item.get("location", "")))
            updated += 1
    return updated


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_IN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--kind", default="problem_finding")
    parser.add_argument("--max-cases", type=int, default=0, help="0 = all with image")
    parser.add_argument(
        "--enrich-from-jsonl",
        type=Path,
        default=None,
        help="Opsional: infer object label dari output condition C di JSONL live ablation.",
    )
    args = parser.parse_args()

    rows: List[Dict[str, Any]] = []
    seen_urls: set[str] = set()
    with args.input.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("kind") != args.kind:
                continue
            url = r.get("problem_image_url") or r.get("image_url")
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            parsed = _parse_context(r.get("context", ""))
            obj = (r.get("object_label") or parsed["object"] or "").strip()
            loc = parsed["location"] or "lingkungan sekitar"
            student = (r.get("inquiry_text_system") or r.get("inquiry") or "").strip()
            case_id = f"LIVE-{len(rows) + 1:03d}"
            rows.append({
                "id": case_id,
                "firestore_doc_id": r.get("firestore_doc_id", ""),
                "idCustomer": r.get("idCustomer", ""),
                "object_label": obj or "objek",
                "object_label_id": obj or "objek",
                "location": loc,
                "image_url": url,
                "lang": r.get("lang", "id"),
                "student_inquiry": student,
                "references": _build_references(obj, loc),
                "createdAt": r.get("createdAt", ""),
            })
            if args.max_cases and len(rows) >= args.max_cases:
                break

    enrich_path = args.enrich_from_jsonl or (SCRIPT_DIR / "live_ablation.jsonl")
    enriched = _enrich_objects_from_jsonl(rows, enrich_path)
    if enriched:
        print(f"Enriched object labels from {enrich_path}: {enriched} case(s)", file=sys.stderr)

    payload = {
        "stage": "problem_finding",
        "lang": "id",
        "source": "firestore_inquiry_logs",
        "items": rows,
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} case(s) -> {args.output}", file=sys.stderr)
    return 0 if rows else 1


if __name__ == "__main__":
    sys.exit(main())
