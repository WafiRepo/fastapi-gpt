#!/usr/bin/env python3
"""
Konversi `docs/Dashboard P-Magiv V2.csv` (log eksperimen siswa) ke JSONL
untuk eval_inquiry_metrics.py.

Satu baris CSV dapat menghasilkan beberapa baris JSONL (PF, PE1, PE2 split, PG).
Referensi:
  - problem_finding: references/problem_finding_id.json
  - problem_exploring: system_q dari dataset augmented (match objek)
  - problem_generating: opsional (generik dari references, sering kosong)

Contoh:
  python export_dashboard_csv_for_eval.py \\
    --csv "docs/Dashboard P-Magiv V2.csv" \\
    --output docs/eval_live/dashboard_eval.jsonl

  python eval_inquiry_metrics.py \\
    --input docs/eval_live/dashboard_eval.jsonl \\
    --output-jsonl docs/eval_live/dashboard_results.jsonl \\
    --output-csv docs/eval_live/dashboard_results.csv \\
    --plots-dir docs/eval_live/eval_plots --bleu-scale 0-100
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = SCRIPT_DIR / "docs" / "Dashboard P-Magiv V2.csv"
DEFAULT_DATASET = SCRIPT_DIR / "docs" / (
    "dataset problem finding and exploring - Augmented_Data_SMA_-_Centripetal_Concept.csv"
)
DEFAULT_PF_REFS = SCRIPT_DIR / "references" / "problem_finding_id.json"
DEFAULT_OUT = SCRIPT_DIR / "docs" / "eval_live" / "dashboard_eval.jsonl"

COL_NO = "NO"
COL_USER = "Unnamed: 1"
COL_PF = "Problem Finding"
COL_OBJ = "object recognition"
COL_PE1 = "Problem Exploring stage 1"
COL_PE2 = "problem exploring stage 2 (problem posing)"
COL_PE2_FB = "feedback problem posing quality"
COL_PG_DIFF = "problem generating                                  problem generating"
COL_PG_TEXT = "Unnamed: 14"

OBJECT_ALIASES: Dict[str, str] = {
    "pencuci sayur": "vegetable washer",
    "portable washer": "vegetable washer",
    "washing machine": "vegetable washer",
    "washer": "vegetable washer",
    "vegetable washer": "vegetable washer",
    "kipas": "fan",
    "fan": "fan",
}


def _clean(s: Any) -> str:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    return str(s).strip()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _pf_references(refs_path: Path, lang: str = "id") -> List[str]:
    data = _load_json(refs_path)
    block = data.get("problem_finding", {})
    if isinstance(block, dict) and lang in block:
        return [str(x) for x in block[lang] if x]
    return []


def _pe_references(refs_path: Path, lang: str = "id") -> List[str]:
    data = _load_json(refs_path)
    block = data.get("problem_exploring", {})
    if isinstance(block, dict) and lang in block:
        return [str(x) for x in block[lang] if x]
    return []


def _normalize_object(name: str) -> str:
    key = name.strip().lower()
    return OBJECT_ALIASES.get(key, key)


def _build_dataset_object_refs(dataset_path: Path) -> Dict[str, List[str]]:
    """Map normalized object -> unique system_q from dataset (stage 1)."""
    if not dataset_path.exists():
        return {}
    df = pd.read_csv(dataset_path, encoding="utf-8")
    out: Dict[str, List[str]] = {}
    skip_inputs = {"input", "sensor data graph", "acceleration vs time", "angular velocity vs time", "angular velocity vs acceleration"}
    for _, row in df.iterrows():
        raw_obj = _clean(row.get("input", "")).lower()
        if not raw_obj or raw_obj in skip_inputs:
            continue
        obj = _normalize_object(raw_obj)
        q = _clean(row.get("system_q", ""))
        if not obj or not q:
            continue
        out.setdefault(obj, [])
        if q not in out[obj]:
            out[obj].append(q)
    return out


def _refs_for_object(
    obj: str,
    stage: str,
    pf_refs: List[str],
    pe_refs: List[str],
    dataset_refs: Dict[str, List[str]],
) -> List[str]:
    norm = _normalize_object(obj)
    if stage == "problem_finding":
        return list(pf_refs)
    if stage == "problem_exploring":
        specific = dataset_refs.get(norm, [])
        if specific:
            return specific[:5]
        return list(pe_refs)
    return []


def _extract_location_from_pf(text: str) -> str:
    """Coba ambil lokasi dari teks PF (playground, rumah, ...)."""
    t = text.lower()
    markers = [
        ("playground", "playground"),
        ("taman", "taman"),
        ("rumah", "rumah"),
        ("ruang kelas", "ruang kelas"),
        ("kelas", "ruang kelas"),
        ("kantor", "kantor"),
        ("dapur", "dapur"),
        ("lapangan", "lapangan"),
        ("jalan", "jalan raya"),
        ("bengkel", "bengkel"),
    ]
    for needle, label in markers:
        if needle in t:
            return label
    return ""


def _build_context(obj: str, pf_text: str = "") -> str:
    parts: List[str] = []
    if obj:
        parts.append(f"Objek: {obj}")
    loc = _extract_location_from_pf(pf_text)
    if loc:
        parts.append(f"Lokasi: {loc}")
    return " | ".join(parts)


def _split_numbered_inquiries(text: str) -> List[str]:
    """Pecah sel PE2 yang berisi '1. ... 2. ...' menjadi beberapa inquiry."""
    text = text.strip()
    if not text:
        return []
    parts = re.split(r"\n\s*(?=\d+\.\s)", text)
    out: List[str] = []
    for p in parts:
        p = re.sub(r"^\d+\.\s*", "", p.strip())
        if len(p) > 15:
            out.append(p)
    if not out and text:
        out = [text]
    return out


def _parse_manual_pe2_scores(feedback: str) -> List[Tuple[Optional[int], Optional[int]]]:
    """
    Ekstrak skor 'Skor inquiry: X/3' dari kolom feedback PE2.
    Return list per sub-inquiry (urutan sama dengan split PE2).
    """
    if not feedback:
        return []
    scores: List[Tuple[Optional[int], Optional[int]]] = []
    for m in re.finditer(r"Skor inquiry\s*:\s*(\d+)\s*/\s*(\d+)", feedback, re.I):
        scores.append((int(m.group(1)), int(m.group(2))))
    return scores


def _read_dashboard(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, header=1, encoding="utf-8")
    df["NO_num"] = pd.to_numeric(df[COL_NO], errors="coerce")
    return df[df["NO_num"].notna()].copy()


def export_dashboard(
    csv_path: Path,
    output_path: Path,
    *,
    refs_path: Path,
    dataset_path: Path,
    lang: str = "id",
    include_pf: bool = True,
    include_pe1: bool = True,
    include_pe2: bool = True,
    include_pg: bool = True,
    metadata_path: Optional[Path] = None,
) -> int:
    df = _read_dashboard(csv_path)
    pf_refs = _pf_references(refs_path, lang)
    pe_refs = _pe_references(refs_path, lang)
    dataset_refs = _build_dataset_object_refs(dataset_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    meta_rows: List[Dict[str, Any]] = []
    written = 0

    with output_path.open("w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            no = int(row["NO_num"])
            uid = _clean(row.get(COL_USER, ""))
            obj = _clean(row.get(COL_OBJ, ""))
            pf = _clean(row.get(COL_PF, ""))
            ctx = _build_context(obj, pf)

            if include_pf and pf:
                refs = _refs_for_object(obj, "problem_finding", pf_refs, pe_refs, dataset_refs)
                item = {
                    "id": f"DASH-PF-{no:03d}",
                    "item_group_id": uid or f"row-{no}",
                    "dashboard_row": no,
                    "stage": "problem_finding",
                    "lang": lang,
                    "inquiry": pf,
                    "context": ctx,
                    "prompt_type": "live_student_experiment",
                    "object_label": obj,
                }
                if refs:
                    item["reference"] = refs
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                written += 1
                meta_rows.append({**item, "pe2_manual_scores": []})

            pe1 = _clean(row.get(COL_PE1, ""))
            if include_pe1 and pe1:
                refs = _refs_for_object(obj, "problem_exploring", pf_refs, pe_refs, dataset_refs)
                item = {
                    "id": f"DASH-PE1-{no:03d}",
                    "item_group_id": uid or f"row-{no}",
                    "dashboard_row": no,
                    "stage": "problem_exploring",
                    "lang": lang,
                    "inquiry": pe1,
                    "context": ctx,
                    "prompt_type": "live_student_experiment",
                    "substage": "stage1",
                    "object_label": obj,
                }
                if refs:
                    item["reference"] = refs
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                written += 1

            pe2 = _clean(row.get(COL_PE2, ""))
            pe2_fb = _clean(row.get(COL_PE2_FB, ""))
            manual_scores = _parse_manual_pe2_scores(pe2_fb)
            if include_pe2 and pe2:
                pe2_parts = _split_numbered_inquiries(pe2)
                refs = _refs_for_object(obj, "problem_exploring", pf_refs, pe_refs, dataset_refs)
                for i, part in enumerate(pe2_parts):
                    suffix = chr(ord("a") + i) if len(pe2_parts) > 1 else ""
                    item = {
                        "id": f"DASH-PE2-{no:03d}{suffix}",
                        "item_group_id": uid or f"row-{no}",
                        "dashboard_row": no,
                        "stage": "problem_exploring",
                        "lang": lang,
                        "inquiry": part,
                        "context": ctx,
                        "prompt_type": "live_student_experiment",
                        "substage": "stage2",
                        "object_label": obj,
                    }
                    if refs:
                        item["reference"] = refs
                    if i < len(manual_scores):
                        num, den = manual_scores[i]
                        item["manual_inquiry_score"] = f"{num}/{den}"
                        item["manual_inquiry_score_num"] = num
                        item["manual_inquiry_score_den"] = den
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
                    written += 1

            pg_diff = _clean(row.get(COL_PG_DIFF, "")).lower()
            pg_text = _clean(row.get(COL_PG_TEXT, ""))
            if include_pg and pg_text:
                item = {
                    "id": f"DASH-PG-{no:03d}",
                    "item_group_id": uid or f"row-{no}",
                    "dashboard_row": no,
                    "stage": "problem_generating",
                    "lang": lang,
                    "inquiry": pg_text,
                    "context": ctx,
                    "prompt_type": "live_student_experiment",
                    "object_label": obj,
                }
                if pg_diff in ("easy", "intermediate", "advanced"):
                    item["difficulty"] = pg_diff
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                written += 1

    if metadata_path:
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with metadata_path.open("w", encoding="utf-8") as mf:
            for m in meta_rows:
                mf.write(json.dumps(m, ensure_ascii=False) + "\n")

    print(f"Wrote {written} eval item(s) -> {output_path}", file=sys.stderr)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description="Export Dashboard CSV to eval JSONL.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output", "-o", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--references", type=Path, default=DEFAULT_PF_REFS)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--lang", default="id", choices=["id", "en"])
    parser.add_argument("--no-pf", action="store_true")
    parser.add_argument("--no-pe1", action="store_true")
    parser.add_argument("--no-pe2", action="store_true")
    parser.add_argument("--no-pg", action="store_true")
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Optional sidecar JSONL with extra fields (default: <output>.meta.jsonl)",
    )
    args = parser.parse_args()

    if not args.csv.exists():
        print(f"ERROR: CSV not found: {args.csv}", file=sys.stderr)
        return 1

    meta = args.metadata
    if meta is None:
        meta = args.output.with_suffix(".meta.jsonl")

    n = export_dashboard(
        args.csv,
        args.output,
        refs_path=args.references,
        dataset_path=args.dataset,
        lang=args.lang,
        include_pf=not args.no_pf,
        include_pe1=not args.no_pe1,
        include_pe2=not args.no_pe2,
        include_pg=not args.no_pg,
        metadata_path=meta,
    )
    return 0 if n > 0 else 2


if __name__ == "__main__":
    sys.exit(main())
