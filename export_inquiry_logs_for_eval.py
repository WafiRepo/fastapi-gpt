#!/usr/bin/env python3
"""
Ekspor entri Firestore `inquiry_logs` ke JSONL untuk dievaluasi
oleh `eval_inquiry_metrics.py` (BLEU/ROUGE/BERTScore/G-eval).

Skema log di Firestore (kind = problem_finding | exploring_stage1 | exploring_stage2):
  idCustomer, kind, inquiryText, userResponse, feedbackSummary,
  sessionId, objectName, experimentLocation, createdAt, ...

Skema JSONL output (cocok untuk eval_inquiry_metrics.py):
  { "id": "...", "stage": "problem_finding" | "problem_exploring",
    "lang": "id" | "en", "inquiry": "...", "context": "...",
    "reference": ["...", "..."], ... }

Pemetaan default:
- kind=problem_finding         -> stage=problem_finding
- kind=exploring_stage1/stage2 -> stage=problem_exploring

Contoh:
  python export_inquiry_logs_for_eval.py --kinds problem_finding \
      --output pf_eval.jsonl --lang id \
      --references references/problem_finding_id.json

  python eval_inquiry_metrics.py --input pf_eval.jsonl \
      --output-jsonl pf_results.jsonl --output-csv pf_results.csv
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from dotenv import load_dotenv  # type: ignore

    script_dir = Path(__file__).resolve().parent
    env_path = script_dir / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        load_dotenv()
except ImportError:
    pass


KIND_TO_STAGE = {
    "problem_finding": "problem_finding",
    "exploring_stage1": "problem_exploring",
    "exploring_stage2": "problem_exploring",
}

DEFAULT_KIND_SUFFIX = {
    "problem_finding": "PF",
    "exploring_stage1": "PE1",
    "exploring_stage2": "PE2",
}


def _init_firestore():
    try:
        import firebase_admin  # type: ignore
        from firebase_admin import credentials, firestore  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "firebase-admin tidak terpasang. Jalankan: pip install firebase-admin"
        ) from exc

    if not firebase_admin._apps:
        cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
        if cred_path and not os.path.isabs(cred_path):
            cred_path = str(Path(__file__).resolve().parent / cred_path)
        if cred_path and os.path.exists(cred_path):
            cred = credentials.Certificate(cred_path)
            firebase_admin.initialize_app(cred)
        else:
            firebase_admin.initialize_app()
    return firestore.client()


def _parse_iso_date(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    s = s.strip()
    if not s:
        return None
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError as exc:
        raise SystemExit(f"Tanggal tidak valid (gunakan ISO 8601): {s} ({exc})")


def _load_references(path: Optional[str]) -> Dict[str, List[str]]:
    """
    File referensi opsional. Format yang didukung:
    1. { "id": ["ref1", "ref2"], "en": ["ref1"] }                # per lang
    2. { "problem_finding": { "id": [...], "en": [...] }, ... }  # per stage/lang
    3. ["ref1", "ref2"]                                          # global list
    """
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"File reference tidak ditemukan: {path}")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"File reference bukan JSON valid: {exc}")

    return data  # type: ignore[return-value]


def _refs_for(data: Any, stage: str, lang: str) -> List[str]:
    if data is None:
        return []
    if isinstance(data, list):
        return [str(x) for x in data if x is not None]
    if isinstance(data, dict):
        if stage in data and isinstance(data[stage], dict):
            return _refs_for(data[stage], stage, lang)
        if lang in data:
            return _refs_for(data[lang], stage, lang)
        if "default" in data:
            return _refs_for(data["default"], stage, lang)
        flat = [v for v in data.values() if isinstance(v, str)]
        return flat
    return []


def _build_context(d: Dict[str, Any]) -> str:
    parts: List[str] = []
    obj = d.get("objectName")
    loc = d.get("experimentLocation")
    if obj:
        parts.append(f"Objek: {obj}")
    if loc:
        parts.append(f"Lokasi: {loc}")
    return " | ".join(parts)


def _detect_lang(text: str, default_lang: str) -> str:
    if not text:
        return default_lang
    sample = text.lower()
    id_markers = (" yang ", " apa ", " bagaimana ", " mengapa ", " kenapa ",
                  " ketika ", " jika ", " dalam ", " akan ", " bisa ", " dapat ",
                  " tidak ", " ini ", " itu ", " adalah ")
    en_markers = (" what ", " how ", " why ", " which ", " can ", " could ",
                  " would ", " when ", " where ", " the ", " is ", " are ",
                  " does ", " do ")
    id_hits = sum(1 for m in id_markers if m in f" {sample} ")
    en_hits = sum(1 for m in en_markers if m in f" {sample} ")
    if id_hits > en_hits:
        return "id"
    if en_hits > id_hits:
        return "en"
    return default_lang


def export_logs(
    db,
    *,
    kinds: List[str],
    user_id: Optional[str],
    start: Optional[datetime],
    end: Optional[datetime],
    limit: Optional[int],
    default_lang: str,
    detect_lang: bool,
    references_data: Any,
    output_path: str,
) -> int:
    col = db.collection("inquiry_logs")
    query = col
    if user_id:
        query = query.where("idCustomer", "==", user_id)

    docs_iter = query.stream()
    written = 0
    skipped = 0
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8") as f:
        for snap in docs_iter:
            if limit is not None and written >= limit:
                break
            d = snap.to_dict() or {}
            kind = str(d.get("kind", "")).strip()
            if kinds and kind not in kinds:
                skipped += 1
                continue
            inquiry = (d.get("inquiryText") or "").strip()
            if not inquiry:
                skipped += 1
                continue

            created_at = d.get("createdAt")
            if isinstance(created_at, datetime):
                ts = created_at if created_at.tzinfo else created_at.replace(tzinfo=timezone.utc)
            else:
                ts_attr = getattr(created_at, "to_datetime", None)
                ts = ts_attr() if callable(ts_attr) else None
                if ts is not None and ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)

            if start and (ts is None or ts < start):
                skipped += 1
                continue
            if end and (ts is None or ts > end):
                skipped += 1
                continue

            stage = KIND_TO_STAGE.get(kind, "problem_finding")
            lang = _detect_lang(inquiry, default_lang) if detect_lang else default_lang
            refs = _refs_for(references_data, stage, lang)

            doc_id = snap.id
            suffix = DEFAULT_KIND_SUFFIX.get(kind, "PF")
            item_id = f"{suffix}-{doc_id}"

            row: Dict[str, Any] = {
                "id": item_id,
                "stage": stage,
                "lang": lang,
                "inquiry": inquiry,
                "context": _build_context(d),
                "prompt_type": "live_log",
            }
            if d.get("kind"):
                row["kind"] = d["kind"]
            if d.get("idCustomer"):
                row["idCustomer"] = d["idCustomer"]
            if d.get("sessionId"):
                row["item_group_id"] = d["sessionId"]
            if ts is not None:
                row["createdAt"] = ts.isoformat()
            if refs:
                row["reference"] = refs

            f.write(json.dumps(row, ensure_ascii=True) + "\n")
            written += 1

    print(
        f"Done. Wrote {written} item(s) to {output_path}. Skipped {skipped} (kind/empty/range).",
        file=sys.stderr,
    )
    return 0 if written > 0 else 2


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Ekspor entri Firestore `inquiry_logs` (problem finding / exploring) "
            "ke JSONL untuk evaluasi (eval_inquiry_metrics.py)."
        )
    )
    parser.add_argument(
        "--kinds",
        nargs="+",
        default=["problem_finding"],
        choices=list(KIND_TO_STAGE.keys()),
        help="Filter berdasarkan field `kind`. Default: problem_finding saja.",
    )
    parser.add_argument("--user-id", help="Filter idCustomer (UID Firebase) tertentu.")
    parser.add_argument(
        "--since",
        help="ISO 8601 (mis. 2025-08-01 atau 2025-08-01T00:00:00+00:00). Inklusif.",
    )
    parser.add_argument(
        "--until",
        help="ISO 8601. Inklusif. Bandingkan dengan field `createdAt` (jika ada).",
    )
    parser.add_argument("--limit", type=int, help="Batasi jumlah entri yang ditulis.")
    parser.add_argument(
        "--lang",
        default="id",
        choices=["id", "en"],
        help="Bahasa default. Jika --detect-lang aktif, ini hanya fallback.",
    )
    parser.add_argument(
        "--detect-lang",
        action="store_true",
        help="Deteksi sederhana bahasa dari teks inquiry (kata penanda).",
    )
    parser.add_argument(
        "--references",
        help=(
            "File JSON daftar reference untuk BLEU/ROUGE/BERTScore. "
            "Format: list, {lang:list}, atau {stage:{lang:list}}."
        ),
    )
    parser.add_argument(
        "--output",
        "-o",
        default="inquiry_logs_eval.jsonl",
        help="Path JSONL output (default: inquiry_logs_eval.jsonl).",
    )
    args = parser.parse_args()

    start = _parse_iso_date(args.since)
    end = _parse_iso_date(args.until)
    refs = _load_references(args.references)

    try:
        db = _init_firestore()
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    return export_logs(
        db,
        kinds=args.kinds,
        user_id=args.user_id,
        start=start,
        end=end,
        limit=args.limit,
        default_lang=args.lang,
        detect_lang=args.detect_lang,
        references_data=refs,
        output_path=args.output,
    )


if __name__ == "__main__":
    sys.exit(main())
