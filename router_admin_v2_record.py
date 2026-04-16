"""
Admin API: backfill questionImageUrl/Path untuk koleksi Firestore **v2_record** saja.

Sumber gambar (berurutan):
1) Field legacy base64 / base64_2 .. base64_5 di dokumen → unggah ke Storage.
2) Opsional: regenerasi dari MySQL data_buffer (t, acc, gyr, gyr_squared). Default: ambil snapshot
   waktu **terdekat** waktu dokumen (`createdAt` / `dateTime` / ID dokumen = millis), bukan buffer terakhir
   yang sama untuk semua record. Mode legacy masih tersedia lewat `mysql_buffer_pick`.

Autentikasi: header Authorization: Bearer (Firebase ID token). User harus
ada di Firestore koleksi user dengan role admin (sama seperti portal admin web).

Parameter ``force_overwrite`` (body JSON): jika True, dokumen yang sudah punya
semua questionImageUrl* tetap diproses dan URL/path diganti (base64 dulu, lalu MySQL).

Lingkungan wajib:
- GOOGLE_APPLICATION_CREDENTIALS atau FIREBASE_CREDENTIALS_PATH — JSON service account
- FIREBASE_STORAGE_BUCKET — default gphysolve.firebasestorage.app

MySQL: pakai koneksi yang sama dengan router_analyze_data (DB_HOST, DB_USER, DB_PASSWORD, DB_NAME).
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import time
import urllib.parse
import uuid
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Tuple

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field

from router_analyze_data import get_db_connection

router_admin_v2_record = APIRouter(prefix="/admin/v2-record", tags=["admin-v2-record"])

V2_RECORD = "v2_record"
BASE64_FIELDS = ["base64", "base64_2", "base64_3", "base64_4", "base64_5"]
URL_FIELDS = [
    "questionImageUrl1",
    "questionImageUrl2",
    "questionImageUrl3",
    "questionImageUrl4",
    "questionImageUrl5",
]
PATH_FIELDS = [
    "questionImagePath1",
    "questionImagePath2",
    "questionImagePath3",
    "questionImagePath4",
    "questionImagePath5",
]


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv

        script_dir = os.path.dirname(os.path.abspath(__file__))
        env_path = os.path.join(script_dir, ".env")
        if os.path.exists(env_path):
            load_dotenv(dotenv_path=env_path)
        else:
            load_dotenv()
    except Exception:
        pass


_load_dotenv()


_firebase_app = None


def _ensure_firebase():
    global _firebase_app
    if _firebase_app is not None:
        return _firebase_app
    try:
        import firebase_admin
        from firebase_admin import credentials
    except ImportError as e:
        raise HTTPException(
            status_code=503,
            detail="firebase-admin tidak terpasang. pip install firebase-admin",
        ) from e

    cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or os.getenv("FIREBASE_CREDENTIALS_PATH")
    if not cred_path or not os.path.isfile(cred_path):
        raise HTTPException(
            status_code=503,
            detail="Set GOOGLE_APPLICATION_CREDENTIALS atau FIREBASE_CREDENTIALS_PATH ke file JSON service account.",
        )

    bucket_name = os.getenv("FIREBASE_STORAGE_BUCKET", "gphysolve.firebasestorage.app")
    if not firebase_admin._apps:
        cred = credentials.Certificate(cred_path)
        _firebase_app = firebase_admin.initialize_app(cred, {"storageBucket": bucket_name})
    else:
        _firebase_app = firebase_admin.get_app()
    return _firebase_app


def _fs_and_storage():
    _ensure_firebase()
    from firebase_admin import firestore, storage

    return firestore.client(), storage.bucket()


def _require_firebase_admin_user(authorization: Optional[str] = Header(None)) -> str:
    """
    Verifikasi ID token Firebase dan pastikan dokumen user/{uid} memiliki role admin.
    Mengembalikan UID admin yang memanggil.
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Kirim header Authorization: Bearer <Firebase ID token>.",
        )
    token = authorization[7:].strip()
    if not token:
        raise HTTPException(status_code=401, detail="Token kosong.")

    _ensure_firebase()
    from firebase_admin import auth as fb_auth
    from firebase_admin import firestore as fb_firestore

    try:
        decoded = fb_auth.verify_id_token(token)
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Token tidak valid: {e}") from e

    uid = decoded.get("uid")
    if not uid:
        raise HTTPException(status_code=401, detail="Token tanpa uid.")

    db = fb_firestore.client()
    snap = db.collection("user").document(uid).get()
    if not snap.exists:
        raise HTTPException(status_code=403, detail="Profil user tidak ditemukan.")
    role = (snap.to_dict() or {}).get("role")
    if role != "admin":
        raise HTTPException(status_code=403, detail="Hanya admin yang boleh mengakses endpoint ini.")
    return uid


def _is_http_url(v: Any) -> bool:
    if v is None:
        return False
    s = str(v).strip()
    return s.startswith("http://") or s.startswith("https://")


def _decode_base64_image_to_bytes(payload: str) -> Optional[bytes]:
    if not payload or not str(payload).strip():
        return None
    data = str(payload).strip()
    if "data:image" in data and "," in data:
        data = data.split(",", 1)[1]
    data = re.sub(r"\s+", "", data)
    try:
        raw = base64.b64decode(data, validate=False)
    except Exception:
        return None
    if not raw:
        return None
    try:
        from PIL import Image

        im = Image.open(io.BytesIO(raw)).convert("RGB")
        out = io.BytesIO()
        im.save(out, format="JPEG", quality=85)
        return out.getvalue()
    except Exception:
        return raw


def _firebase_download_url(bucket_name: str, object_path: str, token: str) -> str:
    enc = urllib.parse.quote(object_path, safe="")
    return f"https://firebasestorage.googleapis.com/v0/b/{bucket_name}/o/{enc}?alt=media&token={token}"


def _upload_jpeg_bytes(
    bucket,
    *,
    user_id: str,
    doc_id: str,
    slot: int,
    jpeg_bytes: bytes,
    content_type: str = "image/jpeg",
) -> Tuple[str, str]:
    token = str(uuid.uuid4())
    filename = f"q_img_{slot}_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}.jpg"
    object_path = f"users/{user_id}/records/{doc_id}/question_images/{filename}"
    blob = bucket.blob(object_path)
    blob.metadata = {"firebaseStorageDownloadTokens": token}
    blob.upload_from_string(jpeg_bytes, content_type=content_type)
    blob.patch()
    url = _firebase_download_url(bucket.name, object_path, token)
    return url, object_path


def _upload_png_bytes(bucket, *, user_id: str, doc_id: str, slot: int, png_bytes: bytes) -> Tuple[str, str]:
    token = str(uuid.uuid4())
    filename = f"q_img_{slot}_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}.png"
    object_path = f"users/{user_id}/records/{doc_id}/question_images/{filename}"
    blob = bucket.blob(object_path)
    blob.metadata = {"firebaseStorageDownloadTokens": token}
    blob.upload_from_string(png_bytes, content_type="image/png")
    blob.patch()
    url = _firebase_download_url(bucket.name, object_path, token)
    return url, object_path


def _parse_doc_time(data: Dict[str, Any]) -> Optional[datetime]:
    if data.get("createdAt") is not None:
        try:
            ca = data["createdAt"]
            if hasattr(ca, "timestamp"):
                return datetime.utcfromtimestamp(float(ca.timestamp()))
            if isinstance(ca, (int, float)):
                ts = float(ca)
                if ts > 1e12:
                    ts = ts / 1000.0
                return datetime.utcfromtimestamp(ts)
            # Firestore REST / dict-shaped timestamp
            if isinstance(ca, dict):
                sec = ca.get("seconds") or ca.get("_seconds")
                if sec is not None:
                    return datetime.utcfromtimestamp(float(sec))
        except Exception:
            pass
    dt = data.get("dateTime")
    if isinstance(dt, str) and dt.strip():
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f"):
            try:
                return datetime.strptime(dt.strip(), fmt)
            except ValueError:
                continue
    return None


def _parse_doc_id_as_millis(doc_id: str) -> Optional[datetime]:
    """ID dokumen dari app sering berupa System.currentTimeMillis() (13 digit)."""
    if not doc_id or not str(doc_id).isdigit():
        return None
    try:
        ms = int(doc_id)
    except ValueError:
        return None
    if ms < 1_000_000_000_000:  # ~2001 in ms; avoid treating short ids as ms
        return None
    return datetime.utcfromtimestamp(ms / 1000.0)


def _resolve_record_target_time(data: Dict[str, Any], doc_id: str) -> Optional[datetime]:
    """Waktu acuan untuk memilih snapshot data_buffer: createdAt/dateTime, lalu doc_id millis."""
    t = _parse_doc_time(data)
    if t is not None:
        return t
    return _parse_doc_id_as_millis(doc_id)


def _group_buffers_by_timestamp(user_id: str) -> List[Tuple[datetime, Dict[str, Any]]]:
    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="MySQL connection failed")
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute(
            """
            SELECT buffer_name, data, timestamp
            FROM data_buffer
            WHERE user_id = %s
            ORDER BY timestamp ASC
            """,
            (user_id,),
        )
        rows = cursor.fetchall()
    finally:
        cursor.close()
        conn.close()

    from collections import defaultdict

    groups: Dict[Any, Dict[str, Any]] = defaultdict(dict)
    ts_map: Dict[Any, datetime] = {}
    for r in rows:
        ts = r["timestamp"]
        if hasattr(ts, "replace"):
            ts_key = ts.replace(microsecond=0)
        else:
            ts_key = ts
        ts_map[ts_key] = ts if isinstance(ts, datetime) else ts_key
        raw = r["data"]
        if raw is None:
            continue
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8", errors="replace")
        try:
            parsed = json.loads(raw) if isinstance(raw, str) else raw
        except Exception:
            continue
        arr = parsed.get("data") if isinstance(parsed, dict) else None
        if not isinstance(arr, list):
            continue
        name = r.get("buffer_name")
        if not name:
            continue
        groups[ts_key][str(name)] = [float(x) for x in arr if isinstance(x, (int, float))]
    out: List[Tuple[datetime, Dict[str, Any]]] = []
    for k, g in groups.items():
        tso = ts_map.get(k, k)
        if isinstance(tso, datetime):
            out.append((tso, g))
        else:
            try:
                out.append((datetime.fromisoformat(str(tso)), g))
            except Exception:
                continue
    out.sort(key=lambda x: x[0])
    return out


def _nearest_buffer(
    groups: List[Tuple[datetime, Dict[str, Any]]], target: Optional[datetime], max_delta_sec: int = 300
) -> Optional[Dict[str, Any]]:
    """Pilih buffer dengan delta terkecil; hanya terima jika delta <= max_delta_sec."""
    if not target or not groups:
        return None
    best = None
    best_d = None
    for ts, g in groups:
        delta = abs((ts - target).total_seconds())
        if best_d is None or delta < best_d:
            best_d = delta
            best = g
    if best is None or best_d is None or best_d > max_delta_sec:
        return None
    return best


def _closest_buffer_group(
    groups: List[Tuple[datetime, Dict[str, Any]]], target: Optional[datetime]
) -> Tuple[Optional[Dict[str, Any]], Optional[float]]:
    """
    Selalu pilih grup snapshot yang paling dekat dengan target (tanpa batas jarak).
    Membedakan record yang berbeda waktu ketika ada beberapa batch di MySQL.
    """
    if not target or not groups:
        return None, None
    best_g = None
    best_d = None
    for ts, g in groups:
        delta = abs((ts - target).total_seconds())
        if best_d is None or delta < best_d:
            best_d = delta
            best_g = g
    return best_g, best_d


def _align_xy(t: List[float], y: List[float]) -> Tuple[List[float], List[float]]:
    import numpy as np

    n = min(len(t), len(y))
    if n < 2:
        return [], []
    t2 = np.array(t[:n], dtype=float)
    y2 = np.array(y[:n], dtype=float)
    return t2.tolist(), y2.tolist()


def _plot_png(t: List[float], y: List[float], ylabel: str, title: str) -> Optional[bytes]:
    if len(t) < 2 or len(y) < 2:
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t, y, "b-", linewidth=1.2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def _table_png(rows: List[List[str]], title: str) -> Optional[bytes]:
    if not rows:
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, min(2 + len(rows) * 0.35, 10)))
    ax.axis("off")
    tbl = ax.table(
        cellText=rows,
        loc="center",
        cellLoc="center",
    )
    tbl.scale(1, 1.2)
    ax.set_title(title)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


class V2RecordBackfillRequest(BaseModel):
    user_id: str = Field(..., description="Firebase idCustomer / UID pemilik record")
    limit: int = Field(40, ge=1, le=200)
    dry_run: bool = Field(False)
    force_overwrite: bool = Field(
        False,
        description="Jika True: timpa questionImageUrl* yang sudah ada (base64 + MySQL). Default False = hanya isi slot kosong.",
    )
    regenerate_from_mysql: bool = Field(
        True,
        description="Jika masih ada slot kosong, coba plot dari data_buffer (timestamp terdekat).",
    )
    mysql_time_window_sec: int = Field(300, ge=30, le=3600)
    mysql_buffer_pick: Literal["closest_by_time", "strict_window_only", "legacy_same_latest_for_all"] = Field(
        "closest_by_time",
        description=(
            "closest_by_time: pilih snapshot data_buffer yang paling dekat waktu dokumen (disarankan; "
            "tiap record beda jika ada banyak batch). "
            "strict_window_only: hanya jika delta <= mysql_time_window_sec. "
            "legacy_same_latest_for_all: perilaku lama — pakai buffer terakhir jika tidak match (sering bikin gambar sama)."
        ),
    )


class V2RecordPreviewRequest(BaseModel):
    user_id: str
    limit_scan: int = Field(500, ge=1, le=5000)


@router_admin_v2_record.post("/preview")
def preview_v2_record_backfill(
    body: V2RecordPreviewRequest,
    _admin_uid: str = Depends(_require_firebase_admin_user),
):
    db, _ = _fs_and_storage()
    uid = body.user_id.strip()
    if not uid:
        raise HTTPException(400, "user_id kosong")

    need = 0
    scanned = 0
    for doc in (
        db.collection(V2_RECORD).where("idCustomer", "==", uid).limit(body.limit_scan).stream()
    ):
        scanned += 1
        data = doc.to_dict() or {}
        if _doc_missing_any_url(data):
            need += 1

    return {
        "collection": V2_RECORD,
        "user_id": uid,
        "scanned": scanned,
        "documents_needing_image_urls": need,
    }


def _doc_missing_any_url(data: Dict[str, Any]) -> bool:
    for k in URL_FIELDS:
        if not _is_http_url(data.get(k)):
            return True
    return False


def _next_missing_slot(
    data: Dict[str, Any],
    slot_urls: Dict[int, str],
    *,
    force_overwrite: bool = False,
) -> Optional[int]:
    """Slot berikutnya untuk diisi. Jika force_overwrite, abaikan URL yang sudah ada di data (kecuali slot sudah diisi di run ini)."""
    for i in range(5):
        slot = i + 1
        if slot in slot_urls:
            continue
        if not force_overwrite and _is_http_url(data.get(URL_FIELDS[i])):
            continue
        return slot
    return None


def _apply_mysql_plots(
    *,
    data: Dict[str, Any],
    doc_id: str,
    uid: str,
    bucket,
    buf: Dict[str, Any],
    slot_urls: Dict[int, str],
    slot_paths: Dict[int, str],
    errors: List[Dict[str, Any]],
    force_overwrite: bool = False,
) -> None:
    t = buf.get("t") or []
    acc = buf.get("acc")
    gyr = buf.get("gyr")
    gsq = buf.get("gyr_squared")
    charts: List[Tuple[Optional[List[float]], str, str]] = [
        (acc, "Acceleration vs t", "a (m/s²)"),
        (gyr, "Angular velocity vs t", "ω (rad/s)"),
        (gsq, "Angular velocity squared vs t", "ω²"),
    ]
    for series, ttl, ylab in charts:
        if not series or len(t) < 2:
            continue
        slot = _next_missing_slot(data, slot_urls, force_overwrite=force_overwrite)
        if slot is None:
            break
        tx, yx = _align_xy(t, series)
        png = _plot_png(tx, yx, ylab, ttl)
        if not png:
            continue
        try:
            u, p = _upload_png_bytes(bucket, user_id=uid, doc_id=doc_id, slot=slot, png_bytes=png)
            slot_urls[slot] = u
            slot_paths[slot] = p
            data[URL_FIELDS[slot - 1]] = u
        except Exception as e:
            errors.append({"doc": doc_id, "slot": slot, "phase": "mysql_plot", "error": str(e)})

    slot = _next_missing_slot(data, slot_urls, force_overwrite=force_overwrite)
    if slot is not None and acc and t:
        tx, yx = _align_xy(t, acc)
        if len(yx) >= 3:
            head = min(12, len(yx))
            rows = [[str(j + 1), f"{tx[j]:.3f}", f"{yx[j]:.4f}"] for j in range(head)]
            rows.insert(0, ["#", "t (s)", "a"])
            png = _table_png(rows, "Data sample (acc)")
            if png:
                try:
                    u, p = _upload_png_bytes(bucket, user_id=uid, doc_id=doc_id, slot=slot, png_bytes=png)
                    slot_urls[slot] = u
                    slot_paths[slot] = p
                    data[URL_FIELDS[slot - 1]] = u
                except Exception as e:
                    errors.append({"doc": doc_id, "slot": slot, "phase": "mysql_table", "error": str(e)})

    slot = _next_missing_slot(data, slot_urls, force_overwrite=force_overwrite)
    if slot is not None and gyr and t:
        tx, yx = _align_xy(t, gyr)
        if len(yx) >= 3:
            head = min(12, len(yx))
            rows = [[str(j + 1), f"{tx[j]:.3f}", f"{yx[j]:.4f}"] for j in range(head)]
            rows.insert(0, ["#", "t (s)", "ω"])
            png = _table_png(rows, "Data sample (gyr)")
            if png:
                try:
                    u, p = _upload_png_bytes(bucket, user_id=uid, doc_id=doc_id, slot=slot, png_bytes=png)
                    slot_urls[slot] = u
                    slot_paths[slot] = p
                    data[URL_FIELDS[slot - 1]] = u
                except Exception as e:
                    errors.append({"doc": doc_id, "slot": slot, "phase": "mysql_table2", "error": str(e)})


@router_admin_v2_record.post("/backfill")
def backfill_v2_record_images(
    body: V2RecordBackfillRequest,
    _admin_uid: str = Depends(_require_firebase_admin_user),
):
    db, bucket = _fs_and_storage()
    uid = body.user_id.strip()
    if not uid:
        raise HTTPException(400, "user_id kosong")

    buffer_groups: Optional[List[Tuple[datetime, Dict[str, Any]]]] = None
    mysql_err: Optional[str] = None
    if body.regenerate_from_mysql:
        try:
            buffer_groups = _group_buffers_by_timestamp(uid)
        except HTTPException:
            raise
        except Exception as e:
            buffer_groups = []
            mysql_err = str(e)

    processed = 0
    updated = 0
    skipped = 0
    errors: List[Dict[str, Any]] = []
    details: List[Dict[str, Any]] = []

    query = db.collection(V2_RECORD).where("idCustomer", "==", uid).limit(body.limit)

    for doc in query.stream():
        processed += 1
        doc_id = doc.id
        data = dict(doc.to_dict() or {})

        if not body.force_overwrite and not _doc_missing_any_url(data):
            skipped += 1
            details.append({"id": doc_id, "status": "skip", "reason": "urls_already_set"})
            continue

        updates: Dict[str, Any] = {}
        slot_urls: Dict[int, str] = {}
        slot_paths: Dict[int, str] = {}
        mysql_pick_note: Optional[Dict[str, Any]] = None

        for i in range(5):
            slot = i + 1
            url_key = URL_FIELDS[i]
            b64_key = BASE64_FIELDS[i]
            if _is_http_url(data.get(url_key)) and not body.force_overwrite:
                continue
            raw_b64 = data.get(b64_key)
            if raw_b64 and str(raw_b64).strip():
                if body.dry_run:
                    slot_urls[slot] = "(dry-run)"
                    continue
                jpeg = _decode_base64_image_to_bytes(str(raw_b64))
                if jpeg:
                    try:
                        u, p = _upload_jpeg_bytes(bucket, user_id=uid, doc_id=doc_id, slot=slot, jpeg_bytes=jpeg)
                        slot_urls[slot] = u
                        slot_paths[slot] = p
                        data[url_key] = u
                    except Exception as e:
                        errors.append({"doc": doc_id, "slot": slot, "phase": "base64", "error": str(e)})

        if body.regenerate_from_mysql and buffer_groups and not body.dry_run:
            target_time = _resolve_record_target_time(data, doc_id)
            buf: Optional[Dict[str, Any]] = None
            if target_time is not None and buffer_groups:
                if body.mysql_buffer_pick == "strict_window_only":
                    buf = _nearest_buffer(buffer_groups, target_time, body.mysql_time_window_sec)
                    mysql_pick_note = {
                        "mode": "strict_window_only",
                        "target_utc": target_time.isoformat() + "Z",
                    }
                elif body.mysql_buffer_pick == "legacy_same_latest_for_all":
                    buf = _nearest_buffer(buffer_groups, target_time, body.mysql_time_window_sec)
                    used_latest_fallback = False
                    if buf is None and buffer_groups:
                        buf = buffer_groups[-1][1]
                        used_latest_fallback = True
                    mysql_pick_note = {
                        "mode": "legacy_same_latest_for_all",
                        "target_utc": target_time.isoformat() + "Z",
                        "used_latest_fallback": used_latest_fallback,
                    }
                else:
                    # default: closest_by_time — tiap dokumen dapat snapshot terdekat (bukan buffer terakhir semua)
                    buf, delta_sec = _closest_buffer_group(buffer_groups, target_time)
                    mysql_pick_note = {
                        "mode": "closest_by_time",
                        "target_utc": target_time.isoformat() + "Z",
                        "delta_sec_to_snapshot": round(delta_sec, 3) if delta_sec is not None else None,
                    }
            if buf:
                _apply_mysql_plots(
                    data=data,
                    doc_id=doc_id,
                    uid=uid,
                    bucket=bucket,
                    buf=buf,
                    slot_urls=slot_urls,
                    slot_paths=slot_paths,
                    errors=errors,
                    force_overwrite=body.force_overwrite,
                )

        for slot, u in slot_urls.items():
            if slot < 1 or slot > 5:
                continue
            updates[URL_FIELDS[slot - 1]] = u
            if slot in slot_paths:
                updates[PATH_FIELDS[slot - 1]] = slot_paths[slot]

        if body.dry_run:
            details.append(
                {
                    "id": doc_id,
                    "status": "dry_run",
                    "would_update": bool(slot_urls),
                    "slots": list(slot_urls.keys()),
                }
            )
            if slot_urls:
                updated += 1
            continue

        if updates:
            try:
                db.collection(V2_RECORD).document(doc_id).update(updates)
                updated += 1
                detail_updated: Dict[str, Any] = {
                    "id": doc_id,
                    "status": "updated",
                    "fields": list(updates.keys()),
                }
                if mysql_pick_note:
                    detail_updated["mysql_pick"] = mysql_pick_note
                details.append(detail_updated)
            except Exception as e:
                errors.append({"doc": doc_id, "phase": "firestore", "error": str(e)})
                details.append({"id": doc_id, "status": "error", "error": str(e)})
        else:
            skipped += 1
            details.append(
                {
                    "id": doc_id,
                    "status": "skip",
                    "reason": "no_base64_and_no_mysql_plots",
                }
            )

    out: Dict[str, Any] = {
        "collection": V2_RECORD,
        "user_id": uid,
        "processed": processed,
        "updated": updated,
        "skipped": skipped,
        "dry_run": body.dry_run,
        "force_overwrite": body.force_overwrite,
        "mysql_buffer_pick": body.mysql_buffer_pick,
        "mysql_buffer_batches": len(buffer_groups) if buffer_groups is not None else 0,
        "details": details[:80],
        "errors": errors[:50],
    }
    if mysql_err:
        out["mysql_warning"] = mysql_err
    return out
