"""
Helpers untuk membaca acc / gyr / t dari `data_buffer` agar satu sesi (uuid sama)
dipakai bersama, bukan ORDER BY per-buffer yang bisa campur batch.
"""
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from mysql.connector import Error
from router_analyze_data import get_db_connection

logger = logging.getLogger(__name__)

# Nama kolom DataFrame — harus sama dengan y_axis_options di generate_*_question
COL_TIME = "Time (s)"
COL_OMEGA = "Angular velocity (ω) rad/s"
COL_ACC = "Centripetal Acceleration (a) m/s^2"


def _parse_data_json(raw: Any) -> Optional[List[float]]:
    if raw is None:
        return None
    if isinstance(raw, (dict, list)):
        if isinstance(raw, dict) and "data" in raw:
            return list(raw["data"])
        if isinstance(raw, list):
            return list(raw)
    s = str(raw)
    if not s or s in ("null", "None"):
        return None
    try:
        o = json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return None
    if isinstance(o, dict) and "data" in o:
        return list(o["data"])
    if isinstance(o, list):
        return o
    return None


def get_aligned_acc_gyr_t(user_id: str) -> Optional[Dict[str, Any]]:
    """
    Ambil satu set buffer acc, gyr, t dengan uuid yang sama (isi terakhir t sebagai acuan waktu).
    Jika uuid tidak tersedia (data lama), jatuh kembali ke 3x ORDER BY timestamp DESC.
    Return: { 'acc': list, 'gyr': list, 't': list, 'buffer_ts': datetime|None, 'uuid': str|None } atau None
    """
    connection = get_db_connection()
    if not connection:
        return None
    cursor = None
    try:
        cursor = connection.cursor(dictionary=True)
        cursor.execute(
            """
            SELECT `timestamp` AS buffer_ts, uuid, data
            FROM data_buffer
            WHERE user_id = %s AND buffer_name = 't'
            ORDER BY `timestamp` DESC
            LIMIT 1
            """,
            (user_id,),
        )
        t_row = cursor.fetchone()
        if not t_row or not t_row.get("data"):
            return None
        t_data = _parse_data_json(t_row["data"])
        if not t_data:
            return None
        buffer_ts = t_row.get("buffer_ts")
        u = t_row.get("uuid")

        if u:
            acc_data = gyr_data = None
            cursor.execute(
                """
                SELECT data FROM data_buffer
                WHERE user_id = %s AND buffer_name = 'acc' AND uuid = %s
                LIMIT 1
                """,
                (user_id, u),
            )
            a = cursor.fetchone()
            if a:
                acc_data = _parse_data_json(a.get("data"))
            cursor.execute(
                """
                SELECT data FROM data_buffer
                WHERE user_id = %s AND buffer_name = 'gyr' AND uuid = %s
                LIMIT 1
                """,
                (user_id, u),
            )
            g = cursor.fetchone()
            if g:
                gyr_data = _parse_data_json(g.get("data"))
            if acc_data and gyr_data:
                if len(t_data) != len(acc_data) or len(t_data) != len(gyr_data):
                    logger.warning(
                        "Aligned buffers length mismatch (uuid=%s): t=%s acc=%s gyr=%s",
                        u, len(t_data), len(acc_data), len(gyr_data),
                    )
                return {
                    "acc": acc_data,
                    "gyr": gyr_data,
                    "t": t_data,
                    "buffer_ts": buffer_ts,
                    "uuid": u,
                }
            logger.info(
                "Fallback: missing acc/gyr for uuid %s, using latest per buffer_name", u
            )

        # Fallback: last row per name (sama lama) — t, acc, gyr masing-masing “latest”
        def latest_buf(name: str) -> Tuple[Optional[List[float]], Any]:
            cursor.execute(
                """
                SELECT `timestamp` AS buffer_ts, data FROM data_buffer
                WHERE user_id = %s AND buffer_name = %s
                ORDER BY `timestamp` DESC
                LIMIT 1
                """,
                (user_id, name),
            )
            r = cursor.fetchone()
            if r and r.get("data") is not None:
                return _parse_data_json(r["data"]), r.get("buffer_ts")
            return None, None

        t_data, buffer_ts = latest_buf("t")
        acc_data, _ = latest_buf("acc")
        gyr_data, _ = latest_buf("gyr")
        if not acc_data or not gyr_data or not t_data:
            return None
        return {
            "acc": acc_data,
            "gyr": gyr_data,
            "t": t_data,
            "buffer_ts": buffer_ts,
            "uuid": u,
        }
    except Error as e:
        logger.error("get_aligned_acc_gyr_t: %s", e)
        return None
    finally:
        if connection and connection.is_connected():
            if cursor:
                try:
                    cursor.close()
                except Exception:
                    pass
            connection.close()


def get_radius_for_buffer_session(user_id: str, buffer_ts) -> Optional[float]:
    """
    Pilih radius: baris di calculated_radius dengan timestamp setelah/ sama dengan
    buffer sensor (radius dihitung/ditulis setelah data sesi ini).
    Fallback: baris terbaru.
    """
    connection = get_db_connection()
    if not connection:
        return None
    cursor = None
    try:
        cursor = connection.cursor(dictionary=True)
        if buffer_ts is not None:
            cursor.execute(
                """
                SELECT radius
                FROM calculated_radius
                WHERE user_id = %s
                  AND `timestamp` >= %s
                ORDER BY `timestamp` ASC
                LIMIT 1
                """,
                (user_id, buffer_ts),
            )
            row = cursor.fetchone()
            if row and row.get("radius") is not None:
                return float(row["radius"])
        cursor.execute(
            """
            SELECT radius
            FROM calculated_radius
            WHERE user_id = %s
            ORDER BY `timestamp` DESC
            LIMIT 1
            """,
            (user_id,),
        )
        row = cursor.fetchone()
        if row and row.get("radius") is not None:
            return float(row["radius"])
        return None
    except Error as e:
        logger.error("get_radius_for_buffer_session: %s", e)
        return None
    finally:
        if connection and connection.is_connected():
            if cursor:
                try:
                    cursor.close()
                except Exception:
                    pass
            connection.close()
