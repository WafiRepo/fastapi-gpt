"""
Database layer for inquiry/personalization: student_profile and misconception_history.
Uses database sensor_data (same as router_analyze_data).
"""
import os
from typing import List, Optional, Dict, Any

from dotenv import load_dotenv

script_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(script_dir, ".env")
if os.path.exists(env_path):
    load_dotenv(dotenv_path=env_path)
else:
    load_dotenv()


def get_db_connection():
    try:
        import mysql.connector
        from mysql.connector import Error
        connection = mysql.connector.connect(
            host=os.getenv("DB_HOST", "127.0.0.1"),
            user=os.getenv("DB_USER", "root"),
            password=os.getenv("DB_PASSWORD", ""),
            database=os.getenv("DB_NAME", "sensor_data"),
        )
        return connection
    except Exception as e:
        print("Error while connecting to MySQL (inquiry_db):", e)
        return None


def ensure_inquiry_tables(connection) -> None:
    """Create student_profile and misconception_history tables if they do not exist."""
    cursor = connection.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS student_profile (
            user_id VARCHAR(64) PRIMARY KEY,
            grade_level VARCHAR(32) DEFAULT NULL,
            iq_rank_order VARCHAR(32) DEFAULT NULL,
            ability_band VARCHAR(16) DEFAULT 'neutral',
            questions_answered INT DEFAULT 0,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS misconception_history (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id VARCHAR(64) NOT NULL,
            misconception_id VARCHAR(32) NOT NULL,
            response_id VARCHAR(64) DEFAULT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_user (user_id)
        )
    """)
    connection.commit()
    cursor.close()


def get_latest_detected_object(user_id: str) -> Optional[Dict[str, Any]]:
    """
    Ambil hasil deteksi objek terbaru dari processed_images untuk user_id.
    Dipakai untuk problem_exploring: konteks inquiry dari objek yang terdeteksi.
    """
    conn = get_db_connection()
    if not conn:
        return None
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            """SELECT id, label, user_id, confidence, center_x, center_y, radius, created_at, image_path
               FROM processed_images WHERE user_id = %s ORDER BY created_at DESC LIMIT 1""",
            (user_id,),
        )
        row = cursor.fetchone()
        cursor.close()
        return row
    except Exception:
        return None
    finally:
        conn.close()


def get_experiment_data_summary(user_id: str) -> Optional[str]:
    """
    Ringkasan data eksperimen dari data_buffer untuk user_id (acc, gyr, t, gyr_squared).
    Dipakai untuk problem_exploring dengan konteks grafik dari sensor (centripetal acceleration).
    """
    conn = get_db_connection()
    if not conn:
        return None
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            """SELECT DISTINCT buffer_name FROM data_buffer
               WHERE user_id = %s AND buffer_name IN ('acc', 'gyr', 't', 'gyr_squared')""",
            (user_id,),
        )
        rows = cursor.fetchall()
        cursor.close()
        if not rows:
            return None
        names = {r["buffer_name"] for r in rows}
        if "acc" in names and "gyr" in names and "t" in names:
            return (
                "The student has collected real-time sensor data from the centripetal acceleration experiment. "
                "Data includes acceleration a (m/s²), angular velocity ω (rad/s), and time t (s). "
                "Four graphs are available: Acceleration vs angular velocity, Acceleration vs ω², Acceleration vs time, Angular velocity vs time. "
                "Ask the student to analyze patterns or trends between centripetal acceleration and angular velocity (or radius) in the graphs."
            )
        return None
    except Exception:
        return None
    finally:
        conn.close()


def get_latest_calculated_radius(user_id: str) -> Optional[Dict[str, Any]]:
    """
    Ambil radius terakhir dari calculated_radius untuk user_id (dari sensor).
    Bisa dipakai bersama processed_images untuk konteks problem_exploring.
    """
    conn = get_db_connection()
    if not conn:
        return None
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            """SELECT id, buffer_name_acc, buffer_name_gyr, radius, user_id, timestamp
               FROM calculated_radius WHERE user_id = %s ORDER BY timestamp DESC LIMIT 1""",
            (user_id,),
        )
        row = cursor.fetchone()
        cursor.close()
        return row
    except Exception:
        return None
    finally:
        conn.close()


def compute_ability_band(iq_rank_order: Optional[str]) -> str:
    """
    P1: Map IQ rank order "A/Total" to ability band (High/Medium/Low).
    Smaller rank = better. r = A/Total; r<=0.33 High, 0.33<r<=0.67 Medium, r>0.67 Low.
    Returns "neutral" if iq_rank_order is missing or invalid.
    """
    if not iq_rank_order or "/" not in iq_rank_order:
        return "neutral"
    parts = iq_rank_order.strip().split("/")
    if len(parts) != 2:
        return "neutral"
    try:
        a, total = int(parts[0].strip()), int(parts[1].strip())
        if total <= 0 or a < 1 or a > total:
            return "neutral"
        r = a / total
        if r <= 0.33:
            return "high"
        if r <= 0.67:
            return "medium"
        return "low"
    except (ValueError, ZeroDivisionError):
        return "neutral"


def get_student_profile(user_id: str) -> Optional[Dict[str, Any]]:
    """Return profile dict or None if not found."""
    conn = get_db_connection()
    if not conn:
        return None
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            "SELECT user_id, grade_level, iq_rank_order, ability_band, questions_answered, updated_at FROM student_profile WHERE user_id = %s",
            (user_id,),
        )
        row = cursor.fetchone()
        cursor.close()
        return row
    finally:
        conn.close()


def upsert_student_profile(
    user_id: str,
    grade_level: Optional[str] = None,
    iq_rank_order: Optional[str] = None,
    ability_band: Optional[str] = None,
    questions_answered: Optional[int] = None,
) -> None:
    """Insert or update student_profile. None values leave existing column unchanged on update."""
    conn = get_db_connection()
    if not conn:
        raise RuntimeError("Database connection failed")
    try:
        ensure_inquiry_tables(conn)
        cursor = conn.cursor()
        existing = get_student_profile(user_id)
        if existing is None:
            ab = ability_band if ability_band is not None else compute_ability_band(iq_rank_order)
            cursor.execute(
                """INSERT INTO student_profile (user_id, grade_level, iq_rank_order, ability_band, questions_answered)
                   VALUES (%s, %s, %s, %s, %s)""",
                (user_id, grade_level or None, iq_rank_order or None, ab, questions_answered if questions_answered is not None else 0),
            )
        else:
            updates = []
            params = []
            if grade_level is not None:
                updates.append("grade_level = %s")
                params.append(grade_level)
            if iq_rank_order is not None:
                updates.append("iq_rank_order = %s")
                params.append(iq_rank_order)
            if ability_band is not None:
                updates.append("ability_band = %s")
                params.append(ability_band)
            elif iq_rank_order is not None:
                updates.append("ability_band = %s")
                params.append(compute_ability_band(iq_rank_order))
            if questions_answered is not None:
                updates.append("questions_answered = %s")
                params.append(questions_answered)
            if updates:
                params.append(user_id)
                cursor.execute(
                    "UPDATE student_profile SET " + ", ".join(updates) + " WHERE user_id = %s",
                    params,
                )
        conn.commit()
        cursor.close()
    finally:
        conn.close()


def get_misconception_history(user_id: str) -> List[str]:
    """Return list of misconception_id for user (order by created_at)."""
    conn = get_db_connection()
    if not conn:
        return []
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT misconception_id FROM misconception_history WHERE user_id = %s ORDER BY created_at ASC",
            (user_id,),
        )
        rows = cursor.fetchall()
        cursor.close()
        return [r[0] for r in rows]
    finally:
        conn.close()


def add_misconception(user_id: str, misconception_id: str, response_id: Optional[str] = None) -> None:
    """Append one detected misconception to history."""
    conn = get_db_connection()
    if not conn:
        raise RuntimeError("Database connection failed")
    try:
        ensure_inquiry_tables(conn)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO misconception_history (user_id, misconception_id, response_id) VALUES (%s, %s, %s)",
            (user_id, misconception_id, response_id),
        )
        conn.commit()
        cursor.close()
    finally:
        conn.close()


def increment_questions_answered(user_id: str) -> None:
    """Increment questions_answered for user. Creates profile if missing."""
    conn = get_db_connection()
    if not conn:
        raise RuntimeError("Database connection failed")
    try:
        ensure_inquiry_tables(conn)
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO student_profile (user_id, questions_answered) VALUES (%s, 1)
               ON DUPLICATE KEY UPDATE questions_answered = questions_answered + 1""",
            (user_id,),
        )
        conn.commit()
        cursor.close()
    finally:
        conn.close()
