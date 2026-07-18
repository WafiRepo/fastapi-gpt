from fastapi import APIRouter, HTTPException, File, UploadFile
from pydantic import BaseModel
import mysql.connector
from mysql.connector import Error
from typing import List, Optional, Set
import json
import os
import logging
import uuid
from dotenv import load_dotenv

# Load environment variables
# Load .env from the same directory as this script
script_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(script_dir, '.env')
if os.path.exists(env_path):
    load_dotenv(dotenv_path=env_path)
else:
    # Fallback: try current directory
    load_dotenv()

router_analyze_data = APIRouter()

# Logger
logger = logging.getLogger("analyze_data_logger")
logger.setLevel(logging.INFO)
os.makedirs("./log_file", exist_ok=True)
_fh = logging.FileHandler("./log_file/analyze_data.log", encoding='utf-8')
_fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(_fh)

# Database model
class BufferData(BaseModel):
    user_id: str
    device_id: int = 1
    acc: List[float] = []
    gyr: List[float] = []
    gyr_squared: List[float] = []
    t: List[float] = []


class SaveExperimentLocationRequest(BaseModel):
    user_id: str
    location: str

# Function to establish database connection
def get_db_connection():
    try:
        connection = mysql.connector.connect(
            host=os.getenv("DB_HOST", "127.0.0.1"),
            user=os.getenv("DB_USER", "root"),
            password=os.getenv("DB_PASSWORD", ""),
            database=os.getenv("DB_NAME", "sensor_data")
        )
        return connection
    except Error as e:
        logger.error(f"Error while connecting to MySQL: {e}")
        return None

_data_buffer_columns_cache: Optional[Set[str]] = None


def _get_data_buffer_columns(connection) -> Set[str]:
    """Cache kolom tabel data_buffer (remote lama mungkin tanpa device_id/uuid)."""
    global _data_buffer_columns_cache
    if _data_buffer_columns_cache is not None:
        return _data_buffer_columns_cache
    cursor = connection.cursor()
    try:
        cursor.execute("SHOW COLUMNS FROM data_buffer")
        _data_buffer_columns_cache = {str(row[0]) for row in cursor.fetchall()}
        logger.info(f"data_buffer columns: {sorted(_data_buffer_columns_cache)}")
        return _data_buffer_columns_cache
    finally:
        cursor.close()


def _insert_buffer_row(
    cursor,
    columns: Set[str],
    user_id: str,
    device_id: int,
    buffer_name: str,
    data_json: str,
    session_uuid: Optional[str],
) -> None:
    """INSERT menyesuaikan skema DB (kompatibel remote lama & lokal baru)."""
    fields = ["user_id", "buffer_name", "data"]
    values: List[object] = [user_id, buffer_name, data_json]
    if "device_id" in columns:
        fields.insert(1, "device_id")
        values.insert(1, device_id)
    if session_uuid and "uuid" in columns:
        fields.append("uuid")
        values.append(session_uuid)
    placeholders = ", ".join(["%s"] * len(fields))
    sql = f"INSERT INTO data_buffer ({', '.join(fields)}) VALUES ({placeholders})"
    cursor.execute(sql, tuple(values))


@router_analyze_data.post("/add_buffer_data/")
async def add_buffer_data(data: BufferData):
    user_id = (data.user_id or "").strip()
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required")

    connection = get_db_connection()
    if not connection:
        raise HTTPException(status_code=500, detail="Database connection failed")

    cursor = None
    session_uuid = str(uuid.uuid4())
    try:
        cursor = connection.cursor()
        columns = _get_data_buffer_columns(connection)

        data_dict = data.dict()
        data_dict.pop("user_id", None)
        device_id = int(data_dict.pop("device_id", 1) or 1)
        inserted = 0
        for buffer_name, buffer_data in data_dict.items():
            if buffer_data:
                rounded_data = [round(float(value), 2) for value in buffer_data]
                data_json = json.dumps({"data": rounded_data})
                logger.info(
                    f"Inserting buffer: {buffer_name}, user={user_id}, device={device_id}, "
                    f"count={len(rounded_data)}, uuid={session_uuid if 'uuid' in columns else '-'}"
                )
                _insert_buffer_row(
                    cursor, columns, user_id, device_id, buffer_name, data_json,
                    session_uuid if "uuid" in columns else None,
                )
                inserted += 1
        if inserted == 0:
            raise HTTPException(status_code=400, detail="No buffer arrays provided (acc, gyr, gyr_squared, t)")
        connection.commit()
        return {
            "message": "Buffer data added successfully",
            "buffers": inserted,
            "uuid": session_uuid if "uuid" in columns else None,
        }

    except HTTPException:
        raise
    except Error as e:
        logger.error(f"Error while inserting data into MySQL: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to add buffer data: {e}")
    finally:
        if cursor:
            cursor.close()
        if connection and connection.is_connected():
            connection.close()


@router_analyze_data.post("/save-experiment-location/")
async def save_experiment_location(req: SaveExperimentLocationRequest):
    """Simpan lokasi eksperimen (Out-of-Class) ke sensor_data.data_buffer untuk user."""
    connection = get_db_connection()
    if not connection:
        raise HTTPException(status_code=500, detail="Database connection failed")
    location_clean = (req.location or "").strip() or "Lokasi tidak diisi"
    cursor = None
    try:
        cursor = connection.cursor()
        data_json = json.dumps({"location": location_clean})
        cursor.execute(
            "INSERT INTO data_buffer (user_id, buffer_name, data) VALUES (%s, %s, %s)",
            (req.user_id, "experiment_location", data_json),
        )
        connection.commit()
        return {"message": "Location saved", "location": location_clean}
    except Error as e:
        logger.error(f"Error saving experiment location: {e}")
        raise HTTPException(status_code=500, detail="Failed to save location")
    finally:
        if cursor:
            cursor.close()
        connection.close()


@router_analyze_data.delete("/delete-buffer-data/")
async def delete_buffer_data(user_id: str):
    """Hapus semua data sensor (data_buffer) milik user tertentu dari MySQL."""
    connection = get_db_connection()
    if not connection:
        raise HTTPException(status_code=500, detail="Database connection failed")
    cursor = None
    try:
        cursor = connection.cursor()
        cursor.execute("DELETE FROM data_buffer WHERE user_id = %s", (user_id,))
        deleted = cursor.rowcount
        connection.commit()
        logger.info(f"Deleted {deleted} rows from data_buffer for user_id={user_id}")
        return {"message": f"Deleted {deleted} buffer rows for user {user_id}", "deleted": deleted}
    except Error as e:
        logger.error(f"Error deleting buffer data for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete buffer data")
    finally:
        if cursor:
            cursor.close()
        connection.close()