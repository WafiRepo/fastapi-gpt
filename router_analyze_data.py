from fastapi import APIRouter, HTTPException, File, UploadFile
from pydantic import BaseModel
import mysql.connector
from mysql.connector import Error
from typing import List
import json
import os
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
        print("Error while connecting to MySQL", e)
        return None

@router_analyze_data.post("/add_buffer_data/")
async def add_buffer_data(data: BufferData):
    connection = get_db_connection()
    if not connection:
        raise HTTPException(status_code=500, detail="Database connection failed")

    try:
        cursor = connection.cursor()

        # Round each value in the lists to 2 decimal places before insertion
        data_dict = data.dict()
        user_id = data_dict.pop("user_id")
        device_id = data_dict.pop("device_id", 1)
        for buffer_name, buffer_data in data_dict.items():
            if buffer_data:
                # Round each value in the buffer_data to 2 decimal places
                rounded_data = [round(value, 2) for value in buffer_data]

                # Convert to JSON format
                data_json = json.dumps({"data": rounded_data})
                print(f"Inserting buffer: {buffer_name} with data: {data_json} for user: {user_id} device: {device_id}")

                # Insert into the database
                cursor.execute(
                    "INSERT INTO data_buffer (user_id, device_id, buffer_name, data) VALUES (%s, %s, %s, %s)",
                    (user_id, device_id, buffer_name, data_json)
                )
        connection.commit()
        return {"message": "Buffer data added successfully"}

    except Error as e:
        print("Error while inserting data into MySQL", e)
        raise HTTPException(status_code=500, detail="Failed to add buffer data")
    finally:
        cursor.close()
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
        print("Error saving experiment location:", e)
        raise HTTPException(status_code=500, detail="Failed to save location")
    finally:
        if cursor:
            cursor.close()
        connection.close()