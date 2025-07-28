from fastapi import APIRouter, HTTPException, File, UploadFile
from pydantic import BaseModel
import mysql.connector
from mysql.connector import Error
from typing import List
import json

router_analyze_data = APIRouter()

# Database model
class BufferData(BaseModel):
    user_id: str
    acc: List[float] = []
    gyr: List[float] = []
    gyr_squared: List[float] = []
    t: List[float] = []

# Function to establish database connection
def get_db_connection():
    try:
        connection = mysql.connector.connect(
            host="127.0.0.1",  # Or use 'localhost'
            user="root",  # Your MySQL username
            password="",  # Your MySQL root password
            database="sensor_data"  # Your MySQL database name
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
        for buffer_name, buffer_data in data_dict.items():
            if buffer_data:
                # Round each value in the buffer_data to 2 decimal places
                rounded_data = [round(value, 2) for value in buffer_data]  

                # Convert to JSON format
                data_json = json.dumps({"data": rounded_data})
                print(f"Inserting buffer: {buffer_name} with data: {data_json} for user: {user_id}")  # Log the data

                # Insert into the database
                cursor.execute(
                    "INSERT INTO data_buffer (user_id, buffer_name, data) VALUES (%s, %s, %s)",
                    (user_id, buffer_name, data_json)
                )
        connection.commit()
        return {"message": "Buffer data added successfully"}

    except Error as e:
        print("Error while inserting data into MySQL", e)
        raise HTTPException(status_code=500, detail="Failed to add buffer data")
    finally:
        cursor.close()
        connection.close()