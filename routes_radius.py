import os
from fastapi import APIRouter, HTTPException, Request, Body, Query
from pydantic import BaseModel
from openai import OpenAI
import openai
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import json
import numpy as np
import io
import base64
import logging
from fastapi.responses import JSONResponse
from mysql.connector import Error
from router_analyze_data import get_db_connection

routes_radius = APIRouter()

# Load environment variables
load_dotenv()

# Setup logging
logger = logging.getLogger(__name__)

def get_latest_buffer_data(buffer_name: str, user_id: str):
    connection = get_db_connection()
    if not connection:
        raise HTTPException(status_code=500, detail="Database connection failed")
    try:
        cursor = connection.cursor(dictionary=True)
        cursor.execute("""
            SELECT timestamp, JSON_UNQUOTE(JSON_EXTRACT(data, '$.data')) AS data
            FROM data_buffer
            WHERE user_id = %s AND buffer_name = %s
            ORDER BY timestamp DESC
            LIMIT 1
        """, (user_id, buffer_name))
        result = cursor.fetchone()
        if result and result['data']:
            result['data'] = json.loads(result['data'])
        return result
    except Error as e:
        raise HTTPException(status_code=500, detail=f"Error while querying MySQL: {str(e)}")
    finally:
        cursor.close()
        connection.close()

@routes_radius.get("/calculate-radius-auto/")
async def calculate_radius_auto(user_id: str = Query(..., description="User ID")):
    logger.info(f"Calculating radius for user_id: {user_id}")
    
    acc_data = get_latest_buffer_data("acc", user_id)
    gyr_squared_data = get_latest_buffer_data("gyr_squared", user_id)
    if not acc_data or not gyr_squared_data:
        raise HTTPException(status_code=404, detail="Data buffer tidak ditemukan untuk user ini")
    
    acc_values = acc_data['data']
    gyr_squared_values = gyr_squared_data['data']
    
    # Log data lengths for debugging
    logger.info(f"Acc data length: {len(acc_values)}, Gyr squared data length: {len(gyr_squared_values)}")
    
    acc_values_filtered = acc_values[5:-5]
    gyr_squared_values_filtered = gyr_squared_values[5:-5]
    
    # Check for empty or invalid data
    if len(acc_values_filtered) == 0 or len(gyr_squared_values_filtered) == 0:
        raise HTTPException(status_code=400, detail="Insufficient data after filtering. Need at least 10 data points.")
    
    # Check for NaN values in input data
    if np.any(np.isnan(acc_values_filtered)) or np.any(np.isnan(gyr_squared_values_filtered)):
        logger.warning(f"NaN values detected in sensor data for user {user_id}")
        raise HTTPException(status_code=400, detail="Invalid sensor data detected (NaN values). Please check your sensors.")
    
    avg_acc = np.mean(acc_values_filtered)
    avg_gyr_squared = np.mean(gyr_squared_values_filtered)
    
    logger.info(f"Average acc: {avg_acc}, Average gyr_squared: {avg_gyr_squared}")
    
    if avg_gyr_squared == 0:
        raise HTTPException(status_code=400, detail="Gyroscope squared average is zero, cannot divide by zero.")
    
    radius = avg_acc / avg_gyr_squared
    radius = radius * 100  # konversi meter ke centimeter
    radius = round(radius, 2)
    
    logger.info(f"Calculated radius: {radius}")
    
    # Validasi nilai radius sebelum menyimpan ke database
    if np.isnan(radius) or np.isinf(radius):
        logger.error(f"Invalid radius calculated: {radius} for user {user_id}")
        raise HTTPException(status_code=400, detail="Invalid radius value calculated (NaN or Infinity). Please check your sensor data.")
    
    # Additional validation for reasonable radius values
    if radius <= 0 or radius > 1000:  # Assuming reasonable range 0-1000 cm
        logger.warning(f"Unusual radius value: {radius} cm for user {user_id}")
    
    connection = get_db_connection()
    if not connection:
        raise HTTPException(status_code=500, detail="Database connection failed")
    
    try:
        cursor = connection.cursor()
        cursor.execute("""
            INSERT INTO calculated_radius (buffer_name_acc, buffer_name_gyr, radius, user_id)
            VALUES (%s, %s, %s, %s)
        """, ("acc", "gyr_squared", radius, user_id))
        connection.commit()
        logger.info(f"Successfully saved radius {radius} for user {user_id}")
        return JSONResponse(content={"radius": radius}, status_code=200)
    except Error as e:
        logger.error(f"Database error for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        if connection and connection.is_connected():
            cursor.close()
            connection.close()
