import cv2
import numpy as np
import base64
from openai import OpenAI
import math
import logging
from collections import Counter
from fastapi import APIRouter, File, UploadFile, FastAPI, Form, HTTPException, Body
from fastapi.responses import JSONResponse
import os
from dotenv import load_dotenv
import mysql.connector
from mysql.connector import Error
from google.cloud import firestore
import requests
from pydantic import BaseModel
from datetime import datetime
import time

# Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load environment variables and OpenAI API key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY is not set")
client = OpenAI(api_key=api_key)

# Initialize FastAPI app and router
app = FastAPI()
router_process_image = APIRouter()

# --- Database Functions ---
def get_db_connection():
    try:
        connection = mysql.connector.connect(
            host="127.0.0.1",
            user="fastapi_user",
            password="SecurePass123!",
            database="sensor_data"
        )
        return connection
    except Error as e:
        logger.error("Error while connecting to MySQL: %s", e)
        return None

def save_to_database(label, confidence, center, radius, image_path, user_id):
    connection = None
    try:
        connection = get_db_connection()
        if connection is None:
            logger.error("Database connection failed.")
            return False

        cursor = connection.cursor()
        # Create table if it doesn't exist (without image_path)
        create_table_query = """
            CREATE TABLE IF NOT EXISTS processed_images (
                id INT AUTO_INCREMENT PRIMARY KEY,
                label VARCHAR(255),
                confidence FLOAT,
                center_x INT,
                center_y INT,
                radius INT,
                image_path VARCHAR(500),
                user_id VARCHAR(64),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        cursor.execute(create_table_query)

        # Check if 'image_path' column exists; if not, add it.
        cursor.execute("SHOW COLUMNS FROM processed_images LIKE 'image_path'")
        result = cursor.fetchone()
        if not result:
            logger.info("Column 'image_path' not found in table. Altering table to add the column.")
            cursor.execute("ALTER TABLE processed_images ADD COLUMN image_path VARCHAR(500)")
        # Check if 'user_id' column exists; if not, add it.
        cursor.execute("SHOW COLUMNS FROM processed_images LIKE 'user_id'")
        result = cursor.fetchone()
        if not result:
            logger.info("Column 'user_id' not found in table. Altering table to add the column.")
            cursor.execute("ALTER TABLE processed_images ADD COLUMN user_id VARCHAR(64)")

        # Insert data into the table
        insert_query = """
            INSERT INTO processed_images (label, confidence, center_x, center_y, radius, image_path, user_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        data = (label, confidence, center["x"], center["y"], radius, image_path, user_id)
        cursor.execute(insert_query, data)
        connection.commit()
        logger.info("Data saved to database successfully.")
        return True
    except Error as e:
        logger.error("Error while saving to MySQL: %s", e)
        return False
    finally:
        if connection is not None and connection.is_connected():
            cursor.close()
            connection.close()

# --- GPT-based Label Prediction Function ---
def get_label_from_gpt(image_bytes: bytes) -> str:
    try:
        logger.debug("Converting image to Base64.")
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        image_url = f"data:image/jpeg;base64,{image_base64}"

        prompt = (
            "This is an image. "
            "Please provide ONE WORD (only one word) that best represents the main object in the image. "
            "Do not give any explanation. Do not add any extra words. Just one word."
        )

        logger.debug("Sending request to GPT-4o.")
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ]
                }
            ],
            max_tokens=10,
            temperature=0.0
        )

        label = response.choices[0].message.content.strip()
        logger.info(f"Label from GPT-4o: {label}")
        return label
    except Exception as e:
        logger.error(f"Failed to get label from GPT-4o: {e}")
        return "unknown"

# --- Helper Function to Draw Text with Background ---
def draw_text_with_background(
    image, text, position, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1,
    color=(0, 0, 255), thickness=2, bg_color=(255, 255, 255), padding=10, bg_offset=(0, 0)
):
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    rect_x1 = position[0] - padding + bg_offset[0]
    rect_y1 = position[1] - padding + bg_offset[1]
    rect_x2 = position[0] + text_width + padding + bg_offset[0]
    rect_y2 = position[1] + text_height + baseline + padding + bg_offset[1]
    cv2.rectangle(image, (rect_x1, rect_y1), (rect_x2, rect_y2), bg_color, -1)
    cv2.putText(image, text, position, font, font_scale, color, thickness)

# --- Helper Function to Detect Dominant (Largest Consistent) Circle ---
def detect_consistent_largest_circle(image, iterations=10):
    detected_circles = []
    image_height, image_width = image.shape[:2]

    for _ in range(iterations):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 2)
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,  # Adjust the dp value
            minDist=75,  # Adjust the minDist
            param1=100,  # Lower the high threshold for edge detection
            param2=100,  # Increase the center detection threshold
            minRadius=int(image_width * 0.2),  # Adjust minRadius
            maxRadius=int(image_width * 0.45)  # Adjust maxRadius
        )
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            detected_circles.extend(circles)

    if not detected_circles:
        return None

    radius_counter = Counter([c[2] for c in detected_circles])
    most_common_radius = radius_counter.most_common(1)[0][0]
    same_radius_circles = [c for c in detected_circles if c[2] == most_common_radius]
    return max(same_radius_circles, key=lambda c: c[2])

# Tambahkan import untuk Firestore dan asumsi fungsi helper
# from google.cloud import firestore
# from firebase_admin import storage

def get_processed_image_path_from_firestore(user_id):
    # Dummy: ganti dengan query Firestore sebenarnya
    # Misal: return hasil query Firestore berupa path file lokal hasil download dari storage
    # Contoh: return f"/tmp/annotated_{user_id}.png"
    return f"image_result/{user_id}/annotated_from_firestore.png"

# Fungsi untuk query Firestore berdasarkan user_id
def get_photo_url_from_firestore(user_id):
    db = firestore.Client()
    docs = db.collection('annotated_images')\
        .where('user_id', '==', user_id)\
        .order_by('created_at', direction=firestore.Query.DESCENDING)\
        .limit(1).stream()
    for doc in docs:
        data = doc.to_dict()
        return data.get('photo_url')
    return None

# Fungsi untuk download dan konversi ke base64
def get_base64_from_url(url):
    response = requests.get(url)
    if response.status_code == 200:
        return base64.b64encode(response.content).decode('utf-8')
    else:
        raise Exception(f"Failed to download image: {response.status_code}")

# --- FastAPI Endpoint for Processing Image ---
@router_process_image.post("/process-image/")
async def process_image(file: UploadFile = File(...), user_id: str = Form(...)):
    try:
        # Simpan file ke folder user dengan nama unik
        user_folder = f'./image_result/{user_id}'
        os.makedirs(user_folder, exist_ok=True)
        # Tambahkan timestamp ke nama file agar unik
        filename, ext = os.path.splitext(file.filename)
        unique_filename = f"{filename}_{int(time.time())}{ext}"
        file_path = os.path.join(user_folder, unique_filename)
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        # Labeling by GPT
        label = get_label_from_gpt(content)
        # Simpan ke database (tanpa deteksi, jadi field lain default/null)
        db_result = save_to_database(
            label=label,
            confidence=None,
            center={"x": None, "y": None},
            radius=None,
            image_path=file_path,
            user_id=user_id
        )
        if not db_result:
            return JSONResponse(content={
                "error": "Gagal menyimpan data ke database. Cek log backend untuk detail error."
            }, status_code=500)
        return JSONResponse(content={
            "label": label,
            "image_path": file_path,
            "message": "Gambar berhasil diupload, disimpan, dilabeli oleh GPT, dan dicatat di database."
        }, status_code=200)
    except Exception as e:
        logging.exception("Error while uploading and labeling image.")
        return JSONResponse(content={"error": str(e)}, status_code=500)

def save_user_image(image_data, user_id, filename):
    user_folder = f'./image_result/{user_id}'
    os.makedirs(user_folder, exist_ok=True)
    file_path = os.path.join(user_folder, filename)
    with open(file_path, 'wb') as f:
        f.write(image_data)
    return file_path

class ImageURLRequest(BaseModel):
    user_id: str
    photo_url: str
    created_at: datetime = None

@router_process_image.post("/save-image-url")
async def save_image_url(data: ImageURLRequest):
    # TODO: Simpan ke database jika diperlukan
    return {"status": "success", "message": "Image URL saved", "photo_url": data.photo_url}

class ProcessImageFromURLRequest(BaseModel):
    user_id: str
    photo_url: str

@router_process_image.post("/process-image-from-url")
async def process_image_from_url(data: ProcessImageFromURLRequest = Body(...)):
    try:
        # Download gambar dari URL
        response = requests.get(data.photo_url)
        if response.status_code != 200:
            return JSONResponse(content={"error": "Gagal mendownload gambar dari URL."}, status_code=400)
        image_bytes = response.content
        # Contoh: konversi ke base64
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        # TODO: Tambahkan proses lain sesuai kebutuhan (deteksi, anotasi, dsb)
        return JSONResponse(content={
            "annotated_image_base64": image_base64,
            "photo_url": data.photo_url,
            "message": "Gambar berhasil diproses dari URL."
        }, status_code=200)
    except Exception as e:
        logging.exception("Error while processing image from URL.")
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Model request untuk update label
class UpdateLabelRequest(BaseModel):
    user_id: str
    image_path: str
    new_label: str

# Fungsi untuk mendapatkan label terbaru berdasarkan ID (lebih reliable)
def get_latest_label_by_id(user_id: str):
    connection = get_db_connection()
    if not connection:
        logger.error("DB connection failed.")
        return None
    try:
        cursor = connection.cursor(dictionary=True)
        cursor.execute("""
            SELECT label
            FROM processed_images
            WHERE user_id = %s
            ORDER BY id DESC
            LIMIT 1
        """, (user_id,))
        row = cursor.fetchone()
        if row:
            return row["label"]
        return None
    except Exception as e:
        logger.error(f"DB Error: {e}")
        return None
    finally:
        cursor.close()
        connection.close()

# Fungsi untuk mendapatkan label terbaru berdasarkan timestamp (fallback)
def get_latest_label_by_timestamp(user_id: str):
    connection = get_db_connection()
    if not connection:
        logger.error("DB connection failed.")
        return None
    try:
        cursor = connection.cursor(dictionary=True)
        cursor.execute("""
            SELECT label
            FROM processed_images
            WHERE user_id = %s
            ORDER BY created_at DESC
            LIMIT 1
        """, (user_id,))
        row = cursor.fetchone()
        if row:
            return row["label"]
        return None
    except Exception as e:
        logger.error(f"DB Error: {e}")
        return None
    finally:
        cursor.close()
        connection.close()

# Fungsi utama yang mencoba kedua metode
def get_latest_label(user_id: str):
    # Coba dengan ID terlebih dahulu (lebih reliable)
    label_by_id = get_latest_label_by_id(user_id)
    if label_by_id:
        logger.info(f"Label found by ID: {label_by_id}")
        return label_by_id
    
    # Fallback ke timestamp jika tidak ada
    label_by_timestamp = get_latest_label_by_timestamp(user_id)
    if label_by_timestamp:
        logger.info(f"Label found by timestamp: {label_by_timestamp}")
        return label_by_timestamp
    
    logger.warning(f"No label found for user {user_id}")
    return None

@router_process_image.post("/overwrite-label/")
async def overwrite_label(data: UpdateLabelRequest):
    try:
        connection = get_db_connection()
        if connection is None:
            return JSONResponse(content={"error": "Gagal koneksi ke database."}, status_code=500)
        cursor = connection.cursor()
        
        # Update label dan timestamp
        logger.info(f"Updating label for user {data.user_id} from path {data.image_path} to '{data.new_label}'")
        update_query = """
            UPDATE processed_images
            SET label = %s, created_at = CURRENT_TIMESTAMP
            WHERE user_id = %s AND image_path = %s
        """
        cursor.execute(update_query, (data.new_label, data.user_id, data.image_path))
        logger.info(f"Label updated successfully. Rows affected: {cursor.rowcount}")
        connection.commit()
        rowcount = cursor.rowcount
        cursor.close()
        connection.close()
        if rowcount == 0:
            return JSONResponse(content={"error": "Data tidak ditemukan atau tidak ada yang diupdate."}, status_code=404)
        return {"status": "success", "message": "Label berhasil dioverwrite dan timestamp diupdate."}
    except Exception as e:
        logging.exception("Error saat overwrite label.")
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Endpoint untuk testing - mendapatkan label terbaru
@router_process_image.get("/get-latest-label/{user_id}")
async def get_latest_label_endpoint(user_id: str):
    try:
        label = get_latest_label(user_id)
        if label:
            return JSONResponse(content={
                "user_id": user_id,
                "latest_label": label,
                "status": "success"
            })
        else:
            return JSONResponse(content={
                "user_id": user_id,
                "latest_label": None,
                "status": "no_label_found"
            }, status_code=404)
    except Exception as e:
        logging.exception("Error getting latest label.")
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Register router with FastAPI app
app.include_router(router_process_image)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("script:app", host="0.0.0.0", port=8000, reload=True)
