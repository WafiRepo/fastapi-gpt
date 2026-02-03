import os
import io
import base64
import uuid
import json
import random
import re
import logging
import glob
from typing import List, Dict
from functools import lru_cache

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Headless backend for plotting
import matplotlib.pyplot as plt
from PyPDF2 import PdfReader

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from openai import OpenAI
import openai

from router_analyze_data import get_db_connection

# --------------------------------------------------------
# Initialize Router
# --------------------------------------------------------
routes_intermediate = APIRouter()

# --------------------------------------------------------
# Load env, init GPT
# --------------------------------------------------------
# Load .env from the same directory as this script
script_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(script_dir, '.env')
if os.path.exists(env_path):
    load_dotenv(dotenv_path=env_path)
else:
    # Fallback: try current directory
    load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

if not api_key:
    raise ValueError("OPENAI_API_KEY is not set")

# --------------------------------------------------------
# Logging Setup
# --------------------------------------------------------
logger = logging.getLogger("intermediate_question_logger")
logger.setLevel(logging.INFO)
os.makedirs("./log_file", exist_ok=True)

file_handler = logging.FileHandler("./log_file/intermediate_question.log")
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# --------------------------------------------------------
# Request Model
# --------------------------------------------------------
class QuestionRequest(BaseModel):
    language: str
    user_id: str

# --------------------------------------------------------
# Utility: Ensure Directory
# --------------------------------------------------------
def ensure_directory(dir_path: str):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

# --------------------------------------------------------
# PDF Processing and RAG Setup
# --------------------------------------------------------
def load_pdf_content(pdf_path: str) -> str:
    """Load and extract text from PDF file."""
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        logger.error(f"Error loading PDF: {e}")
        return ""

# Langchain RAG components removed - not actively used

# --------------------------------------------------------
# GPT Table Interpretation
# --------------------------------------------------------
def interpret_table_data(table_data: list, language: str) -> str:
    """
    Sends table data to GPT-4o for a short summary or interpretation.
    """
    # Convert table_data (list of dicts) to a short string
    data_preview = json.dumps(table_data[:28], indent=2)  # show first 5 rows

    prompt = f"""
    You are an expert data analyst.  Analyze the provided picture experimental data on angular velocity and centripetal acceleration. Identify how centripetal acceleration changes with angular velocity, detect periods of stability or fluctuation. Provide a summary of key trends and potential explanations for observed behaviors in the data
    round the numbers to 2 decimal places.
    Highlight the Steady phase detected of angular velocity (state the value), centripetal acceleration (state the value)   
    Maximum Angular Velocity. 
    Maximum Centripetal Acceleration.
    Behavior of Centripetal Acceleration with Angular Velocity. write in one paragraph consist 5 sentences, the important state the initial steady and latest steady data 
    interpret any key points in {language}:

    Table Data:
    {data_preview}
    """
    try:
        messages = [
            {"role": "system", "content": "You are an expert data analyst."},
            {"role": "user", "content": prompt},
        ]
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=300,
            temperature=0.6
        )
        interpretation = resp.choices[0].message.content.strip()
        logger.info(f"Table interpretation: {interpretation}")
        return interpretation
    except Exception as e:
        logger.error(f"Failed to interpret table data: {e}")
        return f"Failed to interpret table data: {e}"

# --------------------------------------------------------
# GPT Graph Interpretation
# --------------------------------------------------------
def interpret_graph_image(graph_path: str, language: str) -> str:
    """
    Reads a PNG from disk, base64-encodes it, sends it to GPT-4o for interpretation.
    """
    try:
        with open(graph_path, "rb") as f:
            image_data = f.read()
    except Exception as e:
        logger.error(f"Failed to read graph image from disk: {e}")
        return f"Failed reading graph image: {e}"

    image_base64 = base64.b64encode(image_data).decode("utf-8")

    prompt = f"""
    Please analyze this experimental data graph relevant to centripetal acceleration.
    You are an expert data analyst.  Analyze the provided experimental data on angular velocity and centripetal acceleration. Identify how centripetal acceleration changes with angular velocity, detect periods of stability or fluctuation. Provide a summary of key trends and potential explanations for observed behaviors in the data
    round the numbers to 2 decimal places.
    Steady periods detected.
    Fluctuation periods detected.
    Behavior of Centripetal Acceleration with Angular Velocity
    Provide your observations in {language}.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                        },
                    ],
                }
            ],
            max_tokens=300,
            temperature=0.7,
        )
        interpretation_text = response.choices[0].message.content.strip()
        # logger.info(f"Graph interpretation: {interpretation_text}")
        return interpretation_text
    except Exception as e:
        logger.error(f"Error interpreting graph with GPT-4o: {e}")
        return f"Failed to interpret graph: {e}"

# --------------------------------------------------------
# GPT Local Image Interpretation
# --------------------------------------------------------
def interpret_local_image(image_path: str, language: str) -> str:
    """
    Reads a local static image (e.g. processed_image_497_366_311.jpg),
    base64-encodes it, and sends it to GPT-4o for interpretation.
    
    """
    try:
        with open(image_path, "rb") as f:
            img_data = f.read()
    except Exception as e:
        logger.error(f"Failed to read local image {image_path}: {e}")
        return f"Failed to read local image: {e}"

    img_b64 = base64.b64encode(img_data).decode("utf-8")

    prompt = f"""
    The user has an image relevant to centripetal acceleration context.
    In {language}, please describe or interpret any notable features 
    or relevant physical aspects from the image:
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                        },
                    ],
                }
            ],
            max_tokens=300,
            temperature=0.7,
        )
        interpretation = response.choices[0].message.content.strip()
        logger.info(f"Local image interpretation: {interpretation}")
        return interpretation
    except Exception as e:
        logger.error(f"Failed to interpret local image with GPT-4o: {e}")
        return f"Failed to interpret local image: {e}"

# --------------------------------------------------------
# Execute Python Code => Return a single figure
# --------------------------------------------------------
def execute_python_code(code: str, language: str):
    local_vars = {}
    images_b64 = []
    saved_graph_path = None
    graph_interpretation = None

    try:
        logger.info("Executing Python code for graph generation")
        exec(code, {"plt": plt, "np": np, "pd": pd}, local_vars)
        time_list = local_vars.get('time_list', [])
        y_values = local_vars.get('y_values', [])
        selected_y_axis = local_vars.get('selected_y_axis')
        logger.info(f"Data retrieved - Time points: {len(time_list)}, Values: {len(y_values)}, Y-axis: {selected_y_axis}")
        if not time_list or not y_values:
            logger.error("No time_list or y_values found in executed code")
            return images_b64, saved_graph_path, graph_interpretation
        df_data = pd.DataFrame({'Time (s)': time_list, selected_y_axis: y_values})
        auto_phases = detect_phases(time_list, y_values)
        df_phases = pd.DataFrame(auto_phases, columns=['Start Time (s)', 'End Time (s)', 'Phase'])
        logger.info(f"Detected phases: {len(auto_phases)}")
        plt.figure(figsize=(12, 6))
        plt.plot(time_list, y_values, 'b-', label=selected_y_axis)
        colors = {'steady': '#FFFF99', 'increase': '#66CDAA', 'decrease': '#FF6347'}
        for start, end, phase in auto_phases:
            plt.axvspan(start, end, alpha=0.3, color=colors.get(phase, 'gray'), label=f'{phase} phase')
        plt.xlabel('Time (s)')
        plt.ylabel(selected_y_axis)
        plt.title(f'{selected_y_axis} vs Time with Detected Phases')
        plt.grid(True)
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=300, bbox_inches='tight')
        buf.seek(0)
        raw_b64 = base64.b64encode(buf.read()).decode("utf-8")
        data_url = "data:image/png;base64," + raw_b64
        images_b64.append(data_url)
        ensure_directory("./intermediate_graph")
        unique_id = str(uuid.uuid4())[:8]
        saved_graph_path = f"./intermediate_graph/graph_{unique_id}.png"
        with open(saved_graph_path, "wb") as out_file:
            out_file.write(base64.b64decode(raw_b64))
        logger.info(f"Graph saved to {saved_graph_path}")
        graph_interpretation = interpret_graph_image(saved_graph_path, language)
        plt.close('all')
        return images_b64, saved_graph_path, graph_interpretation
    except Exception as e:
        logger.error(f"Error in execute_python_code: {e}")
        plt.close('all')
        return images_b64, saved_graph_path, graph_interpretation

def detect_phases(time_data, y_data, window_size=10):
    phases = []
    data_range = np.max(y_data) - np.min(y_data)
    data_std = np.std(y_data)
    max_y = np.max(y_data)

    steady_threshold = data_std * 0.5
    increase_threshold = data_range * 0.1
    decrease_threshold = -data_range * 0.05

    logger.info(f"Adaptive thresholds - Steady: {steady_threshold:.3f}, Increase: {increase_threshold:.3f}, Decrease: {decrease_threshold:.3f}")

    n = len(time_data)
    for i in range(n - window_size):
        window_time = time_data[i:i + window_size]
        window_y = y_data[i:i + window_size]

        # Hindari steady di 10 data awal/akhir
        if i < 10 or i > n - window_size - 10:
            std_dev = np.inf
        else:
            std_dev = np.std(window_y)

        slope = (window_y[-1] - window_y[0]) / (window_time[-1] - window_time[0])
        mean_y = np.mean(window_y)

        # Steady hanya di puncak
        if std_dev < steady_threshold and mean_y > 0.7 * max_y:
            phase_label = 'steady'
        # Increase
        elif slope > increase_threshold:
            phase_label = 'increase'
        # Decrease, berlaku sampai akhir data (kecuali 10 awal/akhir)
        elif slope < decrease_threshold and i < n - window_size - 10:
            phase_label = 'decrease'
        # Decrease khusus window terakhir (agar penurunan akhir tetap decrease)
        elif slope < decrease_threshold and i >= n - window_size - 10:
            phase_label = 'decrease'
        else:
            continue

        phases.append((window_time[0], window_time[-1], phase_label))

    return phases

def get_latest_label(user_id: str):
    connection = get_db_connection()
    if not connection:
        raise HTTPException(status_code=500, detail="DB connection failed.")
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
        raise HTTPException(status_code=500, detail=f"DB Error: {e}")
    finally:
        cursor.close()
        connection.close()


def get_latest_radius(user_id: str):
    connection = get_db_connection()
    if not connection:
        raise HTTPException(status_code=500, detail="DB connection failed.")
    try:
        cursor = connection.cursor(dictionary=True)
        cursor.execute("""
            SELECT radius
            FROM calculated_radius
            WHERE user_id = %s
            ORDER BY timestamp DESC
            LIMIT 1
        """, (user_id,))
        row = cursor.fetchone()
        if row:
            return row["radius"]
        return None
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB Error: {e}")
    finally:
        cursor.close()
        connection.close()



def get_latest_buffer_data(buffer_name: str, user_id: str):
    connection = get_db_connection()
    if not connection:
        raise HTTPException(status_code=500, detail="DB connection failed.")
    try:
        cursor = connection.cursor(dictionary=True)
        cursor.execute("""
            SELECT timestamp, JSON_UNQUOTE(JSON_EXTRACT(data, '$.data')) AS data
            FROM data_buffer
            WHERE user_id = %s
              AND buffer_name = %s
            ORDER BY timestamp DESC
            LIMIT 1
        """, (user_id, buffer_name))
        row = cursor.fetchone()
        if row and row["data"]:
            row["data"] = json.loads(row["data"])
        return row
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB Error: {e}")
    finally:
        cursor.close()
        connection.close()

def get_latest_image_from_folder(folder_path):
    """Get the latest image from a folder."""
    try:
        image_files = glob.glob(os.path.join(folder_path, "*.jpg")) + glob.glob(os.path.join(folder_path, "*.png"))
        if image_files:
            sorted_files = sorted(image_files, key=os.path.getctime, reverse=True)
            return sorted_files[0]
        return None
    except Exception as e:
        logger.error(f"Error getting latest image: {e}")
        return None

def generate_intermediate_question(
    df,
    object_detected=None,
    language="en",
    latest_radius=None,
    user_id=None
):
    # 1) Check available columns
    logger.info(f"Available columns: {df.columns.tolist()}")
    y_axis_options = [
        col for col in ["Angular velocity (ω) rad/s", "Centripetal Acceleration (a) m/s^2"]
        if col in df.columns
    ]
    if not y_axis_options:
        raise HTTPException(
            status_code=500,
            detail="Neither Acceleration nor Angular velocity found in DataFrame."
        )

    # 2) Randomly pick one
    selected_y_axis = random.choice(y_axis_options)
    logger.info(f"Selected Y-axis column: {selected_y_axis}")

    # 3) Extract time & values
    if selected_y_axis not in df.columns:
        raise HTTPException(status_code=500, detail=f"{selected_y_axis} missing in DataFrame.")

    time_list = df["Time (s)"].tolist()
    y_values = df[selected_y_axis].tolist()

    # 4) Random highlight (optional)
    highlight_time = random.choice(time_list)
    highlight_index = time_list.index(highlight_time)
    highlight_value = y_values[highlight_index]

    # 5) Decide question types (including new combination types)
    question_types = [
        "text+table",
        "text+graph",
        "text+image",
        "text+table+image",
        "text+table+graph",
        "text+graph+image"
    ]
    selected_type = random.choice(question_types)
    logger.info(f"Selected question type: {selected_type}")

    # 6) Construct prefix
    prefix = f"Object detected: {object_detected}. " if object_detected else ""
    if latest_radius is not None:
        # Convert radius from meters to centimeters
        radius_cm = latest_radius
        prefix += f"Latest radius: {radius_cm:.2f} cm.\n"

    # Initialize placeholders
    table_interpretation = None
    graph_interpretation = None
    local_image_interpretation = None
    python_code = None
    graph_images = None
    saved_graph_path = None
    table_data = None
    local_image_base64 = None

    # ----------------------------------------------------------------
    # Few-Shot Prompting Setup
    # ----------------------------------------------------------------
    # 1. Extract all possible questions from the source PDF
    all_questions = extract_questions_from_pdf(pdf_path)
    
    # 2. Get a few random examples to guide the AI
    few_shot_examples = get_few_shot_examples(all_questions, num_examples=55)

    # ----------------------------------------------------------------
    # Pre-computation of assets based on question type
    # ----------------------------------------------------------------
    
    # Table Data and Interpretation
    table_img_base64 = None
    table_img_path = None
    if "table" in selected_type:
        # --- Tambahan: Deteksi phase dan tambahkan ke tabel ---
        phases = detect_phases(df["Time (s)"].tolist(), df[selected_y_axis].tolist())
        phase_col = [''] * len(df)
        for start, end, label in phases:
            for i, t in enumerate(df["Time (s)"]):
                if start <= t <= end:
                    phase_col[i] = label
        df["Phase"] = phase_col
        phase_colors = {'steady': '#FFFF99', 'increase': '#66CDAA', 'decrease': '#FF6347', '': 'white'}

        # table_data harus setelah kolom Phase ditambah
        df_dict = df.to_dict(orient='records')
        table_data = json.loads(json.dumps(df_dict, default=str))
        table_interpretation = interpret_table_data(table_data, language)

        # --- Plot tabel ke gambar dan simpan lokal + base64 dengan warna phase ---
        try:
            # Label dua baris
            col_labels = [
                col.replace('Centripetal Acceleration (a) (m/s^2)', 'Centripetal\nAcceleration (a) (m/s^2)')
                   .replace('Angular velocity (ω) rad/s', 'Angular\nvelocity (ω) rad/s')
                for col in df.columns
            ]
            # Hitung lebar kolom otomatis
            maxlens = [max([len(str(x)) for x in df[col]] + [len(lbl)]) for col, lbl in zip(df.columns, col_labels)]
            total = sum(maxlens)
            colwidths = [0.12 + 0.23 * (l/total) for l in maxlens]
            fig, ax = plt.subplots(figsize=(max(16, len(df.columns)*3), min(2+len(df)*0.6, 15)))
            ax.axis('off')
            tbl = ax.table(cellText=df.values, colLabels=col_labels, colWidths=colwidths, loc='center', cellLoc='center')
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(9)
            tbl.scale(1.2, 1.5)
            # Tidak perlu rotasi label kolom
            for i, phase in enumerate(df["Phase"].tolist()):
                color = phase_colors.get(phase, 'white')
                for j in range(len(df.columns)):
                    tbl[(i+1, j)].set_facecolor(color)
            plt.tight_layout(pad=3.0)
            table_img_dir = f"./table_images/{user_id}"
            os.makedirs(table_img_dir, exist_ok=True)
            table_img_path = os.path.join(table_img_dir, f"table_{uuid.uuid4().hex[:8]}.png")
            plt.savefig(table_img_path, bbox_inches='tight', dpi=200)
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=200)
            buf.seek(0)
            table_img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig)
        except Exception as e:
            logger.error(f"Gagal membuat gambar tabel: {e}")
            table_img_base64 = None
            table_img_path = None

    # Graph Data and Interpretation
    if "graph" in selected_type:
        python_code = f"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Data
selected_y_axis = "{selected_y_axis}"
time_list = {time_list}
y_values = {y_values}
"""
        graph_images, saved_graph_path, graph_interpretation = execute_python_code(python_code, language)
        logger.info(f"Graph Interpretation: {graph_interpretation}")

    # Local Image Interpretation
    if "image" in selected_type:
        user_folder = f"./image_result/{user_id}"
        local_img_path = get_latest_image_from_folder(user_folder)
        if local_img_path:
            local_image_interpretation = interpret_local_image(local_img_path, language)
            with open(local_img_path, "rb") as f:
                local_image_base64 = base64.b64encode(f.read()).decode("utf-8")
        else:
            local_image_interpretation = "No local image was found for interpretation."
            logger.warning(f"No image found in {user_folder} for user {user_id}")
            local_image_base64 = None

    # ----------------------------------------------------------------
    # Prompt generation based on question type
    # ----------------------------------------------------------------
    prompt = ""
    
    # Define language-specific notation rules to be more explicit
    if language == 'id':
        notation_rules = """
**PENTING - ATURAN NOTASI:**
- Untuk percepatan sentripetal, **WAJIB** gunakan notasi `(a)`.
- Untuk kecepatan sudut, gunakan notasi `(ω)`.
- Untuk jari-jari, gunakan notasi `(r)`.
- Setiap kali menyebut jari-jari, tambahkan notasi (r) dan selalu tuliskan satuan sentimeter (cm).
- **JANGAN PERNAH** menggunakan notasi `(a_c)`. Ini adalah konteks pembelajaran yang disederhanakan.
- Do NOT mention "Based on the interpretation" in the question.
"""
    else: # Default to English
        notation_rules = """
**IMPORTANT - NOTATION RULES:**
- For centripetal acceleration, you **MUST** use the notation `(a)`.
- For angular velocity, use the notation `(ω)`.
- Whenever you mention radius, append (r) and always state the unit as centimeter (cm).
- For radius, use the notation `(r)`.
- **NEVER** use the notation `(a_c)`. This is a simplified learning context.
- Do NOT mention "Based on the interpretation" in the question.
"""

    base_prompt_intro = f"""
You are an expert physics educator. Your task is to create a new, unique question based on the provided data.
The new question should follow the style, tone, and structure of the examples given below.

{prefix}
Here are some examples of good questions:
---
{few_shot_examples}
---

Now, using the following new data, create ONE new question.

Rules for the new question:
- The question must be based on the "New Data for Question" provided below.
- It must be in {language}.
- It should be simple, clear, and at an 'intermediate' difficulty level.
- Do not reveal formulas.
- Round all numerical values to 2 decimal places.
{notation_rules}
"""

    if selected_type == "text+table":
        prompt = f"""
{base_prompt_intro}
New Data for Question:
- Table Interpretation: {table_interpretation}

---
Based on the examples and new data, provide ONLY the single new question text as the final output.
"""

    elif selected_type == "text+graph":
        prompt = f"""
{base_prompt_intro}
New Data for Question:
- Graph Interpretation: {graph_interpretation}

---
Based on the examples and new data, provide ONLY the single new question text as the final output.
"""

    elif selected_type == "text+image":
        prompt = f"""
{base_prompt_intro}
New Data for Question:
- Image Interpretation: {local_image_interpretation}

---
Based on the examples and new data, provide ONLY the single new question text as the final output.
"""

    elif selected_type == "text+table+graph":
        prompt = f"""
{base_prompt_intro}
New Data for Question:
- Table Interpretation: {table_interpretation}
- Graph Interpretation: {graph_interpretation}

---
Based on the examples and new data, provide ONLY the single new question text as the final output.
"""

    elif selected_type == "text+table+image":
        prompt = f"""
{base_prompt_intro}
New Data for Question:
- Table Interpretation: {table_interpretation}
- Image Interpretation: {local_image_interpretation}

---
Based on the examples and new data, provide ONLY the single new question text as the final output.
"""

    elif selected_type == "text+graph+image":
        prompt = f"""
{base_prompt_intro}
New Data for Question:
- Graph Interpretation: {graph_interpretation}
- Image Interpretation: {local_image_interpretation}

---
Based on the examples and new data, provide ONLY the single new question text as the final output.
"""

    else:
        raise HTTPException(status_code=500, detail=f"Unsupported question type: {selected_type}")

    try:
        messages = [
            {"role": "system", "content": "You are an expert educator referencing data."},
            {"role": "user", "content": prompt}
        ]
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=400,
            temperature=0.7
        )
        question = resp.choices[0].message.content.strip()
        logger.info(f"Generated question: {question}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating final question: {str(e)}")

    return {
        "question": question,
        "python_code": python_code,
        "graph_images": graph_images,
        "saved_graph_path": saved_graph_path,
        "table_interpretation": table_interpretation,
        "graph_interpretation": graph_interpretation,
        "local_image_interpretation": local_image_interpretation,
        "local_image_base64": local_image_base64,
        "table_data": table_data,
        "table_img_base64": table_img_base64,
        "table_img_path": table_img_path
    }


@routes_intermediate.post("/intermediate-question")
async def intermediate_question(request: QuestionRequest):
    language = request.language
    user_id = request.user_id
    try:
        logger.info(f"Request received. language={language}, user_id={user_id}")

        label = get_latest_label(user_id)
        if not label:
            logger.error("No label in DB.")
            raise HTTPException(status_code=500, detail="No label found.")

        acc_data = get_latest_buffer_data('acc', user_id)
        gyr_data = get_latest_buffer_data('gyr', user_id)
        t_data = get_latest_buffer_data('t', user_id)
        if not acc_data or not gyr_data or not t_data:
            logger.error("Missing buffer data for acc/gyr/t.")
            raise HTTPException(status_code=500, detail="Data not found in DB.")

        radius_val = get_latest_radius(user_id)
        if radius_val is None:
            logger.error("No radius in DB.")
            raise HTTPException(status_code=500, detail="No radius found in DB.")

        # Build DataFrame
        df = pd.DataFrame({
            "Time (s)": t_data["data"],
            "Angular velocity (ω) rad/s": gyr_data["data"],
            "Centripetal Acceleration (a) (m/s^2)": acc_data["data"]
        })
        logger.info(f"DataFrame:\n{df}")

        # Generate question
        result = generate_intermediate_question(
            df=df,
            object_detected=label,
            language=language,
            latest_radius=radius_val,
            user_id=user_id
        )

        # Unpack
        question = result["question"]

        # logger.info(f"Final question: {question}")

        return JSONResponse(content={
            "question": question,
            "python_code": result["python_code"],
            "graph_images": result["graph_images"],
            "saved_graph_path": result["saved_graph_path"],
            "table_interpretation": result["table_interpretation"],
            "graph_interpretation": result["graph_interpretation"],
            "local_image_interpretation": result["local_image_interpretation"],
            "local_image_base64": result["local_image_base64"],
            "table_data": result["table_data"],
            "table_img_base64": result["table_img_base64"],
            "table_img_path": result["table_img_path"]
        })

    except Exception as e:
        logger.error(f"Internal Server Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error. Check server logs for details.")

@lru_cache(maxsize=None)
def extract_questions_from_pdf(pdf_path: str) -> List[str]:
    """
    Extracts all numbered questions from the PDF and returns them as a list.
    The result is cached to avoid reading the file on every request.
    """
    questions = []
    try:
        text = load_pdf_content(pdf_path)
        found_questions = re.findall(r'^\s*\d+\s+(.*)', text, re.MULTILINE)
        questions = [q.strip() for q in found_questions if q.strip()]
        logger.info(f"Extracted {len(questions)} questions from {pdf_path}")
    except Exception as e:
        logger.error(f"Failed to extract questions from PDF {pdf_path}: {e}")
    return questions

def get_few_shot_examples(all_questions: List[str], num_examples: int = 3) -> str:
    """
    Randomly selects a number of questions from the list to use as few-shot examples.
    """
    if not all_questions:
        return "No examples available."
    
    # Ensure we don't request more examples than available
    num_to_sample = min(len(all_questions), num_examples)
    
    selected_examples = random.sample(all_questions, num_to_sample)
    
    # Format the examples for the prompt
    formatted_examples = []
    for i, example in enumerate(selected_examples, 1):
        formatted_examples.append(f"Contoh {i}:\n{example}")
        
    return "\n\n".join(formatted_examples)
