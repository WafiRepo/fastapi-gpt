import os
import io
import base64
import uuid
import json
import random
import logging
import re
import traceback
import glob
from typing import List, Dict
from functools import lru_cache

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Headless backend for plotting on servers
import matplotlib.pyplot as plt
from PyPDF2 import PdfReader

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from openai import OpenAI
import openai

from router_analyze_data import get_db_connection

routes_advance = APIRouter()

# -------------------------------------------------------------------
# Load environment, init GPT
# -------------------------------------------------------------------
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

if not api_key:
    raise ValueError("OPENAI_API_KEY is not set")

# -------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------
logger = logging.getLogger("advanced_question_logger")
logger.setLevel(logging.INFO)
os.makedirs("./log_file", exist_ok=True)

file_handler = logging.FileHandler("./log_file/advanced_question.log")
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# -------------------------------------------------------------------
# PDF Processing and Few-Shot Example Extraction
# -------------------------------------------------------------------
@lru_cache(maxsize=None)
def extract_questions_from_pdf(pdf_path: str) -> List[str]:
    """
    Extracts questions from a PDF file. It assumes questions are numbered
    and start with a number. This is a simplified parser.
    """
    try:
        with open(pdf_path, "rb") as f:
            reader = PdfReader(f)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
        
        # Regex to find numbered questions (e.g., "1. ...", "2. ...")
        # This regex looks for a number, a dot, and then captures the text until the next number.
        questions = re.findall(r'\d+\.\s(.*?)(?=\n\d+\.|\Z)', text, re.DOTALL)
        
        # Clean up whitespace dan biarkan multi-baris
        cleaned_questions = [q.strip() for q in questions]
        
        logger.info(f"Extracted {len(cleaned_questions)} questions from {pdf_path}")
        return cleaned_questions
    except FileNotFoundError:
        logger.error(f"PDF file not found at {pdf_path}")
        return []
    except Exception as e:
        logger.error(f"Failed to extract questions from PDF: {e}")
        return []

def get_few_shot_examples(all_questions: List[str], num_examples: int = 55) -> str:
    """
    Selects a few random questions from the list and formats them
    as a string for the prompt.
    """
    if not all_questions:
        return "No example questions available."
    
    # Ensure we don't request more examples than available
    num_to_sample = min(num_examples, len(all_questions))
    
    # Select random questions
    selected_examples = random.sample(all_questions, num_to_sample)
    
    # Format them into a numbered list string
    return "\n".join(f"{i+1}. {q}" for i, q in enumerate(selected_examples))

def detect_phases(time_data, y_data, window_size=10):
    phases = []
    if len(y_data) < window_size:
        return []

    data_range = np.max(y_data) - np.min(y_data)
    data_std = np.std(y_data)
    max_y = np.max(y_data)

    # Return empty if data range is zero to avoid division by zero or meaningless thresholds
    if data_range == 0:
        return []

    steady_threshold = data_std * 0.5
    increase_threshold = data_range * 0.1
    decrease_threshold = -data_range * 0.05

    logger.info(f"Adaptive thresholds - Steady: {steady_threshold:.3f}, Increase: {increase_threshold:.3f}, Decrease: {decrease_threshold:.3f}")

    n = len(time_data)
    for i in range(n - window_size):
        window_time = time_data[i:i + window_size]
        window_y = y_data[i:i + window_size]

        # Hindari steady di 10 data awal/akhir
        is_edge_data = i < 10 or i > n - window_size - 10
        std_dev = np.std(window_y)

        # Perhitungan slope yang aman
        time_diff = window_time[-1] - window_time[0]
        if time_diff == 0:
            continue
        slope = (window_y[-1] - window_y[0]) / time_diff
        
        mean_y = np.mean(window_y)

        phase_label = None
        # Steady hanya di puncak dan bukan di awal/akhir data
        if not is_edge_data and std_dev < steady_threshold and mean_y > 0.7 * max_y:
            phase_label = 'steady'
        elif slope > increase_threshold:
            phase_label = 'increase'
        elif slope < decrease_threshold:
            phase_label = 'decrease'
        
        if phase_label:
            phases.append((window_time[0], window_time[-1], phase_label))

    return phases

# -------------------------------------------------------------------
# Request Model
# -------------------------------------------------------------------
class QuestionRequest(BaseModel):
    language: str
    user_id: str

# -------------------------------------------------------------------
# Utility Functions
# -------------------------------------------------------------------
def ensure_directory(dir_path: str):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def execute_python_code(code: str):
    """
    Executes the given Python code (usually for plotting),
    returns a list of base64-encoded images, plus a list of file paths
    where the figures were saved.
    """
    execution_scope = {"plt": plt, "np": np}
    images_b64 = []
    saved_paths = []

    try:
        exec(code, execution_scope)

        fig_nums = plt.get_fignums()
        if not fig_nums:
            return images_b64, saved_paths  # No figures => return empty lists

        ensure_directory("./advance_graph")

        for fig_id in fig_nums:
            fig = plt.figure(fig_id)
            buf = io.BytesIO()
            fig.savefig(buf, format="png")
            buf.seek(0)

            img_base64 = base64.b64encode(buf.read()).decode("utf-8")
            images_b64.append(img_base64)

            unique_id = str(uuid.uuid4())[:8]
            saved_graph_path = f"./advance_graph/graph_{unique_id}.png"
            with open(saved_graph_path, "wb") as f:
                f.write(base64.b64decode(img_base64))

            saved_paths.append(saved_graph_path)
            plt.close(fig)

        return images_b64, saved_paths
    except Exception as e:
        error_message = f"Error in execute_python_code: {str(e)}"
        logger.error(f"{error_message}\n{traceback.format_exc()}")
        return [error_message], []

# -------------------------------------------------------------------
# Updated Table Interpretation Function
# -------------------------------------------------------------------
def interpret_table_data(df1: pd.DataFrame, df2: pd.DataFrame, language: str) -> str:
    data1 = df1.to_dict(orient="records")
    data2 = df2.to_dict(orient="records")

    data_preview_1 = json.dumps(data1[:28], indent=2)
    data_preview_2 = json.dumps(data2[:28], indent=2)

    prompt = f"""
You are an expert data analyst. The following are two sets of experimental data on angular velocity and centripetal Centripetal Acceleration. 
Please analyze **all** the data to identify key patterns, trends, and comparisons:

1. How centripetal Centripetal Acceleration (a) changes with angular velocity (ω).
2. Major differences in Centripetal Acceleration patterns or angular velocity patterns.
3. Periods of stability or fluctuation in both experiments.
4. Maximum and minimum values of (a) and (ω).
5. Potential explanations for observed behaviors.
6. Round any numerical examples to 2 decimal places.

Provide a clear, detailed paragraph summarizing your findings in {language}.

**Dataset 1 (preview):**
{data_preview_1}

**Dataset 2 (preview):**
{data_preview_2}
"""
    try:
        messages = [
            {"role": "system", "content": "You are an expert data analyst."},
            {"role": "user", "content": prompt}
        ]
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=500,
            temperature=0.6
        )
        interpretation = resp.choices[0].message.content.strip()
        logger.info(f"Table Interpretation: {interpretation}")
        return interpretation
    except Exception as e:
        logger.error(f"Failed to interpret table data: {e}")
        return f"Failed to interpret table data: {e}"

# -------------------------------------------------------------------
# Graph/Image Interpretation Functions
# -------------------------------------------------------------------
def interpret_comparison_graph(graph_b64: str, language: str) -> str:
    prompt = f"""
    Please analyze 2 experimental data graph relevant to centripetal Centripetal Acceleration.
    You are an expert data analyst. Analyze the provided experimental data on angular velocity and centripetal Centripetal Acceleration. 
    Identify how centripetal Centripetal Acceleration changes with angular velocity, detect periods of increase, steady, decrease. 
    Provide a summary of key trends and potential explanations for observed behaviors in the data.
    Round the numbers to 2 decimal places.
    Provide your observations in {language}.
    """
    try:
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{graph_b64}"}}
                ]
            }
        ]
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=msgs,
            max_tokens=300,
            temperature=0.7
        )
        interpretation = resp.choices[0].message.content.strip()
        logger.info(f"Graph Interpretation: {interpretation}")
        return interpretation
    except Exception as e:
        logger.error(f"Failed to interpret comparison graph: {e}")
        return f"Failed to interpret graph: {e}"

def interpret_local_image(image_path: str, language: str) -> str:
    try:
        with open(image_path, "rb") as f:
            img_data = f.read()
    except Exception as e:
        logger.error(f"Failed to read local image {image_path}: {e}")
        return f"Failed to read local image: {e}"

    img_b64 = base64.b64encode(img_data).decode("utf-8")
    prompt = f"""
    Reads a local static image (e.g. processed_image_497_366_311.jpg),
    base64-encodes it, and sends it to GPT-4o for interpretation.
    Provide your interpretation in {language}, focusing on any relevant physical or experimental context 
    (e.g., centripetal Centripetal Acceleration (a), angular velocity (ω), radius (r)).
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
                ]
            }],
            max_tokens=300,
            temperature=0.7
        )
        interpretation = response.choices[0].message.content.strip()
        logger.info(f"Local Image Interpretation: {interpretation}")
        return interpretation
    except Exception as e:
        logger.error(f"Failed to interpret local image with GPT-4o: {e}")
        return f"Failed to interpret local image: {e}"

def get_latest_image_from_folder(folder_path):
    """Get the latest image from a folder."""
    try:
        image_files = glob.glob(os.path.join(folder_path, "*.jpg")) + glob.glob(os.path.join(folder_path, "*.png"))
        if image_files:
            return max(image_files, key=os.path.getctime)
        return None
    except Exception as e:
        logger.error(f"Error getting latest image: {e}")
        return None

def get_latest_two_images_from_folder(folder_path):
    """Get the two latest images from a folder."""
    try:
        image_files = glob.glob(os.path.join(folder_path, "*.jpg")) + glob.glob(os.path.join(folder_path, "*.png"))
        if len(image_files) >= 2:
            # Sort by creation time and get the 2 latest
            sorted_files = sorted(image_files, key=os.path.getctime, reverse=True)
            return sorted_files[:2]
        elif len(image_files) == 1:
            # If only 1 image, return it twice or handle as needed
            return [image_files[0], None]
        else:
            return [None, None]
    except Exception as e:
        logger.error(f"Error getting latest two images: {e}")
        return [None, None]

# -------------------------------------------------------------------
# DB Fetch Logic
# -------------------------------------------------------------------
def get_latest_label(user_id: str):
    connection = get_db_connection()
    if not connection:
        raise HTTPException(status_code=500, detail="Database connection failed")
    try:
        cursor = connection.cursor(dictionary=True)
        cursor.execute("""
            SELECT label 
            FROM processed_images
            WHERE user_id = %s
            ORDER BY created_at DESC
            LIMIT 1
        """, (user_id,))
        result = cursor.fetchone()
        if result:
            return result['label']
        return None
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error while querying MySQL: {str(e)}")
    finally:
        cursor.close()
        connection.close()

def get_radius_by_id(radius_id: str):
    connection = get_db_connection()
    if not connection:
        raise HTTPException(status_code=500, detail="Database connection failed")
    try:
        cursor = connection.cursor(dictionary=True)
        cursor.execute("""
            SELECT radius
            FROM calculated_radius
            WHERE id = %s
            ORDER BY timestamp DESC
            LIMIT 1
        """, (radius_id,))
        row = cursor.fetchone()
        if row:
            return row["radius"]
        return None
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB Error: {e}")
    finally:
        cursor.close()
        connection.close()

def get_latest_two_buffer_data(buffer_name: str, user_id: str):
    connection = get_db_connection()
    if not connection:
        raise HTTPException(status_code=500, detail="Database connection failed")
    try:
        cursor = connection.cursor(dictionary=True)
        cursor.execute("""
            SELECT timestamp, JSON_UNQUOTE(JSON_EXTRACT(data, '$.data')) AS data
            FROM data_buffer
            WHERE user_id = %s
              AND buffer_name = %s
            ORDER BY timestamp DESC
            LIMIT 2
        """, (user_id, buffer_name))
        rows = cursor.fetchall()
        for row in rows:
            if row and row['data']:
                row['data'] = json.loads(row['data'])
        return rows
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error while querying MySQL: {e}")
    finally:
        cursor.close()
        connection.close()

def fetch_and_process_latest_data(user_id: str):
    # Ambil dua data buffer terbaru untuk masing-masing buffer_name
    acc_rows = get_latest_two_buffer_data("acc", user_id)
    gyr_rows = get_latest_two_buffer_data("gyr", user_id)
    t_rows   = get_latest_two_buffer_data("t", user_id)

    if not (acc_rows and gyr_rows and t_rows) or len(acc_rows) < 2 or len(gyr_rows) < 2 or len(t_rows) < 2:
        raise HTTPException(status_code=500, detail="Kurang dari dua data buffer ditemukan untuk user ini")

    acc_data = [acc_rows[0]['data'], acc_rows[1]['data']]
    gyr_data = [gyr_rows[0]['data'], gyr_rows[1]['data']]
    t_data   = [t_rows[0]['data'],   t_rows[1]['data']]

    # Validasi panjang data
    if len(acc_data[0]) != len(t_data[0]) or len(gyr_data[0]) != len(t_data[0]):
        raise ValueError("Mismatch in length of accelerometer/gyroscope data vs. time data (Experiment 1).")
    if len(acc_data[1]) != len(t_data[1]) or len(gyr_data[1]) != len(t_data[1]):
        raise ValueError("Mismatch in length of accelerometer/gyroscope data vs. time data (Experiment 2).")

    return acc_data, gyr_data, t_data

def get_latest_two_radius(user_id: str):
    connection = get_db_connection()
    if not connection:
        raise HTTPException(status_code=500, detail="Database connection failed")
    try:
        cursor = connection.cursor(dictionary=True)
        cursor.execute("""
            SELECT radius
            FROM calculated_radius
            WHERE user_id = %s
            ORDER BY timestamp DESC
            LIMIT 2
        """, (user_id,))
        rows = cursor.fetchall()
        if rows and len(rows) == 2:
            return rows[0]["radius"], rows[1]["radius"]
        else:
            return None, None
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB Error: {e}")
    finally:
        cursor.close()
        connection.close()

def get_latest_two_labels(user_id: str):
    connection = get_db_connection()
    if not connection:
        raise HTTPException(status_code=500, detail="Database connection failed")
    try:
        cursor = connection.cursor(dictionary=True)
        cursor.execute("""
            SELECT label 
            FROM processed_images
            WHERE user_id = %s
            ORDER BY created_at DESC
            LIMIT 2
        """, (user_id,))
        results = cursor.fetchall()
        labels = [row['label'] for row in results if row and row['label']]
        return labels if labels else [None, None]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error while querying MySQL: {str(e)}")
    finally:
        cursor.close()
        connection.close()

def generate_advanced_question(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    label1=None,
    label2=None,
    language: str = "en",
    latest_radius=None,
    second_latest_radius=None,
    user_id: str = None
):
    df1["Time (s)"] = df1["Time (s)"].round(0).astype(int)
    df2["Time (s)"] = df2["Time (s)"].round(0).astype(int)

    # You can add more columns if needed
    possible_cols = ["Centripetal Acceleration (a) m/s^2", "Angular velocity (ω) rad/s"] #Angular velocity (ω) rad/s / Centripetal Acceleration (a) m/s^2
    selected_col = random.choice(possible_cols)

    if selected_col not in df1.columns or selected_col not in df2.columns:
        raise HTTPException(status_code=500, detail=f"{selected_col} missing in df1 or df2")

    time1 = df1["Time (s)"].tolist()
    y1 = df1[selected_col].tolist()
    time2 = df2["Time (s)"].tolist()
    y2 = df2[selected_col].tolist()

    # ----------------------------------------------------------------
    # Few-Shot Prompting Setup
    # ----------------------------------------------------------------
    # 1. Extract all possible questions from the source PDF
    all_questions = extract_questions_from_pdf("docs/advanced.pdf")

    # 2. Get a few random examples to guide the AI
    few_shot_examples = get_few_shot_examples(all_questions, num_examples=30)

    # --- Find stable periods to pass to the prompt ---
    phases1 = detect_phases(time1, y1)
    phases2 = detect_phases(time2, y2)

    def get_longest_steady_phase(phases):
        steady_phases = [p for p in phases if p[2] == 'steady']
        if not steady_phases:
            return None
        # Find the longest steady phase segment
        longest_phase = max(steady_phases, key=lambda p: p[1] - p[0])
        return (longest_phase[0], longest_phase[1])

    steady_range_1 = get_longest_steady_phase(phases1)
    steady_range_2 = get_longest_steady_phase(phases2)

    steady_info_prompt = "\nIMPORTANT: When referring to a stable or steady period, you must use the following time ranges:"
    if steady_range_1:
        steady_info_prompt += f"\n- For Experiment 1, the steady period is from {steady_range_1[0]:.0f} to {steady_range_1[1]:.0f} seconds."
    else:
        steady_info_prompt += "\n- No clear steady period was detected for Experiment 1."

    if steady_range_2:
        steady_info_prompt += f"\n- For Experiment 2, the steady period is from {steady_range_2[0]:.0f} to {steady_range_2[1]:.0f} seconds."
    else:
        steady_info_prompt += "\n- No clear steady period was detected for Experiment 2."


    # ----------------------------------------------------------------
    # NOW we have 6 question types to choose from:
    # ----------------------------------------------------------------
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

    # ----------------------------------------------------------------
    # Build prefix
    # ----------------------------------------------------------------
    prefix = ""
    if label1 and label2 and label2 != label1:
        prefix += f"Object detected in both experiments: 1) {label1}, 2) {label2}.\n"
    elif label1:
        prefix += f"Object detected in the experiment: {label1}.\n"
    if latest_radius is not None and second_latest_radius is not None:
        radius1_cm = latest_radius * 100
        radius2_cm = second_latest_radius * 100
        prefix += f"Experiment 1 has a radius of {radius1_cm:.2f} cm, and Experiment 2 has a radius of {radius2_cm:.2f} cm.\n"

    # Initialize placeholders
    table_interpretation = None
    graph_interpretation = None
    image1_interpretation = None
    image2_interpretation = None
    saved_graph_path = None
    python_code = None
    graph_images = None
    table_img_base64_1 = None
    table_img_path_1 = None
    table_img_base64_2 = None
    table_img_path_2 = None
    image1_base64 = None
    image2_base64 = None

    # Siapkan string label untuk prompt
    if label1 and label2 and label2 != label1:
        label_info = f"The detected objects in the two images are: 1) {label1}, 2) {label2}."
    elif label1:
        label_info = f"The detected object in the image is: {label1}."
    else:
        label_info = "No detected object label available."

    # We define the phase detection logic as a string to be injected into the plot script.
    # This makes the script self-contained and avoids scope issues with `exec`.
    # The logger call is removed from this version.
    detect_phases_code = """
def detect_phases(time_data, y_data, window_size=10):
    phases = []
    if len(y_data) < window_size:
        return []

    # Using np which is imported in the main script body
    data_range = np.max(y_data) - np.min(y_data)
    data_std = np.std(y_data)
    max_y = np.max(y_data)

    if data_range == 0:
        return []

    steady_threshold = data_std * 0.5
    increase_threshold = data_range * 0.1
    decrease_threshold = -data_range * 0.05

    n = len(time_data)
    for i in range(n - window_size):
        window_time = time_data[i:i + window_size]
        window_y = y_data[i:i + window_size]

        is_edge_data = i < 10 or i > n - window_size - 10
        std_dev = np.std(window_y)

        time_diff = window_time[-1] - window_time[0]
        if time_diff == 0:
            continue
        slope = (window_y[-1] - window_y[0]) / time_diff
        
        mean_y = np.mean(window_y)

        phase_label = None
        if not is_edge_data and std_dev < steady_threshold and mean_y > 0.7 * max_y:
            phase_label = 'steady'
        elif slope > increase_threshold:
            phase_label = 'increase'
        elif slope < decrease_threshold:
            phase_label = 'decrease'
        
        if phase_label:
            phases.append((window_time[0], window_time[-1], phase_label))

    return phases
"""

    # Generate graph assets for ALL question types containing 'graph'
    if "graph" in selected_type:
        python_code = f"""
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

{detect_phases_code}

def plot_with_phase_annotation(time1, vals1, time2, vals2, var_label):
    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f'{{var_label}} vs Time with Detected Phases', fontsize=16)
    fig.supylabel(var_label, fontsize=14)

    phase_colors = {{
        "increase": "#66CDAA",
        "steady": "#FFFF99",
        "decrease": "#FF6347"
    }}
    line_color = 'blue'

    # Create proxy artists for phase legend
    phase_patches = [
        mpatches.Patch(color=phase_colors['increase'], label='Increase', alpha=0.6),
        mpatches.Patch(color=phase_colors['steady'], label='Steady', alpha=0.6),
        mpatches.Patch(color=phase_colors['decrease'], label='Decrease', alpha=0.6)
    ]

    # --- Experiment 1 Subplot ---
    axs[0].plot(time1, vals1, color=line_color, label="Experiment 1")
    axs[0].set_title("Experiment 1")
    axs[0].grid(True, linestyle='--', alpha=0.7)
    
    line_legend_1 = axs[0].legend(loc='upper left')
    axs[0].add_artist(line_legend_1)
    axs[0].legend(handles=phase_patches, loc='upper right')
    
    phases1 = detect_phases(time1, vals1)
    for start, end, label in phases1:
        axs[0].axvspan(start, end, color=phase_colors.get(label, 'grey'), alpha=0.5, ec='none')

    # --- Experiment 2 Subplot ---
    axs[1].plot(time2, vals2, color=line_color, label="Experiment 2")
    axs[1].set_title("Experiment 2")
    axs[1].set_xlabel("Time (s)")
    axs[1].grid(True, linestyle='--', alpha=0.7)
    
    line_legend_2 = axs[1].legend(loc='upper left')
    axs[1].add_artist(line_legend_2)
    axs[1].legend(handles=phase_patches, loc='upper right')

    phases2 = detect_phases(time2, vals2)
    for start, end, label in phases2:
        axs[1].axvspan(start, end, color=phase_colors.get(label, 'grey'), alpha=0.5, ec='none')

    # Common styling
    for ax in axs.flat:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.yaxis.label.set_size(14)
        ax.xaxis.label.set_size(14)
        ax.title.set_size(16)

    # Set y-limits individually
    if vals1:
        axs[0].set_ylim([min(vals1) - 1, max(vals1) + 1])
    if vals2:
        axs[1].set_ylim([min(vals2) - 1, max(vals2) + 1])
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

# --- Data passed to the script ---
time1 = {time1}
vals1 = {y1}
time2 = {time2}
vals2 = {y2}

plot_with_phase_annotation(
    time1, vals1,
    time2, vals2,
    var_label="{selected_col}"
)
"""
        images_b64, paths = execute_python_code(python_code)
        if images_b64 and isinstance(images_b64, list) and len(images_b64) > 0 and not images_b64[0].startswith("Error"):
            graph_images = images_b64
            saved_graph_path = paths[0] if paths else None
            graph_interpretation = interpret_comparison_graph(images_b64[0], language)
        else:
            error_message = images_b64[0] if (images_b64 and len(images_b64) > 0) else "Graph execution failed."
            graph_interpretation = f"No graph generated due to an error: {error_message}"
            logger.error(f"Graph generation failed: {error_message}")
    
    # Generate image assets for ALL question types containing 'image'
    if "image" in selected_type:
        user_folder = f"./image_result/{user_id}"
        image_paths = get_latest_two_images_from_folder(user_folder)
        image_path1, image_path2 = image_paths
        
        # Process first image
        if image_path1:
            image1_interpretation = interpret_local_image(image_path1, language)
            with open(image_path1, "rb") as f:
                image1_base64 = base64.b64encode(f.read()).decode("utf-8")
        else:
            image1_interpretation = "No first image found."
            image1_base64 = None
            
        # Process second image
        if image_path2:
            image2_interpretation = interpret_local_image(image_path2, language)
            with open(image_path2, "rb") as f:
                image2_base64 = base64.b64encode(f.read()).decode("utf-8")
        else:
            image2_interpretation = "No second image found."
            image2_base64 = None

    phase_colors = {'steady': '#FFFF99', 'increase': '#66CDAA', 'decrease': '#FF6347', '': 'white'}

    # Plot tabel untuk SEMUA question type yang mengandung 'table'
    if "table" in selected_type:
        # Untuk advanced, ada dua tabel (df1 dan df2)
        df1_dict = df1.to_dict(orient='records')
        df2_dict = df2.to_dict(orient='records')
        table_data = {'experiment_1': json.loads(json.dumps(df1_dict, default=str)), 'experiment_2': json.loads(json.dumps(df2_dict, default=str))}
        table_interpretation = interpret_table_data(df1, df2, language)

        # --- Tambahan: Deteksi phase dan tambahkan ke tabel untuk kedua eksperimen ---
        phases1 = detect_phases(df1["Time (s)"].tolist(), df1[selected_col].tolist())
        phase_col1 = [''] * len(df1)
        for start, end, label in phases1:
            for i, t in enumerate(df1["Time (s)"]):
                if start <= t <= end:
                    phase_col1[i] = label
        df1["Phase"] = phase_col1
        phases2 = detect_phases(df2["Time (s)"].tolist(), df2[selected_col].tolist())
        phase_col2 = [''] * len(df2)
        for start, end, label in phases2:
            for i, t in enumerate(df2["Time (s)"]):
                if start <= t <= end:
                    phase_col2[i] = label
        df2["Phase"] = phase_col2

        # --- Plot tabel ke gambar dan simpan lokal + base64 dengan warna phase (untuk dua eksperimen) ---
        try:
            # Label dua baris
            col_labels1 = [
                col.replace('Centripetal Acceleration (a) m/s^2', 'Centripetal\nAcceleration (a) m/s^2')
                   .replace('Angular velocity (ω) rad/s', 'Angular\nvelocity (ω) rad/s')
                for col in df1.columns
            ]
            # Hitung lebar kolom otomatis
            maxlens1 = [max([len(str(x)) for x in df1[col]] + [len(lbl)]) for col, lbl in zip(df1.columns, col_labels1)]
            total = sum(maxlens1)
            colwidths1 = [0.12 + 0.23 * (l/total) for l in maxlens1]  # min width 0.12, max 0.35
            fig1, ax1 = plt.subplots(figsize=(max(16, len(df1.columns)*3), min(2+len(df1)*0.6, 15)))
            ax1.axis('off')
            tbl1 = ax1.table(cellText=df1.values, colLabels=col_labels1, colWidths=colwidths1, loc='center', cellLoc='center')
            tbl1.auto_set_font_size(False)
            tbl1.set_fontsize(9)
            tbl1.scale(1.2, 1.5)
            # Tidak perlu rotasi label kolom
            for i, phase in enumerate(df1["Phase"].tolist()):
                color = phase_colors.get(phase, 'white')
                for j in range(len(df1.columns)):
                    tbl1[(i+1, j)].set_facecolor(color)
            plt.tight_layout(pad=3.0)
            table_img_dir = f"./table_images/{user_id}"
            os.makedirs(table_img_dir, exist_ok=True)
            table_img_path_1 = os.path.join(table_img_dir, f"table_exp1_{uuid.uuid4().hex[:8]}.png")
            plt.savefig(table_img_path_1, bbox_inches='tight', dpi=200)
            buf1 = io.BytesIO()
            plt.savefig(buf1, format='png', bbox_inches='tight', dpi=200)
            buf1.seek(0)
            table_img_base64_1 = base64.b64encode(buf1.read()).decode('utf-8')
            plt.close(fig1)
            # Eksperimen 2
            col_labels2 = [
                col.replace('Centripetal Acceleration (a) m/s^2', 'Centripetal\nAcceleration (a) m/s^2')
                   .replace('Angular velocity (ω) rad/s', 'Angular\nvelocity (ω) rad/s')
                for col in df2.columns
            ]
            maxlens2 = [max([len(str(x)) for x in df2[col]] + [len(lbl)]) for col, lbl in zip(df2.columns, col_labels2)]
            total2 = sum(maxlens2)
            colwidths2 = [0.12 + 0.23 * (l/total2) for l in maxlens2]
            fig2, ax2 = plt.subplots(figsize=(max(16, len(df2.columns)*3), min(2+len(df2)*0.6, 15)))
            ax2.axis('off')
            tbl2 = ax2.table(cellText=df2.values, colLabels=col_labels2, colWidths=colwidths2, loc='center', cellLoc='center')
            tbl2.auto_set_font_size(False)
            tbl2.set_fontsize(9)
            tbl2.scale(1.2, 1.5)
            # Tidak perlu rotasi label kolom
            for i, phase in enumerate(df2["Phase"].tolist()):
                color = phase_colors.get(phase, 'white')
                for j in range(len(df2.columns)):
                    tbl2[(i+1, j)].set_facecolor(color)
            plt.tight_layout(pad=3.0)
            table_img_path_2 = os.path.join(table_img_dir, f"table_exp2_{uuid.uuid4().hex[:8]}.png")
            plt.savefig(table_img_path_2, bbox_inches='tight', dpi=200)
            buf2 = io.BytesIO()
            plt.savefig(buf2, format='png', bbox_inches='tight', dpi=200)
            buf2.seek(0)
            table_img_base64_2 = base64.b64encode(buf2.read()).decode('utf-8')
            plt.close(fig2)
        except Exception as e:
            logger.error(f"Gagal membuat gambar tabel: {e}")
            table_img_base64_1 = None
            table_img_path_1 = None
            table_img_base64_2 = None
            table_img_path_2 = None

    base_prompt_intro = f"""
You are an expert physics educator. Your task is to create a new, unique question based on the provided data.
The new question should follow the style, tone, and structure of the examples given below.
- Whenever you mention radius, append (r) and always state the unit as centimeter (cm).

{prefix}
Here are some examples of good questions:
---
{few_shot_examples}
---

Now, using the following new data, create ONE new question.
{steady_info_prompt}
"""
    
    # ----------------------------------------------------------------
    # text+graph
    # ----------------------------------------------------------------
    if selected_type == "text+graph":
        # Graph assets are now pre-computed
        prompt = f"""
{base_prompt_intro}
{label_info}
Advanced Level: Analysis and Comparison
We have a comparison graph with interpretation:
{graph_interpretation}

Task:
You are given 2 datasets from experiments. Create a meaningful problem (1 question) about centripetal Centripetal Acceleration (a), angular velocity (ω), or radius (r). 
Rules:
- Use simple or basic vocabulary.
- Do NOT show any formulas in the question.
- Round all numerical values to 2 decimal places.
- Do NOT mention "Based on the interpretation."
- Choose data points from a steady state in the dataset.
- Whenever you mention centripetal Centripetal Acceleration, angular velocity, or radius, append (a), (ω), or (r).
- Use a subject (I or you or student or classmates)

Please provide the question in {language}.
"""

    # ----------------------------------------------------------------
    # text+table
    # ----------------------------------------------------------------
    elif selected_type == "text+table":
        prompt = f"""
{base_prompt_intro}
{label_info}
Advanced Level: Analysis and Comparison
We have an interpretation from GPT about these 2 experiment tables:
{table_interpretation}

Task: You are given 2 dataset experiments. Your task is to generate a single, clear, meaningful problem using basic vocabulary in {language}.
Guidelines for Generating the Problem:
- Compare or discuss centripetal Centripetal Acceleration (a), angular velocity (ω), and possibly radius (r).
- Round all numerical values to 2 decimal places.
- Do NOT mention phrases like "Based on the interpretation/highlighted data."
- Do NOT explicitly state any formulas in the question.
- Whenever the question mentions the word "centripetal Centripetal Acceleration," "angular velocity," or "radius," follow with (a), (ω), or (r) respectively.
- If the question involves finding radius (r), instruct the user to first eliminate the first 5 and last 5 data points for stable conditions.
- Avoid referencing specific row numbers from the table.
- Use a subject (I or you or student or classmates)

Please provide the final question in {language}.
"""

    # ----------------------------------------------------------------
    # text+image
    # ----------------------------------------------------------------
    elif selected_type == "text+image":
        # Image assets are now pre-computed
        prompt = f"""
{base_prompt_intro}
{label_info}
Advanced Level: Analysis and Comparison
We have two local annotated images of the objects:
1) {image1_interpretation}
2) {image2_interpretation}

Task:
- Create ONE question referencing both experiments and these two images.
- Do NOT show any formulas.
- Always mention 'centripetal Centripetal Acceleration (a)', 'angular velocity (ω)', 'radius (r)'.
- Choose data points from a steady state in the dataset.
- Round all numerical values to 2 decimal places.
- Do NOT say "Based on the interpretation."
- Use a subject (I or you or student or classmates)

Please provide the final question in {language}.
"""

    # ----------------------------------------------------------------
    # text+table+image
    # ----------------------------------------------------------------
    elif selected_type == "text+table+image":
        prompt = f"""
{base_prompt_intro}
{label_info}
Advanced Level: Analysis and Comparison
We have 2 experiments (table data) with interpretation:
{table_interpretation}

We also have two images with interpretation:
1) {image1_interpretation}
2) {image2_interpretation}

Task:
- Create ONE question referencing both the table data (two experiments) and these two images.
- Do NOT show any formulas.
- Always mention 'centripetal Centripetal Acceleration (a)', 'angular velocity (ω)', 'radius (r)'.
- Choose data points from a steady state in the dataset.
- Round all numerical values to 2 decimal places.
- Do NOT say "Based on the interpretation."
- Use a subject (I or you or student or classmates)

Please provide the final question in {language}.
"""

    # ----------------------------------------------------------------
    # text+table+graph
    # ----------------------------------------------------------------
    elif selected_type == "text+table+graph":
        # Graph assets are now pre-computed
        prompt = f"""
{base_prompt_intro}
{label_info}
Advanced Level: Analysis and Comparison
We have 2 experiment tables with interpretation:
{table_interpretation}

We also have a graph with interpretation:
{graph_interpretation}

Task:
- Create ONE question referencing both the table data (two experiments) and this graph.
- Do NOT show any formulas.
- Always mention 'centripetal Centripetal Acceleration (a)', 'angular velocity (ω)', 'radius (r)'.
- Choose data points from a steady state in the dataset.
- Round all numerical values to 2 decimal places.
- Do NOT say "Based on the interpretation."
- Use a subject (I or you or student or classmates)

Please provide the final question in {language}.
"""

    # ----------------------------------------------------------------
    # text+graph+image
    # ----------------------------------------------------------------
    elif selected_type == "text+graph+image":
        # Graph and image assets are pre-computed
        prompt = f"""
{base_prompt_intro}
{label_info}
Advanced Level: Analysis and Comparison
We have a comparison graph with interpretation:
{graph_interpretation}

We also have two local annotated images:
1) {image1_interpretation}
2) {image2_interpretation}

Task:
- Create ONE question referencing both the graph and these two images (and the underlying two experiment datasets).
- Do NOT show any formulas.
- Always mention 'centripetal Centripetal Acceleration (a)', 'angular velocity (ω)', 'radius (r)'.
- Choose data points from a steady state in the dataset.
- Round all numerical values to 2 decimal places.
- Do NOT say "Based on the interpretation."
- Use a subject (I or you or student or classmates)

Please provide the final question in {language}.
"""

    # ----------------------------------------------------------------
    # If none of the above matched for some reason, fallback
    # (Though that shouldn't happen if we covered all question_types)
    # ----------------------------------------------------------------
    else:
        # Image assets are now pre-computed
        prompt = f"""
{base_prompt_intro}
{label_info}
Advanced Level: Analysis and Comparison
We have two local annotated images of the objects:
1) {image1_interpretation}
2) {image2_interpretation}

Task:
- Create ONE question referencing both experiments and these two images.
- Do NOT show any formulas.
- Always mention 'centripetal Centripetal Acceleration (a)', 'angular velocity (ω)', 'radius (r)'.
- Choose data points from a steady state in the dataset.
- Round all numerical values to 2 decimal places.
- Do NOT say "Based on the interpretation."
- Use a subject (I or you or student or classmates)

Please provide the final question in {language}.
"""

    # ----------------------------------------------------------------
    # Send the prompt to GPT-4
    # ----------------------------------------------------------------
    try:
        messages = [
            {"role": "system", "content": "You are an expert educator who creates advanced comparison questions."},
            {"role": "user", "content": prompt}
        ]
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=400, 
            temperature=0.7
        )
        question_text = resp.choices[0].message.content.strip()
        logger.info(f"Final advanced question: {question_text}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating advanced question: {str(e)}")

    # Return all relevant data
    return {
        "question": question_text,
        "python_code": python_code,
        "graph_images": graph_images,
        "saved_graph_path": saved_graph_path,
        "table_interpretation": table_interpretation,
        "graph_interpretation": graph_interpretation,
        "image1_interpretation": image1_interpretation,
        "image2_interpretation": image2_interpretation,
        "experiment_1_data": df1.to_dict(orient="records"),
        "experiment_2_data": df2.to_dict(orient="records"),
        "table_img_base64_1": table_img_base64_1,
        "table_img_path_1": table_img_path_1,
        "table_img_base64_2": table_img_base64_2,
        "table_img_path_2": table_img_path_2,
        "image1_base64": image1_base64,
        "image2_base64": image2_base64,
    }


# -------------------------------------------------------------------
# Endpoint
# -------------------------------------------------------------------
@routes_advance.post("/advanced-question")
async def advanced_question(request: QuestionRequest):
    language = request.language
    user_id = request.user_id
    try:
        logger.info(f"Request received. language={language}, user_id={user_id}")

        # 1) Retrieve the two latest radius values for the user
        radius_val_1, radius_val_2 = get_latest_two_radius(user_id)
        if radius_val_1 is None or radius_val_2 is None:
            logger.error("Less than two radius found in DB for this user.")
            raise HTTPException(status_code=500, detail="Less than two radius found for this user")

        # 2) Retrieve the two latest object labels
        labels = get_latest_two_labels(user_id)
        if not labels or all(l is None for l in labels):
            logger.error("No label found in DB.")
            raise HTTPException(status_code=500, detail="No label found")
        label1 = labels[0] if len(labels) > 0 else None
        label2 = labels[1] if len(labels) > 1 else None

        # 3) Retrieve data buffer terbaru milik user
        acc_data, gyr_data, t_data = fetch_and_process_latest_data(user_id)
        logger.info("Fetched data for advanced question")

        # 4) Build dataframes
        df1 = pd.DataFrame({
            "Time (s)": t_data[0],
            "Centripetal Acceleration (a) m/s^2": acc_data[0],
            "Angular velocity (ω) rad/s": gyr_data[0]
        })
        df2 = pd.DataFrame({
            "Time (s)": t_data[1],
            "Centripetal Acceleration (a) m/s^2": acc_data[1],
            "Angular velocity (ω) rad/s": gyr_data[1]
        })

        if df1.empty or df2.empty:
            logger.error("Either df1 or df2 is empty")
            raise ValueError("df1 or df2 empty, check DB data")

        result = generate_advanced_question(
            df1=df1,
            df2=df2,
            label1=label1,
            label2=label2,
            language=language,
            latest_radius=radius_val_1,
            second_latest_radius=radius_val_2,
            user_id=user_id
        )

        return {
            **result,
            "label1": label1,
            "label2": label2
        }

    except Exception as e:
        logger.error(f"Internal Server Error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")
