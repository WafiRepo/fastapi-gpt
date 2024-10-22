# import os
# from fastapi import APIRouter, File, UploadFile, HTTPException, Form, Depends
# from pydantic import BaseModel
# from openai import OpenAI
# from dotenv import load_dotenv
# import matplotlib.pyplot as plt
# import numpy as np
# import io
# import base64
# import pandas as pd
# from fastapi.responses import JSONResponse
# import random
# from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

# router = APIRouter()

# # Load environment variables
# load_dotenv()
# # Set OpenAI API key
# api_key = os.getenv("OPENAI_API_KEY")
# client = OpenAI(api_key=api_key)

# if not api_key:
#     raise ValueError("OPENAI_API_KEY is not set")

# # Define request models
# class AnswerRequest(BaseModel):
#     question: str
#     language: str

# def generate_question_with_gpt4_few_shot(df, difficulty, language):
#     # Automatically set question_type to 'graph' if the difficulty is 'hard'
#     if difficulty.lower() == 'hard':
#         question_type = 'graph'
#     else:
#         question_type = 'text'

#     # Define the actual columns in the DataFrame
#     actual_columns = ['Acceleration (m/s^2)', 'Angular velocity (rad/s)']
    
#     # Define the LaTeX labels for plotting
#     latex_labels = ['Acceleration $a$ (m/s^{2})', 'Angular velocity $\\omega$ (rad/s)']

#     # Randomly select a column to plot and its corresponding label
#     selected_index = random.randint(0, len(actual_columns) - 1)
#     selected_column = actual_columns[selected_index]
#     selected_label = latex_labels[selected_index]

#     # Randomly select a row from the DataFrame for parameter values
#     row_data = df.sample().iloc[0]

#     # Retrieve the values for plotting
#     time_values = df['Time (s)'].tolist()  # All time values for graph plotting
#     y_values = df[selected_column].tolist()  # The values for the selected column

#     # Randomly select a specific point (time and y-value) to highlight
#     highlight_index = random.randint(0, len(time_values) - 1)
#     highlight_time = time_values[highlight_index]
#     highlight_value = y_values[highlight_index]

#     if question_type == 'graph':
#         # Generate Python code for the graph based on the selected parameter
#         python_code = f"""
# import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

# time = {time_values}
# y_values = {y_values}

# fig, ax = plt.subplots(figsize=(10, 6))

# # Main plot
# ax.plot(time, y_values, 'bo-', label='{selected_label}')
# ax.scatter([{highlight_time}], [{highlight_value}], color='red', s=100, 
#            label='Highlighted point')

# # Zoom-in plot (inset)
# axins = zoomed_inset_axes(ax, 3, loc='center', borderpad=6, bbox_to_anchor=(0.5, 0.5), bbox_transform=ax.transAxes)
# axins.plot(time, y_values, 'bo-')
# axins.scatter([{highlight_time}], [{highlight_value}], color='red', s=100)

# # Set the limits for the zoomed area
# x1, x2 = {highlight_time - 1}, {highlight_time + 1}  # Adjusted X-axis limits for better focus
# y1, y2 = {highlight_value - 0.5}, {highlight_value + 0.5}  # Adjusted Y-axis limits for zoom

# axins.set_xlim(x1, x2)
# axins.set_ylim(y1, y2)

# # Add a grid for better readability
# axins.grid(True)

# # Draw a box and lines connecting the zoom area
# mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5", linestyle="--")

# # Labels and title
# ax.set_xlabel('Time (s)')
# ax.set_ylabel('{selected_label}')
# ax.set_title('{selected_label} vs Time')
# ax.legend()

# plt.show()
# """

#         # Execute the Python code and generate the graph image
#         graph_image = execute_python_code(python_code)

#         # Determine the question content based on the parameter
#         if "Acceleration" in selected_label:
#             question = (
#                 f"Here is the graph showing the relationship between centripetal acceleration $a$ (m/s²) and time (t), "
#                 "based on the values of angular velocity and the formula $a = r \\cdot \\omega^2$.\n\n"
#                 "Graph-Based Question:\n"
#                 "1. From the graph, at what time does the centripetal acceleration reach a specific value?\n"
#                 "2. Describe how the centripetal acceleration changes as time increases. What can you infer about the relationship between angular velocity and centripetal acceleration from this trend?\n"
#                 "3. Identify the point at which centripetal acceleration exceeds a certain value and provide its corresponding time."
#             )
#         elif "Angular velocity" in selected_label:
#             question = (
#                 f"Here is the graph showing the relationship between angular velocity $\\omega$ (rad/s) and time (t).\n\n"
#                 "Graph-Based Question:\n"
#                 "1. At what time does the angular velocity reach a peak value? Describe the changes in angular velocity as time progresses.\n"
#                 "2. What is the relationship between angular velocity and centripetal force observed from the graph?\n"
#                 "3. Identify any point where the angular velocity significantly drops and explain the corresponding time."
#             )
#         else:
#             question = "No suitable graph-based question format found for the selected parameter."

#         return question, python_code, graph_image

#     # Default question generation if not graph
#     examples = f"""
#     Example 1:
#     Be creative in how the question is asked.
#     Time: 5s
#     Radius: 2m
#     {selected_label}: 3 units
#     Difficulty: easy
#     Question: What is the distance covered by an object moving along a circular path with a radius of 2 meters and a given {selected_label.lower()} over a period of 5 seconds?

#     Example 2 (using the previous hard example):
#     Time: 8s
#     Radius: 3m
#     {selected_label}: 2 units
#     Difficulty: medium
#     Question:
#     Imagine an object moving along a circular path with a radius of 3 meters. It has an initial {selected_label.lower()} of 2 units.

#     1. Find the displacement of the object after 8 seconds.
#     2. Determine the final {selected_label.lower()} after 8 seconds.

#     Now, generate a question based on the following inputs:
#     Time: {time_values[-1]}s
#     Radius: {row_data['Radius']}m
#     {selected_label}: {highlight_value} units
#     Difficulty: {difficulty}

#     Please provide the question in {language}.
#     """

#     # Format the prompt with the current inputs
#     prompt = examples

#     # Call OpenAI GPT with the few-shot prompt
#     response = client.chat.completions.create(model="gpt-4",
#     messages=[{"role": "system", "content": prompt}],
#     max_tokens=700,
#     temperature=0.7)

#     output = response.choices[0].message.content.strip()

#     # Extract only the question part and Python code part
#     question_only = output.split("```python")[0].strip()
#     python_code = ""
#     if "```python" in output:
#         code_start = output.find("```python") + len("```python")
#         code_end = output.find("```", code_start)
#         python_code = output[code_start:code_end].strip()

#     return question_only, python_code, None


# def execute_python_code(code: str):
#     local_vars = {}
#     try:
#         exec(code, {"plt": plt, "np": np}, local_vars)

#         # Save the plot to a BytesIO object
#         buf = io.BytesIO()
#         plt.savefig(buf, format="png")
#         buf.seek(0)
#         plt.close()

#         # Encode the image in base64
#         img_base64 = base64.b64encode(buf.read()).decode("utf-8")

#         return img_base64
#     except Exception as e:
#         return str(e)

# @router.post("/generate-question")
# async def generate_question(
#     difficulty: str = Form(...),
#     language: str = Form(...),
#     file: UploadFile = File(...)
# ):
#     try:
#         # Read the uploaded CSV into a DataFrame
#         uploaded_df = pd.read_csv(file.file)

#         # Call the function with the DataFrame and other parameters
#         question_response, python_code, graph_image = generate_question_with_gpt4_few_shot(
#             df=uploaded_df,
#             difficulty=difficulty,
#             language=language,
#         )

#         return {
#             "question": question_response,
#             "python_code": python_code,
#             "graph_image": graph_image
#         }
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


import os
from fastapi import FastAPI, APIRouter, File, UploadFile, HTTPException, Form
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import random
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

app = FastAPI()
router = APIRouter()

# Load Qwen model and tokenizer
model_name = "Qwen/Qwen-7B"  # Ganti dengan model Qwen yang Anda gunakan
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Define request models
class AnswerRequest(BaseModel):
    question: str
    language: str

def generate_question_with_qwen(df, difficulty, language):
    # Automatically set question_type to 'graph' if the difficulty is 'hard'
    if difficulty.lower() == 'hard':
        question_type = 'graph'
    else:
        question_type = 'text'

    actual_columns = ['Acceleration (m/s^2)', 'Angular velocity (rad/s)']
    latex_labels = ['Acceleration $a$ (m/s^{2})', 'Angular velocity $\\omega$ (rad/s)']

    selected_index = random.randint(0, len(actual_columns) - 1)
    selected_column = actual_columns[selected_index]
    selected_label = latex_labels[selected_index]

    row_data = df.sample().iloc[0]
    time_values = df['Time (s)'].tolist()
    y_values = df[selected_column].tolist()

    highlight_index = random.randint(0, len(time_values) - 1)
    highlight_time = time_values[highlight_index]
    highlight_value = y_values[highlight_index]

    if question_type == 'graph':
        python_code = f"""
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

time = {time_values}
y_values = {y_values}

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(time, y_values, 'bo-', label='{selected_label}')
ax.scatter([{highlight_time}], [{highlight_value}], color='red', s=100, 
           label='Highlighted point')

axins = zoomed_inset_axes(ax, 3, loc='center', borderpad=6, bbox_to_anchor=(0.5, 0.5), bbox_transform=ax.transAxes)
axins.plot(time, y_values, 'bo-')
axins.scatter([{highlight_time}], [{highlight_value}], color='red', s=100)

x1, x2 = {highlight_time - 1}, {highlight_time + 1}
y1, y2 = {highlight_value - 0.5}, {highlight_value + 0.5}

axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)

axins.grid(True)
mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5", linestyle="--")

ax.set_xlabel('Time (s)')
ax.set_ylabel('{selected_label}')
ax.set_title('{selected_label} vs Time')
ax.legend()

plt.show()
"""
        graph_image = execute_python_code(python_code)
        
        if "Acceleration" in selected_label:
            question = (
                f"Graph showing centripetal acceleration $a$ (m/s²) vs time.\n"
                "1. At what time does acceleration reach a specific value?\n"
                "2. Describe the change in acceleration over time."
            )
        else:
            question = (
                f"Graph showing angular velocity $\\omega$ (rad/s) vs time.\n"
                "1. At what time does velocity peak?\n"
                "2. Describe changes in velocity over time."
            )

        return question, python_code, graph_image

    prompt = f"""
    Generate a question based on: Time: {time_values[-1]}s, {selected_label}: {highlight_value} units.
    Please provide the question in {language}.
    """

    # Tokenize the input and generate text with the Qwen model
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=200)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return generated_text, "", None

def execute_python_code(code: str):
    local_vars = {}
    try:
        exec(code, {"plt": plt, "np": np}, local_vars)
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plt.close()
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        return img_base64
    except Exception as e:
        return str(e)

@router.post("/generate-question")
async def generate_question(
    difficulty: str = Form(...),
    language: str = Form(...),
    file: UploadFile = File(...)
):
    try:
        # Read the uploaded CSV into a DataFrame
        uploaded_df = pd.read_csv(file.file)
        question_response, python_code, graph_image = generate_question_with_qwen(
            df=uploaded_df, difficulty=difficulty, language=language
        )
        return {
            "question": question_response,
            "python_code": python_code,
            "graph_image": graph_image
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# To run the server, use:
# uvicorn script_name:app --reload
