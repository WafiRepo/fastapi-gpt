# import os
# from fastapi import APIRouter, File, UploadFile, HTTPException, Form, Depends
# from pydantic import BaseModel
# import openai
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
# openai.api_key = api_key

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
# # Determine the question content based on the parameter
# # Determine the question content based on the parameter
#         if "Acceleration" in selected_label:
#             question = (
#                 f"Here is the graph showing the relationship between centripetal acceleration (\alpha) (m/s²) and time (t), "
#                 "based on the values of angular velocity and the formula \\(a = r \\\\cdot \omega^2\\).\n\n"
#                 "Graph-Based Question:\n"
#                 "1. From the graph, at what time does the centripetal acceleration reach a specific value?\n"
#                 "2. Describe how the centripetal acceleration changes as time increases. What can you infer about the relationship between angular velocity and centripetal acceleration from this trend?\n"
#                 "3. Identify the point at which centripetal acceleration exceeds a certain value and provide its corresponding time."
#             )
#         elif "Angular velocity" in selected_label:
#             question = (
#                 f"Here is the graph showing the relationship between angular velocity (\omega) (rad/s) and time (t).\n\n"
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

# # Call OpenAI GPT with the few-shot prompt
#     response = openai.ChatCompletion.create(
#         model="gpt-4",
#         messages=[{"role": "system", "content": prompt}],
#         max_tokens=700,
#         temperature=0.7
#     )

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
from fastapi import APIRouter, File, UploadFile, HTTPException, Form, Depends
from pydantic import BaseModel
from openai import OpenAI
import openai
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
import pandas as pd
from fastapi.responses import JSONResponse
import random

router = APIRouter()

# Load environment variables
load_dotenv()
# Set OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

if not api_key:
    raise ValueError("OPENAI_API_KEY is not set")

# Define request models
class AnswerRequest(BaseModel):
    question: str
    language: str

# Function to execute Python code and return images as base64
def execute_python_code(code: str):
    local_vars = {}
    try:
        exec(code, {"plt": plt, "np": np}, local_vars)

        # Save all open figures to a list of base64 images
        images = []
        for i in plt.get_fignums():
            fig = plt.figure(i)
            buf = io.BytesIO()
            fig.savefig(buf, format="png")
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode("utf-8")
            images.append(img_base64)
            plt.close(fig)

        return images
    except Exception as e:
        return str(e)


# def generate_question_with_gpt4_few_shot(df, comparison_df=None, difficulty="easy", language="en"):
#     # Extract column names for generating questions
#     actual_columns = df.columns.tolist()
#     if 'Time (s)' in actual_columns:
#         actual_columns.remove('Time (s)')
#     if not actual_columns:
#         raise ValueError("No valid data columns found in the dataset.")

#     # Randomly select a column for the question
#     selected_column = random.choice(actual_columns)
#     row_data = df.sample().iloc[0].to_dict()

#     # Define the prompt for 'easy' questions
#     if difficulty.lower() == 'easy':
#         prompt = f"""
# Generate three knowledge and comprehension questions based on the following dataset column:

# Column Name: {selected_column}
# Sample Data: {row_data[selected_column]}

# Questions:
# 1.
# 2.
# 3.
# """
#     # Define prompt for 'intermediate' and 'advanced' difficulties
#     elif difficulty.lower() in ['intermediate', 'advanced']:
#         if comparison_df is None:
#             raise ValueError("Comparison dataset is required for intermediate and advanced difficulties.")

#         time_values_df = df['Time (s)'].tolist()
#         y_values_df = df[selected_column].tolist()

#         time_values_comp = comparison_df['Time (s)'].tolist()
#         y_values_comp = comparison_df[selected_column].tolist()

#         # Generate Python code to plot the datasets with zoomed-in inset plot
#         python_code = f"""
# import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

# # Dataset 1
# time_df = {time_values_df}
# y_values_df = {y_values_df}

# # Dataset 2
# time_comp = {time_values_comp}
# y_values_comp = {y_values_comp}

# fig, ax = plt.subplots(figsize=(10, 6))

# # Plot for Dataset 1
# ax.plot(time_df, y_values_df, 'bo-', label='Dataset 1: {selected_column}')

# # Highlight an example point (randomly chosen)
# highlight_time = {time_values_df[len(time_values_df)//2]}  # Example mid-point highlight
# highlight_value = {y_values_df[len(y_values_df)//2]}

# # Highlight a specific point in red
# ax.scatter([highlight_time], [highlight_value], color='red', s=100, label='Highlighted point')

# # Zoom-in inset plot
# axins = zoomed_inset_axes(ax, 2, loc='upper right')  # Zoom = 2
# axins.plot(time_df, y_values_df, 'bo-')
# axins.scatter([highlight_time], [highlight_value], color='red', s=100)

# # Set limits for zoomed inset plot
# axins.set_xlim(highlight_time - 1, highlight_time + 1)
# axins.set_ylim(highlight_value - 0.5, highlight_value + 0.5)

# # Add grid and marking lines
# axins.grid(True)
# mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

# # Labels and title
# ax.set_xlabel('Time (s)')
# ax.set_ylabel('{selected_column}')
# ax.set_title('{selected_column} vs Time')
# ax.legend()

# plt.show()
# """

#         # Handle questions for intermediate and advanced
#         if difficulty.lower() == 'intermediate':
#             prompt = f"""
# Generate three comparison and analysis questions based on the following datasets and their graphs:

# Dataset 1 Column: {selected_column}
# Dataset 2 Column: {selected_column}

# Questions:
# 1.
# 2.
# 3.
# """
#         elif difficulty.lower() == 'advanced':
#             prompt = f"""
# Generate three synthesis and evaluation questions based on the following datasets and their graphs:

# Dataset 1 Column: {selected_column}
# Dataset 2 Column: {selected_column}

# Questions:
# 1.
# 2.
# 3.
# """

#     else:
#         raise ValueError("Invalid difficulty level specified.")

#     # GPT-4 API call to generate questions
#     try:
#         messages = [
#             {"role": "system", "content": "You are an expert educator who creates insightful questions based on data."},
#             {"role": "user", "content": prompt}
#         ]
#         response = client.chat.completions.create(model="gpt-4",
#         messages=messages,
#         max_tokens=200,
#         temperature=0.7)
#         question = response.choices[0].message.content.strip()
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error generating questions: {str(e)}")

#     # Return questions, Python code for plots, and optionally the generated graph images
#     if difficulty.lower() in ['intermediate', 'advanced']:
#         graph_images = execute_python_code(python_code)
#         return question, python_code, graph_images
#     else:
#         return question, "", None
    
def generate_question_with_gpt4_few_shot(df, comparison_df=None, difficulty="easy", language="en"):
    # Define the actual columns in the DataFrame
    actual_columns = ['Acceleration (m/s^2)', 'Angular velocity (rad/s)']
    
    # Define LaTeX labels for plotting
    latex_labels = ['Acceleration $a$ (m/s^{2})', 'Angular velocity $\\omega$ (rad/s)']

    # Randomly select a column for question generation
    selected_index = random.randint(0, len(actual_columns) - 1)
    selected_column = actual_columns[selected_index]
    selected_label = latex_labels[selected_index]

    # Retrieve time and values for plotting
    time_values = df['Time (s)'].tolist()
    y_values = df[selected_column].tolist()

    # Randomly select a specific time to draw a single vertical line
    highlight_time = random.choice(time_values)

    # Ensure comparison dataset is handled for intermediate and advanced levels
    if comparison_df is not None:
        time_values_comp = comparison_df['Time (s)'].tolist()
        y_values_comp = comparison_df[selected_column].tolist()

        # Find the closest matching time point in Experiment 2 for the selected highlight_time
        closest_time_comp = min(time_values_comp, key=lambda x: abs(x - highlight_time))
        
        # Find the intersection value for the vertical line in both experiments
        highlight_index = time_values.index(highlight_time)
        highlight_value_1 = y_values[highlight_index]  # Experiment 1 value at vertical line

        highlight_index_comp = time_values_comp.index(closest_time_comp)
        highlight_value_2 = y_values_comp[highlight_index_comp]  # Experiment 2 value at vertical line

    if difficulty.lower() == 'easy':
        # Define the prompt for 'easy' difficulty
        prompt = f"""
        Generate three knowledge and comprehension questions based on the following dataset:

        Selected Column: {selected_column}
        Sample Data: {y_values[highlight_index]}
        Difficulty: {difficulty}

        Example:
        - Time: 5 seconds
        - Radius: 2 meters
        - {selected_column}: 3 units

        Question: What is the distance covered by an object moving along a circular path with a radius of 2 meters and a given {selected_column.lower()} over a period of 5 seconds?

        Now, generate a question based on the following inputs:
        Time: {highlight_time} seconds
        {selected_column}: {highlight_value_1} units in Experiment 1, {highlight_value_2} units in Experiment 2.
        Please provide the question in {language}.
        """
    elif difficulty.lower() in ['intermediate', 'advanced']:
        # Generate Python code for graph plotting with a single vertical line and intersection values in the legend
        python_code = f"""
import matplotlib.pyplot as plt

# Experiment 1
time_df = {time_values}
y_values_df = {y_values}

# Experiment 2
time_comp = {time_values_comp}
y_values_comp = {y_values_comp}

fig, ax = plt.subplots(figsize=(10, 6))

# Plot for Experiment 1
ax.plot(time_df, y_values_df, 'bo-', label=f'Experiment 1: {selected_column} (Value: {highlight_value_1:.2f})')
ax.plot(time_comp, y_values_comp, 'go-', label=f'Experiment 2: {selected_column} (Value: {highlight_value_2:.2f})')

# Draw a single vertical line at the selected highlight_time
ax.axvline(x={highlight_time}, color='red', linestyle='--', label=f'Vertical Line at {highlight_time:.2f}s')

# Labels and title
ax.set_xlabel('Time (s)')
ax.set_ylabel('{selected_column}')
ax.set_title('Comparison of {selected_column} between Experiment 1 and Experiment 2')
ax.legend()

# Adjust the layout to avoid cutoff
plt.tight_layout()

# Show the plot
plt.show()
"""

        # Generate questions based on difficulty level using the single vertical line
        if difficulty.lower() == 'intermediate':
            prompt = f"""
            Generate three comparison and analysis questions based on the following datasets and their **graphical representation**, focusing on the **vertical line** in the graph.

            Experiment 1 Column: {selected_column}
            Experiment 2 Column: {selected_column}

            Both datasets have been plotted against time, with a vertical line drawn at a single time point.

            Time: {highlight_time} seconds in Experiment 1 and {closest_time_comp} seconds in Experiment 2

            Questions:
            1. Compare the {selected_column.lower()} values at the vertical line in Experiment 1 and Experiment 2. What conclusions can you draw from the comparison between the values at {highlight_time} seconds in Experiment 1 and {closest_time_comp} seconds in Experiment 2?
            2. Analyze the change in {selected_column.lower()} before and after the vertical line in both experiments. How does the rate of change differ between Experiment 1 and Experiment 2 around this time point?
            3. Considering the vertical line, what might explain the differences in {selected_column.lower()} between Experiment 1 and Experiment 2 at this time point? Provide potential factors that could have influenced these variations.
            """
        elif difficulty.lower() == 'advanced':
            prompt = f"""
            Generate three synthesis and evaluation questions based on the following datasets and their **graphical representation**, focusing on the **vertical line** in the graph.

            Experiment 1 Column: {selected_column}
            Experiment 2 Column: {selected_column}

            Both datasets have been plotted against time, with a vertical line marking a specific time point.

            Questions:
            1. Evaluate the changes in {selected_column.lower()} for Experiment 1 and Experiment 2 at the time marked by the vertical line. What visual trends can be observed, and how do they compare between the two experiments?
            2. Synthesize the insights from the graph and explain how the patterns in {selected_column.lower()} for Experiment 1 and Experiment 2 at this time point could impact the system being analyzed.
            3. Identify any outliers or significant differences at the time of the vertical line. What might be the causes for these differences, and how do they affect the overall comparison of the experiments?
            """
    
    else:
        raise ValueError("Invalid difficulty level specified.")

    # GPT-4 API call to generate questions
    try:
        messages = [
            {"role": "system", "content": "You are an expert educator who creates insightful questions based on data."},
            {"role": "user", "content": prompt}
        ]
        response = client.chat.completions.create(model="gpt-4",
        messages=messages,
        max_tokens=200,
        temperature=0.7)
        question = response.choices[0].message.content.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating questions: {str(e)}")

    # Return questions, Python code for plots, and optionally the generated graph images
    if difficulty.lower() in ['intermediate', 'advanced']:
        graph_images = execute_python_code(python_code)
        return question, python_code, graph_images
    else:
        return question, "", None

# EASY ENDPOINT
@router.post("/easy-question")
async def easy_question(language: str = Form(...), file: UploadFile = File(...)):
    try:
        uploaded_df = pd.read_csv(file.file)
        question_response = generate_question_with_gpt4_few_shot(df=uploaded_df, difficulty="easy", language=language)
        return JSONResponse(content=question_response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# INTERMEDIATE ENDPOINT
@router.post("/intermediate-question")
async def intermediate_question(language: str = Form(...), file1: UploadFile = File(...), file2: UploadFile = File(...)):
    try:
        df1 = pd.read_csv(file1.file)
        df2 = pd.read_csv(file2.file)
        question_response = generate_question_with_gpt4_few_shot(df=df1, comparison_df=df2, difficulty="intermediate", language=language)
        return JSONResponse(content=question_response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ADVANCED ENDPOINT
@router.post("/advanced-question")
async def advanced_question(language: str = Form(...), file1: UploadFile = File(...), file2: UploadFile = File(...)):
    try:
        df1 = pd.read_csv(file1.file)
        df2 = pd.read_csv(file2.file)
        question_response = generate_question_with_gpt4_few_shot(df=df1, comparison_df=df2, difficulty="advanced", language=language)
        return JSONResponse(content=question_response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
