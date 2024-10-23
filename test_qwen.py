

# import os
# from fastapi import FastAPI, APIRouter, File, UploadFile, HTTPException, Form
# from pydantic import BaseModel
# from transformers import AutoTokenizer, AutoModelForCausalLM
# import pandas as pd
# import matplotlib.pyplot as plt
# import io
# import base64
# import random
# from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

# app = FastAPI()
# router = APIRouter()

# # Load Qwen model and tokenizer
# model_name = "Qwen/Qwen-7B"  # Ganti dengan model Qwen yang Anda gunakan
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name)

# # Define request models
# class AnswerRequest(BaseModel):
#     question: str
#     language: str

# def generate_question_with_qwen(df, difficulty, language):
#     # Automatically set question_type to 'graph' if the difficulty is 'hard'
#     if difficulty.lower() == 'hard':
#         question_type = 'graph'
#     else:
#         question_type = 'text'

#     actual_columns = ['Acceleration (m/s^2)', 'Angular velocity (rad/s)']
#     latex_labels = ['Acceleration $a$ (m/s^{2})', 'Angular velocity $\\omega$ (rad/s)']

#     selected_index = random.randint(0, len(actual_columns) - 1)
#     selected_column = actual_columns[selected_index]
#     selected_label = latex_labels[selected_index]

#     row_data = df.sample().iloc[0]
#     time_values = df['Time (s)'].tolist()
#     y_values = df[selected_column].tolist()

#     highlight_index = random.randint(0, len(time_values) - 1)
#     highlight_time = time_values[highlight_index]
#     highlight_value = y_values[highlight_index]

#     if question_type == 'graph':
#         python_code = f"""
# import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

# time = {time_values}
# y_values = {y_values}

# fig, ax = plt.subplots(figsize=(10, 6))

# ax.plot(time, y_values, 'bo-', label='{selected_label}')
# ax.scatter([{highlight_time}], [{highlight_value}], color='red', s=100, 
#            label='Highlighted point')

# axins = zoomed_inset_axes(ax, 3, loc='center', borderpad=6, bbox_to_anchor=(0.5, 0.5), bbox_transform=ax.transAxes)
# axins.plot(time, y_values, 'bo-')
# axins.scatter([{highlight_time}], [{highlight_value}], color='red', s=100)

# x1, x2 = {highlight_time - 1}, {highlight_time + 1}
# y1, y2 = {highlight_value - 0.5}, {highlight_value + 0.5}

# axins.set_xlim(x1, x2)
# axins.set_ylim(y1, y2)

# axins.grid(True)
# mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5", linestyle="--")

# ax.set_xlabel('Time (s)')
# ax.set_ylabel('{selected_label}')
# ax.set_title('{selected_label} vs Time')
# ax.legend()

# plt.show()
# """
#         graph_image = execute_python_code(python_code)

#         if "Acceleration" in selected_label:
#             question = (
#                 f"Graph showing centripetal acceleration $a$ (m/s²) vs time.\n"
#                 "1. At what time does acceleration reach a specific value?\n"
#                 "2. Describe the change in acceleration over time."
#             )
#         else:
#             question = (
#                 f"Graph showing angular velocity $\\omega$ (rad/s) vs time.\n"
#                 "1. At what time does velocity peak?\n"
#                 "2. Describe changes in velocity over time."
#             )

#         return question, python_code, graph_image

#     prompt = f"""
#     Generate a question based on: Time: {time_values[-1]}s, {selected_label}: {highlight_value} units.
#     Please provide the question in {language}.
#     """

#     # Tokenize the input and generate text with the Qwen model
#     inputs = tokenizer(prompt, return_tensors="pt")
#     outputs = model.generate(**inputs, max_length=200)
#     generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

#     return generated_text, "", None

# def execute_python_code(code: str):
#     local_vars = {}
#     try:
#         exec(code, {"plt": plt, "np": np}, local_vars)
#         buf = io.BytesIO()
#         plt.savefig(buf, format="png")
#         buf.seek(0)
#         plt.close()
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
#         question_response, python_code, graph_image = generate_question_with_qwen(
#             df=uploaded_df, difficulty=difficulty, language=language
#         )
#         return {
#             "question": question_response,
#             "python_code": python_code,
#             "graph_image": graph_image
#         }
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# # To run the server, use:
# # uvicorn script_name:app --reload

