# main.py
from fastapi import FastAPI
from routes import router

app = FastAPI()

# Include the CSV route
app.include_router(router)