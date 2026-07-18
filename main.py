# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
# from routes import router
from router_analyze_data import router_analyze_data
from router_process_image import router_process_image
from router_inquiry import router_inquiry
from router_admin_v2_record import router_admin_v2_record
from routes_easy import routes_easy
from routes_intermediate import routes_intermediate
from routes_advance import routes_advance
from routes_radius import routes_radius
import uvicorn
import os

app = FastAPI()

_cors_origins = [o.strip() for o in os.getenv("CORS_ALLOW_ORIGINS", "*").split(",") if o.strip()] or ["*"]
_cors_cred = "*" not in _cors_origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=_cors_cred,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create required directories if not exists
for d in ["evaluation_plot", "image_result"]:
    if not os.path.exists(d):
        os.makedirs(d)

# Serve static folders
app.mount("/evaluation_plot", StaticFiles(directory="evaluation_plot"), name="evaluation_plot")
app.mount("/image_result",    StaticFiles(directory="image_result"),    name="image_result")

# app.include_router(router)
app.include_router(router_analyze_data)
app.include_router(router_process_image)
app.include_router(router_inquiry)
app.include_router(router_admin_v2_record)
app.include_router(routes_easy)
app.include_router(routes_intermediate)
app.include_router(routes_advance)
app.include_router(routes_radius)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
