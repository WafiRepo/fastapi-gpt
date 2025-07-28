# main.py
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
# from routes import router
from router_analyze_data import router_analyze_data
from router_process_image import router_process_image
from routes_easy import routes_easy
from routes_intermediate import routes_intermediate
from routes_advance import routes_advance
from routes_radius import routes_radius
import uvicorn

app = FastAPI()

# Mount static folder for evaluation_plot
app.mount("/evaluation_plot", StaticFiles(directory="evaluation_plot"), name="evaluation_plot")

# app.include_router(router)
app.include_router(router_analyze_data)
app.include_router(router_process_image)
app.include_router(routes_easy)
app.include_router(routes_intermediate)
app.include_router(routes_advance)
app.include_router(routes_radius)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
