from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import joblib
import numpy as np
from pydantic import BaseModel
import os

# Initialize FastAPI app
app = FastAPI(title="Plane Price Prediction API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://plane-price-prediction.onrender.com"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Accept"]
)

# Load model and preprocessors
try:
    regressor = joblib.load("random_forest_regressor_model.joblib")
    ct = joblib.load("column_transformer.joblib")
    sc = joblib.load("standard_scaler.joblib")
    models_loaded = True
except Exception as e:
    print(f"Error loading models: {str(e)}")
    models_loaded = False

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates setup for rendering HTML
templates = Jinja2Templates(directory="templates")

@app.get("/")
async def root():
    return {"status": "ok", "message": "Plane Price Prediction API is running", "models_loaded": models_loaded}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "models_loaded": models_loaded}

# Serve main HTML file
@app.get("/")
def read_root():
    return FileResponse("templates/plane.html")

# Define input data model
class PlaneInput(BaseModel):
    Engine_Type: str
    HP_or_lbs_thr_ea_engine: float
    Fuel_gal_lbs: float
    Empty_Weight_lbs: float
    Range_NM: float

# Prediction endpoint
@app.post("/predict")
async def predict_price(data: PlaneInput):
    if not models_loaded:
        raise HTTPException(status_code=500, detail="Models not loaded properly")
        
    try:
        # Prepare the input data
        input_data = np.array([[data.Engine_Type, data.HP_or_lbs_thr_ea_engine,
                              data.Fuel_gal_lbs, data.Empty_Weight_lbs, data.Range_NM]])

        # Apply transformations
        input_encoded = ct.transform(input_data)
        input_scaled = sc.transform(input_encoded)

        # Predict the price
        predicted_price = regressor.predict(input_scaled)[0]

        return {"predicted_price": float(predicted_price)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
