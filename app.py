from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
from pydantic import BaseModel

# Load model and preprocessors
regressor = joblib.load("random_forest_regressor_model.joblib")
ct = joblib.load("column_transformer.joblib")
sc = joblib.load("standard_scaler.joblib")

app = FastAPI()

# Enable CORS for both local dev and deployed frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://plane-price-prediction.onrender.com",  # Your deployed frontend
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates setup
templates = Jinja2Templates(directory="templates")

# Serve main HTML
@app.get("/", response_class=FileResponse)
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
def predict_price(data: PlaneInput):
    try:
        input_data = np.array([[data.Engine_Type, data.HP_or_lbs_thr_ea_engine,
                                data.Fuel_gal_lbs, data.Empty_Weight_lbs, data.Range_NM]])
        input_encoded = ct.transform(input_data)
        input_scaled = sc.transform(input_encoded)
        predicted_price = regressor.predict(input_scaled)[0]
        return {"predicted_price": predicted_price}

    except Exception as e:
        print(f"Error: {e}")
        return {"error": str(e)}
