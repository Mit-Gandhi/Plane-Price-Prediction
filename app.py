from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np

# Load model and preprocessors
regressor = joblib.load("random_forest_regressor_model.joblib")
ct = joblib.load("column_transformer.joblib")
sc = joblib.load("standard_scaler.joblib")

app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://plane-price-prediction.onrender.com"],  # Your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=FileResponse)
async def serve_home():
    return FileResponse("templates/plane.html")

# Pydantic model
class PlaneInput(BaseModel):
    Engine_Type: str
    HP_or_lbs_thr_ea_engine: float
    Fuel_gal_lbs: float
    Empty_Weight_lbs: float
    Range_NM: float

@app.post("/predict")
async def predict_price(data: PlaneInput):
    try:
        input_data = np.array([[data.Engine_Type,
                                data.HP_or_lbs_thr_ea_engine,
                                data.Fuel_gal_lbs,
                                data.Empty_Weight_lbs,
                                data.Range_NM]])
        input_encoded = ct.transform(input_data)
        input_scaled = sc.transform(input_encoded)
        prediction = regressor.predict(input_scaled)[0]
        return {"predicted_price": prediction}
    except Exception as e:
        return {"error": str(e)}
