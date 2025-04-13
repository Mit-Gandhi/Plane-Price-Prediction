from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import joblib
import numpy as np
from pydantic import BaseModel

# Load model and preprocessors
regressor = joblib.load("random_forest_regressor_model.joblib")
ct = joblib.load("column_transformer.joblib")
sc = joblib.load("standard_scaler.joblib")

# Initialize FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=False,  # Set to False since we're allowing all origins
    allow_methods=["*"],
    allow_headers=["*"]
)

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates setup for rendering HTML
templates = Jinja2Templates(directory="templates")

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
def predict_price(data: PlaneInput):
    try:
        # Prepare the input data
        input_data = np.array([[data.Engine_Type, data.HP_or_lbs_thr_ea_engine,
                                data.Fuel_gal_lbs, data.Empty_Weight_lbs, data.Range_NM]])

        # Apply transformations
        input_encoded = ct.transform(input_data)
        input_scaled = sc.transform(input_encoded)

        # Predict the price
        predicted_price = regressor.predict(input_scaled)[0]

        return {"predicted_price": predicted_price}

    except Exception as e:
        return {"error": str(e)}
