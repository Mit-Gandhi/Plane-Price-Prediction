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

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500", "http://localhost:5500"],  # Update with your frontend origin if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates setup for rendering HTML
templates = Jinja2Templates(directory="templates")

# Serve HTML from templates
@app.get("/", response_class=FileResponse)
def read_root():
    return FileResponse("templates/plane.html")

# Prediction endpoint
class PlaneInput(BaseModel):
    Engine_Type: str
    HP_or_lbs_thr_ea_engine: float
    Fuel_gal_lbs: float
    Empty_Weight_lbs: float
    Range_NM: float

@app.post("/predict")
def predict_price(data: PlaneInput):
    try:
        # Convert input data to a NumPy array
        input_data = np.array([[data.Engine_Type, data.HP_or_lbs_thr_ea_engine,
                                data.Fuel_gal_lbs, data.Empty_Weight_lbs, data.Range_NM]])

        print("Received input data:", input_data)  # Debugging output

        # Apply OneHotEncoding and Scaling
        input_encoded = ct.transform(input_data)
        input_scaled = sc.transform(input_encoded)

        print("Encoded and scaled data:", input_scaled)  # Debugging output

        # Predict the price
        predicted_price = regressor.predict(input_scaled)[0]

        print(f"Predicted price: {predicted_price}")  # Debugging output to verify the prediction

        return {"predicted_price": predicted_price}

    except Exception as e:
        print(f"Error: {e}")  # Print error to server logs
        return {"error": str(e)}  # Return error in response
