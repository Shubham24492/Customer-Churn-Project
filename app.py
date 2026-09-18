from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent
# here we get the path of the model created and saved in churn analysis jupytewr notebook
MODEL_PATH = BASE_DIR / "model" / "churn_model.pkl"

# Customer data model for pydantic validation of input request


class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


# fast api app initialization
app = FastAPI()

# load the model from model path
model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None

# api to health check


@app.get("/")
def read_root():
    return {"message": "Customer Churn App is running"}

# api to predict whether customer will churn based on provided request


@app.post("/predict")
def predict(customer: CustomerData):
    # if model is not found, ask user to run the notebook to generate model
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not found. Run the notebook churn_analysis.ipynb first to create model.",
        )

    try:
        # create dataframe from input request customer data
        payload = customer.model_dump()
        row = pd.DataFrame([payload])
        print(payload.keys())
        print("Data Received", payload)

        # call the model.predict method to predict the outcome Yes or No based on input customer data
        prediction = model.predict(row)[0]
        print("prediction", prediction)

        # calculate the churn probablility
        probability = float(model.predict_proba(
            row)[0, list(model.classes_).index("Yes")])
        print("probability", probability)

        # return response json
        return {
            "prediction": str(prediction),
            "churn_probability": round(probability, 4),
        }

    except (TypeError, ValueError, KeyError) as exc:
        # exception handling
        raise HTTPException(
            status_code=400, detail=f"Invalid customer data: {exc}") from exc
    
