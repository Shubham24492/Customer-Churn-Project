# Telco Customer Churn

# This project predicts whether an IBM Telco customer will churn. 
# The notebook contains the solution: 
# Data Understanding & Preparation, 
# Exploratory Data Analysis, 
# Feature Engineering, 
# Model Evaluation, 
# Model Interpretation, 
# Model Saving & API and 
# Submission Requirements.

# folder structure
customer_churn_project/ 
│ 
├── data/ 
├── notebook/ 
│ └── churn_analysis.ipynb 
├── model/ 
│ └── churn_model.pkl 
├── app.py 
├── requirements.txt 
├── README.md 
└── sample_request.json

## Commands to Run notebook to create model

From customer_churn_project directory:
# create venev for the project 
python -m venv .customer_churn_project

# install dependency
python -m pip install -r requirements.txt

# Run all notebook cells to create model `model/churn_model.pkl`.
jupyter notebook notebook/churn_analysis.ipynb


## Command to Run the server locally to expose API 
python -m fastapi dev app.py


# Then send `sample_request.json` as body to `POST http://127.0.0.1:5000/predict` or use following curl:
# Also I have save the test data to /notebook/test_data.json file you can use it to test the api 
```
curl --location 'http://localhost:8000/predict' \
--header 'Content-Type: application/json' \
--data '{
	"gender": "Female",
	"SeniorCitizen": 0,
	"Partner": "Yes",
	"Dependents": "No",
	"tenure": 125,
	"PhoneService": "Yes",
	"MultipleLines": "No",
	"InternetService": "Fiber optic",
	"OnlineSecurity": "No",
	"OnlineBackup": "Yes",
	"DeviceProtection": "No",
	"TechSupport": "No",
	"StreamingTV": "Yes",
	"StreamingMovies": "Yes",
	"Contract": "Month-to-month",
	"PaperlessBilling": "Yes",
	"PaymentMethod": "Electronic check",
	"MonthlyCharges": 85.5,
	"TotalCharges": 1026.0
}
```

# The response contains `prediction` (`Yes` or `No`) and `churn_probability`. 
{
    "prediction": "Yes",
    "churn_probability": 0.6403
}

## api docs
http://localhost:8000/docs