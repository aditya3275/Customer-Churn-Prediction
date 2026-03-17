from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib

# ── Load model & artifacts ─────────────────────────────────
model = joblib.load("models/best_model.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predicts customer churn probability and risk category",
    version="1.0.0",
)


# ── Input Schema ───────────────────────────────────────────
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


# ── Preprocessing helper ───────────────────────────────────
def preprocess_input(data: CustomerData):
    d = data.dict()

    # Binary encode
    binary_map = {"Yes": 1, "No": 0, "Male": 1, "Female": 0}
    for col in ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]:
        d[col] = binary_map.get(d[col], 0)

    # Scale numerical
    nums = np.array([[d["tenure"], d["MonthlyCharges"], d["TotalCharges"]]])
    scaled = scaler.transform(nums)[0]
    d["tenure"] = scaled[0]
    d["MonthlyCharges"] = scaled[1]
    d["TotalCharges"] = scaled[2]

    # One-hot encode
    multi_cols = {
        "MultipleLines": ["No phone service", "Yes"],
        "InternetService": ["Fiber optic", "No"],
        "OnlineSecurity": ["No internet service", "Yes"],
        "OnlineBackup": ["No internet service", "Yes"],
        "DeviceProtection": ["No internet service", "Yes"],
        "TechSupport": ["No internet service", "Yes"],
        "StreamingTV": ["No internet service", "Yes"],
        "StreamingMovies": ["No internet service", "Yes"],
        "Contract": ["One year", "Two year"],
        "PaymentMethod": [
            "Credit card (automatic)",
            "Electronic check",
            "Mailed check",
        ],
    }

    row = {}
    for col in [
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "tenure",
        "PhoneService",
        "PaperlessBilling",
        "MonthlyCharges",
        "TotalCharges",
    ]:
        row[col] = d[col]

    for col, categories in multi_cols.items():
        for cat in categories:
            row[f"{col}_{cat}"] = 1 if d[col] == cat else 0

    # Feature engineering
    row["charge_per_tenure"] = row["MonthlyCharges"] / (row["tenure"] + 1)
    row["is_new_customer"] = 1 if row["tenure"] < -0.5 else 0
    service_keys = [
        "OnlineSecurity_Yes",
        "OnlineBackup_Yes",
        "DeviceProtection_Yes",
        "TechSupport_Yes",
        "StreamingTV_Yes",
        "StreamingMovies_Yes",
    ]
    row["total_services"] = sum(row.get(k, 0) for k in service_keys)
    row["is_high_value"] = 1 if row["MonthlyCharges"] > 0.5 else 0

    contract_risk = 1.0
    if row.get("Contract_One year"):
        contract_risk = 0.5
    if row.get("Contract_Two year"):
        contract_risk = 0.0
    row["contract_risk"] = contract_risk

    # Align with training columns
    df_row = pd.DataFrame([row])
    df_row = df_row.reindex(columns=feature_columns, fill_value=0)

    return df_row


# ── Routes ─────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "Customer Churn Prediction API is running!"}


@app.post("/predict")
def predict(customer: CustomerData):
    try:
        processed = preprocess_input(customer)
        prob = model.predict_proba(processed)[0][1]
        prediction = int(prob >= 0.5)

        if prob >= 0.7:
            risk = "High Risk"
        elif prob >= 0.4:
            risk = "Medium Risk"
        else:
            risk = "Low Risk"

        return {
            "churn_prediction": "Yes" if prediction == 1 else "No",
            "churn_probability": round(float(prob), 4),
            "risk_category": risk,
            "confidence": f"{round(float(prob) * 100, 2)}%",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "healthy", "model": "Gradient Boosting", "auc": 0.8336}
