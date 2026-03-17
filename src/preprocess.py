import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os


def preprocess(filepath="data/telco_churn.csv"):

    df = pd.read_csv(filepath)

    # ── Drop customerID (useless for ML) ──────────────────
    df.drop(columns=["customerID"], inplace=True)

    # ── Fix TotalCharges (string → float, spaces → NaN) ───
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    print(f"NaNs in TotalCharges after fix: {df['TotalCharges'].isnull().sum()}")

    # ── Fill NaNs with median ──────────────────────────────
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # ── Encode Target ──────────────────────────────────────
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # ── Encode Binary Columns ──────────────────────────────
    binary_cols = [
        "gender",
        "Partner",
        "Dependents",
        "PhoneService",
        "PaperlessBilling",
    ]
    for col in binary_cols:
        df[col] = LabelEncoder().fit_transform(df[col])

    # ── One-Hot Encode Multi-Category Columns ──────────────
    multi_cols = [
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaymentMethod",
    ]
    df = pd.get_dummies(df, columns=multi_cols, drop_first=True)

    # ── Scale Numerical Features ───────────────────────────
    scaler = StandardScaler()
    scale_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    df[scale_cols] = scaler.fit_transform(df[scale_cols])

    # ── Save scaler ────────────────────────────────────────
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.pkl")
    print("Scaler saved to models/scaler.pkl")

    # ── Save processed data ────────────────────────────────
    df.to_csv("data/processed.csv", index=False)
    print(f"Processed data saved! Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    return df


if __name__ == "__main__":
    df = preprocess()
    print("\n✅ Preprocessing Complete!")
    print(df.head())
