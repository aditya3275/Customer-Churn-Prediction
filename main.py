"""
main.py — ChurnGuard AI  (production-ready, DB-backed)
FastAPI application with:
  - PostgreSQL / SQLite via SQLAlchemy
  - Automatic audit logging middleware
  - Full prediction persistence
  - Customer history + audit endpoints
"""

import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from database import Base, engine, get_db
from models import AuditLog, Customer, DashboardSnapshot, Prediction, RetentionAction

# ── Load ML artifacts ──────────────────────────────────────────────────────────
model          = joblib.load("models/best_model.pkl")
scaler         = joblib.load("models/scaler.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")


# ── Create tables on startup ───────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    yield


# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="ChurnGuard AI — Customer Churn Prediction API",
    description=(
        "Predicts customer churn probability, persists every prediction to a database, "
        "and exposes history and audit endpoints."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Audit-logging middleware ───────────────────────────────────────────────────
@app.middleware("http")
async def audit_middleware(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)

    db: Session = next(get_db())
    try:
        customer_id = request.path_params.get("customer_id")
        log = AuditLog(
            endpoint=str(request.url.path),
            method=request.method,
            customer_id=customer_id,
            status_code=response.status_code,
            response_time_ms=elapsed_ms,
        )
        db.add(log)
        db.commit()
    except Exception:
        db.rollback()
    finally:
        db.close()

    response.headers["X-Response-Time-Ms"] = str(elapsed_ms)
    return response


# ─────────────────────────────────────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────────────────────────────────────
class CustomerInput(BaseModel):
    customer_id:      Optional[str] = Field(default=None, description="Leave blank to auto-generate")
    gender:           str
    SeniorCitizen:    int
    Partner:          str
    Dependents:       str
    tenure:           int
    PhoneService:     str
    MultipleLines:    str
    InternetService:  str
    OnlineSecurity:   str
    OnlineBackup:     str
    DeviceProtection: str
    TechSupport:      str
    StreamingTV:      str
    StreamingMovies:  str
    Contract:         str
    PaperlessBilling: str
    PaymentMethod:    str
    MonthlyCharges:   float
    TotalCharges:     float


class PredictionResponse(BaseModel):
    prediction_id:    int
    customer_id:      str
    churn_prediction: str
    churn_probability: float
    risk_category:    str
    confidence_score: float
    lifetime_value:   float
    predicted_at:     datetime


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _risk_category(prob: float) -> str:
    if prob >= 0.7:
        return "High Risk"
    if prob >= 0.4:
        return "Medium Risk"
    return "Low Risk"


def _confidence_score(prob: float) -> float:
    """Normalised distance from the decision boundary (0 = uncertain, 1 = certain)."""
    return round(abs(prob - 0.5) * 2, 4)


def _lifetime_value(monthly: float, tenure: int, churn_prob: float) -> float:
    """Simplified CLV: expected remaining months × monthly charges."""
    expected_remaining = max(0, (1 - churn_prob) * (72 - tenure))
    return round(expected_remaining * monthly, 2)


def _preprocess(data: CustomerInput) -> pd.DataFrame:
    d = data.dict()

    binary_map = {"Yes": 1, "No": 0, "Male": 1, "Female": 0}
    for col in ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]:
        d[col] = binary_map.get(d[col], 0)

    nums = np.array([[d["tenure"], d["MonthlyCharges"], d["TotalCharges"]]])
    scaled = scaler.transform(nums)[0]
    d["tenure"], d["MonthlyCharges"], d["TotalCharges"] = scaled

    multi_cols: Dict[str, List[str]] = {
        "MultipleLines":    ["No phone service", "Yes"],
        "InternetService":  ["Fiber optic", "No"],
        "OnlineSecurity":   ["No internet service", "Yes"],
        "OnlineBackup":     ["No internet service", "Yes"],
        "DeviceProtection": ["No internet service", "Yes"],
        "TechSupport":      ["No internet service", "Yes"],
        "StreamingTV":      ["No internet service", "Yes"],
        "StreamingMovies":  ["No internet service", "Yes"],
        "Contract":         ["One year", "Two year"],
        "PaymentMethod":    [
            "Credit card (automatic)", "Electronic check", "Mailed check"
        ],
    }

    row: Dict[str, Any] = {
        col: d[col]
        for col in [
            "gender", "SeniorCitizen", "Partner", "Dependents",
            "tenure", "PhoneService", "PaperlessBilling",
            "MonthlyCharges", "TotalCharges",
        ]
    }

    for col, cats in multi_cols.items():
        for cat in cats:
            row[f"{col}_{cat}"] = 1 if d[col] == cat else 0

    row["charge_per_tenure"] = row["MonthlyCharges"] / (row["tenure"] + 1)
    row["is_new_customer"]   = 1 if row["tenure"] < -0.5 else 0
    service_keys = [
        "OnlineSecurity_Yes", "OnlineBackup_Yes", "DeviceProtection_Yes",
        "TechSupport_Yes", "StreamingTV_Yes", "StreamingMovies_Yes",
    ]
    row["total_services"] = sum(row.get(k, 0) for k in service_keys)
    row["is_high_value"]  = 1 if row["MonthlyCharges"] > 0.5 else 0

    contract_risk = 1.0
    if row.get("Contract_One year"):
        contract_risk = 0.5
    if row.get("Contract_Two year"):
        contract_risk = 0.0
    row["contract_risk"] = contract_risk

    df = pd.DataFrame([row]).reindex(columns=feature_columns, fill_value=0)
    return df


def _feature_importance(processed_df: pd.DataFrame) -> Dict[str, float]:
    """Return top-10 feature importances from the fitted model."""
    try:
        importances = model.feature_importances_
        pairs = sorted(
            zip(feature_columns, importances),
            key=lambda x: x[1], reverse=True
        )[:10]
        return {k: round(float(v), 6) for k, v in pairs}
    except AttributeError:
        return {}


def _retention_tips(data: CustomerInput, risk: str) -> List[str]:
    tips = []
    if data.Contract == "Month-to-month":
        tips.append("Offer a discounted One Year or Two Year contract upgrade.")
    if data.InternetService == "Fiber optic" and data.OnlineSecurity == "No":
        tips.append("Bundle Online Security — fiber customers churn when they feel unsafe.")
    if data.PaymentMethod == "Electronic check":
        tips.append("Encourage switch to Auto-pay — reduces payment friction.")
    if data.tenure < 6:
        tips.append("New customer detected — offer a loyalty bonus or free service upgrade.")
    if risk == "Low Risk":
        tips.append("Customer is satisfied — consider upselling premium services.")
    return tips


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    return {"message": "ChurnGuard AI API is running!", "version": "2.0.0"}


@app.get("/health", tags=["Health"])
def health():
    return {"status": "healthy", "model": "Gradient Boosting", "auc": 0.8336}


# ── POST /predict ─────────────────────────────────────────────────────────────
@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
def predict(data: CustomerInput, db: Session = Depends(get_db)):
    # ── 1. Run ML model ──────────────────────────────────────────────────────
    try:
        processed  = _preprocess(data)
        prob       = float(model.predict_proba(processed)[0][1])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Model inference failed: {exc}")

    prediction_label = "Yes" if prob >= 0.5 else "No"
    risk             = _risk_category(prob)
    confidence       = _confidence_score(prob)
    clv              = _lifetime_value(data.MonthlyCharges, data.tenure, prob)
    feature_snap     = _feature_importance(processed)
    tips             = _retention_tips(data, risk)

    # ── 2. Upsert customer ───────────────────────────────────────────────────
    customer_id = data.customer_id or str(uuid.uuid4())

    try:
        customer = db.query(Customer).filter(Customer.customer_id == customer_id).first()
        if customer:
            # Update mutable fields
            customer.tenure          = data.tenure
            customer.monthly_charges = data.MonthlyCharges
            customer.total_charges   = data.TotalCharges
            customer.contract_type   = data.Contract
            customer.payment_method  = data.PaymentMethod
            customer.updated_at      = datetime.now(timezone.utc)
        else:
            customer = Customer(
                customer_id     = customer_id,
                gender          = data.gender,
                senior_citizen  = bool(data.SeniorCitizen),
                has_partner     = data.Partner == "Yes",
                has_dependents  = data.Dependents == "Yes",
                tenure          = data.tenure,
                contract_type   = data.Contract,
                monthly_charges = data.MonthlyCharges,
                total_charges   = data.TotalCharges,
                internet_service = data.InternetService,
                payment_method  = data.PaymentMethod,
            )
            db.add(customer)
        db.flush()   # ensure customer row exists before FK insert

        # ── 3. Persist prediction ────────────────────────────────────────────
        pred_row = Prediction(
            customer_id                 = customer_id,
            churn_prediction            = prediction_label,
            churn_probability           = round(prob, 4),
            risk_category               = risk,
            confidence_score            = confidence,
            lifetime_value              = clv,
            feature_importance_snapshot = feature_snap,
        )
        db.add(pred_row)
        db.flush()

        # ── 4. Persist retention actions ─────────────────────────────────────
        for tip in tips:
            db.add(RetentionAction(
                customer_id        = customer_id,
                prediction_id      = pred_row.prediction_id,
                risk_level         = risk,
                recommended_action = tip,
            ))

        db.commit()
        db.refresh(pred_row)

    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database write failed: {exc}")

    return PredictionResponse(
        prediction_id     = pred_row.prediction_id,
        customer_id       = customer_id,
        churn_prediction  = prediction_label,
        churn_probability = round(prob, 4),
        risk_category     = risk,
        confidence_score  = confidence,
        lifetime_value    = clv,
        predicted_at      = pred_row.predicted_at,
    )


# ── GET /customer-history/{customer_id} ───────────────────────────────────────
@app.get("/customer-history/{customer_id}", tags=["History"])
def customer_history(customer_id: str, db: Session = Depends(get_db)):
    customer = db.query(Customer).filter(Customer.customer_id == customer_id).first()
    if not customer:
        raise HTTPException(status_code=404, detail="Customer not found.")

    predictions = (
        db.query(Prediction)
        .filter(Prediction.customer_id == customer_id)
        .order_by(Prediction.predicted_at.desc())
        .limit(10)
        .all()
    )

    return {
        "customer_id": customer_id,
        "total_predictions": len(predictions),
        "history": [
            {
                "prediction_id":    p.prediction_id,
                "churn_prediction": p.churn_prediction,
                "churn_probability": p.churn_probability,
                "risk_category":    p.risk_category,
                "confidence_score": p.confidence_score,
                "lifetime_value":   p.lifetime_value,
                "predicted_at":     p.predicted_at,
            }
            for p in predictions
        ],
    }


# ── GET /audit-logs ───────────────────────────────────────────────────────────
@app.get("/audit-logs", tags=["Admin"])
def get_audit_logs(limit: int = 100, db: Session = Depends(get_db)):
    logs = (
        db.query(AuditLog)
        .order_by(AuditLog.timestamp.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "log_id":           l.log_id,
            "endpoint":         l.endpoint,
            "method":           l.method,
            "customer_id":      l.customer_id,
            "status_code":      l.status_code,
            "response_time_ms": l.response_time_ms,
            "error_message":    l.error_message,
            "timestamp":        l.timestamp,
        }
        for l in logs
    ]


# ── GET /retention-actions/{customer_id} ──────────────────────────────────────
@app.get("/retention-actions/{customer_id}", tags=["History"])
def get_retention_actions(customer_id: str, db: Session = Depends(get_db)):
    actions = (
        db.query(RetentionAction)
        .filter(RetentionAction.customer_id == customer_id)
        .order_by(RetentionAction.created_at.desc())
        .all()
    )
    return [
        {
            "action_id":           a.action_id,
            "prediction_id":       a.prediction_id,
            "risk_level":          a.risk_level,
            "recommended_action":  a.recommended_action,
            "action_taken":        a.action_taken,
            "outcome":             a.outcome,
            "created_at":          a.created_at,
        }
        for a in actions
    ]


# ── GET /dashboard-snapshot ───────────────────────────────────────────────────
@app.get("/dashboard-snapshot", tags=["Admin"])
def dashboard_snapshot(db: Session = Depends(get_db)):
    """Compute a live aggregate snapshot and optionally save it."""
    from sqlalchemy import func, case

    row = db.query(
        func.count(Prediction.prediction_id).label("total"),
        func.avg(Prediction.churn_probability).label("avg_prob"),
        func.sum(
            case((Prediction.risk_category == "High Risk", Customer.monthly_charges), else_=0)
        ).label("revenue_at_risk"),
        func.sum(
            case((Prediction.risk_category == "High Risk", 1), else_=0)
        ).label("high_risk_count"),
    ).join(Customer, isouter=True).one()

    snapshot = DashboardSnapshot(
        total_high_risk_customers = int(row.high_risk_count or 0),
        avg_churn_probability     = round(float(row.avg_prob or 0), 4),
        total_revenue_at_risk     = round(float(row.revenue_at_risk or 0), 2),
        insights={
            "total_predictions": int(row.total or 0),
        },
    )
    db.add(snapshot)
    db.commit()
    db.refresh(snapshot)

    return {
        "snapshot_id":               snapshot.snapshot_id,
        "total_high_risk_customers": snapshot.total_high_risk_customers,
        "avg_churn_probability":     snapshot.avg_churn_probability,
        "total_revenue_at_risk":     snapshot.total_revenue_at_risk,
        "insights":                  snapshot.insights,
        "snapshot_timestamp":        snapshot.snapshot_timestamp,
    }
