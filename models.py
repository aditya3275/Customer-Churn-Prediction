"""
models.py — ChurnGuard AI
SQLAlchemy ORM models for all five database tables.
"""

import os
from datetime import datetime, timezone

from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime,
    ForeignKey, Index, Text,
)
from sqlalchemy.orm import relationship

# JSON type: use native JSONB on PostgreSQL, TEXT-backed JSON on SQLite
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./churnguard.db")
if DATABASE_URL.startswith("postgresql"):
    from sqlalchemy.dialects.postgresql import JSONB as JSONType
else:
    from sqlalchemy.types import JSON as JSONType

from database import Base


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ─────────────────────────────────────────────────────────────────────────────
# 1. customers
# ─────────────────────────────────────────────────────────────────────────────
class Customer(Base):
    __tablename__ = "customers"

    customer_id     = Column(String(64), primary_key=True, index=True)
    gender          = Column(String(10))
    senior_citizen  = Column(Boolean, default=False)
    has_partner     = Column(Boolean, default=False)
    has_dependents  = Column(Boolean, default=False)
    tenure          = Column(Integer)
    contract_type   = Column(String(32))
    monthly_charges = Column(Float)
    total_charges   = Column(Float)
    internet_service = Column(String(32))
    payment_method  = Column(String(64))
    created_at      = Column(DateTime(timezone=True), default=_now)
    updated_at      = Column(DateTime(timezone=True), default=_now, onupdate=_now)

    # relationships
    predictions      = relationship("Prediction",      back_populates="customer",
                                    cascade="all, delete-orphan")
    retention_actions = relationship("RetentionAction", back_populates="customer",
                                    cascade="all, delete-orphan")
    audit_logs       = relationship("AuditLog",        back_populates="customer")


# ─────────────────────────────────────────────────────────────────────────────
# 2. predictions
# ─────────────────────────────────────────────────────────────────────────────
class Prediction(Base):
    __tablename__ = "predictions"

    prediction_id          = Column(Integer, primary_key=True, autoincrement=True)
    customer_id            = Column(String(64), ForeignKey("customers.customer_id"),
                                   nullable=False, index=True)
    churn_prediction       = Column(String(4))           # "Yes" | "No"
    churn_probability      = Column(Float, nullable=False, index=True)
    risk_category          = Column(String(16))          # High | Medium | Low
    confidence_score       = Column(Float)
    lifetime_value         = Column(Float)
    feature_importance_snapshot = Column(JSONType)       # dict persisted as JSON
    predicted_at           = Column(DateTime(timezone=True), default=_now, index=True)

    # relationships
    customer          = relationship("Customer",        back_populates="predictions")
    retention_actions = relationship("RetentionAction", back_populates="prediction",
                                    cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_predictions_customer_predicted", "customer_id", "predicted_at"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3. retention_actions
# ─────────────────────────────────────────────────────────────────────────────
class RetentionAction(Base):
    __tablename__ = "retention_actions"

    action_id           = Column(Integer, primary_key=True, autoincrement=True)
    customer_id         = Column(String(64), ForeignKey("customers.customer_id"),
                                nullable=False, index=True)
    prediction_id       = Column(Integer, ForeignKey("predictions.prediction_id"),
                                nullable=False, index=True)
    risk_level          = Column(String(16))   # High | Medium | Low
    recommended_action  = Column(Text)
    action_taken        = Column(Text, nullable=True)
    outcome             = Column(String(64), nullable=True)   # e.g. "retained" | "churned"
    created_at          = Column(DateTime(timezone=True), default=_now)

    # relationships
    customer   = relationship("Customer",   back_populates="retention_actions")
    prediction = relationship("Prediction", back_populates="retention_actions")


# ─────────────────────────────────────────────────────────────────────────────
# 4. audit_logs
# ─────────────────────────────────────────────────────────────────────────────
class AuditLog(Base):
    __tablename__ = "audit_logs"

    log_id           = Column(Integer, primary_key=True, autoincrement=True)
    endpoint         = Column(String(256))
    method           = Column(String(8))
    customer_id      = Column(String(64), ForeignKey("customers.customer_id"),
                              nullable=True, index=True)
    status_code      = Column(Integer)
    response_time_ms = Column(Float)
    error_message    = Column(Text, nullable=True)
    timestamp        = Column(DateTime(timezone=True), default=_now, index=True)

    # relationship (nullable FK — logs exist even with no customer)
    customer = relationship("Customer", back_populates="audit_logs")


# ─────────────────────────────────────────────────────────────────────────────
# 5. dashboard_snapshots
# ─────────────────────────────────────────────────────────────────────────────
class DashboardSnapshot(Base):
    __tablename__ = "dashboard_snapshots"

    snapshot_id                = Column(Integer, primary_key=True, autoincrement=True)
    total_high_risk_customers  = Column(Integer, default=0)
    avg_churn_probability      = Column(Float, default=0.0)
    total_revenue_at_risk      = Column(Float, default=0.0)
    insights                   = Column(JSONType)    # grouped / segmented metrics
    snapshot_timestamp         = Column(DateTime(timezone=True), default=_now, index=True)
