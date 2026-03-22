"""
database.py — ChurnGuard AI
SQLAlchemy engine + session factory.
Reads DATABASE_URL from the environment (falls back to SQLite for local dev).
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv

load_dotenv()

# ── Connection URL ─────────────────────────────────────────────────────────────
DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    "sqlite:///./churnguard.db",  # SQLite fallback for local development
)

# SQLite needs check_same_thread=False; PostgreSQL ignores this kwarg via connect_args
connect_args: dict = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

# ── Engine ─────────────────────────────────────────────────────────────────────
engine = create_engine(
    DATABASE_URL,
    connect_args=connect_args,
    pool_pre_ping=True,          # verify connections before reuse
    pool_size=10,                # PostgreSQL; ignored by SQLite
    max_overflow=20,
    echo=False,                  # set True to log all SQL in development
)

# ── Session factory ────────────────────────────────────────────────────────────
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# ── Declarative base ───────────────────────────────────────────────────────────
Base = declarative_base()


# ── Dependency (FastAPI) ───────────────────────────────────────────────────────
def get_db():
    """Yield a database session and guarantee cleanup."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
