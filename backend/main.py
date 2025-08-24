# FILE: backend/main.py
from __future__ import annotations

import os
import io
from datetime import datetime
from typing import Optional, Dict, Any, List

import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from database.db_manager import init_db, upsert_monthly_records, fetch_categories, fetch_monthly_records, fetch_series
from utils.data_processor import parse_upload_to_monthly_long, pivot_wide, summarize_wide
from models.arima_model import train_best_sarimax, forecast_with_model, holdout_metrics

# ----------------------
# App / Config
# ----------------------
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./sales.db")
app = FastAPI(title="Medicine Sales ARIMA API", version="1.0.0")

# CORS for dev — restrict in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize DB (creates tables if needed)
init_db(DATABASE_URL)

# ----------------------
# Schemas
# ----------------------
class PredictRequest(BaseModel):
    category: str = Field(..., description="ATC category to forecast (e.g., M01AB)")
    periods: int = Field(..., ge=1, le=36, description="Forecast horizon in months")
    confidence_level: float = Field(0.95, ge=0.5, le=0.99)


# ----------------------
# Endpoints
# ----------------------
@app.get("/api/health")
def health() -> Dict[str, Any]:
    try:
        # simple round-trip
        cats = fetch_categories()
        status = "connected"
    except Exception:
        status = "disconnected"
    return {"status": "ok", "database": status, "time": datetime.utcnow().isoformat() + "Z"}


@app.get("/api/categories")
def categories() -> Dict[str, List[str]]:
    return {"categories": fetch_categories()}


@app.get("/api/data")
def get_data(start_date: Optional[str] = None, end_date: Optional[str] = None, categories: Optional[str] = None) -> Dict[str, Any]:
    rows = fetch_monthly_records(start_date=start_date, end_date=end_date, categories=categories)
    if not rows:
        raise HTTPException(status_code=404, detail="No data found for the given filters.")

    df = pd.DataFrame(rows, columns=["date", "category", "quantity"])
    df["date"] = pd.to_datetime(df["date"])  # ensure Timestamp

    wide = pivot_wide(df)
    summary = summarize_wide(wide)

    # Serialize dates
    wide_out = wide.copy()
    wide_out["date"] = wide_out["date"].dt.strftime("%Y-%m-%d")

    return {"data": wide_out.to_dict(orient="records"), "summary_stats": summary, "categories": fetch_categories()}


@app.post("/api/upload")
def upload(file: UploadFile = File(...)) -> Dict[str, Any]:
    content = file.file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    monthly_long = parse_upload_to_monthly_long(filename=file.filename or "", content=content)
    inserted = upsert_monthly_records(monthly_long)
    cats = fetch_categories()
    return {"message": f"Ingested {inserted} monthly records across {monthly_long['category'].nunique()} categories.", "categories": cats}


@app.post("/api/predict")
def predict(req: PredictRequest) -> Dict[str, Any]:
    y = fetch_series(req.category)
    if len(y) < 6:
        raise HTTPException(status_code=400, detail="Insufficient data for forecasting (need at least 6 months).")

    model, cfg = train_best_sarimax(y)

    dates, mean, lower, upper = forecast_with_model(model, horizon=req.periods, conf_level=req.confidence_level)
    metrics = holdout_metrics(y, model)
    metrics.update({
        "aic": float(cfg.get("aic", float("nan"))),
        "bic": float(cfg.get("bic", float("nan"))),
        "arima_order": list(cfg.get("order", (0, 0, 0))),
        "seasonal_order": list(cfg.get("seasonal_order", (0, 0, 0, 0))),
        "model_valid": True
    })

    return {
        "category": req.category,
        "dates": [d.strftime("%Y-%m-%d") for d in dates],
        "predictions": [float(x) for x in mean],
        "lower_ci": [float(x) for x in lower],
        "upper_ci": [float(x) for x in upper],
        "model_metrics": metrics,
    }


@app.get("/api/stats/{category}")
def stats(category: str) -> Dict[str, Any]:
    y = fetch_series(category)
    df = y.to_frame(name="quantity")
    return {
        "category": category,
        "record_count": int(len(df)),
        "start_date": df.index.min().strftime("%Y-%m-%d"),
        "end_date": df.index.max().strftime("%Y-%m-%d"),
        "avg_quantity": float(df["quantity"].mean()),
        "total_quantity": float(df["quantity"].sum()),
        "min_quantity": float(df["quantity"].min()),
        "max_quantity": float(df["quantity"].max()),
    }






# FILE: backend/utils/validators.py
# (Placeholder for future custom validators & pydantic helpers.)
# Example idea: validate category strings to match expected ATC codes, date ranges, etc.
