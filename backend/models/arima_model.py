# FILE: backend/models/arima_model.py
from __future__ import annotations
from typing import Tuple, Dict, Any, List
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import warnings


def train_best_sarimax(y: pd.Series) -> Tuple[Any, Dict[str, Any]]:
    """Grid-search compact (p,d,q)*(P,D,Q,12) and return best-AIC fit + config."""
    assert isinstance(y.index, pd.DatetimeIndex)
    seasonal = len(y) >= 24

    non_seasonal = [(p, d, q) for p in range(0, 3) for d in range(0, 2) for q in range(0, 3)]
    seasonal_orders = [(P, D, Q, 12) for P in range(0, 2) for D in range(0, 2) for Q in range(0, 2)] if seasonal else [(0, 0, 0, 0)]

    best_model, best_aic, best_cfg = None, np.inf, {}
    warnings.simplefilter("ignore", ConvergenceWarning)

    for order in non_seasonal:
        for s_order in seasonal_orders:
            try:
                model = SARIMAX(y, order=order, seasonal_order=s_order, enforce_stationarity=False, enforce_invertibility=False)
                fitted = model.fit(disp=False)
                if np.isfinite(fitted.aic) and fitted.aic < best_aic:
                    best_model, best_aic = fitted, fitted.aic
                    best_cfg = {"order": order, "seasonal_order": s_order, "aic": float(fitted.aic), "bic": float(fitted.bic)}
            except Exception:
                continue

    if best_model is None:
        raise RuntimeError("Failed to fit ARIMA model on the provided data.")
    return best_model, best_cfg


def forecast_with_model(model, horizon: int, conf_level: float) -> Tuple[pd.DatetimeIndex, np.ndarray, np.ndarray, np.ndarray]:
    fc = model.get_forecast(steps=horizon)
    mean = fc.predicted_mean
    conf = fc.conf_int(alpha=1.0 - conf_level)

    last = model.data.row_labels[-1]
    if not isinstance(last, pd.Timestamp):
        last = pd.to_datetime(last)
    idx = pd.date_range(last + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")

    return idx, mean.values, conf.iloc[:, 0].values, conf.iloc[:, 1].values


def holdout_metrics(y: pd.Series, fitted_model) -> Dict[str, float]:
    if len(y) < 12:
        return {"mape": float("nan"), "rmse": float("nan")}

    k = min(12, max(1, len(y)//3))
    y_train, y_test = y.iloc[:-k], y.iloc[-k:]
    try:
        m, _ = train_best_sarimax(y_train)
        fc = m.get_forecast(steps=k).predicted_mean
        y_pred = pd.Series(fc.values, index=y_test.index)
    except Exception:
        y_pred = pd.Series(fitted_model.get_prediction().predicted_mean.iloc[-k:], index=y_test.index)

    eps = 1e-9
    mape = float(np.mean(np.abs((y_test - y_pred) / np.maximum(eps, y_test))) * 100.0)
    rmse = float(np.sqrt(np.mean((y_test - y_pred) ** 2)))
    return {"mape": mape, "rmse": rmse}
