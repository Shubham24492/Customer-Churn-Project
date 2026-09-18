"""Shared feature engineering for the Telco churn model."""

from __future__ import annotations

import numpy as np
import pandas as pd


SERVICE_COLUMNS = [
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


def add_features(data: pd.DataFrame) -> pd.DataFrame:
    """Clean raw Telco rows and add leakage-safe customer-level features."""
    frame = data.copy()
    frame = frame.drop(columns=["customerID", "Churn"], errors="ignore")

    frame["TotalCharges"] = pd.to_numeric(frame["TotalCharges"], errors="coerce")
    frame["TotalCharges"] = frame["TotalCharges"].fillna(
        frame["tenure"].fillna(0) * frame["MonthlyCharges"]
    )

    frame["tenure_years"] = frame["tenure"] / 12.0
    frame["avg_monthly_charge"] = np.where(
        frame["tenure"] > 0,
        frame["TotalCharges"] / frame["tenure"],
        frame["MonthlyCharges"],
    )
    frame["service_count"] = frame[SERVICE_COLUMNS].eq("Yes").sum(axis=1)
    return frame
