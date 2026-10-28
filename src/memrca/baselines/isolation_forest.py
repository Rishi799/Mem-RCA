"""Normal-only Isolation Forest model for one microservice's metric windows."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer


class ServiceIsolationForest:
    """Fit on normal training windows; return higher scores for anomalies."""

    def __init__(self, random_state: int = 17, n_estimators: int = 100) -> None:
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.columns: list[str] = []
        self.imputer: SimpleImputer | None = None
        self.model: IsolationForest | None = None

    def fit(self, normal_windows: pd.DataFrame) -> "ServiceIsolationForest":
        numeric = normal_windows.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
        self.columns = [col for col in numeric if numeric[col].notna().any()]
        if len(numeric) < 2 or not self.columns:
            raise ValueError("Need at least two normal windows with observed metric values")
        numeric = numeric[self.columns]
        self.imputer = SimpleImputer(strategy="median", add_indicator=True)
        values = self.imputer.fit_transform(numeric)
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=min(256, len(values)),
            contamination="auto",
            random_state=self.random_state,
            n_jobs=-1,
        ).fit(values)
        return self

    def anomaly_score(self, windows: pd.DataFrame) -> np.ndarray:
        if self.imputer is None or self.model is None:
            raise RuntimeError("Fit the model before scoring windows")
        numeric = windows.reindex(columns=self.columns).select_dtypes(include=[np.number])
        numeric = numeric.replace([np.inf, -np.inf], np.nan)
        values = self.imputer.transform(numeric)
        return -self.model.decision_function(values)
