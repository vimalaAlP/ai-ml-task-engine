# create_dummy_models.py
import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

ROOT = Path(__file__).resolve().parent
models_dir = ROOT / "models"
models_dir.mkdir(exist_ok=True)

# Try to use cleaned data if present
data_path = ROOT / "data" / "cleaned_dataset.csv"
if data_path.exists():
    df = pd.read_csv(data_path)
    # pick numeric features that likely exist
    numeric_cols = [c for c in df.columns if df[c].dtype in [int, float] and c not in ('actual_time_hours','delayed')]
    if len(numeric_cols) < 1:
        numeric_cols = df.columns[:3].tolist()
    X = df[numeric_cols].fillna(0).values
    # target for time (if available)
    if 'actual_time_hours' in df.columns:
        y_reg = df['actual_time_hours'].fillna(df['actual_time_hours'].median()).values
        reg = RandomForestRegressor(n_estimators=50, random_state=42)
        reg.fit(X, y_reg)
    else:
        reg = DummyRegressor(strategy="median")
        reg.fit(np.zeros((10,1)), np.zeros(10))

    # target for delayed (if available)
    if 'delayed' in df.columns:
        y_clf = df['delayed'].fillna(0).astype(int).values
        clf = RandomForestClassifier(n_estimators=50, random_state=42)
        clf.fit(X, y_clf)
    else:
        clf = DummyClassifier(strategy="most_frequent")
        clf.fit([[0]]*10, [0]*10)
else:
    # no data -> create simple dummy models
    reg = DummyRegressor(strategy="median")
    reg.fit([[0]]*10, [0]*10)
    clf = DummyClassifier(strategy="most_frequent")
    clf.fit([[0]]*10, [0]*10)

joblib.dump({'model': reg, 'model_version': 'time_v1_placeholder'}, models_dir / 'time_estimator.joblib')
joblib.dump({'model': clf, 'model_version': 'risk_v1_placeholder'}, models_dir / 'risk_predictor.joblib')

print("Placeholder models saved to:", models_dir.resolve())
