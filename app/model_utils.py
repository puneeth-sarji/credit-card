# app/model_utils.py
"""
Model utilities: load data, balance, train RandomForest, single prediction, save/load.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)
import joblib
import os

def load_dataset(file_like):
    """Load CSV into a pandas DataFrame. `file_like` can be path or file-like object."""
    df = pd.read_csv(file_like)
    return df

def basic_checks(df, target_col="Class"):
    if target_col not in df.columns:
        raise ValueError(f"Dataset missing required column: '{target_col}'")
    # Remove non-numeric columns (if any) except the target
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col not in numeric_cols:
        # keep target even if int type detection issues
        numeric_cols.append(target_col)
    return df[numeric_cols]

def balance_undersample(df, target_col="Class", random_state=42):
    """Undersample majority class to match minority class count."""
    legit = df[df[target_col] == 0]
    fraud = df[df[target_col] == 1]
    if fraud.empty:
        raise ValueError("No fraud samples found in dataset (Class == 1).")
    n = len(fraud)
    legit_sample = legit.sample(n=n, random_state=random_state)
    balanced = pd.concat([legit_sample, fraud], axis=0).sample(frac=1, random_state=random_state).reset_index(drop=True)
    return balanced

def train_random_forest(df_balanced, target_col="Class", test_size=0.2, random_state=42, n_estimators=200):
    """
    Train RandomForestClassifier on balanced dataset.
    Returns: model, X_train, X_test, y_train, y_test, y_score, metrics
    """
    X = df_balanced.drop(columns=[target_col], axis=1)
    y = df_balanced[target_col]

    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
    rf.fit(X_train, y_train)

    y_pred_train = rf.predict(X_train)
    y_pred_test = rf.predict(X_test)

    if hasattr(rf, "predict_proba"):
        y_score = rf.predict_proba(X_test)[:, 1]
    else:
        y_score = rf.predict(X_test)

    metrics = {
        "train_accuracy": float(accuracy_score(y_train, y_pred_train)),
        "test_accuracy": float(accuracy_score(y_test, y_pred_test)),
        "precision": float(precision_score(y_test, y_pred_test, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred_test, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred_test, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_score)) if len(np.unique(y_test)) > 1 else None,
        "avg_precision": float(average_precision_score(y_test, y_score)) if len(np.unique(y_test)) > 1 else None,
        "feature_names": feature_names
    }

    return rf, X_train, X_test, y_train, y_test, y_score, metrics

def predict_single(model, feature_cols, values_dict):
    """Predict a single transaction. feature_cols: list, values_dict: {col: val}"""
    row = [float(values_dict.get(c, 0.0)) for c in feature_cols]
    arr = np.array(row).reshape(1, -1)
    pred = int(model.predict(arr)[0])
    prob = float(model.predict_proba(arr)[0][1]) if hasattr(model, "predict_proba") else None
    return {"prediction": pred, "probability": prob}

def save_model(model, path="models/rf_model.pkl"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    return path

def load_model(path="models/rf_model.pkl"):
    return joblib.load(path)
