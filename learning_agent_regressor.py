# Helios V2 — KNN Regressor (Sleep Quality)
# Predicts numeric "Quality of Sleep" using the Kaggle Sleep Health & Lifestyle dataset.

from __future__ import annotations
import os
import json
import re
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Optional, Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.impute import SimpleImputer
import joblib

# ---------------------
# Paths
# ---------------------
BASE_DIR = os.path.dirname(__file__) if '__file__' in globals() else os.getcwd()
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Default target detection (regex for typical Kaggle column name)
TARGET_REGEX = re.compile(r"quality\s*of\s*sleep|sleep\s*quality", re.I)

# ---------------------
# Metadata
# ---------------------
@dataclass
class RegFitMetadata:
    target_name: str
    best_k: int
    numeric_columns: List[str]
    categorical_columns: List[str]
    cv_folds: int
    scorings: Dict[str, float]

# ---------------------
# Core Agent (Regressor)
# ---------------------
class LearningAgentKNNRegressor:


    def __init__(self):
        self.pipeline: Optional[Pipeline] = None
        self.metadata: Optional[RegFitMetadata] = None
        self.df_: Optional[pd.DataFrame] = None        # training subset for neighbor display
        self.X_cols_: Optional[List[str]] = None
        self.target_name_: Optional[str] = None
        self.best_k_: Optional[int] = None

    # ---------- column detection ----------
    def detect_columns(self, df: pd.DataFrame, target: Optional[str] = None) -> Tuple[List[str], List[str], str]:
        # Resolve target column (prefer regex match for Quality of Sleep)
        t = target
        if t is None:
            t = next((c for c in df.columns if TARGET_REGEX.search(c)), None)
        if t is None:
            raise ValueError("Target column for Sleep Quality not found. Expected something like 'Quality of Sleep'.")
        if t not in df.columns:
            raise ValueError(f"Target column '{t}' not found in dataframe.")

        # Parse blood pressure "120/80" -> Systolic/Diastolic if present
        bp = next((c for c in df.columns if re.search(r"blood\s*pressure", c, re.I)), None)
        if bp is not None and df[bp].astype(str).str.contains("/").any():
            bp_split = df[bp].astype(str).str.extract(r"(?P<Systolic>\d+)\s*/\s*(?P<Diastolic>\d+)")
            df["Systolic"] = pd.to_numeric(bp_split["Systolic"], errors="coerce")
            df["Diastolic"] = pd.to_numeric(bp_split["Diastolic"], errors="coerce")

        # Partition numeric vs categorical (excluding target)
        features = [c for c in df.columns if c != t]
        num_cols, cat_cols = [], []
        for c in features:
            if pd.api.types.is_numeric_dtype(df[c]):
                num_cols.append(c)
            else:
                # If most values can coerce to numeric, treat as numeric
                coerced = pd.to_numeric(df[c], errors="coerce")
                if coerced.notna().mean() > 0.9:
                    num_cols.append(c)
                else:
                    cat_cols.append(c)
        return num_cols, cat_cols, t

    # ---------- training ----------
    def train(
        self,
        csv_path: str,
        target: Optional[str] = None,
        k_grid: List[int] = [13],
        cv_folds: int = 5,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Dict[str, Any]:
        df = pd.read_csv(csv_path)

        num_cols, cat_cols, t = self.detect_columns(df, target)
        y = pd.to_numeric(df[t], errors="coerce")
        X = df[num_cols + cat_cols].copy()

        # Drop rows with missing target
        mask = y.notna()
        X, y = X.loc[mask], y.loc[mask]

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=True
        )

        # Preprocessing with imputers
        pre = ColumnTransformer(
            transformers=[
                ("cat", Pipeline(steps=[
                    ("impute", SimpleImputer(strategy="most_frequent")),
                    ("ohe", OneHotEncoder(handle_unknown="ignore"))
                ]), cat_cols),
                ("num", Pipeline(steps=[
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler())
                ]), num_cols),
            ],
            remainder="drop",
        )

        # CV over k; select by lowest RMSE (tie-break by lowest MAE)
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        cv_rows = []
        for k in k_grid:
            rmses, maes = [], []
            for tr_idx, va_idx in kf.split(X_train):
                Xtr, Xva = X_train.iloc[tr_idx], X_train.iloc[va_idx]
                ytr, yva = y_train.iloc[tr_idx], y_train.iloc[va_idx]
                pipe = Pipeline([
                    ("pre", pre),
                    ("reg", KNeighborsRegressor(n_neighbors=k)),
                ])
                pipe.fit(Xtr, ytr)
                yhat = pipe.predict(Xva)
                # Older sklearn may not support squared=False; compute RMSE manually
                rmse = float(np.sqrt(mean_squared_error(yva, yhat)))
                mae  = float(mean_absolute_error(yva, yhat))
                rmses.append(rmse); maes.append(mae)
            cv_rows.append({
                "k": k,
                "rmse": float(np.mean(rmses)),
                "mae": float(np.mean(maes)),
            })
        cv_rows.sort(key=lambda d: (d["rmse"], d["mae"]))
        best = cv_rows[0]
        best_k = int(best["k"])

        # Fit final model on training data only
        pipeline = Pipeline([
            ("pre", pre),
            ("reg", KNeighborsRegressor(n_neighbors=best_k)),
        ])
        pipeline.fit(X_train, y_train)

        # Keep only training rows for neighbor display
        self.df_ = df.loc[X_train.index].copy()

        # Training-set metrics
        y_hat_tr = pipeline.predict(X_train)
        rmse_tr = float(np.sqrt(mean_squared_error(y_train, y_hat_tr)))
        mae_tr  = float(mean_absolute_error(y_train, y_hat_tr))
        r2_tr   = float(r2_score(y_train, y_hat_tr))

        # Test-set metrics
        y_hat_te = pipeline.predict(X_test)
        rmse_te = float(np.sqrt(mean_squared_error(y_test, y_hat_te)))
        mae_te  = float(mean_absolute_error(y_test, y_hat_te))
        r2_te   = float(r2_score(y_test, y_hat_te))

        self.pipeline = pipeline
        self.target_name_ = t
        self.best_k_ = best_k
        self.X_cols_ = num_cols + cat_cols
        self.metadata = RegFitMetadata(
            target_name=t,
            best_k=best_k,
            numeric_columns=num_cols,
            categorical_columns=cat_cols,
            cv_folds=cv_folds,
            scorings={
                "cv_best_rmse": best["rmse"],
                "cv_best_mae": best["mae"],
                "train_rmse": rmse_tr,
                "train_mae": mae_tr,
                "train_r2": r2_tr,
                "test_rmse": rmse_te,
                "test_mae": mae_te,
                "test_r2": r2_te,
            },
        )

        return {
            "selected_k": best_k,
            "cv_table": cv_rows,
            "metrics_train": {
                "rmse": rmse_tr,
                "mae": mae_tr,
                "r2": r2_tr,
            },
            "metrics_test": {
                "rmse": rmse_te,
                "mae": mae_te,
                "r2": r2_te,
            },
        }

    # ---------- single-row prediction ----------
    def predict_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        if self.pipeline is None or self.X_cols_ is None:
            raise RuntimeError("Model not trained/loaded.")
        x = {c: row.get(c, np.nan) for c in self.X_cols_}
        x_df = pd.DataFrame([x], columns=self.X_cols_)
        yhat = float(self.pipeline.predict(x_df)[0])
        return {"pred": yhat}

    # ---------- neighbor explainability ----------
    def explain_neighbors(self, row: Dict[str, Any], n_neighbors: Optional[int] = None) -> pd.DataFrame:
        if self.pipeline is None or self.df_ is None or self.X_cols_ is None:
            raise RuntimeError("Model not trained/loaded.")
        k = int(n_neighbors or self.best_k_ or 5)
        x = {c: row.get(c, np.nan) for c in self.X_cols_}
        x_df = pd.DataFrame([x], columns=self.X_cols_)

        knn: KNeighborsRegressor = self.pipeline.named_steps["reg"]
        pre = self.pipeline.named_steps["pre"]
        x_vec = pre.transform(x_df)
        distances, indices = knn.kneighbors(x_vec, n_neighbors=k, return_distance=True)
        idxs = indices[0].tolist()
        dists = distances[0].tolist()

        out = self.df_.iloc[idxs][self.X_cols_ + [self.target_name_]].copy()
        out.rename(columns={self.target_name_: f"{self.target_name_}_(y)"}, inplace=True)
        out["distance"] = dists
        return out.reset_index(drop=True)

    # ---------- batch CSV prediction ----------
    def predict_csv(self, csv_path: str, output_path: Optional[str] = None) -> pd.DataFrame:
        if self.pipeline is None or self.X_cols_ is None:
            raise RuntimeError("Model not trained/loaded.")
        df = pd.read_csv(csv_path)

        # Ensure all training-time columns exist
        for c in self.X_cols_:
            if c not in df.columns:
                df[c] = np.nan
        x = df[self.X_cols_].copy()

        yhat = self.pipeline.predict(x)
        out = df.copy()
        out["pred"] = yhat
        if output_path:
            out.to_csv(output_path, index=False)
        return out

    # ---------- persistence ----------
    def save(self, base_name: str = "knn_reg_pipeline") -> Tuple[str, str]:
        if self.pipeline is None or self.metadata is None:
            raise RuntimeError("Nothing to save.")
        model_path = os.path.join(MODELS_DIR, f"{base_name}.joblib")
        meta_path = os.path.join(MODELS_DIR, f"{base_name}_metadata.json")
        joblib.dump(self.pipeline, model_path)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(asdict(self.metadata), f, indent=2)
        return model_path, meta_path

    def load(self, base_name: str = "knn_reg_pipeline") -> RegFitMetadata:
        model_path = os.path.join(MODELS_DIR, f"{base_name}.joblib")
        meta_path = os.path.join(MODELS_DIR, f"{base_name}_metadata.json")
        self.pipeline = joblib.load(model_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.metadata = RegFitMetadata(**meta)
        self.best_k_ = self.metadata.best_k
        self.target_name_ = self.metadata.target_name
        self.X_cols_ = self.metadata.numeric_columns + self.metadata.categorical_columns
        return self.metadata