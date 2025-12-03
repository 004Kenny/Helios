# streamlit_app.py — Helios V2 (Manual + Simulated Single Prediction)
from __future__ import annotations
import os
import time
from collections import Counter

import pandas as pd
import numpy as np
import streamlit as st

from learning_agent_regressor import LearningAgentKNNRegressor

st.set_page_config(page_title="Helios V2 — Sleep Quality (KNN Regressor)", layout="wide")

# --- Utils ---
def get_default_dataset_path() -> str:
    base = os.getcwd()
    return os.path.join(base, "data", "Sleep_health_and_lifestyle_dataset.csv")

def parse_k_grid(text: str) -> list[int]:
    """
    Parse a comma-separated k list. Fall back to a sensible odd grid.
    """
    try:
        ks = [int(x.strip()) for x in text.split(",") if x.strip()]
        ks = [k for k in ks if k > 0]
        return ks or [3, 5, 7, 9, 11, 13, 15]
    except Exception:
        return [3, 5, 7, 9, 11, 13, 15]

def badge(text: str, color: str = "#10b981"):  # emerald default
    st.markdown(
        f"""
        <div style="display:inline-block;padding:10px 14px;border-radius:10px;background:{color};color:white;font-weight:600;">
            {text}
        </div>
        """,
        unsafe_allow_html=True,
    )

# --- Simulation helpers ---
def _sample_numeric(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return 0.0
    mu, sigma = float(s.mean()), float(s.std(ddof=0)) or 1.0
    x = np.random.normal(mu, sigma * 0.65)  # narrower than full std
    # clip to observed range with a small margin
    lo, hi = float(s.min()), float(s.max())
    margin = (hi - lo) * 0.05
    return float(np.clip(x, lo - margin, hi + margin))

def _sample_categorical(series: pd.Series) -> str:
    vals = series.dropna().astype(str).tolist()
    if not vals:
        return ""
    counts = Counter(vals)
    choices, weights = zip(*counts.items())
    weights = np.array(weights, dtype=float)
    weights = weights / weights.sum()
    return str(np.random.choice(choices, p=weights))

def simulate_inputs(df_ref: pd.DataFrame | None, cols: list[str]) -> dict:
    """
    Produce a dict of feature->value using the training dataframe distribution.
    Falls back to simple defaults when df_ref is None.
    """
    sim = {}
    for c in cols:
        if df_ref is not None and c in df_ref.columns:
            s = df_ref[c]
            if pd.api.types.is_numeric_dtype(s) or pd.to_numeric(s, errors="coerce").notna().mean() > 0.9:
                sim[c] = _sample_numeric(s)
            else:
                sim[c] = _sample_categorical(s)
        else:
            # conservative fallback
            sim[c] = 0.0
    return sim

def animated_reveal(values: dict, delay_sec: float = 0.15):
    """
    Show key/value pairs one-by-one with a quick animation.
    """
    ph = st.empty()
    rows = []
    for k, v in values.items():
        pretty = f"{v:.2f}" if isinstance(v, (int, float, np.floating)) else str(v)
        rows.append((k, pretty))
        with ph.container():
            st.markdown("#### Incoming wearable data")
            for kk, vv in rows:
                st.write(f"- **{kk}**: {vv}")
        time.sleep(delay_sec)

# --- Session ---
if "knn_agent" not in st.session_state:
    st.session_state.knn_agent = LearningAgentKNNRegressor()
knn_agent: LearningAgentKNNRegressor = st.session_state.knn_agent

# --- Header ---
st.title("Helios V2 — Sleep Quality Prediction (KNN Regressor)")
st.caption(
    "Train on the Kaggle Sleep Health & Lifestyle dataset to predict numeric "
    "'Quality of Sleep' (0–10), then make single or batch predictions with neighbor explanations."
)

# --- Controls row ---
c1, c2, c3, c4 = st.columns([2.2, 1, 1, 1.2])

with c1:
    default_path = get_default_dataset_path()
    data_path = st.text_input("Dataset path", value=default_path)
    up_train = st.file_uploader("Or upload CSV", type=["csv"], key="train_uploader")
    if up_train is not None:
        os.makedirs("data", exist_ok=True)
        uploaded_train_path = os.path.join("data", "_uploaded_sleep_train.csv")
        with open(uploaded_train_path, "wb") as f:
            f.write(up_train.getbuffer())
        data_path = uploaded_train_path
        st.info(f"Using uploaded dataset: {uploaded_train_path}")

with c2:
    cv_folds = st.number_input("CV folds", min_value=3, max_value=10, value=5, step=1)

with c3:
    test_size = st.slider("Test size", min_value=0.05, max_value=0.4, value=0.20, step=0.05)

with c4:
    k_grid_text = st.text_input("k grid", value="3,5,7,9,11,13,15", help="Comma-separated odd ks")

train_clicked = st.button("Train KNN Regressor", type="primary", use_container_width=True)

# --- Train ---
if train_clicked:
    if not os.path.exists(data_path):
        st.error(f"Dataset not found at: {data_path}")
    else:
        try:
            k_grid = parse_k_grid(k_grid_text)
            res = knn_agent.train(
                csv_path=data_path,
                k_grid=k_grid,
                cv_folds=int(cv_folds),
                test_size=float(test_size),
            )
            st.success(f"Training complete. Selected k = {res['selected_k']}")
            with st.expander("Cross-Validation Summary", expanded=False):
                st.dataframe(pd.DataFrame(res["cv_table"]), use_container_width=True)

            mtr = res["metrics_test"]
            mtr_tr = res["metrics_train"]
            st.subheader("Evaluation (hold-out test)")
            a, b, c = st.columns(3)
            a.metric("RMSE", f"{mtr['rmse']:.3f}")
            b.metric("MAE", f"{mtr['mae']:.3f}")
            c.metric("R²", f"{mtr['r2']:.3f}")

            with st.expander("Training & Test metrics"):
                st.write("Train:", mtr_tr)
                st.write("Test:", mtr)
        except Exception as e:
            st.error(f"Training failed: {e}")

st.divider()

# --- Single Prediction (Manual or Simulated) ---
if knn_agent.pipeline is not None and knn_agent.X_cols_ is not None:
    st.subheader("Single Prediction")

    df_ref = getattr(knn_agent, "df_", None)
    feature_cols = list(knn_agent.X_cols_)

    mode_col, act_col = st.columns([1.2, 2])
    with mode_col:
        mode = st.radio("Input mode", ["Manual", "Simulated"], horizontal=True)

    if "sim_inputs" not in st.session_state:
        st.session_state.sim_inputs = {}

    if mode == "Manual":
        # MANUAL FORM
        inputs: dict = {}
        for col in feature_cols:
            if df_ref is not None and col in df_ref.columns:
                s = df_ref[col]
                if pd.api.types.is_numeric_dtype(s) or pd.to_numeric(s, errors="coerce").notna().mean() > 0.9:
                    val = float(pd.to_numeric(s, errors="coerce").median())
                    inputs[col] = st.number_input(col, value=val)
                else:
                    uniq = sorted([str(x) for x in s.dropna().unique().tolist()][:100])
                    default_idx = 0 if uniq else None
                    if uniq:
                        inputs[col] = st.selectbox(col, uniq, index=default_idx)
                    else:
                        inputs[col] = st.text_input(col, value="")
            else:
                inputs[col] = st.text_input(col, value="")

        c_pred1, _ = st.columns([1, 3])
        with c_pred1:
            predict_clicked = st.button("Predict (manual)", use_container_width=True)

        if predict_clicked:
            try:
                out = knn_agent.predict_row(inputs)
                pred_q = out["pred"]
                badge(f"Predicted Sleep Quality: {pred_q:.2f}/10", color="#2563eb")
                with st.expander("Show Nearest Neighbors"):
                    neigh = knn_agent.explain_neighbors(inputs)
                    st.dataframe(neigh, use_container_width=True)
            except Exception as e:
                st.error(f"Prediction failed: {e}")

    else:
        # SIMULATED MODE
        c_sim1, c_sim2, c_sim3 = st.columns([1.2, 1.2, 2])
        with c_sim1:
            n_reveal = st.slider("Reveal speed (ms/field)", 50, 400, 150, step=10)
        with c_sim2:
            jitter = st.slider("Numeric jitter (%)", 0, 30, 10, help="Adds randomness to keep values lively.")
        with c_sim3:
            st.caption("Click **Simulate Inputs + Predict** to auto-generate wearable-like data and run inference.")

        simulate_and_predict = st.button("Simulate Inputs + Predict", type="primary", use_container_width=True)

        if simulate_and_predict:
            # 1) Simulate from training distribution
            sim = simulate_inputs(df_ref, feature_cols)

            # Optional: jitter numeric values slightly
            for k, v in list(sim.items()):
                if isinstance(v, (int, float, np.floating)):
                    j = (jitter / 100.0)
                    sim[k] = float(v * (1.0 + np.random.uniform(-j, j)))

            # 2) Animated reveal (wearable vibe)
            animated_reveal(sim, delay_sec=float(n_reveal) / 1000.0)

            # 3) Predict
            try:
                out = knn_agent.predict_row(sim)
                pred_q = out["pred"]
                badge(f"Predicted Sleep Quality: {pred_q:.2f}/10", color="#2563eb")

                with st.expander("Show Nearest Neighbors"):
                    neigh = knn_agent.explain_neighbors(sim)
                    st.dataframe(neigh, use_container_width=True)
            except Exception as e:
                st.error(f"Prediction failed: {e}")

else:
    st.info("Train the model above to enable single predictions.")

st.divider()

# --- Batch Prediction ---
st.subheader("Batch Prediction (CSV)")
up = st.file_uploader("Upload CSV with the same feature columns", type=["csv"], key="batch_uploader")
if up is not None and knn_agent.pipeline is not None:
    try:
        tmp_path = os.path.join("data", "_batch_upload.csv")
        os.makedirs("data", exist_ok=True)
        with open(tmp_path, "wb") as f:
            f.write(up.getbuffer())
        pred_df = knn_agent.predict_csv(tmp_path)
        st.dataframe(pred_df.head(200), use_container_width=True)
        csv_bytes = pred_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download predictions CSV",
            data=csv_bytes,
            file_name="helios_v2_predictions.csv",
            mime="text/csv",
            use_container_width=True,
        )
    except Exception as e:
        st.error(f"Batch prediction failed: {e}")
elif up is not None:
    st.warning("Train or load a model first.")

st.divider()

# --- Model persistence ---
st.subheader("Model Persistence")
c_save, c_load = st.columns(2)
with c_save:
    if st.button("Save Model", use_container_width=True):
        try:
            model_path, meta_path = knn_agent.save()
            st.success(f"Saved: {model_path}  •  {meta_path}")
        except Exception as e:
            st.error(f"Save failed: {e}")
with c_load:
    if st.button("Load Model", use_container_width=True):
        try:
            meta = knn_agent.load()
            st.success(f"Loaded model with k={meta.best_k} and target={meta.target_name}")
        except Exception as e:
            st.error(f"Load failed: {e}")
