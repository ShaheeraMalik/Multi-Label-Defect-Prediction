# app3.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib

@st.cache_resource
def load_models():
    """Load pre-trained pipelines from disk."""
    return {
        "Logistic Regression": joblib.load("pipe_log.pkl"),
        "SVM":                 joblib.load("pipe_svm.pkl"),
        "DNN (MLPClassifier)": joblib.load("pipe_mlp.pkl"),
    }

def main():
    st.set_page_config(page_title="Multi-Label Defect Predictor", layout="wide")
    st.title("🛠️ Multi-Label Software Defect Prediction")
    st.write(
        "Upload a CSV with a **`report`** column and one column per label, "
        "select your model, and see the predictions with confidences."
    )

    models = load_models()
    model_name = st.sidebar.selectbox("Choose model", list(models.keys()))
    model = models[model_name]

    uploaded = st.file_uploader("📄 Upload defect CSV", type="csv")
    if not uploaded:
        return

    df_input = pd.read_csv(uploaded)
    if "report" not in df_input.columns:
        st.error("CSV must contain a column named `report`.")
        return

    # All the uploaded columns except 'report'
    all_labels = [c for c in df_input.columns if c != "report"]
    st.write(f"Found uploaded label columns (will be truncated/padded): {all_labels}")

    reports = df_input["report"].fillna("").tolist()
    with st.spinner("Predicting…"):
        preds  = model.predict(reports)         # shape (n_samples, n_labels_model)
        probas = model.predict_proba(reports)
        # if output is (n_samples, n_labels, 2), take the positive-class prob
        if probas.ndim == 3:
            probas = probas[:, :, 1]

    n_model_labels = preds.shape[1]
    if len(all_labels) < n_model_labels:
        st.error(
            f"Model produces {n_model_labels} label columns, "
            f"but your CSV only had {len(all_labels)}. "
            "Please include the correct set of label columns."
        )
        return

    # Truncate or pad the uploaded list so it lines up
    label_cols = all_labels[:n_model_labels]
    st.write(f"Using these labels for display: {label_cols}")

    # Build output
    df_display = df_input.copy()
    for i, label in enumerate(label_cols):
        # Yes/No prediction
        df_display[f"{label} (Pred)"]   = np.where(preds[:, i] == 1, "Yes", "No")
        # Confidence
        df_display[f"{label} (Conf %)"] = (probas[:, i] * 100).round(2).astype(str) + "%"

    st.subheader("Prediction Results")
    st.dataframe(df_display, use_container_width=True)

if __name__ == "__main__":
    main()
