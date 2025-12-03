"""
Streamlit frontend for CardShield - Credit Card Fraud Detection.
Run: streamlit run app/main_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
import os
import matplotlib.pyplot as plt

from model_utils import (
    load_dataset, basic_checks, balance_undersample,
    train_random_forest, predict_single, save_model
)
from plots import (
    plot_confusion_matrix, plot_roc_curve,
    plot_precision_recall, plot_feature_importance,
    plot_correlation_heatmap
)

st.set_page_config(page_title="CardShield - Credit Card Fraud Detection", layout="wide")

# ----------------- HEADER with LOGO -----------------
# Attempt to use local asset; if missing, show a small inline svg fallback.
logo_path = "assets/cardshield.png"  # recommended: add a PNG file here

st.markdown("""
    <style>
    .header-container {
        display: flex;
        align-items: center;
        background-color: #0A1F44;
        padding: 12px 18px;
        border-radius: 8px;
        margin-bottom: 18px;
    }
    .header-logo {
        height: 56px;
        margin-right: 14px;
    }
    .header-title {
        color: white;
        font-size: 26px;
        font-weight: 700;
        margin-bottom: 2px;
    }
    .header-tagline {
        color: #B7D0FF;
        font-size: 13px;
        margin-top: -2px;
    }
    </style>
""", unsafe_allow_html=True)

if os.path.exists(logo_path):
    # Simple header with the local image
    st.markdown(
        f"""
        <div class="header-container">
            <img src="{logo_path}" class="header-logo"/>
            <div>
                <div class="header-title">CardShield — Credit Card Fraud Detection</div>
                <div class="header-tagline">AI-powered protection for secure financial transactions</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    # Inline SVG fallback (tiny shield)
    svg = """
    <svg xmlns='http://www.w3.org/2000/svg' width='64' height='64' viewBox='0 0 24 24' fill='none' stroke='white' stroke-width='1.2'>
      <path d='M12 2l7 3v5c0 5-3.6 9.6-7 11-3.4-1.4-7-6-7-11V5l7-3z' fill='#0A84FF'/>
      <path d='M9.5 11l1.5 2 3-3' stroke='white' stroke-width='1.6' stroke-linecap='round' stroke-linejoin='round'/>
    </svg>
    """
    st.markdown(
        f"""
        <div class="header-container">
            <div style="width:56px;height:56px;margin-right:12px">{svg}</div>
            <div>
                <div class="header-title">CardShield — Credit Card Fraud Detection</div>
                <div class="header-tagline">AI-powered protection for secure financial transactions</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
# ----------------------------------------------------


# session state init
if "model" not in st.session_state:
    st.session_state.model = None
if "feature_cols" not in st.session_state:
    st.session_state.feature_cols = None
if "metrics" not in st.session_state:
    st.session_state.metrics = None
if "last_plots" not in st.session_state:
    st.session_state.last_plots = {}

st.sidebar.title("Navigation")
mode = st.sidebar.radio("Choose mode", ["Upload & Train", "Test Transaction", "About / Save Model"])

# helper: save matplotlib fig to bytes for download
def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return buf

# --------------------------
# Upload & Train page
# --------------------------
if mode == "Upload & Train":
    st.header("1) Upload dataset (CSV) and train model")
    uploaded_file = st.file_uploader("Upload CSV (must include column 'Class')", type=["csv"])
    if uploaded_file is not None:
        try:
            df = load_dataset(uploaded_file)
            df = basic_checks(df, target_col="Class")
        except Exception as e:
            st.error(f"Failed to load dataset: {e}")
            st.stop()

        st.subheader("Preview (first 5 rows)")
        st.dataframe(df.head())

        st.subheader("Dataset shape & class counts")
        st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
        st.write(df['Class'].value_counts().rename_axis('Class').reset_index(name='count'))

        if st.button("▶️ Balance + Train Random Forest"):
            with st.spinner("Balancing dataset and training Random Forest..."):
                try:
                    balanced = balance_undersample(df, target_col="Class", random_state=42)
                    model, X_train, X_test, y_train, y_test, y_score, metrics = train_random_forest(
                        balanced, target_col="Class", test_size=0.2, random_state=42, n_estimators=200
                    )
                except Exception as e:
                    st.error(f"Training failed: {e}")
                    st.stop()

                # Save into session
                st.session_state.model = model
                st.session_state.feature_cols = X_train.columns.tolist()
                st.session_state.metrics = metrics
                st.success("Model trained successfully ✅")

                # ---------- Metrics table ----------
                st.subheader("📊 Metrics Summary")

                y_pred_test = model.predict(X_test)
                from sklearn.metrics import confusion_matrix
                cm = confusion_matrix(y_test, y_pred_test)
                try:
                    tn, fp, fn, tp = cm.ravel()
                except Exception:
                    tn = int(cm[0, 0]) if cm.shape == (2, 2) else 0
                    fp = int(cm[0, 1]) if cm.shape == (2, 2) else 0
                    fn = int(cm[1, 0]) if cm.shape == (2, 2) else 0
                    tp = int(cm[1, 1]) if cm.shape == (2, 2) else 0

                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                error_rate = 1 - metrics["test_accuracy"] if metrics.get("test_accuracy") is not None else None

                metrics_table = {
                    "Metric": [
                        "Accuracy",
                        "Precision",
                        "Recall (Sensitivity)",
                        "Specificity",
                        "F1 Score",
                        "Error Rate",
                        "ROC AUC",
                        "Average Precision (AP)"
                    ],
                    "Value": [
                        round(metrics.get("test_accuracy", 0), 4),
                        round(metrics.get("precision", 0), 4),
                        round(metrics.get("recall", 0), 4),
                        round(specificity, 4),
                        round(metrics.get("f1", 0), 4),
                        round(error_rate, 4) if error_rate is not None else None,
                        round(metrics.get("roc_auc", 0), 4) if metrics.get("roc_auc") is not None else None,
                        round(metrics.get("avg_precision", 0), 4) if metrics.get("avg_precision") is not None else None
                    ]
                }
                st.table(pd.DataFrame(metrics_table))

                st.markdown("**Numeric confusion matrix (tn, fp, fn, tp)**")
                st.write({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})

                # ---------- Plots ----------
                st.subheader("Evaluation Plots (Test set)")
                col1, col2 = st.columns(2)

                # Confusion matrix
                with col1:
                    st.markdown("**Confusion Matrix**")
                    fig_cm = plot_confusion_matrix(y_test, y_pred_test)
                    st.pyplot(fig_cm)
                    st.download_button("Download Confusion Matrix (PNG)", data=fig_to_bytes(fig_cm), file_name="confusion_matrix.png", mime="image/png")

                # ROC Curve
                with col2:
                    st.markdown("**ROC Curve**")
                    fig_roc = plot_roc_curve(y_test, y_score)
                    st.pyplot(fig_roc)
                    st.download_button("Download ROC Curve (PNG)", data=fig_to_bytes(fig_roc), file_name="roc_curve.png", mime="image/png")

                col3, col4 = st.columns(2)
                with col3:
                    st.markdown("**Precision-Recall Curve**")
                    fig_pr = plot_precision_recall(y_test, y_score)
                    st.pyplot(fig_pr)
                    st.download_button("Download Precision-Recall (PNG)", data=fig_to_bytes(fig_pr), file_name="precision_recall.png", mime="image/png")

                with col4:
                    st.markdown("**Feature Importances (top 10)**")
                    fig_fi = plot_feature_importance(model, X_train.columns.tolist(), top_n=10)
                    st.pyplot(fig_fi)
                    st.download_button("Download Feature Importances (PNG)", data=fig_to_bytes(fig_fi), file_name="feature_importance.png", mime="image/png")

                st.markdown("**Correlation Heatmap (original uploaded dataset)**")
                numeric_df = df.select_dtypes(include=[np.number])
                fig_corr = plot_correlation_heatmap(numeric_df)
                st.pyplot(fig_corr)
                st.download_button("Download Correlation Heatmap (PNG)", data=fig_to_bytes(fig_corr), file_name="correlation_heatmap.png", mime="image/png")

                # Save last plots into session for future use
                st.session_state.last_plots = {
                    "cm": fig_cm,
                    "roc": fig_roc,
                    "pr": fig_pr,
                    "fi": fig_fi,
                    "corr": fig_corr
                }

# --------------------------
# Test Transaction page
# --------------------------
elif mode == "Test Transaction":
    st.header("2) Test a Single Transaction (use trained model)")

    if st.session_state.model is None or st.session_state.feature_cols is None:
        st.warning("No trained model available. Train a model first on the 'Upload & Train' page.")
    else:
        model = st.session_state.model
        feature_cols = st.session_state.feature_cols

        st.write("Enter feature values for a single transaction (Time, V1..V28, Amount).")
        left, right = st.columns(2)
        inputs = {}
        for i, col in enumerate(feature_cols):
            if i % 2 == 0:
                with left:
                    inputs[col] = st.number_input(col, value=0.0, format="%.6f")
            else:
                with right:
                    inputs[col] = st.number_input(col, value=0.0, format="%.6f")

        if st.button("🔍 Predict"):
            res = predict_single(model, feature_cols, inputs)
            pred = res["prediction"]
            prob = res["probability"]
            if prob is None:
                prob_text = "N/A"
            else:
                prob_text = f"{prob:.3f}"
            if pred == 1:
                st.error(f"⚠️ Model says FRAUDULENT (probability={prob_text})")
            else:
                st.success(f"✅ Model says LEGITIMATE (probability={prob_text})")
            st.markdown("**Note:** prediction uses the Random Forest trained on the balanced sample.")

# --------------------------
# About / Save model page
# --------------------------
else:
    st.header("About & Save Model")
    st.write("This app trains a Random Forest classifier on a user-uploaded credit-card CSV.")
    st.write("After training you can save the trained model to `/models/rf_model.pkl`.")

    if st.session_state.model is None:
        st.info("No trained model in memory.")
    else:
        if st.button("💾 Save trained model to models/rf_model.pkl"):
            path = save_model(st.session_state.model, path="models/rf_model.pkl")
            st.success(f"Model saved to: {path}")
