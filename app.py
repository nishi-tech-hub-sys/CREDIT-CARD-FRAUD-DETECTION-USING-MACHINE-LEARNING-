
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px

# -------------------------------
# PAGE CONFIGURATION
# -------------------------------
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# -------------------------------
# LOAD SAVED FILES
# -------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(MODEL_DIR, "fraud_detection_model.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    threshold = float(joblib.load(os.path.join(MODEL_DIR, "threshold.pkl")))
    features = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))
    metrics = joblib.load(os.path.join(MODEL_DIR, "model_metrics.pkl"))

    return model, scaler, threshold, features, metrics


try:
    model, scaler, threshold, features, metrics = load_artifacts()
except Exception as e:
    st.error(f"Model files could not be loaded: {e}")
    st.stop()


# -------------------------------
# SIDEBAR
# -------------------------------
st.sidebar.title("💳 Fraud Detection")
page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Fraud Detection", "Analytics", "Model Information"]
)

st.sidebar.markdown("---")
st.sidebar.caption("Credit Card Fraud Detection using Random Forest")


# -------------------------------
# DASHBOARD
# -------------------------------
if page == "Dashboard":

    st.title("💳 Credit Card Fraud Detection")
    st.write("Machine Learning based fraud detection using Random Forest.")

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    col1.metric("Selected Model", "Random Forest")
    col2.metric("Decision Threshold", f"{threshold:.4f}")
    col3.metric("Input Features", len(features))

    st.markdown("---")

    st.subheader("Project Overview")

    st.write("""
    This application predicts whether a credit card transaction is
    **Genuine** or **Fraudulent**.

    The system uses a trained Random Forest model and a saved
    StandardScaler for the Time and Amount columns.
    """)

    st.info("Go to Fraud Detection to upload a transaction CSV file.")


# -------------------------------
# FRAUD DETECTION
# -------------------------------
elif page == "Fraud Detection":

    st.title("🔍 Fraud Detection")
    st.write("Upload a CSV file containing credit card transactions.")

    uploaded_file = st.file_uploader(
        "Upload CSV File",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:
            input_df = pd.read_csv(uploaded_file)

            st.subheader("Uploaded Dataset")
            st.dataframe(input_df.head())

            # Remove target column if present
            prediction_df = input_df.drop(
                columns=["Class"],
                errors="ignore"
            ).copy()

            missing = [col for col in features if col not in prediction_df.columns]
            extra = [col for col in prediction_df.columns if col not in features]

            if missing:
                st.error(f"Missing required columns: {missing}")

            elif extra:
                st.error(f"Unexpected columns: {extra}")

            else:
                prediction_df = prediction_df[features]

                if not all(
                    pd.api.types.is_numeric_dtype(prediction_df[col])
                    for col in features
                ):
                    st.error("All feature columns must contain numeric values.")

                elif prediction_df.isnull().any().any():
                    st.error("Dataset contains missing values. Please clean the CSV.")

                elif not np.isfinite(prediction_df.to_numpy(dtype=float)).all():
                    st.error("Dataset contains infinite or invalid numeric values.")

                else:
                    # Scale only Amount and Time
                    prediction_df[["Amount", "Time"]] = scaler.transform(
                        prediction_df[["Amount", "Time"]]
                    )

                    # Predict fraud probability
                    probabilities = model.predict_proba(prediction_df)[:, 1]

                    # Apply saved threshold
                    predictions = (probabilities >= threshold).astype(int)

                    results = input_df.copy()
                    results["Fraud Probability"] = probabilities
                    results["Prediction"] = np.where(
                        predictions == 1,
                        "Fraudulent",
                        "Genuine"
                    )

                    st.markdown("---")
                    st.subheader("Prediction Results")

                    total = len(results)
                    fraud_count = int(predictions.sum())
                    genuine_count = total - fraud_count

                    c1, c2, c3 = st.columns(3)

                    c1.metric("Total Transactions", total)
                    c2.metric("Fraudulent Transactions", fraud_count)
                    c3.metric("Genuine Transactions", genuine_count)

                    st.dataframe(results, use_container_width=True)

                    st.subheader("Prediction Distribution")

                    chart_data = pd.DataFrame({
                        "Transaction Type": ["Genuine", "Fraudulent"],
                        "Count": [genuine_count, fraud_count]
                    })

                    fig = px.pie(
                        chart_data,
                        names="Transaction Type",
                        values="Count",
                        title="Genuine vs Fraudulent Transactions",
                        hole=0.4
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    csv = results.to_csv(index=False).encode("utf-8")

                    st.download_button(
                        "Download Prediction Results",
                        data=csv,
                        file_name="fraud_detection_results.csv",
                        mime="text/csv"
                    )

        except Exception as e:
            st.error(f"Error processing file: {e}")


# -------------------------------
# ANALYTICS
# -------------------------------
elif page == "Analytics":

    st.title("📊 Model Analytics")

    if isinstance(metrics, dict):

        st.subheader("Saved Model Metrics")

        metric_values = {
            key: value
            for key, value in metrics.items()
            if isinstance(value, (int, float, np.integer, np.floating))
        }

        if metric_values:
            cols = st.columns(min(len(metric_values), 4))

            for i, (name, value) in enumerate(metric_values.items()):
                cols[i % len(cols)].metric(
                    name.replace("_", " ").title(),
                    f"{value:.4f}" if isinstance(value, (float, np.floating)) else value
                )
        else:
            st.info("No numeric metrics found in the saved metrics file.")

        st.json(metrics)

    else:
        st.write(metrics)


# -------------------------------
# MODEL INFORMATION
# -------------------------------
elif page == "Model Information":

    st.title("🤖 Model Information")

    st.write("**Algorithm:** Random Forest Classifier")
    st.write("**Preprocessing:** StandardScaler on Time and Amount")
    st.write("**Imbalance Handling:** SMOTE inside the model pipeline")
    st.write(f"**Decision Threshold:** {threshold:.6f}")

    st.subheader("Input Features")
    st.write(features)

    st.subheader("Model Pipeline")
    st.write(model)

    st.markdown("---")
    st.caption("Developed by Rutvi Patel")
