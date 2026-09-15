import streamlit as st
import pandas as pd
import numpy as np
import joblib



# PAGE CONFIGURATION

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)


# LOAD MODEL AND ARTIFACTS


@st.cache_resource
def load_model():
    model = joblib.load("final_xgb_model.pkl")
    return model


@st.cache_data
def load_feature_columns():
    features = joblib.load("feature_columns.pkl")
    return features


@st.cache_data
def load_threshold():
    threshold = joblib.load("threshold.pkl")
    return threshold


@st.cache_data
def load_feature_importance():
    try:
        return pd.read_csv("feature_importance.csv")
    except FileNotFoundError:
        return None


model = load_model()
feature_columns = load_feature_columns()
threshold = load_threshold()
feature_importance = load_feature_importance()


# TITLE

st.title("💳 Credit Card Fraud Detection")

st.markdown(
    """
    This application uses a machine learning model to identify
    potentially fraudulent credit card transactions.
    """
)

st.divider()

# SIDEBAR

st.sidebar.header("Model Information")

st.sidebar.write("**Model:** XGBoost Classifier")
st.sidebar.write("**Imbalance Handling:** Class Weighting")
st.sidebar.write("**Optimized Threshold:** 0.35")

st.sidebar.divider()

st.sidebar.subheader("Test Set Performance")

st.sidebar.metric("Precision", "88.37%")
st.sidebar.metric("Recall", "80.00%")
st.sidebar.metric("F1 Score", "83.98%")
st.sidebar.metric("ROC-AUC", "97.86%")
st.sidebar.metric("PR-AUC", "83.47%")

# TABS

tab1, tab2, tab3 = st.tabs(
    [
        "🔍 Single Transaction",
        "📂 Batch Prediction",
        "📊 Model Insights"
    ]
)


# TAB 1 - SINGLE TRANSACTION

with tab1:

    st.header("Single Transaction Prediction")

    st.info(
        "Enter the transaction features below. "
        "The model will calculate the fraud probability and classify the transaction."
    )

    # Time and Amount

    col1, col2 = st.columns(2)

    with col1:
        time_value = st.number_input(
            "Time",
            min_value=0.0,
            value=0.0,
            step=1.0,
            help="Seconds elapsed since the first transaction in the dataset."
        )

    with col2:
        amount_value = st.number_input(
            "Amount",
            min_value=0.0,
            value=100.0,
            step=1.0,
            help="Transaction amount."
        )

    # V1 - V28

    st.subheader("Anonymized Transaction Features")

    st.caption(
        "V1–V28 are anonymized PCA-transformed features from the dataset."
    )

    input_data = {}

    # Create 4 columns for V1-V28
    feature_cols = st.columns(4)

    for i in range(1, 29):

        column_index = (i - 1) % 4

        with feature_cols[column_index]:

            input_data[f"V{i}"] = st.number_input(
                f"V{i}",
                value=0.0,
                format="%.6f"
            )

    # Add Time and Amount
    input_data["Time"] = time_value
    input_data["Amount"] = amount_value

    # Prediction Button

    st.divider()

    if st.button(
        "🔎 Predict Transaction",
        type="primary",
        use_container_width=True
    ):

        # Create DataFrame
        input_df = pd.DataFrame([input_data])

        # Ensure exact feature order used during training
        input_df = input_df[feature_columns]

        # Predict probability
        fraud_probability = model.predict_proba(
            input_df
        )[0, 1]

        # Apply optimized threshold
        prediction = int(
            fraud_probability >= threshold
        )

        # Display probability

        st.subheader("Prediction Result")

        result_col1, result_col2 = st.columns(2)

        with result_col1:

            st.metric(
                "Fraud Probability",
                f"{fraud_probability:.2%}"
            )

        with result_col2:

            if prediction == 1:

                st.error(
                    "🚨 Potential Fraudulent Transaction"
                )

            else:

                st.success(
                    "✅ Legitimate Transaction"
                )

        # Explanation

        if prediction == 1:

            st.warning(
                f"The predicted fraud probability is "
                f"{fraud_probability:.2%}, which is above the "
                f"classification threshold of {threshold:.2f}."
            )

        else:

            st.success(
                f"The predicted fraud probability is "
                f"{fraud_probability:.2%}, which is below the "
                f"classification threshold of {threshold:.2f}."
            )


# TAB 2 - BATCH PREDICTION

with tab2:

    st.header("Batch Fraud Prediction")

    st.markdown(
        """
        Upload a CSV file containing the transaction features.
        The file must contain the following columns:
        """
    )

    st.code(
        ", ".join(feature_columns)
    )

    uploaded_file = st.file_uploader(
        "Upload Transaction CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:

            uploaded_df = pd.read_csv(uploaded_file)

            st.subheader("Uploaded Data")

            st.dataframe(
                uploaded_df.head(),
                use_container_width=True
            )

            # Check required columns

            missing_columns = [
                col
                for col in feature_columns
                if col not in uploaded_df.columns
            ]

            if missing_columns:

                st.error(
                    "The uploaded CSV is missing the following "
                    f"required columns: {missing_columns}"
                )

            else:

                # Prepare input

                prediction_data = uploaded_df[
                    feature_columns
                ].copy()

                # Check for missing values
                missing_values = prediction_data.isnull().sum().sum()

                if missing_values > 0:

                    st.error(
                        f"The uploaded file contains "
                        f"{missing_values} missing values. "
                        "Please clean the data before prediction."
                    )

                else:

                    # Prediction

                    probabilities = model.predict_proba(
                        prediction_data
                    )[:, 1]

                    predictions = (
                        probabilities >= threshold
                    ).astype(int)

                    # Create results

                    result_df = uploaded_df.copy()

                    result_df["Fraud Probability"] = probabilities

                    result_df["Prediction"] = predictions

                    result_df["Prediction Label"] = np.where(
                        predictions == 1,
                        "Fraud",
                        "Legitimate"
                    )

                    # Summary

                    total_transactions = len(result_df)

                    fraud_transactions = int(
                        predictions.sum()
                    )

                    legitimate_transactions = (
                        total_transactions -
                        fraud_transactions
                    )

                    st.subheader("Prediction Summary")

                    col1, col2, col3 = st.columns(3)

                    with col1:

                        st.metric(
                            "Total Transactions",
                            total_transactions
                        )

                    with col2:

                        st.metric(
                            "Potential Fraud",
                            fraud_transactions
                        )

                    with col3:

                        st.metric(
                            "Legitimate",
                            legitimate_transactions
                        )

                    # Results table

                    st.subheader("Prediction Results")

                    st.dataframe(
                        result_df,
                        use_container_width=True
                    )

                    # Download results

                    csv = result_df.to_csv(
                        index=False
                    ).encode("utf-8")

                    st.download_button(
                        label="⬇️ Download Predictions",
                        data=csv,
                        file_name="fraud_predictions.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

        except Exception as e:

            st.error(
                f"Unable to process the uploaded file: {e}"
            )

# TAB 3 - MODEL INSIGHTS

with tab3:

    st.header("Model Insights")

    st.subheader("Final Model Performance")

    performance_df = pd.DataFrame(
        {
            "Metric": [
                "Precision",
                "Recall",
                "F1 Score",
                "ROC-AUC",
                "PR-AUC"
            ],
            "Score": [
                0.8837,
                0.8000,
                0.8398,
                0.9786,
                0.8347
            ]
        }
    )

    performance_df["Score"] = (
        performance_df["Score"] * 100
    ).round(2)

    performance_df["Score"] = (
        performance_df["Score"].astype(str) + "%"
    )

    st.table(performance_df)

    st.subheader("Classification Threshold")

    st.write(
        f"The optimized classification threshold used by the "
        f"application is **{threshold:.2f}**."
    )

    st.info(
        "A transaction is classified as fraudulent when its "
        f"predicted fraud probability is greater than or equal "
        f"to {threshold:.2f}."
    )

    # Feature Importance

    if feature_importance is not None:

        st.subheader("Top 10 Important Features")

        top_features = feature_importance.head(10).copy()

        top_features["Importance"] = (
            top_features["Importance"].round(4)
        )

        st.dataframe(
            top_features,
            use_container_width=True
        )

        st.bar_chart(
            top_features.set_index("Feature")[
                "Importance"
            ]
        )

    else:

        st.warning(
            "Feature importance file was not found."
        )


# FOOTER

st.divider()

st.caption(
    "Credit Card Fraud Detection | XGBoost Machine Learning Model"
)