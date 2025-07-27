import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model
MODEL_PATH = './saved_models/loan_default_pipeline.joblib'

# Feature engineering (must match training script)
def feature_engineering(df):
    df = df.copy()
    df['income_level'] = pd.cut(df['monthly_income'], bins=[0, 5000, 10000, 15000, 20000, np.inf],
                                labels=['Low', 'Lower-Middle', 'Middle', 'Upper-Middle', 'High'])
    df['payment_behavior'] = df['utility_payment_timeliness'].apply(
        lambda x: 'Good' if x in ['early', 'on-time'] else 'Bad')
    df['loan_to_income_ratio'] = df['loan_amount'] / df['monthly_income']
    df['payment_discipline_score'] = df['utility_payment_timeliness'].map({'early': 1, 'on-time': 0.5, 'late': 0})

    income_level_map = {'Low': 1.0, 'Lower-Middle': 0.8, 'Middle': 0.6, 'Upper-Middle': 0.4, 'High': 0.2}
    df['income_adjusted_loan_ratio'] = df['loan_to_income_ratio'] / df['income_level'].map(income_level_map).astype(float)
    df['reliable_borrower'] = ((df['has_previous_loan'] == 1) & (df['payment_behavior'] == 'Good')).astype(int)
    return df

# Streamlit UI
st.title("📉 Microloan Default Predictor")

uploaded_file = st.file_uploader("Upload borrower CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        st.subheader("📋 Raw Data Preview")
        st.dataframe(data.head())

        # Apply feature engineering
        data_processed = feature_engineering(data)

        # Load trained model
        model = joblib.load(MODEL_PATH)

        # Predict
        predictions = model.predict(data_processed)
        data['Predicted Default'] = predictions

        st.subheader("🔎 Prediction Results")
        st.dataframe(data[['Predicted Default']])

        st.success("✅ Prediction complete!")

    except Exception as e:
        st.error(f"❌ Error: {e}")


# to run:
# after runnning pipeline.py, run the following command:
# ```bash
# streamlit run app.py