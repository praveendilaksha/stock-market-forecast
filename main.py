import streamlit as st 
import pandas as pd
import numpy as np
import yfinance as yf
from prophet import Prophet
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import matplotlib.pyplot as plt
from datetime import date

# --- Streamlit Page Setup ---
st.set_page_config(page_title="AI Finance Predictor by Praveen", layout="wide")

st.title("AI-Powered Predictive Analytics for Stock Market (Research Preview)")
st.caption("Analyze stock data using **Linear Regression**, **Random Forest**, or **XGBoost** — now with Kaggle dataset upload support!")

# --- Sidebar Controls ---
st.sidebar.header("⚙️ Settings")

# Data source choice
data_source = st.sidebar.radio("Choose Data Source:", ["Yahoo Finance", "Upload Kaggle Dataset"])

# --- Load Data ---
if data_source == "Yahoo Finance":
    ticker = st.sidebar.text_input("Enter Stock Symbol", "AAPL")
    start_date = st.sidebar.date_input("Start Date", date(2020, 1, 1))
    end_date = st.sidebar.date_input("End Date", date.today())

    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    df.reset_index(inplace=True)

    if df.empty:
        st.error("No data found for this symbol. Please check the stock code or date range.")
        st.stop()

    st.write(f"### {ticker} Stock Data ({start_date} → {end_date})")
    st.dataframe(df.tail())

else:
    uploaded_file = st.sidebar.file_uploader("📂 Upload Kaggle CSV dataset", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("### Uploaded Dataset Preview")
        st.dataframe(df.head())
    else:
        st.warning("Please upload a Kaggle dataset to continue.")
        st.stop()

# --- Preprocessing ---
if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df.sort_values("Date", inplace=True)

# Target column selection
target_col = st.sidebar.selectbox("Select Target Column (Value to Predict)", options=df.columns, index=len(df.columns)-1)
feature_cols = [c for c in df.columns if c != target_col]

# Drop rows with missing target
df = df.dropna(subset=[target_col])

# --- Feature Scaling (Safe for Mixed Data) ---
numeric_features = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
non_numeric = list(set(feature_cols) - set(numeric_features))

if non_numeric:
    st.warning(f"Ignored non-numeric columns for scaling: {non_numeric}")

scaler = MinMaxScaler()
scaled_part = pd.DataFrame(
    scaler.fit_transform(df[numeric_features + [target_col]]),
    columns=numeric_features + [target_col]
)

# Reattach non-numeric columns (like Date)
for col in non_numeric:
    scaled_part[col] = df[col].values

scaled_df = scaled_part.copy()

# --- Train/Test Split ---
X = scaled_df[numeric_features]
y = scaled_df[target_col]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# --- Model Selection ---
model_choice = st.sidebar.selectbox("Select Model:", ["Linear Regression", "Random Forest", "XGBoost"])

# --- Run Analysis ---
if st.sidebar.button("Run Analysis"):
    with st.spinner("Training model... please wait"):
        # Choose model
        if model_choice == "Linear Regression":
            model = LinearRegression()
        elif model_choice == "Random Forest":
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            model = xgb.XGBRegressor(
                n_estimators=200,
                learning_rate=0.05,
                random_state=42,
                objective='reg:squarederror',
                verbosity=0
            )

        # Train model
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        # Evaluate
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))

        st.success(f"Model Trained Successfully: **{model_choice}**")
        st.write(f"**MAE:** {mae:.4f} | **RMSE:** {rmse:.4f}")

        # --- Plot Results (index-based) ---
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(y_test.values, label="Actual", color="blue")
        ax.plot(preds, label="Predicted", color="red", linestyle="--")
        ax.set_title(f"{model_choice} Prediction vs Actual (Index-based)")
        ax.legend()
        st.pyplot(fig)

        # --- Time-based Prediction Plot ---
        if "Date" in df.columns:
            # Align Date with test indices
            date_col = df["Date"].iloc[-len(y_test):].reset_index(drop=True)

            fig2, ax2 = plt.subplots(figsize=(10, 4))
            ax2.plot(date_col, y_test.values, label="Actual", color="blue")
            ax2.plot(date_col, preds, label="Predicted", color="red", linestyle="--")
            ax2.set_title(f"{model_choice} Prediction vs Actual Over Time")
            ax2.set_xlabel("Date")
            ax2.set_ylabel("Scaled Price")
            ax2.legend()
            plt.xticks(rotation=45)
            st.pyplot(fig2)
        else:
            st.info("Date column not found — time-based chart skipped.")


        # --- Export Predictions ---
        results_df = pd.DataFrame({
            "Actual": y_test.values,
            "Predicted": preds
        })
        csv = results_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Predictions CSV", csv, "predictions.csv", "text/csv")

else:
    st.info("Configure your settings and click **Run Analysis** to start.")

# --- Debug Info ---
with st.expander("Debug Info"):
    st.write("Numeric features used:", numeric_features)
    st.write("Ignored columns:", non_numeric)
    st.write("Target column:", target_col)
    st.write("DataFrame shape:", df.shape)
