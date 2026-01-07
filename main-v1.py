import streamlit as st # type: ignore
import pandas as pd
from prophet import Prophet
import plotly.graph_objs as go
from datetime import datetime
import yfinance as yf # type: ignore
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI Finance Predictor by Praveen", layout="wide")
st.title("AI-Powered Predictive Analytics for Stock Market (Research Preview)")

# Sidebar inputs
stock = st.selectbox("Select stock for prediction", ["AAPL", "GOOG", "MSFT", "TSLA","^CSE"])
years = st.slider("Years of prediction:", 1, 4)
period = years * 365  # days into the future


# Load data (cached)
@st.cache_data
def load_data(ticker):
    data = yf.download(
        ticker, start="2015-01-01", end=datetime.today().strftime("%Y-%m-%d"), progress=False
    )
    if data.empty:
        return pd.DataFrame()
    data = data.reset_index()

    # Fix: flatten multi-index columns (newer yfinance versions sometimes return these)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [' '.join(col).strip() for col in data.columns.values]
    return data


data_load_state = st.text("Loading data...")
df = load_data(stock)
data_load_state.text("Loading data... done!")

if df.empty:
    st.error("No data returned for this ticker. Please check symbol or connection.")
    st.stop()

# Show raw data
st.subheader("Raw Data (Tail)")
st.write(df.tail())

# Detect the correct column name for closing price
close_col = None
for col in df.columns:
    if "Close" in col:
        close_col = col
        break

if close_col is None:
    st.error(f"Couldn't find a 'Close' column in the dataset. Columns found: {list(df.columns)}")
    st.stop()

# Prepare data for Prophet
df_train = df[["Date", close_col]].copy().rename(columns={"Date": "ds", close_col: "y"})

# Ensure correct datatypes
df_train["ds"] = pd.to_datetime(df_train["ds"], errors="coerce")
df_train["y"] = pd.to_numeric(df_train["y"], errors="coerce")

# Drop invalid rows
before_len = len(df_train)
df_train = df_train.dropna(subset=["ds", "y"]).reset_index(drop=True)
after_len = len(df_train)
dropped = before_len - after_len

st.info(f"Prepared {after_len} rows for forecasting. Dropped {dropped} invalid rows.")

# Train Prophet model
try:
    model = Prophet()
    model.fit(df_train)
except Exception as e:
    st.error(f"Prophet model failed to fit: {e}")
    st.stop()

# Create future dataframe and make forecast
future = model.make_future_dataframe(periods=period, freq="D")
forecast = model.predict(future)

# Display forecast data
st.subheader("Forecast Data (Tail)")
st.write(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail())

csv = forecast.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download Full Forecast Data as CSV",
    data=csv,
    file_name=f"{stock}_forecast.csv",
    mime='text/csv'
)


# Plot forecast (Plotly)
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=df_train["ds"], y=df_train["y"], name="Actual"))
fig1.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], name="Forecast"))
fig1.update_layout(
    title=f"{stock} Stock Price Forecast ({years} Year{'s' if years > 1 else ''})",
    xaxis_title="Date",
    yaxis_title="Price (USD)",
    legend=dict(x=0, y=1, traceorder="normal"),
)
st.plotly_chart(fig1, use_container_width=True)

# Prophet's component plots
st.subheader("Forecast Components")
fig2 = model.plot_components(forecast)
st.pyplot(fig2)

# Optional raw debug info
with st.expander("Debug Info"):
    st.write("DataFrame Columns:", list(df.columns))
    st.write("Close column used:", close_col)
    st.write("First rows of df_train:")
    st.write(df_train.head())
