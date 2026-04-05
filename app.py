# 🚀 BULLETPROOF STREAMLIT STOCK APP (Charts + Safe + Stable)

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ---------------- DATA ----------------
@st.cache_data(ttl=3600)
def fetch_history(ticker, period='1y', interval='1d'):
    try:
        data = yf.download(ticker, period=period, interval=interval, progress=False)
        if data is None or data.empty:
            return None
        data = data[['Open','High','Low','Close','Volume']]
        data = data.dropna()
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

# ---------------- PREDICTION ----------------
def make_prediction(history_df, seq_len=60):
    if history_df is None or history_df.shape[0] < seq_len:
        return None, "Not enough data"

    history_df = history_df.dropna()

    if history_df.shape[0] < seq_len:
        return None, "Not enough clean data"

    series = history_df['Close'].iloc[-seq_len:].values

    if len(series) == 0:
        return None, "Empty series"

    last_val = series[-1]

    try:
        last_val = float(last_val)
    except:
        return None, "Invalid last value"

    if np.isnan(last_val):
        return None, "Last value is NaN"

    try:
        pred = float(last_val * (1 + np.random.normal(0, 0.01)))
    except Exception as e:
        return None, f"Prediction error: {e}"

    return pred, "Simulated prediction"

# ---------------- UI ----------------
st.set_page_config(page_title="StockPredictor Pro", layout="wide")

st.title("📈 Stock Predictor PRO (Bulletproof)")

with st.form('form'):
    ticker = st.text_input("Ticker", "AAPL")
    period = st.selectbox("Period", ['1mo','3mo','6mo','1y','2y'])
    interval = st.selectbox("Interval", ['1d','1wk','1h'])
    seq_len = st.number_input("Sequence Length", 10, 200, 60)
    submit = st.form_submit_button("Predict")

if submit:
    history = fetch_history(ticker, period, interval)

    if history is None:
        st.error("No data found. Try another ticker.")
    else:
        st.subheader("📊 Raw Data")
        st.dataframe(history.tail())

        # ---------------- LINE CHART ----------------
        st.subheader("📈 Price Trend")
        st.line_chart(history['Close'])

        # ---------------- SMA ----------------
        history['SMA20'] = history['Close'].rolling(20).mean()
        history['SMA50'] = history['Close'].rolling(50).mean()

        st.subheader("📊 Moving Averages")
        st.line_chart(history[['Close','SMA20','SMA50']].dropna())

        # ---------------- CANDLESTICK ----------------
        st.subheader("🕯️ Candlestick Chart")
        fig = go.Figure(data=[go.Candlestick(
            x=history.index,
            open=history['Open'],
            high=history['High'],
            low=history['Low'],
            close=history['Close']
        )])
        st.plotly_chart(fig, use_container_width=True)

        # ---------------- PREDICTION ----------------
        pred, msg = make_prediction(history, seq_len)

        if pred is None:
            st.warning(msg)
        else:
            last = float(history['Close'].iloc[-1])

            st.subheader("🔮 Prediction")
            st.metric("Predicted Price", f"{pred:.2f}")
            st.write(msg)

            change = (pred - last)/last*100

            if change > 0:
                st.success(f"📈 Expected Increase: +{change:.2f}%")
            else:
                st.error(f"📉 Expected Decrease: {change:.2f}%")

st.markdown("---")
st.caption("⚠️ Not financial advice | Built by Ayush 💀🚀")

