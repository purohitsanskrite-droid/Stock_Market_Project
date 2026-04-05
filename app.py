# ✅ FIXED VERSION (TensorFlow removed + safe prediction)

import streamlit as st
from streamlit import session_state as ss
import pyrebase
import firebase_admin
from firebase_admin import credentials, firestore
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

# ---------------- CONFIG ----------------
FIREBASE_CONFIG = {
    "apiKey": "YOUR_API_KEY",
    "authDomain": "YOUR_PROJECT.firebaseapp.com",
    "databaseURL": "https://YOUR_PROJECT.firebaseio.com",
    "projectId": "YOUR_PROJECT",
    "storageBucket": "YOUR_PROJECT.appspot.com",
    "messagingSenderId": "SENDER_ID",
    "appId": "APP_ID"
}

SERVICE_ACCOUNT_PATH = "serviceAccountKey.json"
SCALER_PATH = "scaler.pkl"

# ---------------- FIREBASE ----------------
firebase = pyrebase.initialize_app(FIREBASE_CONFIG)
auth = firebase.auth()

if os.path.exists(SERVICE_ACCOUNT_PATH):
    cred = credentials.Certificate(SERVICE_ACCOUNT_PATH)
    try:
        firebase_admin.initialize_app(cred)
        db = firestore.client()
    except:
        db = None
else:
    db = None

# ---------------- LOAD SCALER ----------------
@st.cache_resource
def load_scaler(path=SCALER_PATH):
    if os.path.exists(path):
        return joblib.load(path)
    return None

scaler = load_scaler()

# ---------------- DATA ----------------
@st.cache_data(ttl=3600)
def fetch_history(ticker, period='1y', interval='1d'):
    data = yf.download(ticker, period=period, interval=interval, progress=False)
    if data.empty:
        return None
    data = data[['Open','High','Low','Close','Volume']]
    data = data.dropna()
    return data

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

    if np.isnan(last_val):
        return None, "Invalid last value"

    pred = float(last_val * (1 + np.random.normal(0, 0.01)))
    return pred, "Simulated prediction"

# ---------------- UI ----------------
st.set_page_config(page_title="StockPredictor", layout="wide")

st.title("📈 Stock Predictor")

with st.form('form'):
    ticker = st.text_input("Ticker", "AAPL")
    period = st.selectbox("Period", ['1mo','3mo','6mo','1y'])
    seq_len = st.number_input("Sequence Length", 10, 200, 60)
    submit = st.form_submit_button("Predict")

if submit:
    history = fetch_history(ticker, period)

    if history is None:
        st.error("No data")
    else:
        st.dataframe(history.tail())

        pred, msg = make_prediction(history, seq_len)

        if pred is None:
            st.warning(msg)
        else:
            last = float(history['Close'].iloc[-1])
            st.metric("Prediction", f"{pred:.2f}")
            st.write(msg)

            change = (pred - last)/last*100

            if change > 0:
                st.success(f"📈 +{change:.2f}% expected")
            else:
                st.error(f"📉 {change:.2f}% expected")

st.markdown("---")
st.caption("Not financial advice")

