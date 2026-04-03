"""
Streamlit Stock Market Prediction Web App
Filename: streamlit_stock_prediction_app.py

Features included:
- Firebase Authentication (email/password) using pyrebase
- Firestore logging for feedback & contact entries using firebase_admin
- Streamlit pages: Login / Signup, Dashboard (knowledge), Prediction, Feedback, Contact
- Model loading stub (Keras .h5) and scaler (.pkl) usage for predictions
- yfinance data fetch for historical data (for prediction features)
- Clear setup & deployment instructions at the top of this file

IMPORTANT SETUP STEPS (brief):
1. Create Firebase project (https://console.firebase.google.com/) and enable Email/Password auth.
2. Create a Realtime Database (rules open for your dev phase) or use Firestore for storing records.
3. Get Firebase config (apiKey, authDomain, databaseURL, projectId, storageBucket, messagingSenderId, appId).
4. Create a Firebase service account JSON for Firestore and put it in this project (serviceAccountKey.json).
5. Install Python packages: pip install streamlit pyrebase4 firebase-admin yfinance tensorflow scikit-learn pandas joblib
6. Place your trained model file (model.h5) and scaler (scaler.pkl) in the project folder.
7. Run: streamlit run streamlit_stock_prediction_app.py

Note: This is an opinionated template. Replace placeholders (FIREBASE_CONFIG, path to serviceAccountKey.json) with your real values.

"""

import streamlit as st
from streamlit import session_state as ss
import pyrebase
import firebase_admin
from firebase_admin import credentials, firestore
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import os
from datetime import datetime

# ---------------------------
# CONFIGURATION / PLACEHOLDERS
# ---------------------------

# Replace this dict with your Firebase project's config (from Firebase console -> project settings)
FIREBASE_CONFIG = {
    "apiKey": "YOUR_API_KEY",
    "authDomain": "YOUR_PROJECT.firebaseapp.com",
    "databaseURL": "https://YOUR_PROJECT.firebaseio.com",
    "projectId": "YOUR_PROJECT",
    "storageBucket": "YOUR_PROJECT.appspot.com",
    "messagingSenderId": "SENDER_ID",
    "appId": "APP_ID"
}

# Path to Firebase service account JSON (for Firestore admin usage)
SERVICE_ACCOUNT_PATH = "serviceAccountKey.json"  # <- replace with your service account filename

# Model & scaler paths (put your trained artifacts in project folder)
MODEL_PATH = "model.h5"
SCALER_PATH = "scaler.pkl"

# ---------------------------
# INITIALIZE FIREBASE CLIENTS
# ---------------------------

# Initialize pyrebase for client authentication
firebase = pyrebase.initialize_app(FIREBASE_CONFIG)
auth = firebase.auth()

# Initialize firebase_admin for Firestore (server-side)
if os.path.exists(SERVICE_ACCOUNT_PATH):
    cred = credentials.Certificate(SERVICE_ACCOUNT_PATH)
    try:
        firebase_admin.initialize_app(cred)
        db = firestore.client()
    except Exception as e:
        db = None
        print("Firestore init error:", e)
else:
    db = None

# ---------------------------
# UTILITY FUNCTIONS
# ---------------------------

@st.cache_resource
def load_model_and_scaler(model_path=MODEL_PATH, scaler_path=SCALER_PATH):
    model = None
    scaler = None
    if os.path.exists(model_path):
        try:
            model = tf.keras.models.load_model(model_path)
        except Exception as e:
            st.warning(f"Could not load model: {e}")
    else:
        st.info("Model file not found. Prediction page will simulate outputs until you add model.h5")

    if os.path.exists(scaler_path):
        try:
            scaler = joblib.load(scaler_path)
        except Exception as e:
            st.warning(f"Could not load scaler: {e}")
    else:
        st.info("Scaler file not found. Prediction page will simulate outputs until you add scaler.pkl")

    return model, scaler


def save_feedback_to_firestore(user_email, name, rating, message):
    if db is None:
        st.error("Firestore is not configured. Feedback cannot be saved remotely.")
        return False
    try:
        doc_ref = db.collection('feedback').document()
        doc_ref.set({
            'user': user_email,
            'name': name,
            'rating': rating,
            'message': message,
            'timestamp': datetime.utcnow()
        })
        return True
    except Exception as e:
        st.error(f"Failed saving feedback: {e}")
        return False


def save_contact_to_firestore(name, email, subject, message):
    if db is None:
        st.error("Firestore is not configured. Contact cannot be saved remotely.")
        return False
    try:
        doc_ref = db.collection('contacts').document()
        doc_ref.set({
            'name': name,
            'email': email,
            'subject': subject,
            'message': message,
            'timestamp': datetime.utcnow()
        })
        return True
    except Exception as e:
        st.error(f"Failed saving contact: {e}")
        return False


# Simple helper to fetch historical OHLCV using yfinance
@st.cache_data(ttl=3600)
def fetch_history(ticker, period='1y', interval='1d'):
    try:
        data = yf.download(ticker, period=period, interval=interval, progress=False)
        if data.empty:
            return None
        data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None


# Prediction pipeline (stub): expects sequential input prepared by user/model training process
def make_prediction(model, scaler, history_df, seq_len=60):
    # This function should mirror the preprocessing you used when training your model.
    # Example: take last seq_len 'Close' values, scale, reshape to (1, seq_len, features)
    if history_df is None or history_df.shape[0] < seq_len:
        return None, "Not enough historical data for prediction."

    series = history_df['Close'].values[-seq_len:]
    X = series.reshape(-1, 1)
    if scaler is not None:
        try:
            X_scaled = scaler.transform(X)
        except Exception as e:
            return None, f"Scaler transform failed: {e}"
    else:
        X_scaled = (X - np.mean(X)) / (np.std(X) + 1e-8)

    # reshape to (1, seq_len, features)
    X_in = X_scaled.reshape(1, seq_len, 1)

    if model is None:
        # simulated prediction -- naive last value + small random drift
        pred = float(series[-1] * (1 + np.random.normal(0, 0.01)))
        return pred, "Simulated prediction (no model loaded)."

    try:
        raw_pred = model.predict(X_in)
        # if model predicts scaled value, inverse transform
        if scaler is not None:
            try:
                pred_inv = scaler.inverse_transform(np.array(raw_pred).reshape(-1, 1))
                pred_val = float(pred_inv[-1, 0])
            except Exception:
                pred_val = float(raw_pred.ravel()[-1])
        else:
            pred_val = float(raw_pred.ravel()[-1])

        return pred_val, "Model prediction"
    except Exception as e:
        return None, f"Model prediction failed: {e}"


# ---------------------------
# STREAMLIT APP LAYOUT
# ---------------------------

st.set_page_config(page_title="StockPredictor", layout="wide")

# Load model + scaler once
model, scaler = load_model_and_scaler()

# Sidebar: Navigation + Auth
st.sidebar.title("StockPredictor")
page = st.sidebar.radio("Go to", ["Home", "Prediction", "Dashboard", "Feedback", "Contact", "Admin"])

# Authentication forms: Sign Up & Login
with st.sidebar.expander("Account"):
    if 'user' not in ss:
        ss.user = None
    if ss.user:
        st.write("Signed in as:", ss.user.get('email'))
        if st.button("Sign out"):
            ss.user = None
            st.experimental_rerun()
    else:
        auth_mode = st.radio("", ["Login", "Sign up"], index=0)
        if auth_mode == 'Sign up':
            su_email = st.text_input("Email", key='su_email')
            su_password = st.text_input("Password", type='password', key='su_pw')
            su_display = st.text_input("Display name", key='su_name')
            if st.button("Create account"):
                try:
                    user = auth.create_user_with_email_and_password(su_email, su_password)
                    # set displayName in user profile not directly supported by pyrebase easily
                    ss.user = {'email': su_email, 'localId': user['localId']}
                    st.success("Account created — logged in")
                except Exception as e:
                    st.error(f"Signup failed: {e}")
        else:
            li_email = st.text_input("Email", key='li_email')
            li_password = st.text_input("Password", type='password', key='li_pw')
            if st.button("Login"):
                try:
                    user = auth.sign_in_with_email_and_password(li_email, li_password)
                    info = auth.get_account_info(user['idToken'])
                    ss.user = {'email': li_email, 'idToken': user['idToken'], 'localId': user['localId']}
                    st.success("Logged in")
                except Exception as e:
                    st.error(f"Login failed: {e}")

# ---------------------------
# PAGE: Home (Knowledge/Dashboard intro)
# ---------------------------
if page == 'Home':
    st.title("Welcome to StockPredictor")
    st.markdown("""
    **What this app does**
    - Predicts short-term stock prices using your trained model (drop-in model.h5 & scaler.pkl)
    - Lets users sign up / log in via Firebase email/password
    - Dashboard for charts, explanations, and learning resources
    - Feedback & contact forms that log to Firestore

    **Notes & Disclaimers**
    - Predictions are for educational purposes only — NOT financial advice.
    - Always validate model performance with backtests, cross-validation and realistic slippage.
    """)

    st.header("Quick Resources")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Model Validation")
        st.write("Confusion matrix (regression => residuals), MAPE, RMSE, backtesting tips")
    with col2:
        st.subheader("Data Sources")
        st.write("yfinance, AlphaVantage, IEX Cloud. Use official APIs for production")
    with col3:
        st.subheader("Deployment")
        st.write("Host model artifacts on Google Cloud Storage or Firebase Storage. Deploy Streamlit on Streamlit Cloud or GCP.")

# ---------------------------
# PAGE: Prediction
# ---------------------------
if page == 'Prediction':
    st.header("Prediction")
    st.write("Enter a ticker and select history length. The app will fetch historical data, prepare input and call the model.")

    with st.form('pred_form'):
        ticker = st.text_input('Ticker (e.g. AAPL, MSFT, NSE:TCS)', value='AAPL')
        period = st.selectbox('History period', ['1mo', '3mo', '6mo', '1y', '2y', '5y'], index=3)
        interval = st.selectbox('Interval', ['1d', '1wk', '1h'], index=0)
        seq_len = st.number_input('Sequence length for model (timesteps)', min_value=10, max_value=365, value=60)
        submit = st.form_submit_button('Run Prediction')

    if submit:
        with st.spinner('Fetching data...'):
            history = fetch_history(ticker, period=period, interval=interval)

        if history is None:
            st.error('No data retrieved for ticker — check symbol or try different period/interval')
        else:
            st.subheader('Latest data')
            st.dataframe(history.tail(10))

            pred_val, pred_msg = make_prediction(model, scaler, history, seq_len=seq_len)
            if pred_val is None:
                st.warning(pred_msg)
            else:
                st.metric(label=f"Predicted next close for {ticker}", value=f"{pred_val:.2f}")
                st.info(pred_msg)

                # Simple interpretation: compare to last close
                last_close = float(history['Close'].iloc[-1])
                delta = (pred_val - last_close) / last_close * 100
                if delta >= 0:
                    st.success(f"Model expects an increase of {delta:.2f}% from last close ({last_close:.2f})")
                else:
                    st.error(f"Model expects a decrease of {abs(delta):.2f}% from last close ({last_close:.2f})")

# ---------------------------
# PAGE: Dashboard (charts & knowledge)
# ---------------------------
if page == 'Dashboard':
    st.header('Dashboard & Learning')
    st.write('Visualize price history and indicators. Use this as an educational dashboard.')

    dash_ticker = st.text_input('Ticker for dashboard', value='AAPL', key='dash_ticker')
    dash_period = st.selectbox('Period', ['1mo', '3mo', '6mo', '1y', '2y', '5y'], index=3, key='dash_period')
    if st.button('Load Chart'):
        hist = fetch_history(dash_ticker, period=dash_period)
        if hist is None:
            st.error('No data for this ticker')
        else:
            st.line_chart(hist['Close'])
            st.write('Simple indicators (SMA, RSI sample)')
            hist['SMA20'] = hist['Close'].rolling(20).mean()
            hist['SMA50'] = hist['Close'].rolling(50).mean()
            st.line_chart(hist[['Close', 'SMA20', 'SMA50']].dropna())

# ---------------------------
# PAGE: Feedback
# ---------------------------
if page == 'Feedback':
    st.header('Feedback')
    st.write('Your feedback helps improve the model and the product.')

    with st.form('feedback_form'):
        name = st.text_input('Your name')
        email = st.text_input('Email', value=(ss.user['email'] if ss.get('user') else ''))
        rating = st.slider('Rate the app', 1, 5, 4)
        message = st.text_area('Message')
        fb_submit = st.form_submit_button('Send Feedback')

    if fb_submit:
        ok = save_feedback_to_firestore(email, name, rating, message)
        if ok:
            st.success('Thanks — feedback saved.')

# ---------------------------
# PAGE: Contact
# ---------------------------
if page == 'Contact':
    st.header('Contact')
    st.write('Get in touch — we will respond to the email you provide.')

    with st.form('contact_form'):
        cname = st.text_input('Name')
        cemail = st.text_input('Email')
        csubject = st.text_input('Subject')
        cmessage = st.text_area('Message')
        csubmit = st.form_submit_button('Send')

    if csubmit:
        ok = save_contact_to_firestore(cname, cemail, csubject, cmessage)
        if ok:
            st.success('Thanks — contact saved. We will reply to your email.')

# ---------------------------
# PAGE: Admin (simple Firestore viewer)
# ---------------------------
if page == 'Admin':
    st.header('Admin - Firestore Viewer')
    if db is None:
        st.error('Firestore is not configured or service account missing.')
    else:
        col = st.selectbox('Select collection', ['feedback', 'contacts'])
        if st.button('Load'):
            docs = db.collection(col).order_by('timestamp', direction=firestore.Query.DESCENDING).limit(50).stream()
            rows = []
            for d in docs:
                data = d.to_dict()
                data['id'] = d.id
                rows.append(data)
            if rows:
                df = pd.DataFrame(rows)
                st.dataframe(df)
            else:
                st.info('No documents found in that collection.')

# ---------------------------
# Footer / Legal
# ---------------------------
st.markdown('---')
st.caption('This app is a template for building a stock market prediction website. Not financial advice.')
