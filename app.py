# streamlit/app.py

import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import os
from src.data_load import load_nifty30_with_features, get_prices_and_future_returns
from src.train_model import train_and_evaluate
from src.visualisation import show_portfolio_allocation
from src.sentiment_allocation import (
    get_sentiment_adjusted_weights,
    allocate_capital,
    plot_allocation_pie
)

MODEL_DIR = "models"
FEATURES_FILE = os.path.join(MODEL_DIR, "feature_cols.pkl")
LGBM_FILE = os.path.join(MODEL_DIR, "lgbm.pkl")
RIDGE_FILE = os.path.join(MODEL_DIR, "ridge.pkl")

st.set_page_config(page_title="NIFTY30 Prediction Dashboard", layout="wide")
st.title("📈 NIFTY30 Stock Prediction & Analysis")

# ----------------- Step 1: Load Data -----------------
st.subheader("1️⃣ Data Loading")
if st.button("Load NIFTY30 Data"):
    st.write("Fetching data from Yahoo Finance... Please wait ⏳")
    market_snapshot = load_nifty30_with_features()
    st.session_state.market_snapshot = market_snapshot
    st.success(f"Loaded {market_snapshot.shape[0]} rows and {market_snapshot.shape[1]} features.")

# ----------------- Step 2: Prepare Target Returns -----------------
if "market_snapshot" in st.session_state:
    st.subheader("2️⃣ Prepare Dataset for Modeling")
    nifty_30 = sorted(set(col.split("_")[0] for col in st.session_state.market_snapshot.columns))
    nf = get_prices_and_future_returns(
        tickers=nifty_30,
        start_str=st.session_state.market_snapshot.index.min().strftime("%Y-%m-%d"),
        end_str=st.session_state.market_snapshot.index.max().strftime("%Y-%m-%d"),
        market_snapshot=st.session_state.market_snapshot
    )
    st.session_state.nf = nf
    st.success(f"Prepared dataset with {nf.shape[0]} rows.")

# ----------------- Step 3: Train or Load Models -----------------
if "nf" in st.session_state:
    st.subheader("3️⃣ Train or Load Models")
    X = st.session_state.nf[st.session_state.market_snapshot.columns]
    y = st.session_state.nf[[col for col in st.session_state.nf.columns if col not in X.columns]]
    st.session_state.y = y  # store historical/future prices for returns

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Train Model"):
            lgbm, ridge, feature_cols = train_and_evaluate(X, y, save_dir=MODEL_DIR)
            st.session_state.models = (lgbm, ridge)
            st.session_state.feature_cols = feature_cols
            st.success("Model training completed ✅")
    with col2:
        if st.button("Load Saved Models"):
            if os.path.exists(LGBM_FILE) and os.path.exists(RIDGE_FILE) and os.path.exists(FEATURES_FILE):
                lgbm = joblib.load(LGBM_FILE)
                ridge = joblib.load(RIDGE_FILE)
                feature_cols = joblib.load(FEATURES_FILE)
                st.session_state.models = (lgbm, ridge)
                st.session_state.feature_cols = feature_cols
                st.success("Loaded models from disk ✅")
            else:
                st.error("Saved models not found. Please train first.")

# ----------------- Step 4: Portfolio Allocation -----------------
if "models" in st.session_state:
    st.subheader("4️⃣ Portfolio Allocation")
    user_amount = st.number_input("💰 Enter investment amount (₹)", min_value=1000.0, value=100000.0, step=1000.0)

    if st.button("Show Portfolio Allocation"):
        fig_pie, fig_line = show_portfolio_allocation(
            user_amount=user_amount,
            nf=st.session_state.nf,
            y=st.session_state.y,
            lgbm=st.session_state.models[0],
            ridge=st.session_state.models[1],
            feature_cols=st.session_state.feature_cols
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        st.plotly_chart(fig_line, use_container_width=True)

# ----------------- Step 5: Sentiment-Adjusted Portfolio -----------------
if "models" in st.session_state:
    st.subheader("5️⃣ Sentiment-Adjusted Allocation")

    # Inputs
    api_key = st.text_input("🔑 Enter your NewsAPI key", type="password")
    alpha = st.slider("Sentiment Adjustment Factor (alpha)", 0.0, 1.0, 0.5, 0.05)

    # Button click triggers action
    if st.button("Show Sentiment-Adjusted Portfolio"):
        if not api_key:
            st.error("Please enter your NewsAPI key!")
        else:
            # 1️⃣ Prepare features
            nf_pred = st.session_state.nf.reindex(columns=st.session_state.feature_cols, fill_value=0)

            # 2️⃣ Ensemble prediction
            y_pred = 0.7 * st.session_state.models[0].predict(nf_pred) + \
                     0.3 * st.session_state.models[1].predict(nf_pred)
            tickers = st.session_state.y.columns

            # 3️⃣ Sentiment-adjusted weights
            weights_df = get_sentiment_adjusted_weights(
                y_pred=y_pred,
                tickers=tickers,
                api_key=api_key,
                alpha=alpha
            )

            # 4️⃣ Allocate capital
            allocation_df = allocate_capital(weights_df, portfolio_value=user_amount)

            # 5️⃣ Safe pastel pie chart
            colors = px.colors.qualitative.Pastel
            colors = colors * ((len(allocation_df) // len(colors)) + 1)
            fig_pie_sent = px.pie(
                allocation_df,
                names=allocation_df.index,
                values=allocation_df["Allocated Amount (Rs.)"],
                title="📊 Sentiment-Adjusted Portfolio",
                hole=0.4,
                color_discrete_sequence=colors[:len(allocation_df)]
            )
            fig_pie_sent.update_traces(textinfo='label+percent+value', textfont_size=16)
            fig_pie_sent.update_layout(width=900, height=500, title_font_size=22)

            # 6️⃣ Display results
            st.dataframe(allocation_df)
            st.plotly_chart(fig_pie_sent, use_container_width=True)