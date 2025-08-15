# src/features.py

import numpy as np
import pandas as pd
import ta  # technical analysis library

#function to compute technical and signal features

def compute_features(df_price, df_volume, nifty_close):

    df = pd.DataFrame(index=df_price.index)

    # --- Returns ---
    df["log_ret_1d"] = np.log(df_price / df_price.shift(1))
    df["log_ret_5d"] = np.log(df_price / df_price.shift(5))

    # --- Volatility ---
    df["vol_5d"] = df["log_ret_1d"].rolling(5).std()
    df["vol_20d"] = df["log_ret_1d"].rolling(20).std()

    # --- RSI ---
    df["rsi_14"] = ta.momentum.RSIIndicator(close=df_price, window=14).rsi()

    # --- Moving Averages ---
    df["ma_5d"] = df_price.rolling(5).mean()
    df["ma_20d"] = df_price.rolling(20).mean()

    # --- MACD ---
    macd = ta.trend.MACD(close=df_price)
    df["macd_diff"] = macd.macd_diff()

    # --- Bollinger Band Width ---
    bb = ta.volatility.BollingerBands(close=df_price, window=20)
    df["bollinger_width"] = bb.bollinger_wband()

    # --- CCI ---
    df["cci"] = ta.trend.CCIIndicator(high=df_price, low=df_price, close=df_price, window=20).cci()

    # --- ADX ---
    adx = ta.trend.ADXIndicator(high=df_price, low=df_price, close=df_price, window=14)
    df["adx"] = adx.adx()

    # --- Volume Change ---
    df["vol_chg"] = df_volume.pct_change()

    # --- Price Ratios ---
    df["high_low_ratio"] = (df_price.rolling(1).max() - df_price.rolling(1).min()) / df_price.rolling(1).min()
    df["close_open_ratio"] = (df_price - df_price.shift(1)) / df_price.shift(1)

    nifty = pd.DataFrame(index=nifty_close.index)
    nifty["daily_return"] = nifty_close.pct_change()
    nifty["rolling_mean_20d"] = nifty["daily_return"].rolling(20).mean()

    def classify_regime(x):
        if x > 0.001:
            return "bull"
        elif x < -0.001:
            return "bear"
        else:
            return "neutral"

    nifty["regime"] = nifty["rolling_mean_20d"].apply(classify_regime)
    df["market_regime"] = df.index.map(nifty["regime"])
    df = pd.get_dummies(df, columns=["market_regime"])  # One-hot encode

    # --- Cleanup ---
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(method="ffill", inplace=True)
    df.fillna(method="bfill", inplace=True)
    df.fillna(0, inplace=True)

    return df