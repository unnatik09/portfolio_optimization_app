# src/data_load.py

import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
from src.features import compute_features

def load_nifty30_with_features():
    """
    Downloads price and volume data for the top 30 NIFTY50 companies,
    computes features for each, and aggregates into one DataFrame.
    
    Returns:
        pd.DataFrame: Combined feature DataFrame for all tickers
    """
    nifty_30 = [
        'RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS',
        'LT.NS', 'SBIN.NS', 'KOTAKBANK.NS', 'AXISBANK.NS', 'HCLTECH.NS',
        'ITC.NS', 'ASIANPAINT.NS', 'ULTRACEMCO.NS', 'NESTLEIND.NS', 'HINDUNILVR.NS',
        'BAJFINANCE.NS', 'ADANIENT.NS', 'POWERGRID.NS', 'COALINDIA.NS', 'BHARTIARTL.NS',
        'NTPC.NS', 'TITAN.NS', 'ONGC.NS', 'SUNPHARMA.NS', 'TECHM.NS',
        'DIVISLAB.NS', 'WIPRO.NS', 'MARUTI.NS', 'TATAMOTORS.NS', 'BPCL.NS'
    ]

    end_date = datetime.today()
    start_date = end_date - timedelta(days=5*365)
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    nifty_close = yf.download("^NSEI", start=start_str, end=end_str)["Close"]

    all_features = []
    for ticker in nifty_30:
        df = yf.download(ticker, start=start_str, end=end_str, auto_adjust=True)
        if df.empty or 'Close' not in df or 'Volume' not in df:
            print(f"Skipping {ticker} due to missing data.")
            continue

        price = df['Close'].squeeze()
        volume = df['Volume'].squeeze()

        features = compute_features(price, volume, nifty_close)
        features = features.add_prefix(f"{ticker}_")
        all_features.append(features)

    market_snapshot = pd.concat(all_features, axis=1).dropna()
    return market_snapshot


def get_prices_and_future_returns(tickers, start_str, end_str, market_snapshot):
    import yfinance as yf
    import pandas as pd
    
    prices = yf.download(tickers, start=start_str, end=end_str, auto_adjust=True)["Close"]

    future_returns = pd.DataFrame(index=prices.index)
    for ticker in prices.columns:
        try:
            future_price = prices[ticker].shift(-5)
            future_returns[ticker] = ((future_price - prices[ticker]) / prices[ticker])
        except Exception as e:
            print(f"Error in {ticker}: {e}")

    nf = market_snapshot.join(future_returns, how='inner')
    nf = nf.dropna()

    return nf