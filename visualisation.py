import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pypfopt import EfficientFrontier

def show_portfolio_allocation(user_amount, nf, y, lgbm, ridge, feature_cols):
    """
    Portfolio allocation & visualization using predicted returns (mu_pred)
    and historical daily returns from CSV for covariance matrix.
    """

    # 1️⃣ Prepare features
    nf_pred = nf.reindex(columns=feature_cols, fill_value=0)

    # 2️⃣ Ensemble predictions
    y_pred = 0.7 * lgbm.predict(nf_pred) + 0.3 * ridge.predict(nf_pred)
    tickers = list(y.columns)
    y_pred_df = pd.DataFrame(y_pred, columns=tickers, index=nf.index)

    # 3️⃣ Average predicted returns
    mu_pred = y_pred_df.mean()
    mu_pred = mu_pred.dropna()
    mu_pred = mu_pred[mu_pred != 0]  # drop exact zero returns

    # 4️⃣ Historical returns (from CSV)
    try:
        historical_returns = pd.read_csv("daily_returns.csv", index_col=0, parse_dates=True).dropna(axis=1)
    except FileNotFoundError:
        raise FileNotFoundError("daily_returns.csv not found. Please provide the CSV with historical returns.")

    # 5️⃣ Align assets
    assets = list(set(mu_pred.index).intersection(set(historical_returns.columns)))
    mu = mu_pred.loc[assets]
    mu = mu[mu > 1e-6]  # optional: drop tiny returns
    assets = mu.index

    # 6️⃣ Covariance matrix with small regularization
    S = historical_returns[assets].cov()
    S += np.eye(len(assets)) * 1e-6  # stabilize

    # 7️⃣ Optimize portfolio
    ef = EfficientFrontier(mu, S)
    try:
        weights_series = pd.Series(ef.max_sharpe(), index=assets)
    except:
        # fallback: equal weights if optimizer fails
        n = len(assets)
        weights_series = pd.Series(1/n, index=assets)

    weights_series = weights_series[weights_series > 0]

    # 8️⃣ Pie chart
    fig_pie = px.pie(
        weights_series,
        names=weights_series.index,
        values=weights_series.values * user_amount,
        title="📊 Portfolio Allocation (Optimized Weights)",
        hole=0.4
    )
    fig_pie.update_traces(textinfo='label+percent+value', textfont_size=16)
    fig_pie.update_layout(width=1200, height=500, title_font_size=22)

    # 9️⃣ Portfolio cumulative returns
    portfolio_returns = historical_returns[weights_series.index] @ weights_series
    cumulative_returns = (1 + portfolio_returns).cumprod()

    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(
        x=cumulative_returns.index,
        y=cumulative_returns.values,
        mode='lines',
        line=dict(color='royalblue', width=3)
    ))
    fig_line.update_layout(
        title="📈 Cumulative Portfolio Return (Ensemble Model)",
        xaxis_title="Date",
        yaxis_title="Portfolio Value",
        template="plotly_white",
        width=1200,
        height=500,
        showlegend=False
    )

    return fig_pie, fig_line