import pandas as pd
import numpy as np
from datetime import datetime
from newsapi import NewsApiClient
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from pypfopt.efficient_frontier import EfficientFrontier
from tqdm import tqdm
import plotly.express as px

def get_sentiment_adjusted_weights(y_pred, tickers, api_key, alpha=0.5):
    """
    Fetches sentiment for tickers, adjusts predicted returns, and calculates portfolio weights.
    """
    today_str = datetime.today().strftime('%Y-%m-%d')

    # Load FinBERT
    model_name = "yiyanghkust/finbert-tone"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    sentiment_model = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

    # Fetch sentiment from NewsAPI
    newsapi = NewsApiClient(api_key=api_key)
    sentiment_today = {}

    for ticker in tqdm(tickers):
        query = ticker.replace(".NS", "") + " stock"
        try:
            articles = newsapi.get_everything(
                q=query,
                from_param=today_str,
                to=today_str,
                language='en',
                page_size=10
            )
            headlines = [a['title'] for a in articles['articles']]
            if not headlines:
                sentiment_today[ticker] = 0
                continue

            results = sentiment_model(headlines)
            score = sum(
                1 if r['label'].lower() == 'positive' else -1 if r['label'].lower() == 'negative' else 0
                for r in results
            )
            sentiment_today[ticker] = round(score / len(results), 4)
        except Exception as e:
            sentiment_today[ticker] = 0

    # Adjust returns
    mu_pred = pd.Series(y_pred.mean(axis=0), index=tickers)
    mu_sent = mu_pred * (1 + alpha * pd.Series(sentiment_today))

    # Optimize Portfolio
    returns = pd.read_csv("daily_returns.csv", index_col=0, parse_dates=True).dropna(axis=1)
    cov_matrix = returns[tickers].cov()

    ef = EfficientFrontier(mu_sent, cov_matrix)
    weights_sent = ef.max_sharpe()
    cleaned_weights = ef.clean_weights()

    weights_df = pd.DataFrame.from_dict(cleaned_weights, orient='index', columns=["Weight"])
    weights_df.index.name = "Ticker"

    return weights_df


def allocate_capital(weights_df, portfolio_value=100_000):
    """
    Calculates allocation in Rs. and returns DataFrame.
    """
    allocation_df = weights_df[weights_df["Weight"] > 0].copy()
    allocation_df["Allocated Amount (Rs.)"] = allocation_df["Weight"] * portfolio_value
    allocation_df = allocation_df.sort_values("Allocated Amount (Rs.)", ascending=False)
    return allocation_df


def plot_allocation_pie(allocation_df):
    """
    Creates pastel blue allocation pie chart.
    """
    pastel_blues = px.colors.sequential.Blues[len(allocation_df) + 2][1:]
    fig_pie = px.pie(
        allocation_df,
        names=allocation_df.index,
        values=allocation_df["Allocated Amount (Rs.)"],
        hole=0.4,
        color_discrete_sequence=pastel_blues,
        title="📊 Portfolio Allocation"
    )
    fig_pie.update_traces(textinfo='label+percent+value', textfont_size=16)
    fig_pie.update_layout(width=900, height=500, title_font_size=22)
    return fig_pie