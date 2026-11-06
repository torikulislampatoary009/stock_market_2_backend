"""Flask API - Stock Market Volatility Predictor."""
import os
import re
import time
import logging

import numpy as np
import pandas as pd
import joblib
from flask import Flask, jsonify, request

import db
from features import FEATURES, build_features, _flatten

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("app")

app = Flask(__name__)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "volatility_model.joblib")
_bundle = None
_price_cache = {}          # ticker -> (timestamp, dataframe)
CACHE_SECONDS = 3600
TICKER_RE = re.compile(r"^[A-Z0-9.\-\^=]{1,12}$")

# ---------- CORS (set CORS_ORIGINS="https://your-frontend.onrender.com" in Render; default "*") ----------
ALLOWED = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]


@app.after_request
def add_cors(resp):
    origin = request.headers.get("Origin", "")
    if "*" in ALLOWED:
        resp.headers["Access-Control-Allow-Origin"] = "*"
    elif origin in ALLOWED:
        resp.headers["Access-Control-Allow-Origin"] = origin
        resp.headers["Vary"] = "Origin"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return resp


@app.route("/api/<path:_any>", methods=["OPTIONS"])
def preflight(_any):
    return ("", 204)


# ---------- helpers ----------
def get_bundle():
    global _bundle
    if _bundle is None:
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


def fetch_prices(ticker):
    cached = _price_cache.get(ticker)
    if cached and time.time() - cached[0] < CACHE_SECONDS:
        return cached[1]
    import yfinance as yf
    df = yf.download(ticker, period="2y", interval="1d", auto_adjust=True, progress=False)
    if df is None or df.empty:
        raise ValueError(f"No data found for ticker '{ticker}'.")
    df = _flatten(df).dropna(subset=["Close"])
    if len(df) < 60:
        raise ValueError(f"Not enough history for '{ticker}'.")
    _price_cache[ticker] = (time.time(), df)
    return df


def vol_level(annual):
    if annual < 0.15: return "Low"
    if annual < 0.30: return "Moderate"
    if annual < 0.50: return "High"
    return "Extreme"


def init_db_safely():
    try:
        db.init_db()
        log.info("Database ready.")
    except Exception as e:  # app still works without DB (predictions just aren't saved)
        log.warning("Database not available: %s", e)


init_db_safely()


# ---------- routes ----------
@app.route("/")
def index():
    return jsonify(name="Stock Market Volatility Predictor API", endpoints=["/api/health", "/api/predict", "/api/history", "/api/model-info"])


@app.route("/api/health")
def health():
    try:
        db.ping(); db_ok = True; db_msg = "connected"
    except Exception as e:
        db_ok = False; db_msg = str(e)[:150]
    return jsonify(status="ok", model_loaded=os.path.exists(MODEL_PATH), database=db_msg, database_ok=db_ok)


@app.route("/api/model-info")
def model_info():
    b = get_bundle()
    return jsonify(features=b["features"], metrics=b["metrics"])


@app.route("/api/predict", methods=["POST"])
def predict():
    body = request.get_json(silent=True) or {}
    ticker = str(body.get("ticker", "")).strip().upper()
    if not TICKER_RE.match(ticker):
        return jsonify(error="Please enter a valid ticker symbol, e.g. AAPL."), 400
    try:
        raw = fetch_prices(ticker)
    except ValueError as e:
        return jsonify(error=str(e)), 404
    except Exception as e:
        log.exception("Yahoo fetch failed")
        return jsonify(error="Could not fetch data from Yahoo Finance right now. Try again in a minute."), 502

    feats = build_features(raw)
    last = feats.dropna()
    if last.empty:
        return jsonify(error="Not enough clean data to build features."), 422
    x = last.iloc[[-1]][FEATURES]          # most recent complete row - uses only past data

    b = get_bundle()
    pred_ret = float(b["ret_model"].predict(x)[0])
    pred_vol = float(max(b["vol_model"].predict(x)[0], 0.0))
    last_close = float(raw["Close"].iloc[-1])
    pred_close = last_close * float(np.exp(pred_ret))
    annual = pred_vol * float(np.sqrt(252))

    result = {
        "ticker": ticker,
        "as_of_date": str(raw.index[-1].date()),
        "last_close": round(last_close, 2),
        "predicted_close": round(pred_close, 2),
        "predicted_change_pct": round((pred_close / last_close - 1) * 100, 3),
        "predicted_volatility": round(pred_vol, 5),
        "annualized_volatility": round(annual, 4),
        "volatility_level": vol_level(annual),
    }

    saved = True
    try:
        db.save_prediction(result)
    except Exception as e:
        saved = False
        log.warning("Could not save prediction: %s", e)

    tail = raw.tail(120)
    close = raw["Close"]
    ma20 = close.rolling(20).mean().tail(120)
    vol20 = (np.log(close).diff().rolling(20).std() * np.sqrt(252)).tail(120)
    result["saved_to_db"] = saved
    result["chart"] = {
        "dates": [str(d.date()) for d in tail.index],
        "close": [round(float(v), 2) for v in tail["Close"]],
        "ma20": [None if pd.isna(v) else round(float(v), 2) for v in ma20],
        "vol20": [None if pd.isna(v) else round(float(v), 4) for v in vol20],
    }
    return jsonify(result)


@app.route("/api/history")
def history():
    try:
        limit = max(1, min(int(request.args.get("limit", 20)), 100))
        ticker = request.args.get("ticker", "").strip().upper() or None
        return jsonify(db.get_history(limit, ticker))
    except Exception as e:
        log.warning("History failed: %s", e)
        return jsonify(error="Database is not reachable. Check the DB_* environment variables on Render."), 503


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=False)
