"""
Train the model and save it to model/volatility_model.joblib

  Real data (needs internet):  python train.py
                               python train.py --tickers AAPL MSFT TSLA --years 10
  Offline demo data:           python train.py --synthetic
"""
import argparse, os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error

from features import FEATURES, build_features, add_targets

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "volatility_model.joblib")
DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "JPM",
                   "XOM", "WMT", "KO", "DIS", "NFLX", "INTC", "AMD", "BA"]


def load_real(tickers, years):
    import yfinance as yf
    frames = {}
    for t in tickers:
        df = yf.download(t, period=f"{years}y", interval="1d", auto_adjust=True, progress=False)
        if df is not None and len(df) > 300:
            frames[t] = df
            print(f"  {t}: {len(df)} rows")
    return frames


def load_synthetic(n_tickers=40, n_days=2500, seed=7):
    """GARCH(1,1) price paths - gives realistic volatility clustering for offline demos."""
    rng = np.random.default_rng(seed)
    frames = {}
    for i in range(n_tickers):
        omega, alpha, beta = 2e-6 * rng.uniform(0.5, 2), rng.uniform(0.05, 0.12), rng.uniform(0.82, 0.9)
        mu = rng.uniform(0.0, 0.0006)
        var = omega / (1 - alpha - beta)
        rets = np.empty(n_days)
        for t in range(n_days):
            e = rng.standard_t(6) / np.sqrt(6 / 4)
            r = mu + np.sqrt(var) * e
            rets[t] = r
            var = omega + alpha * (r - mu) ** 2 + beta * var
        close = 100 * np.exp(np.cumsum(rets))
        rng_hl = np.abs(rets) * rng.uniform(1.2, 2.2, n_days) + 0.003
        idx = pd.bdate_range("2014-01-01", periods=n_days)
        frames[f"SYN{i}"] = pd.DataFrame({
            "Close": close, "High": close * (1 + rng_hl / 2), "Low": close * (1 - rng_hl / 2),
            "Volume": np.exp(15 + 6 * np.abs(rets) / 0.01 * 0.1 + rng.normal(0, 0.3, n_days)),
        }, index=idx)
    return frames


def make_dataset(frames):
    parts = []
    for name, raw in frames.items():
        d = add_targets(build_features(raw)).dropna()
        d["ticker"] = name
        parts.append(d)
    return pd.concat(parts).sort_index(kind="stable")


def chrono_split(data, test_frac=0.2):
    """Per-ticker chronological split: last 20% of each ticker's history is the test set."""
    tr, te = [], []
    for _, g in data.groupby("ticker"):
        g = g.sort_index()
        cut = int(len(g) * (1 - test_frac))
        tr.append(g.iloc[:cut]); te.append(g.iloc[cut:])
    return pd.concat(tr), pd.concat(te)


def new_model():
    return GradientBoostingRegressor(n_estimators=150, max_depth=3, learning_rate=0.05,
                                     subsample=0.8, min_samples_leaf=40, random_state=42)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    ap.add_argument("--years", type=int, default=10)
    args = ap.parse_args()

    print("Loading data...")
    frames = load_synthetic() if args.synthetic else load_real(args.tickers, args.years)
    if not frames:
        raise SystemExit("No data loaded.")
    data = make_dataset(frames)
    train, test = chrono_split(data)
    print(f"Train rows: {len(train)}  Test rows: {len(test)}")

    # Time-series cross-validation (expanding window, no shuffling) on the training part
    tr_sorted = train.sort_index(kind="stable")
    cv_scores = []
    for fold, (a, b) in enumerate(TimeSeriesSplit(n_splits=4, gap=FORWARD_GAP).split(tr_sorted), 1):
        m = new_model().fit(tr_sorted.iloc[a][FEATURES], tr_sorted.iloc[a]["y_vol"])
        mae = mean_absolute_error(tr_sorted.iloc[b]["y_vol"], m.predict(tr_sorted.iloc[b][FEATURES]))
        cv_scores.append(float(mae)); print(f"  CV fold {fold}: vol MAE={mae:.5f}")

    ret_model = new_model().fit(train[FEATURES], train["y_ret"])
    vol_model = new_model().fit(train[FEATURES], train["y_vol"])

    # Evaluation on the untouched, later-in-time test set
    pred_ret = ret_model.predict(test[FEATURES])
    pred_vol = vol_model.predict(test[FEATURES])
    # price-level error vs. the "tomorrow = today" baseline (price relative to today's close = 1)
    mape_model = float(np.mean(np.abs(np.expm1(pred_ret) - np.expm1(test["y_ret"]))) * 100)
    mape_naive = float(np.mean(np.abs(np.expm1(test["y_ret"])) ) * 100)
    vol_mae = float(mean_absolute_error(test["y_vol"], pred_vol))
    vol_naive = float(mean_absolute_error(test["y_vol"], test["vol_5"]))
    direction = float(np.mean(np.sign(pred_ret) == np.sign(test["y_ret"])) * 100)

    metrics = {
        "trained_on": "synthetic GARCH data" if args.synthetic else "Yahoo Finance: " + ", ".join(frames),
        "train_rows": int(len(train)), "test_rows": int(len(test)),
        "price_error_pct_model": round(mape_model, 4),
        "price_error_pct_naive": round(mape_naive, 4),
        "direction_accuracy_pct": round(direction, 2),
        "volatility_mae_model": round(vol_mae, 6),
        "volatility_mae_naive_vol5": round(vol_naive, 6),
        "cv_vol_mae": [round(x, 6) for x in cv_scores],
    }
    print(metrics)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump({"features": FEATURES, "ret_model": ret_model, "vol_model": vol_model,
                 "metrics": metrics}, MODEL_PATH, compress=3)
    print("Saved", MODEL_PATH, f"({os.path.getsize(MODEL_PATH)/1024:.0f} KB)")


FORWARD_GAP = 5  # gap between train/validation folds so overlapping 5-day targets can't leak

if __name__ == "__main__":
    main()
