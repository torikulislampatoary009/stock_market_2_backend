"""
Feature engineering for the Stock Market Volatility Predictor.

LEAKAGE RULES (the "handle time-series data leakage" part of the project):
  1. Every feature at row t uses only data up to and including day t
     (rolling windows look backwards only).
  2. Targets are created with shift(-k), so they live in the future relative to the row.
  3. Features are scale-free (returns / ratios), so no global statistics are needed.
  4. Train/test split is chronological - never shuffled.
"""
import numpy as np
import pandas as pd

FEATURES = [
    "ret_1", "ret_lag_1", "ret_lag_2", "abs_ret_1",
    "ret_mean_5", "ret_mean_10", "ret_mean_20",
    "vol_5", "vol_10", "vol_20", "vol_ratio_5_20",
    "ma_ratio_5", "ma_ratio_10", "ma_ratio_20",
    "hl_range", "volume_chg", "rsi_14",
]

FORWARD_VOL_DAYS = 5  # volatility target = std of the next 5 daily returns


def _flatten(df: pd.DataFrame) -> pd.DataFrame:
    """yfinance can return MultiIndex columns (field, ticker) - flatten to fields."""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    return df


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    raw = _flatten(raw)
    close = raw["Close"].astype(float)
    d = pd.DataFrame(index=raw.index)

    d["ret_1"] = np.log(close).diff()
    d["ret_lag_1"] = d["ret_1"].shift(1)
    d["ret_lag_2"] = d["ret_1"].shift(2)
    d["abs_ret_1"] = d["ret_1"].abs()

    for w in (5, 10, 20):
        d[f"ret_mean_{w}"] = d["ret_1"].rolling(w).mean()          # rolling average of returns
        d[f"vol_{w}"] = d["ret_1"].rolling(w).std()                # rolling volatility
        d[f"ma_ratio_{w}"] = close / close.rolling(w).mean() - 1   # price vs rolling average
    d["vol_ratio_5_20"] = d["vol_5"] / d["vol_20"].replace(0, np.nan)

    if {"High", "Low"}.issubset(raw.columns):
        d["hl_range"] = (raw["High"] - raw["Low"]) / close
    else:
        d["hl_range"] = d["abs_ret_1"]

    if "Volume" in raw.columns:
        vol = raw["Volume"].astype(float).replace(0, np.nan)
        d["volume_chg"] = np.log(vol).diff().clip(-2, 2)
    else:
        d["volume_chg"] = 0.0

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    d["rsi_14"] = (100 - 100 / (1 + rs)) / 100

    d = d.replace([np.inf, -np.inf], np.nan)
    return d[FEATURES]


def add_targets(feats: pd.DataFrame) -> pd.DataFrame:
    """Targets are shifted INTO THE FUTURE so a row never sees its own answer."""
    out = feats.copy()
    out["y_ret"] = feats["ret_1"].shift(-1)                                            # next-day log return
    out["y_vol"] = feats["ret_1"].rolling(FORWARD_VOL_DAYS).std().shift(-FORWARD_VOL_DAYS)  # next-5-day volatility
    return out
