from __future__ import annotations
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Callable, Optional, Iterable
import numpy as np
import pandas as pd

# ============================================================
# strategy.py
# Minimal, flexible strategy interface + ready-made strategies.
# Works with ANY dataframe produced by DataLoader;  
# Decide which columns a strategy can see.
# ============================================================

class Strategy(ABC):
    @abstractmethod
    def signal(self, df: pd.DataFrame) -> pd.Series | pd.DataFrame:
        """Produce either a target exposure Series indexed like df 
            or a dataframe with entry exit and direction.

        Values typically in [-1, 1] (short/flat/long) or arbitrary weights.
        IMPORTANT: Do NOT shift here; the backtester applies trade lag.
        """
        ...


# ------------------------------------------------------------
# Function-based strategy wrapper (optionally column-limited)
# ------------------------------------------------------------

@dataclass
class FunctionStrategy(Strategy):
    func: Callable[[pd.DataFrame], pd.Series]
    use_cols: Optional[Iterable[str]] = None  # if set, only these columns are passed to func

    def signal(self, df: pd.DataFrame) -> pd.Series:
        view = df if self.use_cols is None else df[self.use_cols]
        s = self.func(view)
        if not isinstance(s, pd.Series):
            raise TypeError("FunctionStrategy.func must return a pandas Series")
        return s.reindex(df.index)


def function_strategy(
    func: Callable[[pd.DataFrame], pd.Series],
    *,
    use_cols: Optional[Iterable[str]] = None,
) -> Strategy:
    """Factory (can be used as a decorator) to wrap a plain function(df)->Series
    into a Strategy, optionally restricting visible columns to `use_cols`.

    Examples
    --------
    strat = function_strategy(my_alpha, use_cols=["close","rsi","alpha1"])  
    sig = strat.signal(df)

    @function_strategy  # no use_cols; function sees full df
    def my_fn(df):
        return (df["rsi"] < 30).astype(float) - (df["rsi"] > 70).astype(float)
    sig = my_fn.signal(df)
    """
    return FunctionStrategy(func=func, use_cols=use_cols)


# ------------------------------------------------------------
# Built-in strategies (work with arbitrary loader columns)
# ------------------------------------------------------------

@dataclass
class BuyAndHold(Strategy):
    weight: float = 1.0
    def signal(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(self.weight, index=df.index)


@dataclass
class SMACross(Strategy):
    fast: int = 10
    slow: int = 50
    price_col: str = "close"
    def signal(self, df: pd.DataFrame) -> pd.Series:
        px = pd.to_numeric(df[self.price_col], errors="coerce")
        ma_f = px.rolling(self.fast, min_periods=self.fast).mean()
        ma_s = px.rolling(self.slow, min_periods=self.slow).mean()
        raw = pd.Series(np.sign(ma_f - ma_s))
        return raw.fillna(0.0)



@dataclass
class EMACross(Strategy):
    fast: int = 12
    slow: int = 26
    price_col: str = "close"
    def signal(self, df: pd.DataFrame) -> pd.Series:
        px = pd.to_numeric(df[self.price_col], errors="coerce")
        ema_f = px.ewm(span=self.fast, adjust=False).mean()
        ema_s = px.ewm(span=self.slow, adjust=False).mean()
        raw = pd.Series(np.sign(ema_f - ema_s))
        return raw.fillna(0.0)


@dataclass
class ZScoreReversion(Strategy):
    lookback: int = 50
    z_entry: float = 1.0
    price_col: str = "close"
    def signal(self, df: pd.DataFrame) -> pd.Series:
        px = pd.to_numeric(df[self.price_col], errors="coerce")
        ma = px.rolling(self.lookback, min_periods=self.lookback).mean()
        sd = px.rolling(self.lookback, min_periods=self.lookback).std()
        z = (px - ma) / sd.replace(0, np.nan)
        s = pd.Series(0.0, index=df.index)
        s[z < -self.z_entry] = 1.0
        s[z >  self.z_entry] = -1.0
        return s.fillna(0.0)


@dataclass
class RSIBands(Strategy):
    period: int = 14
    lower: float = 30
    upper: float = 70
    price_col: str = "close"
    def signal(self, df: pd.DataFrame) -> pd.Series:
        px = pd.to_numeric(df[self.price_col], errors="coerce")
        chg = px.diff()
        up  = chg.clip(lower=0.0).rolling(self.period).mean()
        dn  = (-chg.clip(upper=0.0)).rolling(self.period).mean()
        rs  = up / dn.replace(0, np.nan)
        rsi = 100 - 100/(1+rs)
        s = pd.Series(0.0, index=df.index)
        s[rsi < self.lower] =  1.0
        s[rsi > self.upper] = -1.0
        return s.fillna(0.0)


# ------------------------------------------------------------
# Optional helpers for signal post-processing
# ------------------------------------------------------------

def clip_weights(sig: pd.Series, *, min_w: float = -1.0, max_w: float = 1.0) -> pd.Series:
    return sig.clip(min_w, max_w)


def de_churn(sig: pd.Series, *, min_hold: int = 1) -> pd.Series:
    if min_hold <= 1:
        return sig
    sig = sig.fillna(0.0)
    out = sig.copy()
    last = 0.0
    hold = 0
    for i, v in enumerate(sig.values):
        if v != 0:
            last, hold = v, min_hold
        elif hold > 0:
            v, hold = last, hold - 1
        out.iat[i] = v
    return out

'''
if __name__ == "__main__":
    # smoke test
    idx = pd.date_range("2024-01-01", periods=200, freq="D")
    df = pd.DataFrame({"close": np.cumsum(np.random.randn(len(idx))) + 100}, index=idx)

    # built-in
    sig1 = SMACross(fast=20, slow=50).signal(df)

    # function-based, limited to specific loader features
    def my_alpha(view_df):
        return (view_df["close"].pct_change().fillna(0).rolling(3).mean() > 0).astype(float)

    strat = function_strategy(my_alpha, use_cols=["close"])  # only 'close' passed into my_alpha
    sig2 = strat.signal(df)

    print(sig1.tail())
    print(sig2.tail())
'''