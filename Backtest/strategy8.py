
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from typing import Optional

# Import your abstract base Strategy (unchanged)
from Strategy import Strategy


@dataclass
class _S8Base(Strategy):
    """
    Shared prep & parameters for Strategy 8 variants.
    Produces LONG-ONLY signals: 1.0 = enter long, 0.0 = flat.
    Do NOT shift inside .signal(); the backtester handles trade lag/holding.
    """
    window: int = 10000                  # rolling window (bars)
    threshold_pct: float = 0.1        # use (100 - threshold_pct)% quantile of |ret|
    price_col: str = "close"
    interval_minutes: int = 5
    hold_minutes: int = 120

    def _prep(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.price_col not in df.columns:
            raise KeyError(f"Missing price column '{self.price_col}'")
        out = pd.DataFrame(index=df.index)
        px = pd.to_numeric(df[self.price_col], errors="coerce")
        ret = px.pct_change()
        out["ret"] = ret
        out["abs_ret"] = ret.abs()
        # rolling lower-quantile threshold on |ret| (e.g., 90th percentile when threshold_pct=10)
        q = 1.0 - float(self.threshold_pct) * 0.01
        out["q_abs"] = out["abs_ret"].rolling(int(self.window), min_periods=int(self.window)).quantile(q)
        # precompute holding window (bars)
        self._shift_periods = max(1, int(self.hold_minutes // max(1, self.interval_minutes)))
        # rolling MA/STD often used by 8.2_improved
        out["ma"] = px.rolling(self.window).mean()
        out["std"] = px.rolling(self.window).std()
        return out

    

@dataclass
class Strategy8_1(_S8Base):
    """
    Rule:
      ret < 0  AND  ret < - Quantile_{(1 - threshold)}(|ret|) over `window`.
      If triggered when long, extend the holding period
    """
    def signal(self, df: pd.DataFrame) -> pd.DataFrame:
        x = self._prep(df)
        sig = (x["ret"] < 0) & (x["ret"] < -x["q_abs"])
        entry = sig
        # exit = sig.shift(self._shift_periods, fill_value=0)
        # extend holding if already long
        # so we dont need to set the event exit signal here
        # the extend mode will handle it 
        # as long as we set the fixed holding period
        exit = pd.Series(False, index=df.index)
        side = sig.astype(float).fillna(0.0)
        sig = pd.DataFrame({
            "entry": entry.astype(float).fillna(0.0),
            "exit": exit.astype(float).fillna(0.0),
            "side": side
        }, index=df.index)
        return sig


@dataclass
class Strategy8_2(_S8Base):
    """
    Same trigger as 8_1.
    Open a new position every time the trigger fires, even if already long.
    """
    def signal(self, df: pd.DataFrame) -> pd.DataFrame:
        x = self._prep(df)
        sig = (x["ret"] < 0) & (x["ret"] < -x["q_abs"])
        entry = sig
        exit = sig.shift(self._shift_periods, fill_value=0)
        #exit = pd.Series(False, index=df.index)
        side = sig.astype(float).fillna(0.0)
        sig = pd.DataFrame({
            "entry": entry.astype(float).fillna(0.0),
            "exit": exit.astype(float).fillna(0.0),
            "side": side
        }, index=df.index)
        return sig
    
@dataclass
class Strategy8_2_Improved(_S8Base):
    hard_drop: float = 0.05   # require at least -5% bar move (tweak to taste)
    k_std: float = 5.0        # MA band: px <= ma + k_std*std

    def signal(self, df: pd.DataFrame) -> pd.DataFrame:
        x = self._prep(df)
        base = (x["ret"] < 0) & (x["ret"] < -x["q_abs"])
        sig = base & (x["ret"] <= -self.hard_drop) & (df[self.price_col] <= x["ma"] + self.k_std * x["std"])
        entry = sig
        exit = sig.shift(self._shift_periods, fill_value=0)
        #exit = pd.Series(False, index=df.index)
        side = sig.astype(float).fillna(0.0)
        sig = pd.DataFrame({
            "entry": entry.astype(float).fillna(0.0),
            "exit": exit.astype(float).fillna(0.0),
            "side": side
        }, index=df.index)
        return sig
        
    
    
@dataclass
class Strategy8_3(_S8Base):
    """
    Rule:
      8_1 trigger AND today's |ret| exceeds the *max* of the previous
      `hold_minutes` worth of |ret| (lookback excludes today).
        => |ret_t| > max_{t-1..t-hold} |ret|
    """
    def signal(self, df: pd.DataFrame) -> pd.DataFrame:
        x = self._prep(df)
        prev_abs_max = x["abs_ret"].shift(1).rolling(self._shift_periods, min_periods=self._shift_periods).max()
        sig = (x["ret"] < 0) & (x["ret"] < -x["q_abs"]) & (x["abs_ret"] > prev_abs_max)
        entry = sig
        exit = sig.shift(self._shift_periods, fill_value=0)
        #exit = pd.Series(False, index=df.index)
        side = sig.astype(float).fillna(0.0)
        sig = pd.DataFrame({
            "entry": entry.astype(float).fillna(0.0),
            "exit": exit.astype(float).fillna(0.0),
            "side": side
        }, index=df.index)
        return sig


@dataclass
class Strategy8_4(_S8Base):
    """
    Rule:
      8_1 trigger AND today's negative return is *more negative* than the
      minimum of (-ret) over the previous holding window (excluding today):
        => ret_t <  min_{t-1..t-hold} (-ret)
    """
    def signal(self, df: pd.DataFrame) -> pd.DataFrame:
        x = self._prep(df)
        # previous 2h max of ret is equivalent to min of -ret
        prev_neg_min = (-x["ret"]).shift(1).rolling(self._shift_periods, min_periods=self._shift_periods).min()
        sig = (x["ret"] < 0) & (x["ret"] < -x["q_abs"]) & (x["ret"] < prev_neg_min)
        entry = sig
        exit = sig.shift(self._shift_periods, fill_value=0)
        #exit = pd.Series(False, index=df.index)
        side = sig.astype(float).fillna(0.0)
        sig = pd.DataFrame({
            "entry": entry.astype(float).fillna(0.0),
            "exit": exit.astype(float).fillna(0.0),
            "side": side
        }, index=df.index)
        return sig