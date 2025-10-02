from __future__ import annotations
from dataclasses import dataclass,field
from typing import Optional, Tuple, List, Dict
import numpy as np
import pandas as pd

# ============================================================
# cost & portfolio/execution (single-asset). Now with optional
# stop-loss / take-profit and a trade log.
# ============================================================

@dataclass
class CostModel:
    """
    Simple proportional trading cost model in basis points (bps).
    Costs are charged on per-bar gross turnover (absolute units traded).
    """
    fee_bps: float = 1.0
    slip_bps: float = 2.0

    @property
    def total_bps(self) -> float:
        return float(self.fee_bps) + float(self.slip_bps)

    def cost_rate(self, turnover: pd.Series) -> pd.Series:
        """
        Parameters
        ----------
        turnover : pd.Series
            Absolute traded units per bar (e.g., |Δposition| or gross flow).

        Returns
        -------
        pd.Series
            Per-bar cost rate (as a return), same index as turnover.
        """
        return turnover * (self.total_bps / 1e4)


# ------------------------------------------------------------
# Exit policy: stop loss / take profit / max hold / trailing
# ------------------------------------------------------------

@dataclass
class ExitPolicy:
    """Bar-close exit rules evaluated while in a position.

    All percentages are expressed as decimals (e.g., 0.02 = 2%).
    - stop_loss_pct: hard stop if PnL ≤ -stop_loss_pct
    - take_profit_pct: take profit if PnL ≥ take_profit_pct
    - max_hold_bars: exit after this many bars in position
    - trailing_stop_pct: exit if drawdown from peak PnL ≥ trailing_stop_pct

    Note: evaluated at bar closes; intrabar exits are not modeled here.
    """
    stop_loss_pct: Optional[float] = None
    take_profit_pct: Optional[float] = None
    max_hold_bars: Optional[int] = None
    trailing_stop_pct: Optional[float] = None

    def check(
        self,
        entry_price: float,
        side: int,
        price_now: float,
        bar_age: int,
        peak_price_since_entry: float,
    ) -> Tuple[bool, Optional[str]]:
        """Return (should_exit, reason). side in {+1,-1}.
        PnL is computed vs. entry on bar close.
        """
        pnl = side * (price_now / entry_price - 1.0)
        # hard stop
        if self.stop_loss_pct is not None and pnl <= -abs(self.stop_loss_pct):
            return True, "stop_loss"
        # take profit
        if self.take_profit_pct is not None and pnl >= abs(self.take_profit_pct):
            return True, "take_profit"
        # max holding period
        if self.max_hold_bars is not None and bar_age >= int(self.max_hold_bars):
            return True, "max_hold"
        # trailing stop based on peak move in favorable direction
        if self.trailing_stop_pct is not None:
            # compute peak PnL in favorable direction using peak_price_since_entry
            peak_pnl = side * (peak_price_since_entry / entry_price - 1.0)
            if peak_pnl > 0:
                drawdown_from_peak = peak_pnl - pnl
                if drawdown_from_peak >= abs(self.trailing_stop_pct):
                    return True, "trailing_stop"
        return False, None


# ------------------------------------------------------------
# Execution policy (lag, bounds, optional exit policy)
# ------------------------------------------------------------

@dataclass
class ExecutionPolicy:
    """
    Execution assumptions & event-conflict semantics.

    trade_lag    : bars between signal and execution (1 = next bar)
    bound        : clamp Series signal to [-bound, +bound] if provided (Series path only)
    exit_policy  : optional price-based ExitPolicy (used in event runner if enabled)

    entry_conflict : behavior when an entry arrives while already in position
        - "ignore"  : do nothing (default)
        - "extend"  : refresh scheduled exit to now + fixed_hold_bars
        - "pyramid" : open a new lot (qty = lot_size)
        - "flip"    : close all open lots, then open a new lot with desired side

    fixed_hold_bars : if set, each entry schedules an exit at entry_bar + H (used by extend/pyramid)
    exit_scope      : "all" (default) or "side" (explicit exit closes all lots or only lots on current net side)
    lot_size        : size per entry for pyramiding
    force_close_eod : if True, close any open lots at the last bar
    apply_event_exit_policy : if True, check exit_policy per-lot during event runs

    exit_lot_policy : for explicit exit, which candidates to close ("all" | "fifo" | "lifo")
    exit_lot_count  : for explicit exit, number of lots to close (None => all candidates)
    """
    trade_lag: int = 1
    bound: Optional[float] = 1.0
    exit_policy: Optional[ExitPolicy] = None

    # Event conflict controls
    entry_conflict: str = "ignore"        # "ignore" | "extend" | "pyramid" | "flip"
    fixed_hold_bars: Optional[int] = None
    exit_scope: str = "all"               # "all" | "side"
    lot_size: float = 1.0
    force_close_eod: bool = False
    apply_event_exit_policy: bool = True
    exit_lot_policy: str = "all"   # "all" | "fifo" | "lifo"
    exit_lot_count: Optional[int] = None  # None => close all candidates; e.g. 1 => close one lot

    # Helpers for Series signals
    def bound_signal(self, signal: pd.Series, allow_short: bool) -> pd.Series:
        sig = signal.copy()
        if not allow_short:
            sig = sig.clip(lower=0.0, upper=1.0)
        elif self.bound is not None:
            sig = sig.clip(-self.bound, self.bound)
        return sig

    def executed_position(self, bounded_signal: pd.Series) -> pd.Series:
        # Avoid look-ahead: execute with lag
        return bounded_signal.shift(self.trade_lag).fillna(0.0)


# ------------------------------------------------------------
# Portfolio with optional bar-by-bar engine for exits & trade log
# ------------------------------------------------------------

@dataclass
class Portfolio:
    """
    Single-asset portfolio.

    - Series signal -> fast MTM path (no trade log).
    - DataFrame events -> universal event runner (trade log produced).

    Event schema expected (aligned to price index):
      - 'enter' (bool): entry trigger
      - 'side'  (int): +1 / -1 for intended entry
      - 'exit'  (bool, optional): explicit exit trigger (closes lots per exit_scope)
      - 'qty'   (float, optional): per-entry size (defaults to exec.lot_size)
    """
    initial_equity: float = 1
    price_col: str = "close"  # the price column to use for PnL calculations
    allow_short: bool = True
    accounting: str = "trade"  # "mtm" or "trade"
    cost: CostModel = field(default_factory=CostModel)
    exec: ExecutionPolicy = field(default_factory=ExecutionPolicy)

    # runtime
    _trade_log: Optional[pd.DataFrame] = None # populated when events (DataFrame) are used

    # -------------- public API --------------
    def run(self, df: pd.DataFrame, signal) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Return the ledger DataFrame (and attach trade_log in attrs if built).
        Use `get_trade_log()` afterward to retrieve the trade log.
        """
        if isinstance(signal, pd.Series):
            # Position-style strategy -> MTM compounding; no trade log
            ledger, trade_log  = self._run_vectorized(df, signal)
            self._trade_log = trade_log
            return ledger, trade_log
            
        elif isinstance(signal, pd.DataFrame):
            # event-style strategy -> force trade-based accounting
            ledger, trade_log =  self._run_events_trade(df, signal)
            self._trade_log = trade_log
            return ledger, trade_log
        else:
            raise TypeError("signal must be a pandas Series (positions) or DataFrame (events)")

    

    def get_trade_log(self) -> Optional[pd.DataFrame]:
        return getattr(self, "_trade_log", None)

    # -------------- vectorized path (no exits policy) --------------
    def _run_vectorized(self, df: pd.DataFrame, signal: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if self.price_col not in df.columns:
            raise KeyError(f"price_col '{self.price_col}' not in dataframe")
        px = pd.to_numeric(df[self.price_col], errors="coerce").astype(float)
        sig = signal.reindex(px.index)
        sig = self.exec.bound_signal(sig, allow_short=self.allow_short)
        pos  = self.exec.executed_position(sig)

        ret_asset = px.pct_change().fillna(0.0)
        turnover = (pos - pos.shift(1).fillna(0.0)).abs()
        cost_drag = self.cost.cost_rate(turnover)
        ret_gross = pos * ret_asset
        ret_net = ret_gross - cost_drag
        equity = (1.0 + ret_net).cumprod() * float(self.initial_equity)
        dd = equity / equity.cummax() - 1.0
        out = pd.DataFrame({
            "price": px,
            "signal_raw": sig.fillna(0.0),
            "position": pos.fillna(0.0),
            "turnover": turnover.fillna(0.0),
            "ret_asset": ret_asset,
            "ret_gross": ret_gross.fillna(0.0),
            "cost_rate": cost_drag.fillna(0.0),
            "ret_net": ret_net.fillna(0.0),
            "equity": equity,
            "drawdown": dd,
        })
        
        trade_log = pd.DataFrame()  # empty DataFrame; no trade log in vectorized path
        # trade_log = self._build_trade_log_from_positions(px, pos)
        self._trade_log = trade_log
        return out , trade_log 
    


    def _run_events_trade(self, df: pd.DataFrame, events: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Event-driven backtest with generic conflict modes and optional ExitPolicy.
        Expected columns in `events` (aligned to price index):
          - 'enter' (bool): entry trigger
          - 'side'  (int): +1 / -1 for intended entry
          - 'exit'  (bool, optional): explicit exit trigger (closes lots per exit_scope)
          - 'qty'   (float, optional): per-entry size (defaults to exec.lot_size)

        Behavior knobs from ExecutionPolicy:
          - entry_conflict: 'ignore' | 'extend' | 'pyramid' | 'flip'
          - fixed_hold_bars: scheduled exit at entry + H (for extend/pyramid)
          - exit_scope: 'all' or 'side' (how explicit exit closes)
          - lot_size, force_close_eod, trade_lag
          - apply_event_exit_policy + exit_policy: per-lot price-based exits (SL/TP/time/trailing)
        """
        if self.price_col not in df.columns:
            raise KeyError(f"price_col '{self.price_col}' not in dataframe")

        px = pd.to_numeric(df[self.price_col], errors="coerce").astype(float)
        idx = px.index
        ev = events.reindex(px.index).copy()
        if "side" not in ev:
            raise KeyError("events must include 'side' column (+1 long, -1 short on entry)")
        ev["side"] = ev["side"].fillna(0).astype(int)

        if "qty" in ev:
            ev["qty"] = pd.to_numeric(ev["qty"], errors="coerce").fillna(self.exec.lot_size).astype(float)
        else:
            ev["qty"] = float(self.exec.lot_size)

        entry_flag = self.exec.executed_position(ev["entry"])
        exit_flag = self.exec.executed_position(ev["exit"])
        entry_qty = self.exec.executed_position(ev["qty"])
        entry_side = self.exec.executed_position(ev["side"]).astype(int)

        # -- config --
        mode = (self.exec.entry_conflict or "ignore").lower()
        fixed_hold = self.exec.fixed_hold_bars
        exit_scope = (self.exec.exit_scope or "all").lower()   # "all" or "side"
        force_eod = bool(self.exec.force_close_eod)
        ep = self.exec.exit_policy if bool(self.exec.apply_event_exit_policy) else None

        # -- state --
        # Lots: each is a dict with keys:
        # side:int, qty:float, entry_i:int, entry_price:float, sched_exit_i:Optional[int], peak_price:float
        lots: List[Dict] = []
        trade_rows: List[Dict] = []

        # per-bar outputs
        n = len(idx)
        pos = np.zeros(n, dtype=float)
        flow = np.zeros(n, dtype=float)   # GROSS traded units per bar (entries + exits)

        # --- helpers ---
        def net_side_from_lots() -> int:
            if not lots: return 0
            s = sum(l["side"] * l["qty"] for l in lots)
            return int(np.sign(s))

        def close_lot(i_bar: int, reason: str, lot: Dict) -> None:
            """Close a single lot at bar i_bar."""
            exit_p = float(px.iat[i_bar])
            pnl = float(lot["side"]) * (exit_p / float(lot["entry_price"]) - 1.0) * float(lot["qty"])
            trade_rows.append({
                "entry_time": idx.take([int(lot["entry_i"])])[0],
                "entry_price": float(lot["entry_price"]),
                "exit_time": idx[i_bar],
                "exit_price": float(exit_p),
                "side": int(lot["side"]),
                "qty": float(lot["qty"]),
                "bars_held": int(i_bar - int(lot["entry_i"])),
                "pnl": float(pnl),
                "exit_reason": str(reason),
            })
            # gross flow: closing lot trades 'qty' units
            flow[i_bar] += abs(float(lot["qty"]))

        def schedule_exit_i(entry_i: int) -> Optional[int]:
            if fixed_hold is None: 
                return None
            return min(entry_i + int(fixed_hold), n - 1)
        

        # --- main loop ---
        for i, t in enumerate(idx):
            price = float(px.iat[i])

            # 0) Update peak prices for trailing stops (per-lot)
            for lot in lots:
                if lot["side"] == +1:
                    lot["peak_price"] = max(lot["peak_price"], price)
                else:
                    # for shorts, track "best" (lowest) price
                    lot["peak_price"] = min(lot["peak_price"], price)

            # 1) Scheduled exits (fixed_hold_bars)
            if lots and fixed_hold is not None:
                still = []
                for lot in lots:
                    se = lot.get("sched_exit_i")
                    if se is not None and i >= int(se):
                        close_lot(i, "scheduled_exit", lot)
                    else:
                        still.append(lot)
                lots = still

            # 2) ExitPolicy exits (per-lot, price-based)
            if lots and ep is not None:
                still = []
                for lot in lots:
                    bar_age = i - int(lot["entry_i"])
                    should_exit, reason = ep.check(
                        entry_price=float(lot["entry_price"]),
                        side=int(lot["side"]),
                        price_now=price,
                        bar_age=bar_age,
                        peak_price_since_entry=float(lot["peak_price"]),
                    )
                    if should_exit:
                        close_lot(i, reason or "exit_policy", lot)
                    else:
                        still.append(lot)
                lots = still

            # 3) Explicit exit event
            if bool(exit_flag.iat[i]) and lots:
                if exit_scope == "side":
                    net_side = net_side_from_lots()
                    close_mask = lambda l: int(l["side"]) == int(net_side)
                else:  # "all"
                    close_mask = lambda l: True

                # split candidates vs others
                candidates = [l for l in lots if close_mask(l)]
                others     = [l for l in lots if not close_mask(l)]

                # choose which candidates to close based on policy
                policy = (self.exec.exit_lot_policy or "all").lower()
                count  = self.exec.exit_lot_count  # None => close all candidates

                if not candidates:
                    # nothing to do
                    pass
                else:
                    if count is None or policy == "all":
                        to_close = candidates
                    else:
                        # order candidates
                        if policy in ("fifo", "first"):
                            candidates.sort(key=lambda l: int(l["entry_i"]))  # oldest first
                        elif policy in ("lifo", "last"):
                            candidates.sort(key=lambda l: int(l["entry_i"]), reverse=True)  # newest first
                        else:
                            # fallback to all if unknown policy
                            to_close = candidates
                            closed_ids = {id(x) for x in to_close}
                            lots = [l for l in lots if id(l) not in closed_ids] + []
                            # ensure we don't double-append below
                            continue
                        to_close = candidates[: int(count)]

                    # close selected lots
                    closed_ids = set()
                    for lot in to_close:
                        close_lot(i, "event_exit", lot)
                        closed_ids.add(id(lot))

                    # keep the rest
                    remaining_candidates = [l for l in candidates if id(l) not in closed_ids]
                    lots = others + remaining_candidates

            # 4) Entry event (conflict resolution)
            if bool(entry_flag.iat[i]) and int(entry_side.iat[i]) != 0:
                desired = int(entry_side.iat[i])
                qty = float(entry_qty.iat[i]) if not np.isnan(float(entry_qty.iat[i])) else float(self.exec.lot_size)

                if not self.allow_short and desired < 0:
                    desired = 0

                net_side = net_side_from_lots()

                if net_side == 0:
                    # open first lot
                    if desired != 0 and qty != 0.0:
                        lots.append({
                            "side": desired,
                            "qty": qty,
                            "entry_i": i,
                            "entry_price": price,
                            "sched_exit_i": schedule_exit_i(i),
                            "peak_price": price,  # init for trailing stop
                        })
                        flow[i] += abs(qty)  # opening adds gross flow

                elif net_side == desired:
                    if mode == "ignore":
                        pass
                    elif mode == "extend":
                        if fixed_hold is not None:
                            for lot in lots:
                                if int(lot["side"]) == desired:
                                    lot["sched_exit_i"] = schedule_exit_i(i)
                    elif mode == "pyramid":
                        if qty != 0.0:
                            lots.append({
                                "side": desired,
                                "qty": qty,
                                "entry_i": i,
                                "entry_price": price,
                                "sched_exit_i": schedule_exit_i(i),
                                "peak_price": price,
                            })
                            flow[i] += abs(qty)
                    elif mode == "flip":
                        # same-side + flip is a no-op by definition; keep consistent
                        pass

                elif net_side == -desired:
                    if mode == "flip":
                        # Close all current lots, then open one on desired side
                        if lots:
                            for lot in list(lots):
                                close_lot(i, "signal_flip", lot)
                            lots = []
                        if desired != 0 and qty != 0.0:
                            lots.append({
                                "side": desired,
                                "qty": qty,
                                "entry_i": i,
                                "entry_price": price,
                                "sched_exit_i": schedule_exit_i(i),
                                "peak_price": price,
                            })
                            flow[i] += abs(qty)
                    else:
                        # default: ignore the opposite signal while holding
                        pass

            # 5) Position snapshot (after all actions on bar)
            if lots:
                pos[i] = float(sum(l["side"] * l["qty"] for l in lots))
            else:
                pos[i] = 0.0

        # 6) Force close at end of data if requested
        if force_eod and lots:
            last_i = n - 1
            for lot in lots:
                close_lot(last_i, "eod_close", lot)
            lots = []

        # --- build trade log ---
        trade_log = pd.DataFrame(trade_rows, columns=[
            "entry_time","entry_price","exit_time","exit_price","side","qty","bars_held","pnl","exit_reason"
        ])
        if len(trade_log):
            trade_log = trade_log.astype({
                "entry_price":"float64","exit_price":"float64","side":"int64","qty":"float64",
                "bars_held":"int64","pnl":"float64","exit_reason":"object"
            })

        # --- ledger (trade accounting + costs) ---
        pos_s = pd.Series(pos, index=idx, name="position", dtype="float64")
        # GROSS turnover: sum of absolute entries+exits on each bar
        turnover = pd.Series(flow, index=idx, name="turnover", dtype="float64")
        cost_rate = self.cost.cost_rate(turnover).astype("float64")

        # Realized returns posted on exit bars (vectorized)
        ret_net = pd.Series(0.0, index=idx, dtype="float64")
        if len(trade_log):
            realized = trade_log["side"].astype(float) * (
                trade_log["exit_price"] / trade_log["entry_price"] - 1.0
            ) * trade_log.get("qty", 1.0)
            realized_by_bar = realized.groupby(trade_log["exit_time"]).sum()
            ret_net = ret_net.add(realized_by_bar.reindex(ret_net.index).fillna(0.0))

        # subtract costs
        ret_net = ret_net - cost_rate

        ret_asset = px.pct_change().fillna(0.0)  # informational only in trade accounting
        equity_additive = ret_net.cumsum() + float(self.initial_equity)
        equity_compound = (1.0 + ret_net).cumprod() * float(self.initial_equity)
        dd_additive = equity_additive / equity_additive.cummax() - 1.0
        dd_compound = equity_compound / equity_compound.cummax() - 1.0

        out = pd.DataFrame({
            "price": px.astype(float),
            "position": pos_s,
            "turnover": turnover,
            "ret_asset": ret_asset.astype(float),
            "cost_rate": cost_rate,
            "ret_net": ret_net,
            "equity_additive": equity_additive.astype(float),
            "drawdown_additive": dd_additive.astype(float),
            "equity_compound": equity_compound.astype(float),
            "drawdown_compound": dd_compound.astype(float),
        })
        return out, trade_log
    






    
# ============================================================
# smoke test
# ============================================================
'''
if __name__ == "__main__":
    idx = pd.date_range("2024-01-01", periods=200, freq="D")
    px = pd.Series(np.cumsum(np.random.randn(len(idx))) + 100, index=idx, name="close")
    df = pd.DataFrame({"close": px})

    # sample signal: long when price above its 10-day mean
    sig = (px > px.rolling(10).mean()).astype(float)

    port = Portfolio(
        initial_equity=1_000_000,
        price_col="close",
        cost=CostModel(1.0, 2.0),
        exec=ExecutionPolicy(trade_lag=1, bound=1.0, exit_policy=ExitPolicy(stop_loss_pct=0.05, take_profit_pct=0.1, max_hold_bars=20)),
    )

    ledger = port.run(df, sig)
    print(ledger.tail())
    print("trade log:", port.get_trade_log().head())
'''
