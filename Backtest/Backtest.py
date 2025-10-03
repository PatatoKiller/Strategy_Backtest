from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, TYPE_CHECKING, TypedDict, List, Tuple, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# For type-checkers like Pylance; avoids runtime import cycles
if TYPE_CHECKING:
    from Portfolio import Portfolio
    from Strategy import Strategy

# ============================================================
# Backtester: orchestrates Strategy -> Portfolio, computes metrics,
# and provides plotting helpers. Metrics and plotting are separated
# into dedicated methods to keep things modular.
# ============================================================

class RunResult(TypedDict):
    '''
    Result of a backtest run.
    '''
    name: str
    summary: Dict[str, float]
    signals: pd.Series | pd.DataFrame
    ledger: pd.DataFrame
    trade_log: Optional[pd.DataFrame]
    ret_trade_desc: pd.Series

@dataclass
class Backtester:
    asset_name: str
    df: pd.DataFrame
    portfolio: "Portfolio"
    periods_per_year: int =252                      # set ~18144 for 5-minute bars
    days_per_year: float|int = 252.0                    # for CAGR calc from equity
    history: Optional[List[RunResult]] = None       # keep all run results for aggregation

    # ------ main run ------
    def run(self, strategy: "Strategy", name: Optional[str]=None) -> RunResult:
        """Run strategy through portfolio and return results dict.
        Returns dict: {"summary", "signals", "ledger", "trade_log", "ret_trade_desc"}
        where ret_trade_desc is `describe()` of per-bar returns when in a trade.
        """
        # 1) signals
        sig = strategy.signal(self.df)

        # 2) execute through portfolio
        ledger, trade_log = self.portfolio.run(self.df, sig)
        # trade_log = getattr(self.portfolio, "_trade_log", None)

        # 3) per-bar returns (force numeric for type-checkers & robustness)
        r = pd.to_numeric(ledger["ret_net"], errors="coerce").astype(float).fillna(0.0)
        in_trade = ledger["position"].fillna(0.0) != 0.0

        # 4) metrics
        summary = self._compute_metrics(ledger, trade_log, r, self.days_per_year)
        ret_trade_desc = trade_log['pnl'].describe() if trade_log is not None else pd.Series(dtype=float)

        result: RunResult ={
            "name": name or getattr(strategy, "__class__", type(strategy)).__name__,
            "summary": summary,
            "signals": sig,
            "ledger": ledger,
            "trade_log": trade_log,
            "ret_trade_desc": ret_trade_desc,
        }  
        # store in historyif self.history is None:
        if self.history is None:
            self.history = []
        self.history.append(result)
        return result # type: ignore[return-value]

    # ------ metrics ------
    def _compute_metrics(self, ledger: pd.DataFrame, trade_log: Optional[pd.DataFrame], r: pd.Series, days_per_year: int | float) -> Dict[str, float]:
        r = pd.to_numeric(r, errors="coerce").astype(float).fillna(0.0)
        n = int(len(r))
        if n == 0:
            return {k: np.nan for k in [
                "CAGR_add","CAGR_comp","AnnReturn","AnnVol","Sharpe","MaxDD_add","MaxDD_comp",
                "NumTrades","HitRate","AvgWin","AvgLoss","PnlRatio","MaxSingleLoss","AvgTurnover"
            ]}

        # annualized return, annual_vol, annaual sharpe over bar returns
        ann_ret = r.mean() * self.periods_per_year if n else np.nan
        ann_vol = r.std() * np.sqrt(self.periods_per_year) if n else np.nan
        sharpe = (ann_ret-0.03) / ann_vol if (ann_vol and ann_vol > 0) else np.nan

        # CAGR from equity
        eq_additive = ledger["equity_additive"].astype(float)
        eq_compound = ledger["equity_compound"].astype(float)
        # years = max(n / self.periods_per_year, 0.0)
        years = (ledger.index[-1] - ledger.index[0]) / pd.Timedelta('1 day') / days_per_year
        total_ret_additive = float(eq_additive.iloc[-1] / eq_additive.iloc[0] - 1.0) if len(eq_additive) > 1 else np.nan
        total_ret_compound = float(eq_compound.iloc[-1] / eq_compound.iloc[0] - 1.0) if len(eq_compound) > 1 else np.nan
        cagr_add = float(np.exp(np.log1p(total_ret_additive) / years) - 1.0) if years > 0  else np.nan
        cagr_comp = float(np.exp(np.log1p(total_ret_compound) / years) - 1.0) if years > 0 else np.nan

        maxdd_add = float(ledger["drawdown_additive"].min()) if "drawdown_additive" in ledger else np.nan
        maxdd_comp = float(ledger["drawdown_compound"].min()) if "drawdown_compound" in ledger else np.nan
        avg_to = float(ledger["turnover"].mean()) if "turnover" in ledger else np.nan

        # trade-level stats
        num_trades = hit = avg_win = avg_loss = pnl_ratio = max_single_loss = np.nan
        if trade_log is not None and len(trade_log):
            pnl = trade_log["pnl"].astype(float)
            num_trades = int(len(pnl))
            wins = pnl[pnl > 0]
            losses = pnl[pnl <= 0]
            hit = float(len(wins) / len(pnl)) if len(pnl) else np.nan
            avg_win = float(wins.mean()) if len(wins) else np.nan
            avg_loss = float(losses.mean()) if len(losses) else np.nan
            pnl_ratio = float(avg_win / abs(avg_loss)) if (avg_win == avg_win and avg_loss == avg_loss and avg_loss < 0) else np.nan
            max_single_loss = float(losses.min()) if len(losses) else np.nan

        return {
            "CAGR_add": float(cagr_add) if cagr_add == cagr_add else np.nan,
            "CAGR_comp": float(cagr_comp) if cagr_comp == cagr_comp else np.nan,
            "AnnReturn": float(ann_ret) if ann_ret == ann_ret else np.nan,
            "AnnVol": float(ann_vol) if ann_vol == ann_vol else np.nan,
            "Sharpe": float(sharpe) if sharpe == sharpe else np.nan,
            "MaxDD_add": float(maxdd_add) if maxdd_add == maxdd_add else np.nan,
            "MaxDD_comp": float(maxdd_comp) if maxdd_comp == maxdd_comp else np.nan,
            "AvgTurnover": float(avg_to) if avg_to == avg_to else np.nan,
            "NumTrades": int(num_trades) if num_trades == num_trades else 0,
            "WinRate": float(hit) if hit == hit else np.nan,
            "AvgWin": float(avg_win) if avg_win == avg_win else np.nan,
            "AvgLoss": float(avg_loss) if avg_loss == avg_loss else np.nan,
            "PnlRatio": float(pnl_ratio) if pnl_ratio == pnl_ratio else np.nan,
            "MaxSingleLoss": float(max_single_loss) if max_single_loss == max_single_loss else np.nan,
        }

    # ------ plotting helpers ------
    def plot_return_histogram(self, results: RunResult, bins: int = 50, abs_range: float = 0.2):
        """Histogram of per-bar strategy returns when in a trade."""
        ledger = results["ledger"]
        r = ledger["ret_net"].fillna(0.0)
        in_trade = ledger["ret_net"] != 0.0
        r_trade = r[in_trade]
        if len(r_trade) == 0:
            print("No in-trade returns to plot.")
            return
        plt.figure(figsize=(10,5))
        plt.hist(r_trade, bins=bins, range=(-abs_range,abs_range))
        plt.title("Strategy Return Histogram (in-trade bars) On " + self.asset_name)
        plt.xlabel("Return")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()

    def plot_return_timeseries(self, results: RunResult):
        """Time series of per-bar strategy returns when in a trade."""
        ledger = results["ledger"]
        r = ledger["ret_net"].fillna(0.0)
        
        if len(r) == 0:
            print("No in-trade returns to plot.")
            return
        plt.figure(figsize=(10,5))
        r.plot()
        plt.title("Strategy Returns Series (in-trade bars) On " + self.asset_name)
        plt.xlabel("Time")
        plt.ylabel("Return")
        plt.tight_layout()
        plt.show()

    def plot_equity_and_price(self, results: RunResult, price_col: Optional[str] = None, method: str = "compound"):
        """Plot compounded equity and asset price with entry/exit markers on twin axes.
        price_col: column in self.df to use as price; defaults to "close" if None.
        IMPORTANT: the price_col is not the portfolio's execution price, self.portfolio.price_col
        """
        ledger = results["ledger"]
        trade_log = results["trade_log"]
        df = self.df
        pcol = price_col  or "close"
        if pcol not in df.columns:
            raise KeyError(f"price_col '{pcol}' not in df")

        fig, ax1 = plt.subplots(figsize=(10,5))
        if method == "additive":
            ax1.plot(ledger.index, ledger["equity_additive"], 'g-', alpha=0.8, label='Strategy Net Value(Additive)')
            # ledger["equity_additive"].plot(ax=ax1, color='g-', alpha=0.6, label='Strategy Net Value(Additive)')
        elif method == "compound":
            ax1.plot(ledger.index, ledger["equity_compound"], 'g-', alpha=0.8, label='Strategy Net Value(Compound)')
            # ledger["equity_compound"].plot(ax=ax1, color='g-', alpha=0.6, label='Strategy Net Value(Compound)')
        else:
            raise ValueError("method must be 'additive' or 'compound'")
        ax1.set_ylabel("Net Value")
        ax1.set_title("Strategy Net Value & Price with Entries/Exits On " + self.asset_name)

        ax2 = ax1.twinx()
        ax2.plot(df.index, df[pcol], 'b-', alpha=0.8, label =self.asset_name+' Price')
        # df[pcol].plot(ax=ax2, color='b-', alpha=0.6, label =self.asset_name+' Price')
        ax2.set_ylabel("Price")

        # Entry/exit markers
        if trade_log is not None and len(trade_log):
            ax2.scatter(trade_log["entry_time"], trade_log["entry_price"], marker="^", s=70, color='green',
                   edgecolors='black', linewidths=0.5, label='Entry')
            ax2.scatter(trade_log["exit_time"], trade_log["exit_price"], marker="v", s=70, color='red',
                   edgecolors='black', linewidths=0.5, label='Exit')

        ax1.legend(loc='upper left')
        ax2.legend(loc='center left')

        plt.tight_layout()
        plt.show()


    def plot_drawdown_simple(self, results: RunResult, method: str = "compound"):
        ledger = results["ledger"]
        plt.figure(figsize=(10,5))
        if method == "additive":
            ledger["drawdown_additive"].plot(label="Drawdown (Additive)")
        elif method == "compound":
            ledger["drawdown_compound"].plot(label="Drawdown (Compound)")
        plt.title("Strategy Drawdown")
        plt.xlabel("Time")
        plt.ylabel("Drawdown")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_drawdown(self, results: RunResult, method: str = "compound"):
        ledger = results["ledger"]
        figsize = (10,5)
        if method == "additive":
            if "equity_additive" not in ledger.columns:
                raise KeyError("ledger missing 'equity_additive'")
            eq = ledger["equity_additive"].astype(float)
        elif method == "compound":
            if "equity_compound" not in ledger.columns:
                raise KeyError("ledger missing 'equity_compound'")
            eq = ledger["equity_compound"].astype(float)
        else:
            raise ValueError("method must be 'additive' or 'compound'")
        if len(eq) == 0:
            print("Nothing to plot: empty equity series.")
            return
        
        nav = eq / float(eq.iloc[0])  # normalized net equity curve starting at 1.0

        # ---- recompute drawdown from normalized equity ----
        roll_max = nav.cummax()
        dd = (nav / roll_max) - 1.0   # <= 0
        dd = dd.fillna(0.0)

        # ---- find max drawdown stats (peak, trough, recovery) ----
        def _max_dd_info(nav_series: pd.Series):
            cm = nav_series.cummax()
            dds = nav_series / cm - 1.0
            trough = dds.idxmin()
            if trough is None:
                return None
            pre = nav_series.loc[:trough]
            # last time we were at a running peak before trough
            peak = (pre[pre == pre.cummax()].index[-1]
                    if len(pre) else nav_series.index[0])
            peak_val = nav_series.loc[peak]
            post = nav_series.loc[trough:]
            recovery = post[post >= peak_val].index.min() if (post >= peak_val).any() else None
            return dict(peak=peak, trough=trough, recovery=recovery, maxdd=float(dds.min()))

        info = _max_dd_info(nav)
        # ---- convert to NumPy for plotting (appeases type checkers) ----
        x = nav.index.to_numpy()
        y_nav = nav.to_numpy(dtype=float)
        y_dd = dd.to_numpy(dtype=float)

        # ---- plot ----
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[2.2, 1.3], hspace=0.08)

        # Top: normalized equity
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(x, y_nav, linewidth=1.6)
        ax1.set_ylabel("Normalized Net Value")
        
        ax1.grid(True, which="both", alpha=0.25, linewidth=0.8)
        ax1.set_title("Strategy Drawdown On " + self.asset_name)

        # Mark peak/trough/recovery
        if info:
            ax1.scatter([info["peak"]], [nav.loc[info["peak"]]], s=28)
            ax1.scatter([info["trough"]], [nav.loc[info["trough"]]], s=28)
            if info["recovery"] is not None:
                ax1.scatter([info["recovery"]], [nav.loc[info["recovery"]]], s=28)

        # Bottom: underwater drawdown
        ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
        ax2.fill_between(x, y_dd, 0.0, alpha=0.8)
        ax2.set_ylim( (float(np.min(y_dd)) * 1.05), 0.0)
        ax2.set_ylabel("Drawdown "+method)
        ax2.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax2.grid(True, alpha=0.25, linewidth=0.8)
        ax2.set_xlabel("Time")

        if info:
            trough = info["trough"]
            maxdd = info["maxdd"]  # negative
            ax2.axvline(trough, linestyle="--", linewidth=1.0, alpha=1.0, color='red')
            peak, rec = info["peak"], info["recovery"]
            duration = (rec - peak) if (rec is not None) else None
            dur_txt = f", duration: {duration}" if duration is not None else ", duration: open"
            ax2.text(trough, dd.loc[trough], f"Max DD: {maxdd:.2%}{dur_txt}",
                        va="bottom", ha="left", fontsize=9)

        # Hide top x tick labels for cleanliness
        for lbl in ax1.get_xticklabels():
            lbl.set_visible(False)

        # fig.tight_layout()
        plt.show()




    # ============================================================
    # Aggregation helpers: for multiple strategy on the same asset
    # ============================================================

    def aggregate_results(self,
        sort_by: Optional[str] = "Sharpe",
        ascending: bool = False
    ) -> pd.DataFrame:
        """
        Collect summaries from multiple RunResult objects from history into a leaderboard DataFrame.

        Parameters
        ----------
        sort_by : str | None
            Metric column to sort by (e.g., "Sharpe", "CAGR", "AnnReturn"). If None, no sorting.
        ascending : bool
            Sort order.

        Returns
        -------
        pd.DataFrame
            Index = strategy name; Columns = metrics from `summary`.
        """
        if not self.history:
            raise ValueError("No run results in history to aggregate.")
        rows: List[Dict[str, float|str]] = []
        for res in self.history:
            row: Dict[str, float | str] = {"name": res["name"]}
            row.update(res["summary"])
            rows.append(row)

        df = pd.DataFrame(rows).set_index("name")
        if sort_by and sort_by in df.columns:
            df = df.sort_values(sort_by, ascending=ascending)
        return df
    ## ============================================================
    # Example
    ## ============================================================
    # bt = Backtester(asset_name=asset_name, df=df_btc, portfolio=port_btc, periods_per_year=252)
    #  --- run multiple strategies on the same asset ---
    # res1 = bt.run(SMACross(fast=20, slow=100, price_col="close"), name="SMA 20/100")
    # res2 = bt.run(SMACross(fast=50, slow=200, price_col="close"), name="SMA 50/200")
    # res3 = bt.run(EMACross(fast=12, slow=26,  price_col="close"), name="EMA 12/26")
    #  --- aggregate & sort by Sharpe (desc) ---
    # leaderboard = aggregate_results([res1, res2, res3], sort_by="Sharpe", ascending=False)
    # print(leaderboard)



    # ============================================================
    # Aggregation helpers: for same strategy on multiple assets
    # ============================================================
    @staticmethod
    def run_strategy_on_assets(
        strategy: "Strategy",
        assets: Dict[str, Tuple[pd.DataFrame, "Portfolio"]],
        *,
        periods_per_year: int = 252,
        name: Optional[str] = None,
        ) -> Tuple[pd.DataFrame, Dict[str, RunResult]]:
        """Run the same strategy across multiple assets and aggregate metrics.


        Parameters
        ----------
        strategy : Strategy
        The signal generator to test on all assets.
        assets : dict[str, (df, portfolio)]
        Mapping from asset_name -> (DataFrame, Portfolio). You can reuse the same
        Portfolio instance if execution settings are identical; otherwise supply
        per-asset portfolio objects.
        periods_per_year : int Annualization factor (use ~18144 for 5-min bars).
        name : str | None Optional display name for the strategy in the results.

        Returns
        -------
        leaderboard : pd.DataFrame
        Rows = assets, columns = metrics from RunResult["summary"].
        results_by_asset : dict[str, RunResult]
        Full results (ledger, trade_log, etc.) keyed by asset.
        """
        rows: List[Dict[str, float]] = []
        results_by_asset: Dict[str, RunResult] = {}


        for asset, (df, port) in assets.items():
            bt = Backtester(asset_name=asset, df=df, portfolio=port, periods_per_year=periods_per_year)
            res = bt.run(strategy, name=name)
            results_by_asset[asset] = res


            row: Dict[str, Any] = {"asset": asset}
            row.update(res["summary"]) # type: ignore[arg-type]
            rows.append(row)


        leaderboard = pd.DataFrame(rows).set_index("asset") if rows else pd.DataFrame()
        return leaderboard, results_by_asset
    ## ============================================================
    # Example 
    ## ============================================================
    # assets = {
    #     "BTCUSDT": (df_btc, port_btc),
    #     "ETHUSDT": (df_eth, port_eth),
    #     "AAPL":    (df_aapl, port_aapl),
    # }

    # strat = SMACross(fast=50, slow=200, price_col="close")

    # board, res_by_asset = run_strategy_on_assets(
    #     strategy=strat,
    #     assets=assets,
    #     periods_per_year=252,
    #     name="SMA(50/200)"
    # )

    # print(board.sort_values("Sharpe", ascending=False))
