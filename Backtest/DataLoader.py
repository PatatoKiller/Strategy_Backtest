import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Union, Tuple, List
from pandas.tseries.frequencies import to_offset

@dataclass
class DataLoader:
    """
    Simple, flexible data loader for backtesting.
    - Loads CSV or Parquet
    - Ensures DateTimeIndex
    - Drops duplicate timestamps (keeps first appearance)
    - Optionally checks that time gaps equal an expected frequency
    - Allows selecting an arbitrary list of columns to keep, not only OHLCV
    """
    filepath: str # path to CSV or Parquet file
    date_col: str = "date"  # name of the date column in the source file
    cols_to_keep: Optional[List[str]] = None  # list of columns to keep after loading; keep all if None
    tz: Optional[str] = None  # e.g., 'UTC'
    unit: Optional[str] = None  # e.g., 'ms' if date_col is in milliseconds since epoch
    rename_map: Optional[Dict[str, str]] = None  # map input column names to new names
    expected_freq: Optional[str] = None  # e.g., '5T', '1H'; if provided, validate grid

    def load(self) -> pd.DataFrame:
        """Load and clean data into a DataFrame with selected columns.

        Returns
        -------
        DataFrame
            Indexed by datetime (tz-aware if tz provided) and containing only
            the requested columns (or all columns if cols_to_keep is None).
            Adds attrs with validation info when expected_freq is set:
              - df.attrs['expected_freq']
              - df.attrs['n_duplicate_index_dropped']
              - df.attrs['n_bad_intervals']
        """
        # 1) Read file
        if str(self.filepath).endswith((".parquet", ".pq")):
            df = pd.read_parquet(self.filepath)
        else:
            df = pd.read_csv(self.filepath)

        # 2) Optional rename
        if self.rename_map:
            df = df.rename(columns=self.rename_map)

        # 3) Parse datetime and set index
        dt_series = pd.to_datetime(df[self.date_col], errors='coerce', unit= self.unit)
        if self.tz:
            if getattr(dt_series.dt, 'tz', None) is not None:
                dt_series = dt_series.dt.tz_convert(self.tz)
            else:
                dt_series = dt_series.dt.tz_localize(self.tz)
        df[self.date_col] = dt_series
        df = df.set_index(self.date_col, drop=True)

        # 4) Sort and drop duplicate timestamps (keep first appearance)
        df = df.sort_index()
        before = len(df)
        df = df[~df.index.duplicated(keep='first')]
        n_dups = before - len(df)

        # 5) Keep only specified columns if provided
        if self.cols_to_keep is not None:
            missing = [c for c in self.cols_to_keep if c not in df.columns]
            if missing:
                raise ValueError(f"Missing required columns: {missing}")
            df = df[self.cols_to_keep].copy()

        # 6) (Optional) Validate time grid
        n_bad = 0
        if self.expected_freq:
            off = to_offset(self.expected_freq)
            # Only allow fixed ticks (e.g., '5T', '1H', '1D'); calendar-based like 'M' vary by month.
            if not hasattr(off, "nanos"):
                raise ValueError(
                    "expected_freq must be a fixed tick (e.g., '5T', '1H', '1D'), not calendar-based like 'M' or 'BM'."
                )
            expected_td = pd.to_timedelta(off.nanos, unit="ns")
            diffs = df.index.to_series().diff()  # first diff is NaT
            n_bad = int((diffs != expected_td).sum())
            # only keep the good rows 
            mask  = diffs.isna() | (diffs == expected_td) 
            df = df.loc[mask]
            df = df.sort_index()


        # 7) Attach validation attrs
        df.attrs['expected_freq'] = self.expected_freq
        df.attrs['n_duplicate_index_dropped'] = int(n_dups)
        df.attrs['n_bad_intervals'] = int(n_bad)

        return df

    # Convenience method if you want a quick validation report without raising
    def validate(self, df: pd.DataFrame) -> Tuple[int, int]:
        """Return (n_duplicate_index_dropped, n_bad_intervals) from attrs or recompute if missing."""
        n_dups = df.attrs.get('n_duplicate_index_dropped')
        n_bad = df.attrs.get('n_bad_intervals')
        if n_dups is None or n_bad is None:
            # recompute
            before = len(df)
            n_dups = before - len(df[~df.index.duplicated(keep='first')])
            n_bad = 0
            if self.expected_freq:
                off = to_offset(self.expected_freq)
                # Only allow fixed ticks (e.g., '5T', '1H', '1D'); calendar-based like 'M' vary by month.
                if not hasattr(off, "nanos"):
                    raise ValueError(
                        "expected_freq must be a fixed tick (e.g., '5T', '1H', '1D'), not calendar-based like 'M' or 'BM'."
                    )
                expected_td = pd.to_timedelta(off.nanos, unit="ns")
                diffs = df.index.to_series().diff().iloc[1:]  # first diff is NaT
                n_bad = int((diffs != expected_td).sum())

        return int(n_dups or 0), int(n_bad or 0)

'''
if __name__ == "__main__":
    # example usage
    loader = DataLoader(
        filepath="data.csv",
        date_col="timestamp",
        cols_to_keep=["open", "high", "low", "close", "volume", "feature1", "feature2"],
        tz="UTC",
        rename_map={"Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"},
        expected_freq='5T',
        strict_timegrid=False,
    )
    df = loader.load()
    print(df.head())
    print({k: df.attrs[k] for k in ['expected_freq','n_duplicate_index_dropped','n_bad_intervals']})
'''