import numpy as np
import pandas as pd

def compute_return_series(series: pd.Series, dates: pd.Series, return_type: str, scale: float) -> pd.Series:
    """Compute scaled return series with an aligned date index."""
    values = series.to_numpy(dtype=float, copy=False)
    orig_idx = series.index.to_numpy()
    if values.size < 2:
        return pd.Series(dtype=float)
    prev = values[:-1]
    curr = values[1:]
    with np.errstate(divide="ignore", invalid="ignore"):
        if return_type == "simple":
            raw = np.where(prev != 0.0, curr / prev - 1.0, np.nan)
        else:
            raw = np.log(curr) - np.log(prev)
    valid = np.isfinite(raw)
    if not valid.any():
        return pd.Series(dtype=float)
    scaled = raw[valid] * scale
    aligned_positions = orig_idx[1:][valid]
    if len(aligned_positions):
        aligned_dates = dates.iloc[aligned_positions]
    else:
        aligned_dates = pd.Index([], dtype=object)
    return pd.Series(scaled, index=aligned_dates)
