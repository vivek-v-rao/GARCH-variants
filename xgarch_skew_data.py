import argparse
import math
import sys
import time
import csv
import atexit
from collections import defaultdict
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from garch_util import (ModelRow, nagarch_cond_sd, igarch_cond_sd, fit_arch_model,
    st_cond_sd, fit_nagarch, fit_igarch, fit_constant_vol, fit_st,
    fit_nagarch_model, fit_igarch_model, fit_st_model,
    compute_loglik_select_values, compute_metric_order)
from stats_util import (summary_stats,
    print_summary_table, print_autocorr_table, compute_aicc, compute_bic,
    sorted_indices, compute_ranks, print_selects, residual_summary_table,
    cond_sd_summary_table)
from returns_util import compute_return_series
import minimize

K_PI = math.pi
OBS_PER_YEAR = 252

@dataclass
class CriteriaSummary:
    aicc_order: List[str] = field(default_factory=list)
    bic_order: List[str] = field(default_factory=list)

@dataclass
class ColumnSummary:
    name: str
    criteria: CriteriaSummary

def print_model_table(rows: List[ModelRow], aicc_vals: List[float], bic_vals: List[float],
                      loglik_ranks: List[int], aicc_ranks: List[int], bic_ranks: List[int], precision: int = 6) -> None:
    """Print the fitted-model parameter table with ranks."""
    headers = ["uncond_sd", "mu", "omega", "alpha", "beta", "gamma", "shift", "skew", "dof",
               "loglik", "n_params", "AICC", "BIC", "loglik_rank", "AICC_rank", "BIC_rank"]
    print(f"{'model':<25}" + "".join(f"{h:>20}" for h in headers))
    for idx, row in enumerate(rows):
        values = [row.uncond_sd, row.mu, row.omega, row.alpha, row.beta, row.gamma, row.shift, row.skew, row.dof,
                  row.loglik]
        formatted = []
        for val in values:
            if math.isfinite(val):
                formatted.append(f"{val:20.{precision}f}")
            else:
                formatted.append(f"{'NA':>20}")
        n_params_str = f"{row.n_params:20d}"
        aicc_val = aicc_vals[idx]
        bic_val = bic_vals[idx]
        formatted_aicc = f"{aicc_val:20.{precision}f}" if math.isfinite(aicc_val) else f"{'NA':>20}"
        formatted_bic = f"{bic_val:20.{precision}f}" if math.isfinite(bic_val) else f"{'NA':>20}"
        print(f"{row.label:<25}" + "".join(formatted) + n_params_str + formatted_aicc + formatted_bic +
              f"{loglik_ranks[idx]:20d}{aicc_ranks[idx]:20d}{bic_ranks[idx]:20d}")

        # Below, the models not from the arch package are labeled
        # "custom". They may be slower to estimate.
ALLOWED_MODELS = {
    "nagarch_normal", # custom
    "nagarch_student_t", # custom
    "nagarch_skew_student_t", # custom
    "garch_normal",
    "garch_student_t",
    "garch_skew_student_t",
    "gjr_normal",
    "gjr_student_t",
    "gjr_skew_student_t",
    "egarch_normal",
    "egarch_student_t",
    "egarch_skew_student_t",
    "igarch_normal", # custom
    "igarch_student_t", # custom
    "igarch_skew_student_t", # custom
    "st_normal", # custom
    "st_student_t", # custom
    "st_skew_student_t", # custom
    "constant_vol", # custom
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the script."""
    parser = argparse.ArgumentParser(description="Fit GARCH-family models using arch")
    parser.add_argument("--file", default="prices.csv")
    parser.add_argument("--columns", default="")
    parser.add_argument(
        "--models",
        default=(
#            "nagarch_normal,nagarch_student_t,nagarch_skew_student_t,garch_student_t,garch_skew_student_t,garch_normal,"
            "nagarch_normal,nagarch_student_t,nagarch_skew_student_t,garch_student_t,garch_skew_student_t,garch_normal,"
            "gjr_student_t,gjr_skew_student_t,gjr_normal,egarch_student_t,egarch_skew_student_t,egarch_normal,"
#            "igarch_student_t,igarch_skew_student_t,igarch_normal,st_student_t,st_skew_student_t,st_normal,constant_vol"
            "igarch_student_t,igarch_normal,st_student_t,st_normal,constant_vol"
        ),
    )
    parser.add_argument("--max-columns", type=int, default=-1)
    parser.add_argument("--min-rows", type=int, default=250)
    parser.add_argument("--scale", type=float, default=100.0)
    parser.add_argument("--returns", choices=("log", "simple"), default="log",
                          help="return calculation: log (default) or simple percentage change")
    parser.add_argument("--no-demean", action="store_true")
    parser.add_argument("--no-resid-stats", action="store_true")
    parser.add_argument("--cond-sd-stats", action="store_true", default=True)
    parser.add_argument("--sd-corr", action="store_true", default=True)
    parser.add_argument("--no-sd-corr", action="store_false", dest="sd_corr")
    parser.add_argument("--sd-corr-file", default="sd_corr.csv")
    parser.add_argument("--return-summary", action="store_true", dest="return_summary", default=True)
    parser.add_argument("--no-return-summary", action="store_false", dest="return_summary")
    parser.add_argument("--return-corr", action="store_true", dest="return_corr", default=True)
    parser.add_argument("--no-return-corr", action="store_false", dest="return_corr")
    parser.add_argument("--autocorr-lags", type=int, default=5)
    parser.add_argument("--vols-csv", action="store_true", dest="vols_csv", default=True)
    parser.add_argument("--no-vols-csv", action="store_false", dest="vols_csv")
    parser.add_argument("--vols-dir", default="")
    parser.add_argument("--param-csv", default="model_params.csv",
                        help="Path for aggregated model parameter CSV")
    parser.add_argument("--no-param-csv", action="store_true",
                        help="Disable writing the parameter summary CSV")
    return parser.parse_args()


def main() -> int:
    """Coordinate data loading, model fitting, and reporting."""
    minimize.maxiter_max = 100 # reducing this boosts speed but may worsen model fits
    minimize.fatol = 1e-2 # Function Value Absolute Tolerance -- increase this to boost speed
    param_headers = ["symbol", "model", "uncond_sd", "mu", "omega", "alpha", "beta", "gamma", "shift",
                     "skew", "dof", "loglik", "n_params", "AICC", "BIC", "loglik_rank", "AICC_rank",
                     "BIC_rank"]
    param_writer = None
    param_csv_file = None
    param_csv_path: Optional[Path] = None
    param_rows_written = False
    def fmt_csv_value(value: float, precision: int = 6) -> str:
        return f"{value:.{precision}f}" if math.isfinite(value) else "NA"
    args = parse_args()
    print_each_fit_time = False
    if getattr(args, 'no_cond_sd_stats', False):
        args.cond_sd_stats = False
    sd_corr_path = Path(args.sd_corr_file) if getattr(args, 'sd_corr', True) else None
    if sd_corr_path and sd_corr_path.exists():
        sd_corr_path.unlink()
    elif not getattr(args, 'sd_corr', True):
        args.sd_corr = False
    if args.scale <= 0:
        print("--scale must be positive", file=sys.stderr)
        return 1
    args.returns = args.returns.lower()
    if args.returns not in ("log", "simple"):
        print("--returns must be log or simple", file=sys.stderr)
        return 1
    start_time = time.perf_counter()
    timing_records = []
    model_metrics = defaultdict(lambda: {'aicc': [], 'bic': []})
    try:
        raw = pd.read_csv(args.file)
    except Exception as exc:
        print(f"failed to read {args.file}: {exc}", file=sys.stderr)
        return 1
    if raw.shape[1] < 2:
        print("expecting at least one price column", file=sys.stderr)
        return 1
    date_col = pd.to_datetime(raw.iloc[:, 0], errors="coerce")
    prices = raw.iloc[:, 1:]
    prices.columns = [col.strip() for col in prices.columns]
    n_rows, n_cols = prices.shape
    print("max Nelder-Mead iterations, function tolerance:", minimize.maxiter_max, minimize.fatol)
    print(f"loaded price data from {args.file} with {n_rows} rows and {n_cols} columns")
    if date_col.notna().any():
        first = date_col.dropna().iloc[0]
        last = date_col.dropna().iloc[-1]
        first_str = first.strftime("%Y-%m-%d") if isinstance(first, pd.Timestamp) else str(first)
        last_str = last.strftime("%Y-%m-%d") if isinstance(last, pd.Timestamp) else str(last)
        print(f"date range: {first_str} to {last_str}")
    print(f"return scaling factor: {args.scale}")
    print(f"demean returns: {'no' if args.no_demean else 'yes'}")
    print(f"return type: {args.returns}")
    requested = [name.strip() for name in args.models.split(',') if name.strip()]
    for name in requested:
        if name not in ALLOWED_MODELS:
            print(f"unknown model: {name}", file=sys.stderr)
            return 1
    columns = prices.columns.tolist()
    if args.columns.strip():
        subset = []
        for name in args.columns.split(','):
            name = name.strip()
            if name:
                if name not in columns:
                    print(f"unknown column {name}", file=sys.stderr)
                    return 1
                if name not in subset:
                    subset.append(name)
        columns = subset

    if (not args.no_param_csv) and args.param_csv:
        param_csv_path = Path(args.param_csv).expanduser()
        try:
            param_csv_path.parent.mkdir(parents=True, exist_ok=True)
            param_csv_file = param_csv_path.open('w', newline='', encoding='utf-8')
        except OSError as exc:
            print(f"failed to open parameter CSV {args.param_csv}: {exc}", file=sys.stderr)
            return 1
        else:
            atexit.register(param_csv_file.close)
            param_writer = csv.writer(param_csv_file)
            param_writer.writerow(param_headers)

    if args.max_columns > 0 and len(columns) > args.max_columns:
        columns = columns[: args.max_columns]
        print(f"limiting to first {args.max_columns} column(s)")
    return_series_map: Dict[str, pd.Series] = {}
    for column in columns:
        series = prices[column].dropna()
        if series.empty:
            return_series_map[column] = pd.Series(dtype=float)
            continue
        return_series_map[column] = compute_return_series(series, date_col, args.returns, args.scale)

    has_return_data = any(not ser.empty for ser in return_series_map.values())
    if args.return_summary and has_return_data:
        print(f"\n==== return summary (annualized, obs_year = {OBS_PER_YEAR}) ====")
        header = f"{'symbol':<12}{'ann_return':>15}{'ann_vol':>15}{'skew':>12}{'kurtosis':>12}{'min':>12}{'max':>12}"
        print(header)
        def fmt_val(value: float, width: int) -> str:
            return f"{value:>{width}.4f}" if math.isfinite(value) else f"{'NA':>{width}}"
        for column in columns:
            series = return_series_map.get(column, pd.Series(dtype=float))
            values = series.to_numpy(dtype=float) if series is not None else np.array([], dtype=float)
            if values.size:
                mean_val = float(np.mean(values))
                ann_return = mean_val * OBS_PER_YEAR
                std_val = float(np.std(values, ddof=0))
                ann_vol = std_val * math.sqrt(OBS_PER_YEAR) if values.size > 1 else math.nan
                centered = values - mean_val
                m2 = float(np.mean(centered ** 2))
                if m2 > 0.0:
                    m3 = float(np.mean(centered ** 3))
                    m4 = float(np.mean(centered ** 4))
                    skew = m3 / (m2 ** 1.5)
                    kurt = m4 / (m2 ** 2)
                else:
                    skew = math.nan
                    kurt = math.nan
                min_val = float(np.min(values))
                max_val = float(np.max(values))
            else:
                ann_return = ann_vol = skew = kurt = min_val = max_val = math.nan
            row = f"{column:<12}{fmt_val(ann_return, 15)}{fmt_val(ann_vol, 15)}{fmt_val(skew, 12)}{fmt_val(kurt, 12)}{fmt_val(min_val, 12)}{fmt_val(max_val, 12)}"
            print(row)

    if args.return_corr and sum(1 for ser in return_series_map.values() if not ser.empty) >= 2:
        series_list = [ser.rename(col) for col, ser in return_series_map.items() if not ser.empty]
        returns_df = pd.concat(series_list, axis=1, join="outer") if series_list else pd.DataFrame()
        if not returns_df.empty and returns_df.shape[1] >= 2:
            corr = returns_df.corr()
            if corr.shape[1] >= 2:
                print("\ncorrelations")
                width = max(12, max(len(col) for col in corr.columns))
                header = f"{'':<{width}}" + ''.join(f"{col:>{width}}" for col in corr.columns)
                print(header)
                for name in corr.index:
                    row = f"{name:<{width}}"
                    for val in corr.loc[name].to_numpy():
                        row += f"{val:>{width}.3f}" if math.isfinite(val) else f"{'NA':>{width}}"
                    print(row)
    summaries: List[ColumnSummary] = []
    models_fit_counts: List[int] = []
    wrote_sd_corr = False
    vols_paths: List[str] = []
    vols_dir_path: Optional[Path] = Path(args.vols_dir).expanduser() if getattr(args, "vols_dir", "") else None
    if args.vols_csv and vols_dir_path:
        vols_dir_path.mkdir(parents=True, exist_ok=True)
    for column in columns:
        series = prices[column].dropna()
        values = series.to_numpy(dtype=float, copy=False)
        price_obs = values.size
        if values.size > 1:
            if args.returns == "simple":
                prev = values[:-1]
                curr = values[1:]
                with np.errstate(divide="ignore", invalid="ignore"):
                    returns = np.where(prev != 0.0, curr / prev - 1.0, np.nan)
            else:
                returns = np.diff(np.log(values))
        else:
            returns = np.array([], dtype=float)
        returns = returns * args.scale
        series_dates = date_col.iloc[series.index]
        aligned_prices = values[1:] if values.size > 1 else np.array([])
        aligned_date_vals = series_dates.iloc[1:] if series_dates.size > 1 else series_dates.iloc[0:0]
        aligned_date_strings: List[str] = []
        for val in aligned_date_vals:
            if pd.isna(val):
                aligned_date_strings.append("")
            elif isinstance(val, pd.Timestamp):
                aligned_date_strings.append(val.strftime("%Y-%m-%d"))
            else:
                aligned_date_strings.append(str(val))
        stats = summary_stats(returns)
        print(f"\n==== symbol: {column} ====")
        print(f"price observations: {price_obs}, {args.returns} returns used: {returns.size}")
        print_summary_table(stats)
        print()
        print_autocorr_table(returns, args.autocorr_lags)
        adjusted_returns = returns if args.no_demean else returns - returns.mean()
        if adjusted_returns.size < args.min_rows:
            print(f"not enough data (need {args.min_rows})")
            continue
        model_rows: List[ModelRow] = []
        residual_entries: List[Tuple[str, np.ndarray]] = []
        cond_sd_entries: List[Tuple[str, np.ndarray]] = []
        for model in requested:
            start_fit = time.perf_counter()
            if print_each_fit_time:
                print("model, time =", model, start_fit) # debug
            try:
                dist = "normal"
                base = model
                if model.endswith("_skew_student_t"):
                    base = model[: -len("_skew_student_t")]
                    dist = "skew-student"
                elif model.endswith("_student_t"):
                    base = model[: -len("_student_t")]
                    dist = "student-t"
                elif model.endswith("_normal"):
                    base = model[: -len("_normal")]
                if base == "constant_vol":
                    row = fit_constant_vol(adjusted_returns)
                elif base == "egarch":
                    row = fit_arch_model(adjusted_returns, "egarch", dist)
                elif base == "garch":
                    row = fit_arch_model(adjusted_returns, "garch", dist)
                elif base == "gjr":
                    row = fit_arch_model(adjusted_returns, "gjr", dist)
                elif base == "igarch":
                    row = fit_igarch_model(adjusted_returns, dist)
                elif base == "nagarch":
                    row = fit_nagarch_model(adjusted_returns, dist)
                elif base == "st":
                    row = fit_st_model(adjusted_returns, dist)
                else:
                    continue
            except Exception as exc:
                print(f"failed to fit {model}: {exc}", file=sys.stderr)
                sys.exit(1)
            elapsed = time.perf_counter() - start_fit
            timing_records.append({"symbol": column, "model": row.label, "seconds": elapsed, "n_params": row.n_params})
            model_rows.append(row)
            residual_entries.append((row.label, row.std_resid))
            if row.cond_sd is not None:
                cond_sd_entries.append((row.label, row.cond_sd))

        if not model_rows:
            print("no models fitted")
            continue
        n_obs = adjusted_returns.size
        aicc_vals = [compute_aicc(row.loglik, row.n_params, n_obs) for row in model_rows]
        bic_vals = [compute_bic(row.loglik, row.n_params, n_obs) for row in model_rows]
        for row, aicc_val, bic_val in zip(model_rows, aicc_vals, bic_vals):
            model_metrics[row.label]['aicc'].append(aicc_val)
            model_metrics[row.label]['bic'].append(bic_val)
        loglik_vals = [row.loglik for row in model_rows]
        loglik_ranks = compute_ranks(loglik_vals, ascending=False)
        aicc_ranks = compute_ranks(aicc_vals)
        bic_ranks = compute_ranks(bic_vals)
        print()
        print_model_table(model_rows, aicc_vals, bic_vals, loglik_ranks, aicc_ranks, bic_ranks)
        if param_writer:
            for idx, row in enumerate(model_rows):
                csv_row = [
                    column,
                    row.label,
                    fmt_csv_value(row.uncond_sd),
                    fmt_csv_value(row.mu),
                    fmt_csv_value(row.omega),
                    fmt_csv_value(row.alpha),
                    fmt_csv_value(row.beta),
                    fmt_csv_value(row.gamma),
                    fmt_csv_value(row.shift),
                    fmt_csv_value(row.skew),
                    fmt_csv_value(row.dof),
                    fmt_csv_value(row.loglik),
                    str(row.n_params),
                    fmt_csv_value(aicc_vals[idx]),
                    fmt_csv_value(bic_vals[idx]),
                    str(loglik_ranks[idx]),
                    str(aicc_ranks[idx]),
                    str(bic_ranks[idx]),
                ]
                param_writer.writerow(csv_row)
            param_rows_written = True
        names = [row.label for row in model_rows]
        loglik_select_vals = compute_loglik_select_values(model_rows)
        print_selects("loglik", names, loglik_select_vals)
        print_selects("AICC", names, aicc_vals)
        print_selects("BIC", names, bic_vals)
        if not args.no_resid_stats:
            residual_summary_table(residual_entries)
        if args.cond_sd_stats and cond_sd_entries:
            cond_sd_summary_table(cond_sd_entries)
            if args.sd_corr:
                named_entries = [(name, values) for name, values in cond_sd_entries
                                  if name != "constant_vol" and values is not None]
                if len(named_entries) >= 2:
                    lengths = [len(vals) for _, vals in named_entries if len(vals) > 0]
                    if lengths:
                        min_len = min(lengths)
                        if min_len > 0:
                            aligned = np.column_stack([vals[:min_len] for _, vals in named_entries])
                            corr = np.corrcoef(aligned, rowvar=False)
                            sd_corr_path = Path(args.sd_corr_file)
                            with sd_corr_path.open('a', encoding='utf-8') as fh:
                                fh.write(f"symbol,{column}\n")
                                headers = ','.join(['model'] + [name for name, _ in named_entries])
                                fh.write(headers + "\n")
                                for (name, _), row in zip(named_entries, corr):
                                    row_str = ','.join(f"{val:.6f}" for val in row)
                                    fh.write(f"{name},{row_str}\n")
                                fh.write("\n")
                            wrote_sd_corr = True
        if args.vols_csv and cond_sd_entries:
            lengths = [adjusted_returns.size, aligned_prices.shape[0], len(aligned_date_strings)]
            lengths.extend(len(vals) for _, vals in cond_sd_entries)
            min_len = min(lengths) if lengths and all(lengths) else 0
            if min_len > 0:
                return_key = "simple_return" if args.returns == "simple" else "log_return"
                data = {
                    "date": aligned_date_strings[:min_len],
                    "price": aligned_prices[:min_len],
                    return_key: adjusted_returns[:min_len],
                }
                for name, values in cond_sd_entries:
                    data[name] = values[:min_len]
                out_df = pd.DataFrame(data)
                out_name = f"{column.lower()}_vols.csv"
                out_path = (vols_dir_path / out_name) if vols_dir_path else Path(out_name)
                try:
                    out_df.to_csv(out_path, index=False, float_format="%.6f")
                except OSError as exc:
                    print(f"failed to write volatility series to {out_path}: {exc}", file=sys.stderr)
                else:
                    vols_paths.append(str(out_path))
        summaries.append(ColumnSummary(name=column,
                                       criteria=CriteriaSummary(
                                           aicc_order=compute_metric_order(model_rows, aicc_vals),
                                           bic_order=compute_metric_order(model_rows, bic_vals))))
        models_fit_counts.append(len(model_rows))
    print("\n==== model ranks by asset and information criterion ====")
    if not summaries:
        print("no fitted models to summarize")
    else:
        name_width = max(len(entry.name) for entry in summaries)
        model_names = [model
                       for entry in summaries
                       for model in (entry.criteria.aicc_order + entry.criteria.bic_order)]
        col_width = max((len(model) for model in model_names), default=0) + 2
        col_width = max(col_width, 18)
        def format_row(models):
            if not models:
                return "(none)"
            return ' '.join(f"{model:<{col_width}}" for model in models)
        for entry in summaries:
            prefix_aicc = f"{entry.name:<{name_width}} AICC: "
            prefix_bic = f"{entry.name:<{name_width}}  BIC: "
            print(prefix_aicc + format_row(entry.criteria.aicc_order))
            print(prefix_bic + format_row(entry.criteria.bic_order))

    def tally_orders(entries, key):
        counts = {}
        max_rank = 0
        for entry in entries:
            order = getattr(entry.criteria, key)
            if not order:
                continue
            if len(order) > max_rank:
                max_rank = len(order)
            for idx, name in enumerate(order):
                vec = counts.setdefault(name, [])
                if len(vec) <= idx:
                    vec.extend([0] * (idx + 1 - len(vec)))
                vec[idx] += 1
        for vec in counts.values():
            if len(vec) < max_rank:
                vec.extend([0] * (max_rank - len(vec)))
        return counts, max_rank


    def print_tally(label, counts, max_rank, averages):
        print(f"\n==== {label} rank counts ====")
        if not counts or max_rank == 0:
            print("no data")
            return
        def sort_key(item):
            avg_val = averages.get(item[0], float('nan'))
            if not math.isfinite(avg_val):
                avg_val = float('inf')
            return (avg_val, tuple([-c for c in item[1][:max_rank]]), item[0])
        rows = sorted(counts.items(), key=sort_key)
        col_width = 10
        header = f"{'model':<25}{(label + '_avg'):>15}"
        for rank in range(1, max_rank + 1):
            header += f" {('#' + str(rank)).rjust(col_width)}"
        print(header)
        for name, vec in rows:
            avg_val = averages.get(name, float('nan'))
            avg_text = f"{avg_val:15.3f}" if math.isfinite(avg_val) else f"{'NA':>15}"
            line = f"{name:<25}{avg_text}"
            for idx in range(max_rank):
                line += f" {vec[idx]:>{col_width}d}"
            print(line)

    aicc_counts, aicc_max = tally_orders(summaries, 'aicc_order')
    bic_counts, bic_max = tally_orders(summaries, 'bic_order')
    def compute_avg(metric_key):
        averages = {}
        for model, metrics in model_metrics.items():
            vals = [v for v in metrics[metric_key] if math.isfinite(v)]
            averages[model] = float(np.mean(vals)) if vals else float('nan')
        return averages
    avg_aicc = compute_avg('aicc')
    avg_bic = compute_avg('bic')
    print_tally('AICC', aicc_counts, aicc_max, avg_aicc)
    print_tally('BIC', bic_counts, bic_max, avg_bic)
    timing_df = pd.DataFrame(timing_records, columns=['symbol', 'model', 'seconds', 'n_params']) if timing_records else pd.DataFrame(columns=['symbol', 'model', 'seconds', 'n_params'])
    if not timing_df.empty:
        totals = timing_df.groupby('model')['seconds'].sum().sort_values(ascending=False)
        param_map = timing_df.drop_duplicates('model').set_index('model')['n_params']
        print("\n==== total fit time by model (seconds) ====")
        print(f"{'model':<25} {'seconds':>12} {'n_params':>10}")
        for name, total_sec in totals.items():
            n_params = param_map.get(name, float('nan'))
            param_str = f"{int(n_params):>10d}" if math.isfinite(n_params) else f"{'NA':>10}"
            print(f"{name:<25} {total_sec:12.3f} {param_str}")
    if args.sd_corr and wrote_sd_corr:
        print(f"\nconditional sd correlation matrices written to {args.sd_corr_file}")

    elapsed = time.perf_counter() - start_time
    asset_count = len(models_fit_counts)
    total_models = sum(models_fit_counts)
    avg_models = (total_models / asset_count) if asset_count else float('nan')
    avg_models_int = int(round(avg_models)) if math.isfinite(avg_models) else float('nan')
    sec_per_model = (elapsed / total_models) if total_models else float('nan')

    def fmt(value: float) -> str:
        return f"{value:.3f}" if math.isfinite(value) else 'NA'

    if args.vols_csv and vols_paths:
        print("\nvolatility series written to:")
        for path in vols_paths:
            print(f"  {path}")
    if param_writer and param_csv_path:
        if param_csv_file:
            param_csv_file.flush()
        notice = "model parameter CSV written to" if param_rows_written else "parameter CSV initialized at"
        print(f"\n{notice} {param_csv_path}")

    labels = ['assets', 'models/asset', 'total models', 'seconds', 'sec/model']
    values = [str(asset_count),
        (str(avg_models_int) if math.isfinite(avg_models_int) else 'NA'),
        str(total_models), fmt(elapsed), fmt(sec_per_model)]
    width = 15
    print()
    print(8*" ", ''.join(f"{label:<{width}}" for label in labels))
    print(''.join(f"{value:>{width}}" for value in values))
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
