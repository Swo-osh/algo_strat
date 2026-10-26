import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf


def fetch_market_data(start: str, end: str | None = None) -> pd.DataFrame:
    """Download and merge market, SPY, and short-term Treasury ETF data."""
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    tickers = ["^VIX", "^VIX3M", "SPY", "SHY"]
    frames = {}

    for ticker in tickers:
        raw = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
        if raw.empty:
            raise ValueError(f"No data returned for {ticker} between {start} and {end}.")
        price_column = "Adj Close" if ticker in {"SPY", "SHY"} else "Close"
        frame = raw[[price_column]].copy()
        frame.columns = [ticker]
        frames[ticker] = frame

    merged = frames["^VIX"].join(frames["^VIX3M"], how="inner").join(frames["SPY"], how="inner").join(frames["SHY"], how="inner")
    merged.columns = ["VIX", "VIX3M", "SPY", "SHY"]
    merged = merged.dropna().sort_index()
    return merged


def compute_term_structure(df: pd.DataFrame) -> pd.DataFrame:
    """Add ratio and signal columns to the data frame."""
    result = df.copy()
    result["ratio"] = result["VIX"] / result["VIX3M"]
    result["is_backwardation"] = result["ratio"] > 1.0
    return result


def detect_entry_points(df: pd.DataFrame, threshold: float = 1.1) -> pd.Series:
    """Return True on the first day when the ratio crosses above the threshold."""
    series = df["ratio"]
    previous = series.shift(1)
    return (series > threshold) & (previous <= threshold)


def event_study(
    df: pd.DataFrame,
    threshold: float = 1.1,
    exit_threshold: float = 1.0,
    max_hold_days: int = 15,
) -> pd.DataFrame:
    """Compute trade-level returns for each backwardation event."""
    data = df.copy()
    data["entry_signal"] = detect_entry_points(data, threshold=threshold)

    trades = []
    last_exit_index = -1
    for idx in range(len(data)):
        if idx <= last_exit_index or not data["entry_signal"].iloc[idx]:
            continue

        entry_index = idx + 1
        if entry_index >= len(data):
            continue

        entry_date = data.index[entry_index]
        entry_price = data["SPY"].iloc[entry_index]
        exit_index = entry_index
        exit_reason = "max_hold"

        while exit_index < len(data) and (exit_index - entry_index) < max_hold_days:
            if data["ratio"].iloc[exit_index] < exit_threshold:
                exit_reason = "reversion"
                break
            exit_index += 1

        if exit_reason == "reversion":
            exit_index += 1
            if exit_index >= len(data):
                continue
        elif exit_index >= len(data):
            exit_index = len(data) - 1

        exit_date = data.index[exit_index]
        exit_price = data["SPY"].iloc[exit_index]
        holding_days = (exit_date - entry_date).days
        trade_return = exit_price / entry_price - 1.0

        trades.append(
            {
                "entry_date": entry_date,
                "exit_date": exit_date,
                "holding_days": holding_days,
                "entry_ratio": data["ratio"].iloc[idx],
                "exit_ratio": data["ratio"].iloc[exit_index],
                "trade_return": trade_return,
                "exit_reason": exit_reason,
            }
        )
        last_exit_index = exit_index

    if not trades:
        raise ValueError(f"No backwardation events found with threshold {threshold}.")

    trade_df = pd.DataFrame(trades)
    trade_df["equity_curve"] = (1 + trade_df["trade_return"]).cumprod()
    return trade_df


def compute_sharpe_ratio(returns: pd.Series | np.ndarray, risk_free_rate: float = 0.0, annualization_factor: float = 252.0) -> float:
    """Compute the annualized Sharpe ratio for a return series."""
    series = pd.Series(returns, dtype=float).dropna()
    if series.empty:
        return 0.0

    excess_returns = series - risk_free_rate
    std_dev = excess_returns.std(ddof=1)
    if std_dev <= 0:
        return 0.0

    return float((excess_returns.mean() / std_dev) * np.sqrt(annualization_factor))


def summarize_return_series(returns: pd.Series) -> dict[str, float]:
    """Summarize a daily return series for portfolio comparisons."""
    returns = returns.dropna()
    equity = (1.0 + returns).cumprod()
    total_return = float(equity.iloc[-1] - 1.0) if not equity.empty else 0.0
    annualized_return = (
        float((1.0 + total_return) ** (252.0 / len(returns)) - 1.0)
        if not returns.empty and total_return > -1.0
        else 0.0
    )
    drawdown = equity / equity.cummax() - 1.0 if not equity.empty else pd.Series(dtype=float)
    return {
        "return_pct": total_return * 100.0,
        "annualized_return_pct": annualized_return * 100.0,
        "volatility_pct": float(returns.std(ddof=1) * np.sqrt(252.0) * 100.0),
        "sharpe": compute_sharpe_ratio(returns),
        "max_drawdown_pct": float(drawdown.min() * 100.0) if not drawdown.empty else 0.0,
    }


def compute_daily_strategy_returns(
    data: pd.DataFrame,
    trade_df: pd.DataFrame,
    transaction_cost_bps: float = 0.0,
) -> pd.Series:
    """Build daily returns using SPY during trades and SHY otherwise."""
    daily_market_returns = data["SPY"].pct_change().fillna(0.0)
    daily_treasury_returns = data["SHY"].pct_change().fillna(0.0)
    position = pd.Series(0.0, index=data.index)

    for trade in trade_df.itertuples():
        active_period = (data.index >= trade.entry_date) & (data.index <= trade.exit_date)
        position.loc[active_period] = 1.0

    invested = position.shift(1).fillna(0.0)
    portfolio_returns = invested * daily_market_returns + (1.0 - invested) * daily_treasury_returns
    turnover = position.diff().abs().fillna(position.iloc[0])
    return portfolio_returns - turnover * (transaction_cost_bps / 10_000.0)


def compute_market_exposure(data: pd.DataFrame, trade_df: pd.DataFrame) -> float:
    """Return the percentage of sessions spent holding SPY."""
    position = pd.Series(False, index=data.index)
    for trade in trade_df.itertuples():
        active_period = (data.index >= trade.entry_date) & (data.index <= trade.exit_date)
        position.loc[active_period] = True
    return float(position.mean() * 100.0)


def summarize_trades(
    trade_df: pd.DataFrame,
    daily_strategy_returns: pd.Series | None = None,
    market_exposure_pct: float | None = None,
    transaction_cost_bps: float = 0.0,
    benchmark_window: int = 15,
    notional: float = 100_000.0,
) -> dict:
    """Create a compact performance summary for the strategy."""
    returns = trade_df["trade_return"].dropna()
    if returns.empty:
        raise ValueError("No valid trade returns available.")

    benchmark = np.nanmean(returns) * 0.0
    pnl = returns * notional
    sharpe_ratio = compute_sharpe_ratio(daily_strategy_returns) if daily_strategy_returns is not None else 0.0
    portfolio_return = (
        float((1.0 + daily_strategy_returns).prod() - 1.0)
        if daily_strategy_returns is not None
        else 0.0
    )
    annualized_return = (
        float((1.0 + portfolio_return) ** (252.0 / len(daily_strategy_returns)) - 1.0)
        if daily_strategy_returns is not None and portfolio_return > -1.0
        else 0.0
    )
    annualized_volatility = (
        float(daily_strategy_returns.std(ddof=1) * np.sqrt(252.0))
        if daily_strategy_returns is not None
        else 0.0
    )
    max_drawdown = 0.0
    if daily_strategy_returns is not None:
        equity = (1.0 + daily_strategy_returns).cumprod()
        drawdown = equity / equity.cummax() - 1.0
        max_drawdown = float(drawdown.min())
    summary = {
        "total_trades": int(len(returns)),
        "win_rate": float((returns > 0).mean()),
        "average_trade_return": float(returns.mean()),
        "average_trade_pnl_usd": float(pnl.mean()),
        "total_pnl_usd": float(pnl.sum()),
        "median_trade_return": float(returns.median()),
        "best_trade": float(returns.max()),
        "worst_trade": float(returns.min()),
        "average_holding_days": float(trade_df["holding_days"].mean()),
        "sharpe_ratio": sharpe_ratio,
        "portfolio_return_pct": portfolio_return * 100.0,
        "annualized_return_pct": annualized_return * 100.0,
        "annualized_volatility_pct": annualized_volatility * 100.0,
        "max_drawdown_pct": max_drawdown * 100.0,
        "market_exposure_pct": float(market_exposure_pct or 0.0),
        "transaction_cost_bps": float(transaction_cost_bps),
        "gross_portfolio_return_usd": portfolio_return * notional,
        "final_portfolio_value_usd": (1.0 + portfolio_return) * notional,
        "benchmark_15d_mean_return": benchmark,
        "notional_per_trade_usd": float(notional),
    }
    return summary


def evaluate_period(
    data: pd.DataFrame,
    threshold: float,
    exit_threshold: float,
    max_hold_days: int,
    transaction_cost_bps: float,
    notional: float,
) -> dict[str, float] | None:
    """Evaluate one parameter set on an already downloaded data window."""
    try:
        trades = event_study(
            data,
            threshold=threshold,
            exit_threshold=exit_threshold,
            max_hold_days=max_hold_days,
        )
    except ValueError:
        return None

    daily_returns = compute_daily_strategy_returns(data, trades, transaction_cost_bps)
    summary = summarize_trades(
        trades,
        daily_strategy_returns=daily_returns,
        market_exposure_pct=compute_market_exposure(data, trades),
        transaction_cost_bps=transaction_cost_bps,
        notional=notional,
    )
    return summary


def print_sensitivity_report(
    data: pd.DataFrame,
    transaction_cost_bps: float,
    notional: float,
) -> None:
    """Print results across reasonable entry and holding-period choices."""
    print("\n=== Parameter Sensitivity ===")
    print("entry_threshold | max_hold_days | trades | return_pct | sharpe | max_drawdown_pct")
    for threshold in (1.00, 1.05, 1.10, 1.20):
        for max_hold_days in (10, 20, 30):
            summary = evaluate_period(
                data,
                threshold,
                1.00,
                max_hold_days,
                transaction_cost_bps,
                notional,
            )
            if summary is None:
                print(f"{threshold:14.2f} | {max_hold_days:13d} | no trades")
                continue
            print(
                f"{threshold:14.2f} | {max_hold_days:13d} | "
                f"{summary['total_trades']:6.0f} | {summary['portfolio_return_pct']:10.2f} | "
                f"{summary['sharpe_ratio']:6.2f} | {summary['max_drawdown_pct']:16.2f}"
            )


def print_period_report(
    data: pd.DataFrame,
    periods: list[tuple[str, str, str]],
    threshold: float,
    exit_threshold: float,
    max_hold_days: int,
    transaction_cost_bps: float,
    notional: float,
) -> None:
    """Print out-of-sample and crisis-period results where data is available."""
    print("\n=== Period Analysis ===")
    print("period       | trades | return_pct | annualized_pct | sharpe | max_drawdown_pct")
    for name, start, end in periods:
        period_data = data.loc[(data.index >= start) & (data.index < end)]
        summary = evaluate_period(
            period_data,
            threshold,
            exit_threshold,
            max_hold_days,
            transaction_cost_bps,
            notional,
        ) if not period_data.empty else None
        if summary is None:
            print(f"{name:12} | no trades or unavailable data")
            continue
        print(
            f"{name:12} | {summary['total_trades']:6.0f} | "
            f"{summary['portfolio_return_pct']:10.2f} | "
            f"{summary['annualized_return_pct']:14.2f} | "
            f"{summary['sharpe_ratio']:6.2f} | {summary['max_drawdown_pct']:16.2f}"
        )


def plot_results(
    data: pd.DataFrame,
    trade_df: pd.DataFrame,
    daily_strategy_returns: pd.Series,
    benchmark_returns: dict[str, pd.Series],
    entry_threshold: float,
    output_path: str,
) -> None:
    """Save a two-panel chart showing the ratio and portfolio returns."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=False)

    axes[0].plot(data.index, data["ratio"], color="tab:blue", linewidth=1.8, label="VIX / VIX3M")
    axes[0].axhline(1.0, color="black", linestyle="--", linewidth=1, alpha=0.8, label="Backwardation = 1.0")
    axes[0].axhline(
        entry_threshold,
        color="tab:red",
        linestyle=":",
        linewidth=1.2,
        label=f"Entry threshold = {entry_threshold:.2f}",
    )
    axes[0].set_title("VIX Term Structure Backwardation")
    axes[0].set_ylabel("Ratio")
    axes[0].legend(loc="upper left")
    axes[0].grid(alpha=0.25)

    trade_equity = (1 + trade_df["trade_return"]).cumprod()
    portfolio_equity = (1 + daily_strategy_returns).cumprod()
    axes[1].plot(
        trade_df["exit_date"],
        trade_equity,
        color="tab:green",
        linewidth=1.8,
        marker="o",
        label="Event-trade equity",
    )
    axes[1].plot(
        portfolio_equity.index,
        portfolio_equity,
        color="tab:purple",
        linewidth=2,
        label="SPY/SHY portfolio equity",
    )
    for name, returns in benchmark_returns.items():
        axes[1].plot(
            returns.index,
            (1.0 + returns).cumprod(),
            linewidth=1.5,
            linestyle="--",
            label=f"{name} buy-and-hold",
        )
    axes[1].set_title("Strategy and Portfolio Equity")
    axes[1].set_ylabel("Growth of $1")
    axes[1].set_xlabel("Date")
    axes[1].legend(loc="best")
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_file, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest the VIX term-structure mean reversion strategy.")
    parser.add_argument("--start", default="2004-01-01", help="Start date for market data.")
    parser.add_argument("--end", default=None, help="Optional end date (default is today).")
    parser.add_argument("--threshold", type=float, default=1.05, help="Backwardation trigger ratio; lower values produce more trade signals.")
    parser.add_argument("--exit-threshold", type=float, default=1.0, help="Exit when ratio falls below this value.")
    parser.add_argument("--max-hold-days", type=int, default=20, help="Maximum hold period for each trade.")
    parser.add_argument("--notional-usd", type=float, default=100_000.0, help="Dollar notional per trade.")
    parser.add_argument("--transaction-cost-bps", type=float, default=5.0, help="Cost per SPY entry or exit in basis points.")
    parser.add_argument("--analysis", action="store_true", help="Print sensitivity, out-of-sample, and crisis-period reports.")
    parser.add_argument("--plot-path", default="strategy_backtest.png", help="Where to save the output chart.")
    args = parser.parse_args()

    market_data = fetch_market_data(start=args.start, end=args.end)
    data = compute_term_structure(market_data)
    trades = event_study(
        data,
        threshold=args.threshold,
        exit_threshold=args.exit_threshold,
        max_hold_days=args.max_hold_days,
    )
    trades["notional_usd"] = args.notional_usd
    trades["pnl_usd"] = trades["trade_return"] * trades["notional_usd"]
    daily_strategy_returns = compute_daily_strategy_returns(
        data,
        trades,
        transaction_cost_bps=args.transaction_cost_bps,
    )
    market_exposure_pct = compute_market_exposure(data, trades)
    benchmark_returns = {
        "SPY": data["SPY"].pct_change().fillna(0.0),
        "SHY": data["SHY"].pct_change().fillna(0.0),
    }
    summary = summarize_trades(
        trades,
        daily_strategy_returns=daily_strategy_returns,
        market_exposure_pct=market_exposure_pct,
        transaction_cost_bps=args.transaction_cost_bps,
        notional=args.notional_usd,
    )
    for name, returns in benchmark_returns.items():
        for metric, value in summarize_return_series(returns).items():
            summary[f"{name.lower()}_benchmark_{metric}"] = value

    print("\n=== VIX Term Structure Mean-Reversion Strategy ===")
    print(f"Sample window: {data.index.min().date()} to {data.index.max().date()}")
    print(f"Backwardation threshold: {args.threshold:.2f}")
    print(f"Exit threshold: {args.exit_threshold:.2f}")
    print(f"Max hold days: {args.max_hold_days}")
    print(f"Notional per trade: ${args.notional_usd:,.0f}")
    print("\nTrade summary:")
    for key, value in summary.items():
        print(f"- {key}: {value}")

    if args.analysis:
        print_sensitivity_report(data, args.transaction_cost_bps, args.notional_usd)
        print_period_report(
            data,
            [
                ("development", "2010-01-01", "2019-01-01"),
                ("out_of_sample", "2019-01-01", "2026-01-01"),
                ("2007_2009", "2007-01-01", "2010-01-01"),
                ("2020", "2020-01-01", "2021-01-01"),
                ("2022", "2022-01-01", "2023-01-01"),
            ],
            args.threshold,
            args.exit_threshold,
            args.max_hold_days,
            args.transaction_cost_bps,
            args.notional_usd,
        )

    plot_results(
        data,
        trades,
        daily_strategy_returns,
        benchmark_returns,
        args.threshold,
        args.plot_path,
    )
    print(f"\nChart saved to {Path(args.plot_path).resolve()}")


if __name__ == "__main__":
    main()
