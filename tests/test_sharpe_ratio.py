import pandas as pd

from main import compute_sharpe_ratio, summarize_trades


def test_compute_sharpe_ratio_positive_for_profitable_series():
    returns = pd.Series([0.10, 0.05, -0.02, 0.08, 0.04])
    sharpe = compute_sharpe_ratio(returns)

    assert sharpe > 0.0


def test_summary_includes_sharpe_ratio_key():
    trade_df = pd.DataFrame(
        {
            "trade_return": [0.12, -0.04, 0.10, 0.08],
            "holding_days": [10, 12, 9, 15],
        }
    )

    summary = summarize_trades(trade_df)

    assert "sharpe_ratio" in summary
