# momentum-trader

An educational SPY momentum day-trading research script. The code trains a LightGBM classifier on engineered price and volume features, evaluates it with expanding-window walk-forward validation, and can run a paper-trading loop using the saved model.

## Repository contents

- momentum.py contains data preparation, feature engineering, labels, walk-forward training, backtesting, position sizing, and the live paper-trading loop.
- spy_lgb_model.pkl is the saved LightGBM model artifact.
- spy_scaler.pkl is the fitted RobustScaler.
- spy_config.pkl stores the calibrated prediction threshold and configuration.

The binary pickle files are generated or consumed by momentum.py. They are Python-specific artifacts and should only be loaded from trusted sources.

## Strategy overview

The script uses SPY data and builds returns, volatility, ATR, moving-average crossovers, RSI, MACD, Bollinger-band, volume, candle-shape, and time-of-day features. The target is whether the next ten one-minute bars rise by at least 0.05%. An expanding-window walk-forward procedure trains fold models, calibrates a probability threshold from out-of-fold predictions, and then fits a final model on the complete training set.

For the live mode, the script limits each trade with a 0.20% stop, a 0.40% target, a 1.5% equity risk budget, and a 20% maximum position.

## Setup

Python 3.10 or newer is recommended. Install the dependencies in a virtual environment:

~~~bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell
.venv\Scripts\Activate.ps1
pip install numpy pandas lightgbm scikit-learn yfinance joblib
~~~

## Usage

~~~bash
# Train, validate, and save the model artifacts
python momentum.py train

# Load the saved artifacts and run the paper-trading loop
python momentum.py live

# Train and then start paper trading
python momentum.py both
~~~

The script downloads SPY daily data and expands each session into synthetic one-minute bars with a seeded Brownian-bridge approximation. Those bars are useful for repeatable experimentation, but they are not real intraday observations. Review the data path and model artifacts before treating any result as meaningful.

## Outputs and configuration

Training writes or replaces spy_lgb_model.pkl, spy_scaler.pkl, and spy_config.pkl in the repository root. Console logs include fold metrics, the calibrated threshold, prediction distributions, and trade results. The live path is paper trading; no broker integration or order submission is implemented.

## Limitations

This repository is for education and research. The synthetic intraday data, single-ticker design, and historical backtest can produce results that do not generalize to live markets. Paper-trade for an extended period, inspect for leakage, and independently validate any strategy before using real capital.
