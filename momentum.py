"""
SPY ML Momentum Day Trader  v2
==============================
Fixes vs v1:
  - Final model/scaler fitted on FULL training set (no fold leakage into backtest)
  - Label: up ≥0.05% in next 10 bars (~10 min) → ~45-50% base rate, balanced classes
  - Equity accounting corrected (cost deducted on entry, cost+pnl returned on exit)
  - Threshold auto-calibrated from fold OOF predictions
  - Backtest prints per-trade log + distribution of prediction confidence

USAGE:
  python spy_ml_trader.py train   # fetch, train, backtest, save
  python spy_ml_trader.py live    # load saved model, paper-trade market hours
  python spy_ml_trader.py both    # train then immediately go live

DISCLAIMER: Educational/research only. Paper-trade ≥3 months before real capital.
"""

import warnings, os, sys, time, math, logging, datetime, joblib
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
from zoneinfo import ZoneInfo
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import RobustScaler

# ─────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────
TICKER           = "SPY"
STARTING_CAP     = 1_600.0
RISK_PER_TRADE   = 0.015         # 1.5% equity risked per trade
MAX_POSITION     = 0.20          # max 20% equity per trade
STOP_LOSS_PCT    = 0.002         # 0.20% stop
TAKE_PROFIT_PCT  = 0.004         # 0.40% target  (2:1 R:R)
PRED_THRESHOLD   = 0.55          # starting threshold (auto-tuned after WFV)
PRED_FREQ_SEC    = 30            # re-predict every 30 s live
FORWARD_BARS     = 10            # label horizon: ~10 min on 1-min bars
LABEL_MIN_RETURN = 0.0005        # ≥ 0.05% rise = positive label

ET           = ZoneInfo("America/New_York")
MARKET_OPEN  = datetime.time(9, 30)
MARKET_CLOSE = datetime.time(16, 0)

MODEL_PATH  = "spy_lgb_model.pkl"
SCALER_PATH = "spy_scaler.pkl"
CFG_PATH    = "spy_config.pkl"   # saves calibrated threshold

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("SPY-ML")


# ─────────────────────────────────────────────────────────────────
# 1. DATA
# ─────────────────────────────────────────────────────────────────

def fetch_data(start="2020-01-01", end="2026-01-01") -> pd.DataFrame:
    import yfinance as yf
    log.info(f"Downloading SPY daily data {start} → {end} ...")
    df = yf.download(TICKER, start=start, end=end, interval="1d",
                     auto_adjust=True, progress=False)
    df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower()
                  for c in df.columns]
    df = df[["open","high","low","close","volume"]].dropna()
    log.info(f"  {len(df)} daily bars downloaded")
    return expand_to_intraday(df)


def expand_to_intraday(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Expand daily OHLCV → 1-min synthetic bars via Brownian bridge."""
    MINS = 390
    records = []
    for date, row in daily_df.iterrows():
        o, h, l, c, vol = (float(row[x]) for x in ["open","high","low","close","volume"])
        np.random.seed(int(pd.Timestamp(date).timestamp()) % (2**31))
        increments = np.random.randn(MINS)
        bridge = np.cumsum(increments)
        bridge -= bridge[-1]                          # pin end
        span   = max(abs(bridge.max()), abs(bridge.min()), 1e-9)
        bridge = bridge / span * (c - o)
        prices = np.clip(o + bridge, l, h)
        vol_shape = np.random.exponential(1.0, MINS)
        vol_shape /= vol_shape.sum()
        for i, price in enumerate(prices):
            ts = pd.Timestamp(date) + pd.Timedelta(hours=9, minutes=30+i)
            spread = price * 0.0001
            records.append({
                "timestamp": ts,
                "open":   round(price - spread/2, 4),
                "high":   round(max(price, o if i==0 else prices[i-1]) + abs(np.random.randn())*spread, 4),
                "low":    round(min(price, o if i==0 else prices[i-1]) - abs(np.random.randn())*spread, 4),
                "close":  round(price, 4),
                "volume": float(vol * vol_shape[i]),
            })
    df = pd.DataFrame(records).set_index("timestamp")
    log.info(f"  Expanded to {len(df):,} intraday bars")
    return df


# ─────────────────────────────────────────────────────────────────
# 2. FEATURES
# ─────────────────────────────────────────────────────────────────

FEATURE_COLS = [
    "ret_1","ret_3","ret_5","ret_10","ret_20","ret_60",
    "vol_5","vol_10","vol_20",
    "atr_10","atr_20",
    "sma_cross_5_20","sma_cross_10_60","close_vs_sma20",
    "rsi_14",
    "macd_hist",
    "bb_pos","bb_width",
    "vol_ratio","vwap_dev",
    "body","wick_up","wick_dn",
    "time_frac","is_first30","is_last30",
]


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    # Returns
    for n in [1, 3, 5, 10, 20, 60]:
        d[f"ret_{n}"] = d["close"].pct_change(n)

    # Volatility (rolling std of 1-bar returns)
    for n in [5, 10, 20]:
        d[f"vol_{n}"] = d["ret_1"].rolling(n).std()

    # ATR
    d["tr"]    = (d["high"] - d["low"]).abs()
    d["atr_10"] = d["tr"].rolling(10).mean()
    d["atr_20"] = d["tr"].rolling(20).mean()

    # SMAs & crossovers
    for n in [5, 10, 20, 60]:
        d[f"sma_{n}"] = d["close"].rolling(n).mean()
    d["sma_cross_5_20"]  = (d["sma_5"]  - d["sma_20"])  / (d["sma_20"]  + 1e-9)
    d["sma_cross_10_60"] = (d["sma_10"] - d["sma_60"])  / (d["sma_60"]  + 1e-9)
    d["close_vs_sma20"]  =  d["close"]  / (d["sma_20"]  + 1e-9) - 1

    # RSI-14
    delta = d["close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    d["rsi_14"] = 100 - 100 / (1 + gain / (loss + 1e-9))

    # MACD histogram only (avoids multicollinearity)
    ema12 = d["close"].ewm(span=12, adjust=False).mean()
    ema26 = d["close"].ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    sig   = macd.ewm(span=9, adjust=False).mean()
    d["macd_hist"] = macd - sig

    # Bollinger
    bb_mid = d["close"].rolling(20).mean()
    bb_std = d["close"].rolling(20).std()
    d["bb_pos"]   = (d["close"] - bb_mid) / (2 * bb_std + 1e-9)
    d["bb_width"] = (2 * bb_std) / (bb_mid + 1e-9)

    # Volume
    avg_vol = d["volume"].rolling(20).mean()
    d["vol_ratio"] = d["volume"] / (avg_vol + 1e-9)
    cumvol  = d["volume"].rolling(20).sum()
    cumvwap = (d["close"] * d["volume"]).rolling(20).sum()
    d["vwap_dev"] = d["close"] / (cumvwap / (cumvol + 1e-9) + 1e-9) - 1

    # Candle
    d["body"]    = (d["close"] - d["open"]) / (d["open"].abs() + 1e-9)
    hi_oc = d[["open","close"]].max(axis=1)
    lo_oc = d[["open","close"]].min(axis=1)
    d["wick_up"] = (d["high"] - hi_oc) / (d["open"].abs() + 1e-9)
    d["wick_dn"] = (lo_oc - d["low"])   / (d["open"].abs() + 1e-9)

    # Time features
    mins = d.index.hour * 60 + d.index.minute
    d["time_frac"]  = (mins - 570) / 390.0        # 0=open, 1=close
    d["is_first30"] = ((mins >= 570) & (mins < 600)).astype(float)
    d["is_last30"]  = ((mins >= 930) & (mins < 960)).astype(float)

    return d


def make_labels(df: pd.DataFrame,
                forward_bars: int = FORWARD_BARS,
                min_ret: float = LABEL_MIN_RETURN) -> pd.Series:
    fwd = df["close"].shift(-forward_bars) / df["close"] - 1
    return (fwd >= min_ret).astype(int)


# ─────────────────────────────────────────────────────────────────
# 3. WALK-FORWARD VALIDATION
# ─────────────────────────────────────────────────────────────────

LGBM_PARAMS = {
    "objective":         "binary",
    "metric":            "auc",
    "learning_rate":     0.03,
    "num_leaves":        31,
    "min_data_in_leaf":  200,
    "feature_fraction":  0.75,
    "bagging_fraction":  0.75,
    "bagging_freq":      5,
    "lambda_l1":         0.2,
    "lambda_l2":         0.2,
    "scale_pos_weight":  1.0,   # adjust if label is imbalanced
    "verbose":           -1,
}


def walk_forward_train(train_df: pd.DataFrame, n_splits: int = 5):
    """
    Expanding-window WFV.
    Returns (final_model, final_scaler, calibrated_threshold).
    final_model/scaler are trained on the COMPLETE train_df.
    """
    n         = len(train_df)
    fold_size = n // (n_splits + 1)

    log.info(f"\n{'─'*62}")
    log.info(f"Walk-Forward Validation  ({n_splits} folds, ~{fold_size:,} bars/fold)")
    log.info(f"{'─'*62}")

    X_all = train_df[FEATURE_COLS]
    y_all = train_df["label"]

    oof_probs = np.full(n, np.nan)

    for fold in range(n_splits):
        tr_end    = fold_size * (fold + 1)
        val_start = tr_end
        val_end   = val_start + fold_size
        if val_end > n:
            break

        scaler_fold = RobustScaler()
        X_tr = scaler_fold.fit_transform(X_all.iloc[:tr_end])
        y_tr = y_all.iloc[:tr_end].values
        X_vl = scaler_fold.transform(X_all.iloc[val_start:val_end])
        y_vl = y_all.iloc[val_start:val_end].values

        # Class-weight balance per fold
        pos_rate = y_tr.mean()
        params   = {**LGBM_PARAMS,
                    "scale_pos_weight": (1 - pos_rate) / (pos_rate + 1e-9)}

        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dval   = lgb.Dataset(X_vl, label=y_vl, reference=dtrain)

        model_fold = lgb.train(
            params, dtrain,
            num_boost_round=600,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(60, verbose=False),
                       lgb.log_evaluation(-1)],
        )

        preds = model_fold.predict(X_vl)
        oof_probs[val_start:val_end] = preds

        # Metrics at default 0.5 cut
        bp = (preds >= 0.5).astype(int)
        auc  = roc_auc_score(y_vl, preds)
        prec = precision_score(y_vl, bp, zero_division=0)
        rec  = recall_score(y_vl, bp, zero_division=0)
        f1   = f1_score(y_vl, bp, zero_division=0)
        log.info(
            f"  Fold {fold+1}/{n_splits} | train={tr_end:6,} | val={fold_size:5,} | "
            f"AUC={auc:.3f}  Prec={prec:.3f}  Rec={rec:.3f}  F1={f1:.3f}"
        )

    # ── Calibrate threshold from OOF predictions ──────────────────
    valid_mask = ~np.isnan(oof_probs)
    oof_y      = y_all.values[valid_mask]
    oof_p      = oof_probs[valid_mask]

    best_thresh, best_f1 = PRED_THRESHOLD, 0.0
    for t in np.arange(0.40, 0.75, 0.01):
        bp  = (oof_p >= t).astype(int)
        f1  = f1_score(oof_y, bp, zero_division=0)
        prec = precision_score(oof_y, bp, zero_division=0)
        if f1 > best_f1 and prec >= 0.52:   # require ≥52% precision
            best_f1, best_thresh = f1, t

    log.info(f"\n  OOF calibrated threshold → {best_thresh:.2f}  (F1={best_f1:.3f})")
    log.info(f"  OOF label rate           → {oof_y.mean():.1%}")
    log.info(f"{'─'*62}\n")

    # ── Final model: fit on ALL train data ────────────────────────
    log.info("Training final model on complete training set ...")
    pos_rate   = y_all.mean()
    params_fin = {**LGBM_PARAMS,
                  "scale_pos_weight": (1 - pos_rate) / (pos_rate + 1e-9)}

    final_scaler = RobustScaler()
    X_final = final_scaler.fit_transform(X_all)

    # Use best n_estimators from last fold as guide
    final_model = lgb.train(
        params_fin,
        lgb.Dataset(X_final, label=y_all.values),
        num_boost_round=400,   # conservative fixed rounds on full data
        callbacks=[lgb.log_evaluation(-1)],
    )
    log.info("  Final model trained.")

    return final_model, final_scaler, best_thresh


# ─────────────────────────────────────────────────────────────────
# 4. POSITION SIZING  (half-Kelly)
# ─────────────────────────────────────────────────────────────────

def size_position(equity: float, confidence: float, price: float) -> dict:
    p = float(confidence)
    b = TAKE_PROFIT_PCT / STOP_LOSS_PCT          # reward:risk = 2.0
    kelly_f = max(0.0, (p * b - (1 - p)) / b) * 0.5
    kelly_f = min(kelly_f, MAX_POSITION)

    shares_kelly    = int(kelly_f * equity / price)
    max_loss_shares = int((equity * RISK_PER_TRADE) / (price * STOP_LOSS_PCT))
    shares = max(0, min(shares_kelly, max_loss_shares))

    return {
        "shares":    shares,
        "cost":      round(shares * price, 2),
        "stop":      round(price * (1 - STOP_LOSS_PCT), 2),
        "target":    round(price * (1 + TAKE_PROFIT_PCT), 2),
        "kelly_pct": round(kelly_f * 100, 2),
        "conf_pct":  round(p * 100, 2),
    }


# ─────────────────────────────────────────────────────────────────
# 5. BACKTEST
# ─────────────────────────────────────────────────────────────────

def backtest(df: pd.DataFrame, model, scaler, threshold: float,
             equity: float = STARTING_CAP) -> pd.DataFrame:
    log.info(f"\nBacktesting on {len(df):,} test bars  (threshold={threshold:.2f}) ...")

    probs    = model.predict(scaler.transform(df[FEATURE_COLS]))
    prices   = df["close"].values
    position = None
    trades   = []
    eq       = equity
    cash     = equity

    # show prediction distribution
    p50, p75, p90 = np.percentile(probs, [50, 75, 90])
    above_thresh  = (probs >= threshold).sum()
    log.info(f"  Pred dist → p50={p50:.3f}  p75={p75:.3f}  p90={p90:.3f}  "
             f"above_thresh={above_thresh} ({above_thresh/len(probs)*100:.1f}%)")

    for i, (idx, row) in enumerate(df.iterrows()):
        price = float(prices[i])

        # ── Exit
        if position:
            reason = None
            if price <= position["stop"]:
                reason = "stop"
            elif price >= position["target"]:
                reason = "target"

            if reason:
                proceeds = position["shares"] * price
                pnl      = proceeds - position["cost"]
                cash    += proceeds
                eq       = cash  # (no open position)
                trades.append({
                    "time":   idx,
                    "entry":  position["entry"],
                    "exit":   round(price, 4),
                    "shares": position["shares"],
                    "pnl":    round(pnl, 2),
                    "equity": round(eq, 2),
                    "reason": reason,
                    "conf":   position["conf"],
                })
                position = None

        # ── Entry
        if position is None:
            prob = float(probs[i])
            if prob >= threshold:
                sz = size_position(cash, prob, price)
                if sz["shares"] > 0 and sz["cost"] <= cash:
                    cash -= sz["cost"]
                    position = {
                        "shares": sz["shares"],
                        "entry":  price,
                        "cost":   sz["cost"],
                        "stop":   sz["stop"],
                        "target": sz["target"],
                        "conf":   prob,
                    }

    # close any open position at last bar
    if position:
        price    = float(prices[-1])
        proceeds = position["shares"] * price
        pnl      = proceeds - position["cost"]
        cash    += proceeds
        eq       = cash
        trades.append({
            "time":   df.index[-1],
            "entry":  position["entry"],
            "exit":   round(price, 4),
            "shares": position["shares"],
            "pnl":    round(pnl, 2),
            "equity": round(eq, 2),
            "reason": "end",
            "conf":   position["conf"],
        })

    if not trades:
        log.info("  ⚠  No trades triggered. Try lowering PRED_THRESHOLD.")
        return pd.DataFrame()

    t = pd.DataFrame(trades)
    wins     = (t["pnl"] > 0).sum()
    pnl_arr  = t["pnl"].values
    w_arr    = pnl_arr[pnl_arr > 0]
    l_arr    = pnl_arr[pnl_arr < 0]
    eq_curve = t["equity"]
    drawdown = eq_curve - eq_curve.cummax()

    log.info(f"\n{'═'*60}")
    log.info(f"  BACKTEST RESULTS")
    log.info(f"{'─'*60}")
    log.info(f"  Total trades   : {len(t)}")
    log.info(f"  Win rate       : {wins}/{len(t)}  ({wins/len(t)*100:.1f}%)")
    log.info(f"  Total P&L      : ${pnl_arr.sum():+.2f}")
    log.info(f"  Starting equity: ${equity:.2f}")
    log.info(f"  Final equity   : ${eq_curve.iloc[-1]:.2f}")
    log.info(f"  Return         : {(eq_curve.iloc[-1]/equity - 1)*100:+.2f}%")
    if len(w_arr):
        log.info(f"  Avg win        : ${w_arr.mean():.2f}")
    if len(l_arr):
        log.info(f"  Avg loss       : ${l_arr.mean():.2f}")
    if len(w_arr) and len(l_arr):
        log.info(f"  Profit factor  : {w_arr.sum() / abs(l_arr.sum()):.2f}")
    log.info(f"  Max drawdown   : ${drawdown.min():.2f}")
    log.info(f"{'═'*60}")

    # Per-trade log (last 20)
    print("\n── Last 20 trades ──────────────────────────────────")
    print(t.tail(20).to_string(index=False))

    return t


# ─────────────────────────────────────────────────────────────────
# 6. LIVE PAPER TRADER
# ─────────────────────────────────────────────────────────────────

class PaperTrader:
    def __init__(self, model, scaler, threshold: float,
                 equity: float = STARTING_CAP):
        self.model     = model
        self.scaler    = scaler
        self.threshold = threshold
        self.cash      = equity
        self.position  = None
        self.trades    = []

    def _in_market(self) -> bool:
        now = datetime.datetime.now(ET).time()
        return MARKET_OPEN <= now < MARKET_CLOSE

    def _fetch(self) -> pd.DataFrame:
        import yfinance as yf
        df = yf.download(TICKER, period="5d", interval="1m",
                         auto_adjust=True, progress=False)
        df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower()
                      for c in df.columns]
        return df[["open","high","low","close","volume"]].dropna()

    def _check_exit(self, price: float):
        if not self.position:
            return
        pos = self.position
        reason = None
        if price <= pos["stop"]:
            reason = "STOP"
        elif price >= pos["target"]:
            reason = "TARGET"
        elif not self._in_market():
            reason = "EOD"

        if reason:
            proceeds = pos["shares"] * price
            pnl      = proceeds - pos["cost"]
            self.cash += proceeds
            self.trades.append({
                "time":   datetime.datetime.now(ET).strftime("%H:%M:%S"),
                "entry":  pos["entry"], "exit": price,
                "shares": pos["shares"], "pnl": round(pnl, 2),
                "equity": round(self.cash, 2), "reason": reason,
            })
            log.info(f"  ◀ EXIT {reason:6s} | entry={pos['entry']:.2f} "
                     f"exit={price:.2f} pnl=${pnl:+.2f} cash=${self.cash:.2f}")
            self.position = None

    def tick(self):
        if not self._in_market():
            log.info("Outside market hours.")
            return

        try:
            raw  = self._fetch()
            feat = make_features(raw).dropna(subset=FEATURE_COLS)
        except Exception as e:
            log.error(f"Data error: {e}")
            return

        if feat.empty:
            return

        price = float(feat["close"].iloc[-1])
        self._check_exit(price)

        if self.position:
            pos = self.position
            log.info(f"  ◈ HOLD | price={price:.2f} stop={pos['stop']:.2f} "
                     f"target={pos['target']:.2f} cash=${self.cash:.2f}")
            return

        X    = self.scaler.transform(feat[FEATURE_COLS].iloc[[-1]])
        prob = float(self.model.predict(X)[0])
        sz   = size_position(self.cash, prob, price)

        log.info(f"  ◇ SCAN | price={price:.2f} conf={prob*100:.1f}% "
                 f"(need≥{self.threshold*100:.0f}%) size={sz['shares']}sh "
                 f"cost=${sz['cost']:.2f} kelly={sz['kelly_pct']:.1f}%")

        if prob >= self.threshold and sz["shares"] > 0 and sz["cost"] <= self.cash:
            self.cash -= sz["cost"]
            self.position = {
                "shares": sz["shares"], "entry": price,
                "cost": sz["cost"], "stop": sz["stop"],
                "target": sz["target"], "conf": prob,
            }
            log.info(f"  ▶ ENTER {sz['shares']}sh @ {price:.2f} | "
                     f"stop={sz['stop']:.2f} target={sz['target']:.2f} "
                     f"cash=${self.cash:.2f}")

    def run(self):
        log.info("=" * 62)
        log.info(f"  SPY Live Paper Trader | cash=${self.cash:.2f} "
                 f"threshold={self.threshold:.2f}")
        log.info(f"  Stop={STOP_LOSS_PCT*100:.2f}% Target={TAKE_PROFIT_PCT*100:.2f}% "
                 f"Interval={PRED_FREQ_SEC}s")
        log.info("=" * 62)

        while True:
            try:
                self.tick()
            except Exception as e:
                log.error(f"Tick: {e}")

            if datetime.datetime.now(ET).time() >= MARKET_CLOSE:
                if self.position:
                    raw = self._fetch()
                    self._check_exit(float(raw["close"].iloc[-1]))
                self._summary()
                break
            time.sleep(PRED_FREQ_SEC)

    def _summary(self):
        log.info(f"\n{'═'*60}")
        if not self.trades:
            log.info("  No trades today.")
        else:
            df   = pd.DataFrame(self.trades)
            wins = (df["pnl"] > 0).sum()
            log.info(f"  Trades: {len(df)}  Win%: {wins/len(df)*100:.0f}%  "
                     f"PnL: ${df['pnl'].sum():+.2f}  "
                     f"Equity: ${df['equity'].iloc[-1]:.2f}")
            print(df.to_string(index=False))
        log.info(f"{'═'*60}")


# ─────────────────────────────────────────────────────────────────
# 7. MAIN
# ─────────────────────────────────────────────────────────────────

def train_pipeline():
    intra = fetch_data("2020-01-01", "2026-01-01")

    log.info("Engineering features ...")
    feat = make_features(intra)
    feat["label"] = make_labels(feat)
    feat = feat.dropna(subset=FEATURE_COLS + ["label"])
    label_rate = feat["label"].mean()
    log.info(f"  Dataset: {len(feat):,} bars | label rate: {label_rate:.1%}")

    if label_rate < 0.25 or label_rate > 0.75:
        log.warning(f"  Label rate {label_rate:.1%} is imbalanced — "
                    "consider adjusting LABEL_MIN_RETURN or FORWARD_BARS")

    split    = int(len(feat) * 0.8)
    train_df = feat.iloc[:split].copy()
    test_df  = feat.iloc[split:].copy()
    log.info(f"  Train: {len(train_df):,}  |  Test: {len(test_df):,}")

    model, scaler, threshold = walk_forward_train(train_df)
    backtest(test_df, model, scaler, threshold)

    joblib.dump(model,     MODEL_PATH)
    joblib.dump(scaler,    SCALER_PATH)
    joblib.dump(threshold, CFG_PATH)
    log.info(f"\nSaved → {MODEL_PATH}, {SCALER_PATH}, {CFG_PATH}")
    return model, scaler, threshold


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "train"

    if mode == "train":
        train_pipeline()

    elif mode == "live":
        for p in [MODEL_PATH, SCALER_PATH, CFG_PATH]:
            if not os.path.exists(p):
                log.error(f"Missing {p} — run 'train' first.")
                sys.exit(1)
        model     = joblib.load(MODEL_PATH)
        scaler    = joblib.load(SCALER_PATH)
        threshold = joblib.load(CFG_PATH)
        PaperTrader(model, scaler, threshold, STARTING_CAP).run()

    elif mode == "both":
        model, scaler, threshold = train_pipeline()
        PaperTrader(model, scaler, threshold, STARTING_CAP).run()

    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()