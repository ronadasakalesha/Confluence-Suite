"""
Confluence Suite Bot — Python conversion of the Pine Script indicator.
Fetches 1-hour BTC and ETH candles from Delta Exchange public API,
computes all indicators, detects Strong Buy / Strong Sell signals,
and sends results to a Telegram bot.

No order placement code is included.
"""

import os
import time
import math
import logging
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from dotenv import load_dotenv

# ─────────────────────────────────────────────────────────────
# ENV / LOGGING
# ─────────────────────────────────────────────────────────────
load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# SETTINGS  (mirror Pine Script defaults)
# ─────────────────────────────────────────────────────────────
SENSITIVITY  = 5          # Signal Sensitivity
ATR_LENGTH   = 10         # Signal Tuner (atrLength)
STRONG_THRESHOLD = 0.76   # NN score threshold for signal

# Adaptive Supertrend
MIN_MULT = max(SENSITIVITY - 4, 1)
MAX_MULT = min(SENSITIVITY, 26)
STEP_ST  = 0.5
PERF_ALPHA = 10
MAX_ITER = 250
MAX_DATA = 2500

# Smart Trail
ST_LEN    = 24
ST_SMOOTH = 9
ST_MULT1  = 0.618
ST_MULT2  = 1.0

# Reversal Zones (NPC)
NPC_LEN   = 24
NPC_H     = 8.0
NPC_R     = 2.0
NPC_INNER = 1.5
NPC_OUTER = 2.5

# Delta Exchange API
# Base URL: India exchange (api.india.delta.exchange) per official docs
DELTA_BASE    = "https://api.india.delta.exchange"
SYMBOL_MAP    = {"BTC": "BTCUSD", "ETH": "ETHUSD"}
# Resolution must be a string like "1m", "5m", "15m", "1h", "4h", "1d"
CANDLE_RESOLUTION = "1h"  # 1-hour candles
CANDLE_LIMIT      = 500   # candles to fetch (API max is 2000)

# Seconds per candle for each string resolution
_RESOLUTION_SECONDS = {
    "1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800,
    "1h": 3600, "2h": 7200, "4h": 14400, "6h": 21600,
    "8h": 28800, "12h": 43200, "1d": 86400,
}

# Polling interval: run once per hour, a few seconds after candle close
POLL_INTERVAL_SECONDS = 3600


# ─────────────────────────────────────────────────────────────
# DELTA EXCHANGE — DATA FETCH
# ─────────────────────────────────────────────────────────────

def fetch_ohlcv(symbol: str, resolution: str = "1h", limit: int = 500) -> pd.DataFrame:
    """
    Fetch OHLCV candles from Delta Exchange public API.
    - resolution: string e.g. "1h", "5m", "1d"  (as per Delta Exchange docs)
    - limit:      number of candles to fetch (API max = 2000)
    Returns a DataFrame with columns: time, open, high, low, close, volume.
    Candles are sorted oldest → newest.
    """
    now   = int(time.time())
    secs  = _RESOLUTION_SECONDS.get(resolution, 3600)   # seconds per candle
    start = now - limit * secs

    url    = f"{DELTA_BASE}/v2/history/candles"
    params = {
        "resolution": resolution,
        "symbol":     symbol,
        "start":      str(start),
        "end":        str(now),
    }
    log.debug(f"GET {url} params={params}")
    resp = requests.get(url, params=params, timeout=15)

    if not resp.ok:
        # Log the full response body so we see Delta's error message clearly
        log.error(f"Delta API error {resp.status_code}: {resp.text[:500]}")
        resp.raise_for_status()

    data = resp.json()

    candles = data.get("result", [])
    if not candles:
        raise ValueError(f"No candle data returned for {symbol} — response: {data}")

    df = pd.DataFrame(candles)
    # Delta returns: time (unix), open, high, low, close, volume
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df = df.sort_values("time").reset_index(drop=True)
    log.info(f"Fetched {len(df)} candles for {symbol} ({resolution}) up to "
             f"{datetime.fromtimestamp(df['time'].iloc[-1], tz=timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    return df


# ─────────────────────────────────────────────────────────────
# HELPER MATH
# ─────────────────────────────────────────────────────────────

def wma(series: np.ndarray, length: int) -> np.ndarray:
    """Weighted Moving Average."""
    if length <= 0 or len(series) == 0:
        return np.full(len(series), np.nan)
    weights = np.arange(1, length + 1, dtype=float)
    result = np.full(len(series), np.nan)
    for i in range(length - 1, len(series)):
        window = series[i - length + 1: i + 1]
        result[i] = np.dot(window, weights) / weights.sum()
    return result


def ema(series: np.ndarray, length: int) -> np.ndarray:
    """Exponential Moving Average."""
    result = np.full(len(series), np.nan)
    if length <= 0 or len(series) == 0:
        return result
    k = 2.0 / (length + 1)
    for i in range(len(series)):
        if np.isnan(series[i]):
            continue
        if np.isnan(result[i - 1]) if i > 0 else True:
            result[i] = series[i]
        else:
            result[i] = series[i] * k + result[i - 1] * (1 - k)
    return result


def sma(series: np.ndarray, length: int) -> np.ndarray:
    """Simple Moving Average."""
    result = np.full(len(series), np.nan)
    for i in range(length - 1, len(series)):
        result[i] = np.mean(series[i - length + 1: i + 1])
    return result


def rma(series: np.ndarray, length: int) -> np.ndarray:
    """Wilder's RMA (Pine Script ta.rma)."""
    result = np.full(len(series), np.nan)
    alpha = 1.0 / length
    for i in range(len(series)):
        if np.isnan(series[i]):
            continue
        if np.isnan(result[i - 1]) if i > 0 else True:
            result[i] = series[i]
        else:
            result[i] = alpha * series[i] + (1 - alpha) * result[i - 1]
    return result


def stdev(series: np.ndarray, length: int) -> np.ndarray:
    """Rolling standard deviation (population ddof=0 like Pine Script)."""
    result = np.full(len(series), np.nan)
    for i in range(length - 1, len(series)):
        result[i] = np.std(series[i - length + 1: i + 1], ddof=0)
    return result


def highest(series: np.ndarray, length: int) -> np.ndarray:
    result = np.full(len(series), np.nan)
    for i in range(length - 1, len(series)):
        result[i] = np.max(series[i - length + 1: i + 1])
    return result


def lowest(series: np.ndarray, length: int) -> np.ndarray:
    result = np.full(len(series), np.nan)
    for i in range(length - 1, len(series)):
        result[i] = np.min(series[i - length + 1: i + 1])
    return result


def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, length: int) -> np.ndarray:
    """Average True Range (Wilder RMA)."""
    n = len(close)
    tr = np.full(n, np.nan)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i - 1]),
                    abs(low[i]  - close[i - 1]))
    return rma(tr, length)


def vwma(series: np.ndarray, volume: np.ndarray, length: int) -> np.ndarray:
    result = np.full(len(series), np.nan)
    for i in range(length - 1, len(series)):
        s = series[i - length + 1: i + 1]
        v = volume[i - length + 1: i + 1]
        vs = v.sum()
        result[i] = (s * v).sum() / vs if vs > 0 else np.nan
    return result


def linreg(series: np.ndarray, length: int, offset: int = 0) -> np.ndarray:
    """Pine Script ta.linreg — rolling linear regression value at bar i - offset."""
    result = np.full(len(series), np.nan)
    for i in range(length - 1 + offset, len(series)):
        y = series[i - offset - length + 1: i - offset + 1]
        if len(y) < length:
            continue
        x = np.arange(length, dtype=float)
        xm, ym = x.mean(), y.mean()
        denom = ((x - xm) ** 2).sum()
        if denom == 0:
            result[i] = ym
        else:
            slope = ((x - xm) * (y - ym)).sum() / denom
            intercept = ym - slope * xm
            result[i] = intercept + slope * (length - 1 - offset)
    return result


def crossover(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """True at index i when a[i-1] < b[i-1] and a[i] >= b[i]."""
    result = np.zeros(len(a), dtype=bool)
    for i in range(1, len(a)):
        result[i] = (a[i - 1] < b[i - 1]) and (a[i] >= b[i])
    return result


def crossunder(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    result = np.zeros(len(a), dtype=bool)
    for i in range(1, len(a)):
        result[i] = (a[i - 1] > b[i - 1]) and (a[i] <= b[i])
    return result


def rsi(series: np.ndarray, length: int = 14) -> np.ndarray:
    delta = np.diff(series, prepend=np.nan)
    gain  = np.where(delta > 0, delta, 0.0)
    loss  = np.where(delta < 0, -delta, 0.0)
    avg_gain = rma(gain, length)
    avg_loss = rma(loss, length)
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = np.where(avg_loss > 0, avg_gain / avg_loss, np.inf)
    return 100 - 100 / (1 + rs)


def alma(series: np.ndarray, window: int, offset: float, sigma: float) -> np.ndarray:
    """Arnaud Legoux Moving Average."""
    result = np.full(len(series), np.nan)
    m = offset * (window - 1)
    s = window / sigma
    weights = np.exp(-((np.arange(window) - m) ** 2) / (2 * s * s))
    weights /= weights.sum()
    for i in range(window - 1, len(series)):
        result[i] = np.dot(series[i - window + 1: i + 1], weights)
    return result


# ─────────────────────────────────────────────────────────────
# 1. SMART TRAIL
# ─────────────────────────────────────────────────────────────

def calc_trend_flow_line(close: np.ndarray, length: int) -> np.ndarray:
    hma_len = int(length * 100 / 24 * 20 / 100)
    hma_half = max(hma_len // 2, 1)
    sqrt_len = max(int(math.floor(math.sqrt(hma_len))), 1)
    wma1 = wma(close, hma_half)
    wma2 = wma(close, hma_len)
    raw  = 2 * wma1 - wma2
    hma_val = wma(raw, sqrt_len)

    w1 = max(round(length / 3), 1)
    w2 = max(round((length - w1) / 3), 1)
    dwma = wma(wma(close, w2), w1)

    n = len(close)
    result = np.full(n, np.nan)
    for i in range(n):
        vals = [hma_val[i], dwma[i], hma_val[i]]
        if not any(np.isnan(v) for v in vals):
            result[i] = np.mean(vals)
    return result


def get_smart_trail(close: np.ndarray, length: int, smooth: int,
                    m1: float, m2: float):
    tfl  = calc_trend_flow_line(close, length)
    dev  = stdev(close, length)
    n    = len(close)

    bull      = np.full(n, False)
    inner_arr = np.full(n, np.nan)
    outer_arr = np.full(n, np.nan)
    dir_arr   = [""] * n

    for i in range(smooth, n):
        b = tfl[i] > tfl[i - smooth] if not np.isnan(tfl[i]) and not np.isnan(tfl[i - smooth]) else False
        bull[i]      = b
        d1           = m1 * dev[i] if not np.isnan(dev[i]) else 0.0
        d2           = m2 * dev[i] if not np.isnan(dev[i]) else 0.0
        inner_arr[i] = tfl[i] - d1 if b else tfl[i] + d1
        outer_arr[i] = tfl[i] - d2 if b else tfl[i] + d2
        dir_arr[i]   = "long" if b else "short"

    return inner_arr, outer_arr, dir_arr, tfl, bull


# ─────────────────────────────────────────────────────────────
# 2. TREND CATCHER — Super Smoother filter
# ─────────────────────────────────────────────────────────────

def super_smoother(src: np.ndarray, length: int) -> np.ndarray:
    a1 = math.exp(-math.sqrt(2) * math.pi / length)
    b1 = 2 * a1 * math.cos(math.sqrt(2) * math.pi / length)
    c3 = -(a1 ** 2)
    c2 = b1
    c1 = 1 - c2 - c3
    result = np.full(len(src), np.nan)
    for i in range(len(src)):
        s_prev1 = result[i - 1] if i > 0 and not np.isnan(result[i - 1]) else (src[i - 1] if i > 0 else src[i])
        s_prev2 = result[i - 2] if i > 1 and not np.isnan(result[i - 2]) else (src[i - 2] if i > 1 else src[i])
        result[i] = c1 * src[i] + c2 * s_prev1 + c3 * s_prev2
    return result


def get_trend_catcher(close: np.ndarray):
    ss = super_smoother(close, 50)
    bullish = np.zeros(len(close), dtype=bool)
    for i in range(1, len(close)):
        bullish[i] = ss[i] > ss[i - 1]
    return ss, bullish


# ─────────────────────────────────────────────────────────────
# 3. TREND TRACER
# ─────────────────────────────────────────────────────────────

def get_trend_tracer(high: np.ndarray, low: np.ndarray, close: np.ndarray):
    length = 20
    hi     = highest(high, length)
    lo     = lowest(low,  length)
    mid    = (hi + lo) / 2
    bull   = close >= mid
    return mid, bull


# ─────────────────────────────────────────────────────────────
# 4. TREND STRENGTH
# ─────────────────────────────────────────────────────────────

def get_trend_strength(close: np.ndarray, volume: np.ndarray, high: np.ndarray,
                       low: np.ndarray, length: int = 14) -> np.ndarray:
    vm   = vwma(close, volume, length)
    at   = atr(high, low, close, length)
    raw  = np.where(at > 0, (close - vm) / at, 0.0)
    raw  = np.clip(raw, -2.0, 2.0)
    return raw / 2.0 * 100.0


# ─────────────────────────────────────────────────────────────
# 5. VOLATILITY METRIC
# ─────────────────────────────────────────────────────────────

def get_volatility(close: np.ndarray, high: np.ndarray, low: np.ndarray) -> np.ndarray:
    at  = atr(high, low, close, 14)
    raw = np.where(close > 0, at / close * 100, 0.0)
    return np.minimum(raw * 25, 100.0)


# ─────────────────────────────────────────────────────────────
# 6. SQUEEZE METRIC
# ─────────────────────────────────────────────────────────────

def get_squeeze(close: np.ndarray, high: np.ndarray, low: np.ndarray,
                bb_len: int = 45, kc_len: int = 20) -> np.ndarray:
    bb_mult = 2.0
    kc_mult = 1.5

    basis    = sma(close, bb_len)
    dev      = stdev(close, bb_len)
    bb_upper = basis + bb_mult * dev
    bb_lower = basis - bb_mult * dev

    true_range = atr(high, low, close, kc_len)
    kc_upper   = basis + kc_mult * true_range
    kc_lower   = basis - kc_mult * true_range

    sqz_on = (bb_lower > kc_lower) & (bb_upper < kc_upper)

    hi_kc  = highest(high, kc_len)
    lo_kc  = lowest(low,   kc_len)
    mid    = (hi_kc + lo_kc) / 2
    mid    = (mid + basis) / 2
    val    = linreg(close - mid, bb_len, 0)
    ab_val = np.abs(val)
    max_v  = highest(ab_val, 100)
    norm   = np.where(max_v > 0, ab_val / max_v * 100, 0.0)
    return np.where(sqz_on, norm, norm * 0.5)


# ─────────────────────────────────────────────────────────────
# 7. REVERSAL ZONES — Nadaraya-Watson Kernel Regression (NPC)
# ─────────────────────────────────────────────────────────────

def npc_weight(i: int, h: float, r: float) -> float:
    d = i * i
    return (1 + d / (2 * r * h * h)) ** (-r)


def get_reversal_zones(src: np.ndarray, high: np.ndarray, low: np.ndarray,
                       close: np.ndarray):
    n        = len(src)
    yhat_arr = np.full(n, np.nan)
    lo_outer = np.full(n, np.nan)
    hi_outer = np.full(n, np.nan)
    at       = atr(high, low, close, NPC_LEN)

    for i in range(NPC_LEN - 1, n):
        num, den = 0.0, 0.0
        for j in range(NPC_LEN):
            w   = npc_weight(j, NPC_H, NPC_R)
            num += src[i - j] * w
            den += w
        yhat = num / den if den > 0 else src[i]
        yhat_arr[i] = yhat

        err = sum(abs(src[i - j] - yhat) for j in range(NPC_LEN))
        npc_vol = (err / NPC_LEN + at[i]) / 2 if not np.isnan(at[i]) else err / NPC_LEN

        hi_outer[i] = yhat + npc_vol * NPC_OUTER   # SELL SL
        lo_outer[i] = yhat - npc_vol * NPC_OUTER   # BUY  SL

    return yhat_arr, hi_outer, lo_outer


# ─────────────────────────────────────────────────────────────
# 8. ADAPTIVE SUPERTREND (K-Means over ATR multipliers)
# ─────────────────────────────────────────────────────────────

def adaptive_supertrend(close: np.ndarray, high: np.ndarray, low: np.ndarray,
                        length: int = ATR_LENGTH,
                        min_mult: float = MIN_MULT,
                        max_mult: float = MAX_MULT,
                        step: float     = STEP_ST,
                        perf_alpha: int = PERF_ALPHA,
                        max_iter: int   = MAX_ITER,
                        max_data: int   = MAX_DATA):
    n       = len(close)
    at      = atr(high, low, close, length)
    hl2     = (high + low) / 2

    factors = np.arange(min_mult, max_mult + step / 2, step)
    k_cnt   = len(factors)

    # Per-factor state
    upper = hl2.copy()
    lower = hl2.copy()
    trend = np.zeros(k_cnt, dtype=int)
    perf  = np.zeros(k_cnt, dtype=float)
    output = np.zeros(k_cnt, dtype=float)
    # Working arrays for each factor across bars
    upper_f  = np.full((k_cnt, n), np.nan)
    lower_f  = np.full((k_cnt, n), np.nan)
    trend_f  = np.zeros((k_cnt, n), dtype=int)
    output_f = np.full((k_cnt, n), np.nan)
    perf_f   = np.full((k_cnt, n), 0.0)

    alpha = 2.0 / (perf_alpha + 1)

    # Initialise
    for k, fac in enumerate(factors):
        upper_f[k, 0] = hl2[0]
        lower_f[k, 0] = hl2[0]

    for i in range(1, n):
        prev_close = close[i - 1]
        cur_at     = at[i] if not np.isnan(at[i]) else 0.0
        for k, fac in enumerate(factors):
            up  = hl2[i] + cur_at * fac
            dn  = hl2[i] - cur_at * fac

            prev_upper = upper_f[k, i - 1] if not np.isnan(upper_f[k, i - 1]) else up
            prev_lower = lower_f[k, i - 1] if not np.isnan(lower_f[k, i - 1]) else dn

            new_upper = min(up, prev_upper) if prev_close < prev_upper else up
            new_lower = max(dn, prev_lower) if prev_close > prev_lower else dn

            upper_f[k, i] = new_upper
            lower_f[k, i] = new_lower

            prev_trend = trend_f[k, i - 1]
            if close[i] > new_upper:
                trend_f[k, i] = 1
            elif close[i] < new_lower:
                trend_f[k, i] = 0
            else:
                trend_f[k, i] = prev_trend

            prev_out   = output_f[k, i - 1] if not np.isnan(output_f[k, i - 1]) else new_upper
            output_f[k, i] = new_lower if trend_f[k, i] == 1 else new_upper

            diff = math.copysign(1, prev_close - prev_out) if prev_out != 0 else 0
            ret  = (close[i] - close[i - 1]) * diff
            perf_f[k, i] = perf_f[k, i - 1] + alpha * (ret - perf_f[k, i - 1])

    # K-Means — select best factor cluster per bar
    target_factor_arr = np.full(n, np.nan)
    perf_idx_arr      = np.full(n, np.nan)
    den_st_arr        = atr(high, low, close, perf_alpha)  # ema of |close diff|
    den_st_arr2 = np.full(n, np.nan)
    abs_diff = np.abs(np.diff(close, prepend=np.nan))
    den_st_arr2 = ema(abs_diff, perf_alpha)

    for i in range(n - 1, n - 1 - 1, -1):   # process last bar only (real-time)
        pass

    # Build per-bar arrays
    os_arr    = np.zeros(n, dtype=int)
    upper_ts  = np.full(n, np.nan)
    lower_ts  = np.full(n, np.nan)
    ts_arr    = np.full(n, np.nan)
    perf_ama  = np.full(n, np.nan)

    # We compute K-Means for the last `max_data` bars at the final bar
    # For each bar compute target factor
    for i in range(n):
        bar_start = max(0, i - max_data + 1)
        perf_slice  = perf_f[:, i]
        factor_arr  = factors

        # K-means on perf_slice
        p25 = float(np.percentile(perf_slice, 25))
        p50 = float(np.percentile(perf_slice, 50))
        p75 = float(np.percentile(perf_slice, 75))
        centroids = [p25, p50, p75]

        for _ in range(max_iter):
            clusters = [[], [], []]
            fac_clusters = [[], [], []]
            for ki, pv in enumerate(perf_slice):
                dists = [abs(pv - c) for c in centroids]
                idx = int(np.argmin(dists))
                clusters[idx].append(pv)
                fac_clusters[idx].append(factor_arr[ki])
            new_centroids = [float(np.mean(c)) if c else centroids[j] for j, c in enumerate(clusters)]
            if new_centroids == centroids:
                break
            centroids = new_centroids

        # "Best" = cluster index 2 (highest perf)
        best_cluster = 2
        fac_list = fac_clusters[best_cluster]
        perf_list = clusters[best_cluster]
        target_factor = float(np.mean(fac_list)) if fac_list else factor_arr[-1]
        perf_val      = float(np.mean(perf_list)) if perf_list else 0.0
        den           = den_st_arr2[i] if not np.isnan(den_st_arr2[i]) else 1e-9

        target_factor_arr[i] = target_factor
        perf_idx_arr[i]      = max(perf_val, 0) / den if den > 0 else 0.0

    # Final adaptive supertrend using target_factor_arr
    for i in range(n):
        tf = target_factor_arr[i]
        if np.isnan(tf) or np.isnan(at[i]):
            upper_ts[i] = hl2[i]
            lower_ts[i] = hl2[i]
            continue

        up = hl2[i] + at[i] * tf
        dn = hl2[i] - at[i] * tf

        prev_close = close[i - 1] if i > 0 else close[i]
        prev_upper = upper_ts[i - 1] if i > 0 and not np.isnan(upper_ts[i - 1]) else up
        prev_lower = lower_ts[i - 1] if i > 0 and not np.isnan(lower_ts[i - 1]) else dn

        upper_ts[i] = min(up, prev_upper) if prev_close < prev_upper else up
        lower_ts[i] = max(dn, prev_lower) if prev_close > prev_lower else dn

        if close[i] > upper_ts[i]:
            os_arr[i] = 1
        elif close[i] < lower_ts[i]:
            os_arr[i] = 0
        else:
            os_arr[i] = os_arr[i - 1] if i > 0 else 0

        ts_arr[i] = lower_ts[i] if os_arr[i] != 0 else upper_ts[i]

        # perf_ama
        pidx = perf_idx_arr[i]
        if np.isnan(pidx):
            pidx = 0.0
        if i == 0 or np.isnan(perf_ama[i - 1]):
            perf_ama[i] = ts_arr[i]
        else:
            perf_ama[i] = perf_ama[i - 1] + pidx * (ts_arr[i] - perf_ama[i - 1])

    return os_arr, ts_arr, perf_ama


# ─────────────────────────────────────────────────────────────
# 9. NEURAL NETWORK GRADER
# ─────────────────────────────────────────────────────────────

def nn_tanh(x: float) -> float:
    ex = math.exp(2 * x)
    return (ex - 1) / (ex + 1)


def dema(series: np.ndarray, length: int) -> np.ndarray:
    e1 = ema(series, length)
    return 2 * e1 - ema(e1, length)


def nn_amf_score(close: np.ndarray, high: np.ndarray, low: np.ndarray, is_buy: bool, i: int) -> float:
    at14 = atr(high, low, close, 14)
    fast = dema(close, 7)
    n_val = (1.5 * (close[i] - fast[i]) / at14[i]) if at14[i] > 0 else 0.0
    n_arr = n_val  # scalar proxy
    line = float(sma(np.array([n_val * 100] * 10), 10)[-1])
    sig  = float(ema(np.array([line] * 10), 10)[-1])
    cross = 1.0 if line > sig else (-1.0 if line < sig else 0.0)
    if is_buy:
        raw = (1.0 if cross > 0 or line > sig else (-0.6 if line < sig else 0.0))
    else:
        raw = (1.0 if cross < 0 or line < sig else (-0.6 if line > sig else 0.0))
    return max(-1.0, min(1.0, raw))


def nn_amf_score_arr(close: np.ndarray, high: np.ndarray, low: np.ndarray, is_buy: bool) -> np.ndarray:
    n    = len(close)
    at14 = atr(high, low, close, 14)
    fast = dema(close, 7)
    n_arr  = np.where(at14 > 0, 1.5 * (close - fast) / at14, 0.0)
    line  = sma(n_arr * 100, 10)
    sig   = ema(line, 10)
    result = np.full(n, 0.0)
    for i in range(n):
        if np.isnan(line[i]) or np.isnan(sig[i]):
            result[i] = 0.0
            continue
        cross = 1.0 if (i > 0 and line[i] > sig[i] and line[i-1] <= sig[i-1]) else \
               (-1.0 if (i > 0 and line[i] < sig[i] and line[i-1] >= sig[i-1]) else 0.0)
        if is_buy:
            raw = (1.0 if cross > 0 or line[i] > sig[i] else (-0.6 if line[i] < sig[i] else 0.0))
        else:
            raw = (1.0 if cross < 0 or line[i] < sig[i] else (-0.6 if line[i] > sig[i] else 0.0))
        result[i] = max(-1.0, min(1.0, raw))
    return result


def nn_alma_score_arr(close: np.ndarray, high: np.ndarray, low: np.ndarray, is_buy: bool) -> np.ndarray:
    n    = len(close)
    at14 = atr(high, low, close, 14)
    fast = alma(close, 20, 0.85, 6.0)
    slow = alma(close, 20, 0.77, 6.0)
    gap  = np.abs(fast - slow) / np.where(at14 > 0, at14, np.inf)
    wide = gap > 0.1
    result = np.full(n, 0.0)
    for i in range(n):
        if np.isnan(fast[i]) or np.isnan(slow[i]):
            continue
        if is_buy:
            result[i] = (0.5 if fast[i] > slow[i] else (-0.8 if wide[i] else -0.2))
        else:
            result[i] = (0.5 if fast[i] < slow[i] else (-0.8 if wide[i] else -0.2))
    return result


def nn_swing_score_arr(high: np.ndarray, low: np.ndarray, is_buy: bool) -> np.ndarray:
    n   = len(high)
    rh  = highest(high, 20)
    rl  = lowest(low,   20)
    ph2 = np.full(n, np.nan)
    pl2 = np.full(n, np.nan)
    for i in range(20, n):
        ph2[i] = np.max(high[i - 40: i - 20]) if i >= 40 else np.max(high[:i-20]) if i > 20 else np.nan
        pl2[i] = np.min(low[i  - 40: i - 20]) if i >= 40 else np.min(low[:i-20])  if i > 20 else np.nan
    result = np.full(n, -0.1)
    for i in range(n):
        if np.isnan(rh[i]) or np.isnan(rl[i]) or np.isnan(ph2[i]) or np.isnan(pl2[i]):
            continue
        bull_ = rh[i] > ph2[i] and rl[i] > pl2[i]
        bear_ = rh[i] < ph2[i] and rl[i] < pl2[i]
        if is_buy:
            result[i] = (0.6 if bull_ else (-0.7 if bear_ else -0.1))
        else:
            result[i] = (0.6 if bear_ else (-0.7 if bull_ else -0.1))
    return result


def nn_regime_score_arr(high: np.ndarray, low: np.ndarray, close: np.ndarray, is_buy: bool) -> np.ndarray:
    n    = len(close)
    tr   = np.full(n, np.nan)
    pdm  = np.full(n, np.nan)
    mdm  = np.full(n, np.nan)
    for i in range(1, n):
        tr[i]  = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        pdm_r  = high[i] - high[i-1]
        mdm_r  = low[i-1] - low[i]
        pdm[i] = max(pdm_r, 0.0) if pdm_r > mdm_r else 0.0
        mdm[i] = max(mdm_r, 0.0) if mdm_r > pdm_r else 0.0
    atr30 = rma(tr,  30)
    with np.errstate(divide='ignore', invalid='ignore'):
        pdi   = np.where(atr30 > 0, 100 * rma(pdm, 30) / atr30, 0.0)
        mdi   = np.where(atr30 > 0, 100 * rma(mdm, 30) / atr30, 0.0)
        denom = pdi + mdi
        dx    = np.where(denom > 0, np.abs(pdi - mdi) / denom * 100, 0.0)
    adx   = rma(dx, 30)
    result = np.full(n, -0.2)
    for i in range(n):
        if np.isnan(adx[i]):
            continue
        trend_ = adx[i] >= 60
        bull_  = trend_ and pdi[i] > mdi[i]
        bear_  = trend_ and mdi[i] > pdi[i]
        if is_buy:
            result[i] = (0.7 if bull_ else (-0.8 if bear_ else -0.2))
        else:
            result[i] = (0.7 if bear_ else (-0.8 if bull_ else -0.2))
    return result


def nn_sr_score_arr(high: np.ndarray, low: np.ndarray, close: np.ndarray, is_buy: bool) -> np.ndarray:
    """Support/Resistance score — pivot high/low lookback."""
    n    = len(close)
    at14 = atr(high, low, close, 14)
    lb   = 10

    # Detect pivot highs / lows
    ph_arr = np.full(n, np.nan)
    pl_arr = np.full(n, np.nan)
    for i in range(lb, n - lb):
        if high[i] == np.max(high[i-lb:i+lb+1]):
            ph_arr[i] = high[i]
        if low[i]  == np.min(low[i-lb:i+lb+1]):
            pl_arr[i] = low[i]

    res = np.full(n, np.nan)
    sup = np.full(n, np.nan)
    for i in range(1, n):
        res[i] = ph_arr[i] if not np.isnan(ph_arr[i]) else res[i-1]
        sup[i] = pl_arr[i] if not np.isnan(pl_arr[i]) else sup[i-1]

    result = np.full(n, 0.0)
    for i in range(5, n):
        if np.isnan(at14[i]) or at14[i] == 0:
            continue
        rd = abs(close[i] - res[i]) / at14[i] if not np.isnan(res[i]) else 999.0
        sd = abs(close[i] - sup[i]) / at14[i] if not np.isnan(sup[i]) else 999.0
        n_res = rd <= 0.5
        n_sup = sd <= 0.5
        brk_res = (not np.isnan(res[i])
                   and close[i]   >  res[i]
                   and close[i-5] <= res[i])
        brk_sup = (not np.isnan(sup[i])
                   and close[i]   <  sup[i]
                   and close[i-5] >= sup[i])
        if is_buy:
            result[i] = (0.8 if brk_res else (0.5 if n_sup else (-0.6 if n_res else 0.0)))
        else:
            result[i] = (0.8 if brk_sup else (0.5 if n_res else (-0.6 if n_sup else 0.0)))
    return result


def nn_volume_score_arr(close: np.ndarray, high: np.ndarray, low: np.ndarray,
                        volume: np.ndarray, is_buy: bool) -> np.ndarray:
    n  = len(close)
    uv = np.zeros(n)
    dv = np.zeros(n)
    for i in range(1, n):
        v = volume[i] if not np.isnan(volume[i]) else 0.0
        c, o, h, l = close[i], close[i], high[i], low[i]  # approx
        if (c - l) > (h - c):
            uv[i] = v
        elif (c - l) < (h - c):
            dv[i] = -v
        elif close[i] > close[i-1]:
            uv[i] = v
        elif close[i] < close[i-1]:
            dv[i] = -v
        else:
            uv[i] = max(uv[i-1], 0)
            dv[i] = min(dv[i-1], 0)
    tot   = uv + np.abs(dv)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(tot > 0, (uv - np.abs(dv)) / tot, 0.0)
    return ratio if is_buy else -ratio


def nn_score(amf, alma_, sr, swing, regime, vol) -> float:
    h1 = nn_tanh(1.2*amf + 0.4*alma_ + 0.5*sr + 0.4*swing + 0.7*regime + 0.8*vol + 0.10)
    h2 = nn_tanh(1.4*amf + 0.3*alma_ + 0.6*sr + 0.3*swing + 0.8*regime + 0.6*vol + 0.05)
    h3 = nn_tanh(1.6*amf + 0.4*alma_ + 0.4*sr + 0.5*swing + 0.6*regime + 1.0*vol + 0.15)
    raw = 0.4*h1 + 0.4*h2 + 0.4*h3
    return 1.0 / (1.0 + math.exp(-raw))


def nn_grade(sc: float) -> str:
    if sc >= 0.80: return "A+"
    if sc >= 0.76: return "A"
    if sc >= 0.65: return "B"
    if sc >= 0.37: return "C"
    if sc >= 0.28: return "D"
    return "F"


def nn_label(sc: float, is_buy: bool) -> str:
    strength = ("Strong" if sc >= 0.76 else "Good" if sc >= 0.65 else
                "Moderate" if sc >= 0.37 else "Weak")
    return f"{strength} {'Buy' if is_buy else 'Sell'}"


# ─────────────────────────────────────────────────────────────
# 10. EXIT MARKERS
# ─────────────────────────────────────────────────────────────

def get_exit_markers(close: np.ndarray, high: np.ndarray, low: np.ndarray):
    rsi14     = rsi(close, 14)
    ema_fast  = ema(close, 9)
    ema_slow  = ema(close, 21)
    ema_cross_dn = crossunder(ema_fast, ema_slow)
    ema_cross_up = crossover(ema_fast, ema_slow)
    rsi_high  = highest(rsi14, 5)
    rsi_low   = lowest(rsi14,  5)
    bearish   = ema_cross_dn & (rsi_high > 60)
    bullish   = ema_cross_up & (rsi_low  < 40)
    return bearish, bullish


# ─────────────────────────────────────────────────────────────
# MAIN SIGNAL COMPUTATION
# ─────────────────────────────────────────────────────────────

def compute_signals(df: pd.DataFrame):
    close  = df["close"].values.astype(float)
    high   = df["high"].values.astype(float)
    low    = df["low"].values.astype(float)
    volume = df["volume"].values.astype(float)
    hlc3   = (high + low + close) / 3

    n = len(close)
    if n < 300:
        raise ValueError("Not enough data to compute signals (need ≥ 300 bars)")

    # Indicators
    smart_trail_inner, smart_trail_outer, st_dir, st_tfl, st_bull = \
        get_smart_trail(close, ST_LEN, ST_SMOOTH, ST_MULT1, ST_MULT2)
    tc_line, tc_bull = get_trend_catcher(close)
    tt_mid, tt_bull  = get_trend_tracer(high, low, close)
    ts_metric        = get_trend_strength(close, volume, high, low)
    vol_metric       = get_volatility(close, high, low)
    sqz_metric       = get_squeeze(close, high, low)
    _, npc_hi_outer, npc_lo_outer = get_reversal_zones(hlc3, high, low, close)
    os_ts, ts, perf_ama = adaptive_supertrend(close, high, low)
    bearish_exit, bullish_exit = get_exit_markers(close, high, low)

    # NN score arrays (buy and sell sides)
    amf_buy  = nn_amf_score_arr(close,  high, low, True)
    amf_sell = nn_amf_score_arr(close,  high, low, False)
    alma_buy  = nn_alma_score_arr(close, high, low, True)
    alma_sell = nn_alma_score_arr(close, high, low, False)
    sr_buy    = nn_sr_score_arr(high, low, close, True)
    sr_sell   = nn_sr_score_arr(high, low, close, False)
    swing_buy  = nn_swing_score_arr(high, low, True)
    swing_sell = nn_swing_score_arr(high, low, False)
    regime_buy  = nn_regime_score_arr(high, low, close, True)
    regime_sell = nn_regime_score_arr(high, low, close, False)
    vol_buy  = nn_volume_score_arr(close, high, low, volume, True)
    vol_sell = nn_volume_score_arr(close, high, low, volume, False)

    # ── Signal detection on the LAST CONFIRMED (closed) candle ────────────────
    # We use index -2 (second last) since -1 is the live incomplete candle
    i = n - 2

    def prev(arr, idx): return arr[idx - 1] if idx > 0 else arr[idx]

    os_now  = int(os_ts[i])
    os_prev = int(os_ts[i - 1]) if i > 0 else os_now

    signal    = None
    sc        = None
    sl        = None
    tp1       = None
    tp2       = None
    tp3       = None
    entry     = close[i]
    direction = None

    # STRONG BUY condition: os_ts crossed up (0→1)
    if os_now > os_prev:
        sc_buy = nn_score(amf_buy[i], alma_buy[i], sr_buy[i],
                          swing_buy[i], regime_buy[i], vol_buy[i])
        if sc_buy >= STRONG_THRESHOLD:
            signal    = "STRONG BUY"
            sc        = sc_buy
            direction = "LONG"
            sl        = float(npc_lo_outer[i])
            rk        = entry - sl
            tp1       = entry + 1.0 * rk
            tp2       = entry + 2.0 * rk
            tp3       = entry + 3.0 * rk

    # STRONG SELL condition: os_ts crossed down (1→0)
    elif os_now < os_prev:
        sc_sell = nn_score(amf_sell[i], alma_sell[i], sr_sell[i],
                           swing_sell[i], regime_sell[i], vol_sell[i])
        if sc_sell >= STRONG_THRESHOLD:
            signal    = "STRONG SELL"
            sc        = sc_sell
            direction = "SHORT"
            sl        = float(npc_hi_outer[i])
            rk        = sl - entry
            tp1       = entry - 1.0 * rk
            tp2       = entry - 2.0 * rk
            tp3       = entry - 3.0 * rk

    # Dashboard values (last confirmed bar)
    ts_val    = float(ts_metric[i])
    ts_label  = ("Strongly Bullish"  if ts_val >= 75  else
                 "Mildly Bullish"    if ts_val >= 30  else
                 "Neutral"           if ts_val >= -30 else
                 "Mildly Bearish"    if ts_val >= -75 else
                 "Strongly Bearish")
    vol_text  = ("Stable"   if vol_metric[i] < 30 else
                 "Moderate" if vol_metric[i] < 80 else
                 "Volatile")
    sqz_val   = float(sqz_metric[i])

    result = {
        "entry":        entry,
        "signal":       signal,
        "nn_score":     sc,
        "nn_grade":     nn_grade(sc) if sc is not None else None,
        "nn_label":     nn_label(sc, direction == "LONG") if sc is not None else None,
        "direction":    direction,
        "sl":           sl,
        "tp1":          tp1,
        "tp2":          tp2,
        "tp3":          tp3,
        "trend_bias":   ts_label,
        "trend_str":    round(ts_val, 2),
        "volatility":   vol_text,
        "squeeze_pct":  round(sqz_val, 1),
        "smart_trail":  st_dir[i],
        "tc_bullish":   bool(tc_bull[i]),
        "tt_bullish":   bool(tt_bull[i]),
        "exit_long":    bool(bearish_exit[i] if i >= 0 else False),
        "exit_short":   bool(bullish_exit[i] if i >= 0 else False),
    }
    return result


# ─────────────────────────────────────────────────────────────
# TELEGRAM
# ─────────────────────────────────────────────────────────────

def send_telegram(msg: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        log.warning("Telegram credentials not set — skipping message.")
        return
    url  = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "Markdown"}
    try:
        r = requests.post(url, data=data, timeout=10)
        r.raise_for_status()
        log.info("Telegram message sent.")
    except Exception as e:
        log.error(f"Failed to send Telegram message: {e}")


def format_signal_message(coin: str, sig: dict, candle_time: str) -> str:
    if sig["signal"] is None:
        return None

    emoji = "🟢" if "BUY" in sig["signal"] else "🔴"
    lines = [
        f"{emoji} *{sig['signal']} — {coin}*",
        f"🕐 Candle: `{candle_time}` (1h closed)",
        f"📊 Entry:  `${sig['entry']:,.2f}`",
        f"",
        f"🎯 *Targets*",
        f"  TP1: `${sig['tp1']:,.2f}`",
        f"  TP2: `${sig['tp2']:,.2f}`",
        f"  TP3: `${sig['tp3']:,.2f}`",
        f"  SL:  `${sig['sl']:,.2f}`",
        f"",
        f"🧠 *AI Signal Quality*",
        f"  Grade: `{sig['nn_grade']}` | Score: `{sig['nn_score']:.3f}`",
        f"  Label: `{sig['nn_label']}`",
        f"",
        f"📈 *Market Context*",
        f"  Trend Bias:    `{sig['trend_bias']}`",
        f"  Trend Strength: `{sig['trend_str']:+.1f}%`",
        f"  Volatility:    `{sig['volatility']}`",
        f"  Squeeze:       `{sig['squeeze_pct']:.1f}%`",
        f"  Smart Trail:   `{sig['smart_trail'].upper()}`",
    ]
    if sig["exit_long"]:
        lines.append("\n⚠️ *Long Exit signal active (EMA cross + RSI)*")
    if sig["exit_short"]:
        lines.append("\n⚠️ *Short Exit signal active (EMA cross + RSI)*")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────

def run_once():
    for coin, symbol in SYMBOL_MAP.items():
        log.info(f"Processing {coin} ({symbol}) …")
        try:
            df  = fetch_ohlcv(symbol, CANDLE_RESOLUTION, CANDLE_LIMIT)
            sig = compute_signals(df)

            # Candle timestamp (second-to-last bar = confirmed closed)
            idx   = len(df) - 2
            ts_unix = int(df["time"].iloc[idx])
            ts_str  = datetime.fromtimestamp(ts_unix, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

            if sig["signal"]:
                msg = format_signal_message(coin, sig, ts_str)
                log.info(f"\n{msg}")
                send_telegram(msg)
            else:
                log.info(
                    f"{coin} | No signal | Trend: {sig['trend_bias']} | "
                    f"Str: {sig['trend_str']:+.1f}% | Vol: {sig['volatility']} | "
                    f"Sqz: {sig['squeeze_pct']:.1f}%"
                )
        except Exception as e:
            log.error(f"Error processing {coin}: {e}", exc_info=True)


def main():
    log.info("Confluence Suite Bot started — 1h BTC/ETH on Delta Exchange")
    log.info(f"Settings: sensitivity={SENSITIVITY}, atr_len={ATR_LENGTH}, "
             f"nn_threshold={STRONG_THRESHOLD}")
    while True:
        run_once()
        log.info(f"Sleeping {POLL_INTERVAL_SECONDS}s until next candle …")
        time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
