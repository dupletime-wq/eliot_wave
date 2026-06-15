from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


# =============================================================================
# Config
# =============================================================================

APP_TITLE = "Elliott Wave Lab"
DEFAULT_TICKERS = "AAPL, QQQ, SPY, 005930.KS, 000660.KS"
OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
DEFAULT_PERIOD = "10y"
DEFAULT_INTERVAL = "1d"
DEFAULT_SENSITIVITIES = (0.03, 0.05, 0.08)
DEFAULT_ATR_MULTIPLE = 1.5
MAX_PIVOTS = 80
MAX_ANALYSIS_BARS = 2600
TOP_SCENARIOS = 12
DISCLAIMER = (
    "Research tool only. Elliott Wave counts are probabilistic rule scores, "
    "not investment advice or guaranteed forecasts."
)
FIB_RETRACEMENTS = np.array([0.236, 0.382, 0.5, 0.618, 0.786, 1.0])
FIB_EXTENSIONS = np.array([0.618, 1.0, 1.272, 1.382, 1.618, 2.0, 2.618])


# =============================================================================
# Data models
# =============================================================================


@dataclass(frozen=True)
class WavePivot:
    index: int
    date: pd.Timestamp
    price: float
    kind: str


@dataclass
class WaveCandidate:
    pattern: str
    direction: str
    labels: tuple[str, ...]
    pivots: tuple[WavePivot, ...]
    score_raw: float
    target: float | None
    invalidation: float | None
    rule_hits: list[str] = field(default_factory=list)
    rule_misses: list[str] = field(default_factory=list)
    extension: str = ""
    alternation: str = ""
    summary: str = ""


@dataclass
class ScenarioScore:
    candidate: WaveCandidate
    probability_pct: float
    indicator_bonus: float
    indicator_notes: list[str]
    confidence_pct: float
    relative_weight_pct: float = 0.0


@dataclass
class BacktestStats:
    pattern: str
    samples: int
    target_hit_rate: float
    invalidation_hit_rate: float
    unresolved_rate: float
    avg_bars_held: float
    expectancy_r: float
    avg_return_pct: float


@dataclass
class TickerAnalysis:
    ticker: str
    df: pd.DataFrame
    indicators: pd.DataFrame
    pivots: list[WavePivot]
    scenarios: list[ScenarioScore]
    elapsed_ms: float
    errors: list[str] = field(default_factory=list)


# =============================================================================
# Utility
# =============================================================================


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    if denominator is None or denominator == 0 or pd.isna(denominator):
        return default
    return float(numerator) / float(denominator)


def signed_direction(a: float, b: float) -> int:
    return 1 if b >= a else -1


def direction_name(direction: int | str) -> str:
    if direction in (1, "up", "bullish"):
        return "Bullish"
    if direction in (-1, "down", "bearish"):
        return "Bearish"
    return "Neutral"


def nearest_fib_score(value: float, fibs: Iterable[float], tolerance: float = 0.18) -> tuple[float, float]:
    if value is None or pd.isna(value) or math.isinf(value):
        return 0.0, float("nan")
    arr = np.array(list(fibs), dtype=float)
    idx = int(np.argmin(np.abs(arr - value)))
    nearest = float(arr[idx])
    distance = abs(float(value) - nearest)
    score = clamp(1.0 - distance / tolerance, 0.0, 1.0)
    return score, nearest


def parse_tickers(ticker_text: str) -> list[str]:
    tokens = re.split(r"[\s,;]+", ticker_text.strip())
    seen: set[str] = set()
    tickers: list[str] = []
    for token in tokens:
        symbol = token.strip().upper()
        if not symbol:
            continue
        symbol = symbol.replace("/", "-")
        if symbol not in seen:
            seen.add(symbol)
            tickers.append(symbol)
    return tickers


def candidate_symbols_for_input(symbol: str) -> list[str]:
    if re.fullmatch(r"\d{6}", symbol):
        return [f"{symbol}.KS", f"{symbol}.KQ"]
    return [symbol]


def format_price(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "-"
    if abs(float(value)) >= 1000:
        return f"{float(value):,.0f}"
    return f"{float(value):,.2f}"


def price_move_pct(current: float, target: float | None, direction: str) -> float | None:
    if target is None or pd.isna(target) or current <= 0:
        return None
    if direction == "Bearish":
        return (current - float(target)) / current * 100
    return (float(target) - current) / current * 100


def pattern_name_ko(pattern: str) -> str:
    if "Impulse" in pattern:
        if "Truncated" in pattern:
            return "충격파 5파 절단 후보"
        return "충격파 1-2-3-4-5 후보"
    if "Ending Diagonal" in pattern:
        return "종결 다이애고널 후보"
    if "Triangle" in pattern:
        return "삼각 조정 A-B-C-D-E 후보"
    if "Zigzag" in pattern:
        return "ABC 지그재그 조정 후보"
    if "Flat" in pattern:
        return "ABC 플랫 조정 후보"
    if "Triple Three" in pattern:
        return "복합 조정 W-X-Y-X-Z 후보"
    if "W-X-Y" in pattern:
        return "복합 조정 W-X-Y 후보"
    return pattern


def direction_ko(direction: str) -> str:
    if direction == "Bullish":
        return "상방"
    if direction == "Bearish":
        return "하방"
    return "중립"


def confidence_ko(value: float) -> str:
    if value >= 82:
        return "높음"
    if value >= 68:
        return "보통 이상"
    if value >= 55:
        return "보통"
    return "낮음"


def build_korean_guide(analysis: TickerAnalysis, scenario: ScenarioScore) -> dict[str, str]:
    candidate = scenario.candidate
    current = float(analysis.indicators["Close"].iloc[-1])
    move_pct = price_move_pct(current, candidate.target, candidate.direction)
    invalid_pct = price_move_pct(current, candidate.invalidation, "Bearish" if candidate.direction == "Bullish" else "Bullish")
    pattern_ko = pattern_name_ko(candidate.pattern)
    direction_text = direction_ko(candidate.direction)
    confidence_text = confidence_ko(scenario.confidence_pct)

    if move_pct is None:
        target_sentence = "목표가 계산이 불안정해서 목표 구간은 표시하지 않습니다."
    elif candidate.direction == "Bullish":
        if move_pct > 0:
            target_sentence = f"현재가 기준 1차 목표가는 {format_price(candidate.target)}로, 약 {move_pct:.1f}% 상방 여력이 있습니다."
        else:
            target_sentence = f"목표가 {format_price(candidate.target)}가 이미 현재가 근처이거나 아래라서, 추가 상승 여력은 낮게 봅니다."
    elif move_pct > 0:
        target_sentence = f"이 후보는 상승보다 조정/하락 시나리오에 가깝고, 하방 목표는 {format_price(candidate.target)} 부근입니다."
    else:
        target_sentence = f"하방 목표가 {format_price(candidate.target)}가 이미 충족됐을 수 있어 재카운트가 필요합니다."

    if candidate.invalidation is None or pd.isna(candidate.invalidation):
        risk_sentence = "명확한 무효화 가격이 없어 포지션 판단에는 보수적으로 접근해야 합니다."
    else:
        risk_word = "이탈" if candidate.direction == "Bullish" else "돌파"
        risk_sentence = f"{format_price(candidate.invalidation)} {risk_word} 시 이 파동 카운트는 약해집니다."
        if invalid_pct is not None and invalid_pct > 0:
            risk_sentence += f" 현재가와 무효화선의 거리는 약 {invalid_pct:.1f}%입니다."

    if "Impulse" in candidate.pattern:
        state_sentence = "추세 방향의 5파 구조가 가장 유력한 후보입니다. 연장 파동이 확인되면 목표가는 더 멀어질 수 있지만, 5파 후반이면 변동성 확대와 되돌림도 같이 봐야 합니다."
    elif "Diagonal" in candidate.pattern:
        state_sentence = "다이애고널은 추세 막바지에서 자주 나오는 쐐기형 구조입니다. 추가 상승이 가능해도 급한 되돌림 리스크를 함께 봐야 합니다."
    elif "Triangle" in candidate.pattern:
        state_sentence = "삼각 조정은 에너지를 모으는 구간으로 해석합니다. E파 부근 이후에는 방향성 돌파가 중요합니다."
    elif "Flat" in candidate.pattern or "Zigzag" in candidate.pattern:
        state_sentence = "ABC 조정 구조 후보입니다. 조정이 끝나는지, 아니면 더 복잡한 W-X-Y로 이어지는지 확인이 필요합니다."
    else:
        state_sentence = "복합 조정 후보입니다. 가격 목표보다 시간 소모와 박스권 돌파 여부가 더 중요합니다."

    if candidate.extension:
        extension_sentence = f"연장 판단: {candidate.extension}. 연장파 이후에는 비연장 파동과의 균형, 피보나치 확장 목표를 같이 봅니다."
    else:
        extension_sentence = "뚜렷한 연장파 신호는 약합니다."

    return {
        "headline": f"{analysis.ticker}는 현재 {pattern_ko}로 해석될 가능성이 가장 큽니다.",
        "state": f"방향은 {direction_text}, 종합 기술 확률은 {scenario.probability_pct:.1f}%입니다. 근거 점수는 {confidence_text}({scenario.confidence_pct:.1f}%)이고, 후보 내 비중은 {scenario.relative_weight_pct:.1f}%입니다.",
        "target": target_sentence,
        "risk": risk_sentence,
        "wave": state_sentence,
        "extension": extension_sentence,
        "indicator": "보조지표 확인: " + ", ".join(scenario.indicator_notes[:4]),
    }


# =============================================================================
# Data
# =============================================================================


def normalize_ohlcv_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=OHLCV_COLUMNS)

    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [str(col[-1]) for col in out.columns]
    out.columns = [str(col).strip().title().replace("Adj Close", "Close") for col in out.columns]

    rename_map = {}
    for column in out.columns:
        compact = column.replace(" ", "").lower()
        if compact == "adjclose":
            rename_map[column] = "Close"
        elif compact in {"open", "high", "low", "close", "volume"}:
            rename_map[column] = column.title()
    out = out.rename(columns=rename_map)

    missing = [column for column in OHLCV_COLUMNS if column not in out.columns]
    for column in missing:
        out[column] = np.nan if column != "Volume" else 0

    out = out[OHLCV_COLUMNS].apply(pd.to_numeric, errors="coerce")
    out = out.dropna(subset=["Close", "High", "Low"])
    out["Open"] = out["Open"].fillna(out["Close"])
    out["Volume"] = out["Volume"].fillna(0)
    out = out.loc[~out.index.duplicated(keep="last")].sort_index()
    out.index = pd.to_datetime(out.index).tz_localize(None) if getattr(out.index, "tz", None) else pd.to_datetime(out.index)
    return out


def split_yfinance_frame(raw: pd.DataFrame, requested: list[str]) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    if raw is None or raw.empty:
        return {ticker: pd.DataFrame(columns=OHLCV_COLUMNS) for ticker in requested}

    if isinstance(raw.columns, pd.MultiIndex):
        levels = raw.columns.names
        level0 = [str(item) for item in raw.columns.get_level_values(0).unique()]
        level1 = [str(item) for item in raw.columns.get_level_values(1).unique()]
        level0_has_prices = len(set(OHLCV_COLUMNS).intersection(level0)) >= 3
        level1_has_prices = len(set(OHLCV_COLUMNS).intersection(level1)) >= 3

        for ticker in requested:
            frame = pd.DataFrame(columns=OHLCV_COLUMNS)
            try:
                if level0_has_prices:
                    frame = raw.xs(ticker, axis=1, level=1, drop_level=True)
                elif level1_has_prices:
                    frame = raw.xs(ticker, axis=1, level=0, drop_level=True)
                else:
                    if ticker in level0:
                        frame = raw.xs(ticker, axis=1, level=0, drop_level=True)
                    elif ticker in level1:
                        frame = raw.xs(ticker, axis=1, level=1, drop_level=True)
            except (KeyError, ValueError):
                frame = pd.DataFrame(columns=OHLCV_COLUMNS)
            frames[ticker] = normalize_ohlcv_frame(frame)
        return frames

    only = requested[0] if requested else "UNKNOWN"
    frames[only] = normalize_ohlcv_frame(raw)
    return frames


@st.cache_data(ttl=60 * 60, show_spinner=False)
def load_price_data(ticker_text: str, period: str, refresh_nonce: int = 0) -> tuple[dict[str, pd.DataFrame], list[str], float]:
    del refresh_nonce
    started = time.perf_counter()
    inputs = parse_tickers(ticker_text)
    input_to_candidates = {symbol: candidate_symbols_for_input(symbol) for symbol in inputs}
    all_candidates = list(dict.fromkeys(candidate for candidates in input_to_candidates.values() for candidate in candidates))
    warnings: list[str] = []
    selected: dict[str, pd.DataFrame] = {}

    if not all_candidates:
        return selected, ["No ticker symbols were provided."], 0.0

    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("Install yfinance first: pip install yfinance") from exc

    try:
        raw = yf.download(
            all_candidates,
            period=period,
            interval=DEFAULT_INTERVAL,
            auto_adjust=True,
            group_by="column",
            threads=True,
            progress=False,
        )
    except Exception as exc:
        return {}, [f"yfinance download failed: {type(exc).__name__}: {exc}"], 0.0

    downloaded = split_yfinance_frame(raw, all_candidates)
    for input_symbol, candidates in input_to_candidates.items():
        chosen_symbol = None
        chosen_frame = pd.DataFrame(columns=OHLCV_COLUMNS)
        for candidate in candidates:
            frame = downloaded.get(candidate, pd.DataFrame(columns=OHLCV_COLUMNS))
            if not frame.empty:
                chosen_symbol = candidate
                chosen_frame = frame
                break
        if chosen_symbol is None:
            warnings.append(f"{input_symbol}: no daily OHLCV data returned.")
            continue
        if input_symbol != chosen_symbol:
            warnings.append(f"{input_symbol}: mapped to {chosen_symbol}.")
        selected[chosen_symbol] = chosen_frame.tail(MAX_ANALYSIS_BARS)

    elapsed_ms = (time.perf_counter() - started) * 1000
    return selected, warnings, elapsed_ms


# =============================================================================
# Indicators
# =============================================================================


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=max(2, window // 3)).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=max(2, span // 3)).mean()


def true_range(df: pd.DataFrame) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    prev_close = df["Close"].shift(1)
    parts = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1)
    return parts.max(axis=1)


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    return true_range(df).rolling(window, min_periods=max(2, window // 2)).mean()


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    macd_line = ema(series, 12) - ema(series, 26)
    signal = ema(macd_line, 9)
    hist = macd_line - signal
    return macd_line, signal, hist


def adx(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index)
    tr = true_range(df)
    atr_w = tr.rolling(window, min_periods=window).sum()
    plus_di = 100 * plus_dm.rolling(window, min_periods=window).sum() / atr_w.replace(0, np.nan)
    minus_di = 100 * minus_dm.rolling(window, min_periods=window).sum() / atr_w.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.rolling(window, min_periods=window).mean()


def obv(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df["Close"].diff()).fillna(0)
    return (direction * df["Volume"].fillna(0)).cumsum()


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = out["Close"]
    out["SMA20"] = sma(close, 20)
    out["SMA50"] = sma(close, 50)
    out["SMA200"] = sma(close, 200)
    out["EMA21"] = ema(close, 21)
    out["RSI14"] = rsi(close, 14)
    out["MACD"], out["MACD_SIGNAL"], out["MACD_HIST"] = macd(close)
    out["ATR14"] = atr(out, 14)
    out["ADX14"] = adx(out, 14)
    out["OBV"] = obv(out)
    out["OBV_SLOPE"] = out["OBV"].diff(10)
    out["VOL_MA50"] = out["Volume"].rolling(50, min_periods=10).mean()
    out["REL_VOLUME"] = out["Volume"] / out["VOL_MA50"].replace(0, np.nan)
    mid = sma(close, 20)
    std = close.rolling(20, min_periods=10).std()
    out["BB_WIDTH"] = (4 * std) / mid.replace(0, np.nan)
    return out


# =============================================================================
# Pivots
# =============================================================================


def make_pivot(df: pd.DataFrame, index: int, price: float, kind: str) -> WavePivot:
    index = int(clamp(index, 0, len(df) - 1))
    return WavePivot(index=index, date=pd.Timestamp(df.index[index]), price=float(price), kind=kind)


def compress_pivots(pivots: list[WavePivot]) -> list[WavePivot]:
    if not pivots:
        return []
    pivots = sorted(pivots, key=lambda item: item.index)
    compressed: list[WavePivot] = []
    for pivot in pivots:
        if compressed and pivot.index == compressed[-1].index:
            if pivot.kind == compressed[-1].kind:
                better_high = pivot.kind == "H" and pivot.price >= compressed[-1].price
                better_low = pivot.kind == "L" and pivot.price <= compressed[-1].price
                if better_high or better_low:
                    compressed[-1] = pivot
            continue
        if compressed and pivot.kind == compressed[-1].kind:
            prev = compressed[-1]
            if pivot.kind == "H" and pivot.price >= prev.price:
                compressed[-1] = pivot
            elif pivot.kind == "L" and pivot.price <= prev.price:
                compressed[-1] = pivot
            continue
        compressed.append(pivot)
    return compressed


def extract_zigzag_pivots(
    df: pd.DataFrame,
    reversal_pct: float = 0.05,
    atr_multiple: float = DEFAULT_ATR_MULTIPLE,
    max_pivots: int = MAX_PIVOTS,
) -> list[WavePivot]:
    if df.empty or len(df) < 10:
        return []

    data = df.reset_index(drop=True)
    high = data["High"].to_numpy(dtype=float)
    low = data["Low"].to_numpy(dtype=float)
    close = data["Close"].to_numpy(dtype=float)
    atr_values = atr(df, 14).bfill().ffill().fillna(0).to_numpy(dtype=float)

    def threshold(anchor_price: float, i: int) -> float:
        pct_threshold = abs(anchor_price) * reversal_pct
        atr_threshold = atr_values[i] * atr_multiple if i < len(atr_values) else 0
        return max(pct_threshold, atr_threshold, 1e-9)

    pivots: list[WavePivot] = []
    start_price = close[0]
    trend = 0
    candidate_high_idx = 0
    candidate_high = high[0]
    candidate_low_idx = 0
    candidate_low = low[0]

    for i in range(1, len(data)):
        if trend == 0:
            if high[i] >= start_price + threshold(start_price, i):
                trend = 1
                pivots.append(make_pivot(df, candidate_low_idx, candidate_low, "L"))
                candidate_high_idx, candidate_high = i, high[i]
            elif low[i] <= start_price - threshold(start_price, i):
                trend = -1
                pivots.append(make_pivot(df, candidate_high_idx, candidate_high, "H"))
                candidate_low_idx, candidate_low = i, low[i]
            else:
                if high[i] > candidate_high:
                    candidate_high_idx, candidate_high = i, high[i]
                if low[i] < candidate_low:
                    candidate_low_idx, candidate_low = i, low[i]
            continue

        if trend == 1:
            if high[i] > candidate_high:
                candidate_high_idx, candidate_high = i, high[i]
            if low[i] <= candidate_high - threshold(candidate_high, i):
                pivots.append(make_pivot(df, candidate_high_idx, candidate_high, "H"))
                trend = -1
                candidate_low_idx, candidate_low = i, low[i]
        else:
            if low[i] < candidate_low:
                candidate_low_idx, candidate_low = i, low[i]
            if high[i] >= candidate_low + threshold(candidate_low, i):
                pivots.append(make_pivot(df, candidate_low_idx, candidate_low, "L"))
                trend = 1
                candidate_high_idx, candidate_high = i, high[i]

    if trend == 1:
        pivots.append(make_pivot(df, candidate_high_idx, candidate_high, "H"))
    elif trend == -1:
        pivots.append(make_pivot(df, candidate_low_idx, candidate_low, "L"))

    pivots = compress_pivots(pivots)
    if len(pivots) > max_pivots:
        pivots = pivots[-max_pivots:]
    return pivots


# =============================================================================
# Wave rules
# =============================================================================


def pivot_prices(pivots: tuple[WavePivot, ...] | list[WavePivot]) -> list[float]:
    return [float(pivot.price) for pivot in pivots]


def swing_lengths(prices: list[float]) -> list[float]:
    return [abs(prices[i + 1] - prices[i]) for i in range(len(prices) - 1)]


def signed_swings(prices: list[float], direction: int) -> list[float]:
    return [direction * (prices[i + 1] - prices[i]) for i in range(len(prices) - 1)]


def target_from_candidate(pattern: str, prices: list[float], direction: int) -> tuple[float | None, float | None]:
    current = prices[-1]
    if len(prices) >= 6 and ("Impulse" in pattern or "Diagonal" in pattern):
        w1 = abs(prices[1] - prices[0])
        w3 = abs(prices[3] - prices[2])
        if "Truncated" in pattern:
            target = prices[3] + direction * (0.382 * max(w1, w3))
        else:
            target = prices[4] + direction * max(w1, 0.618 * w3)
        invalidation = prices[4]
        return float(target), float(invalidation)
    if len(prices) >= 4 and ("ABC" in pattern or "Zigzag" in pattern or "Flat" in pattern):
        a_len = abs(prices[1] - prices[0])
        target = prices[2] + direction * max(a_len, abs(prices[1] - prices[2]))
        invalidation = prices[0]
        return float(target), float(invalidation)
    if "Triangle" in pattern and len(prices) >= 6:
        height = max(prices) - min(prices)
        target = current + direction * height * 0.75
        invalidation = min(prices) if direction == 1 else max(prices)
        return float(target), float(invalidation)
    if ("W-X-Y" in pattern or "Triple Three" in pattern) and len(prices) >= 6:
        height = max(prices) - min(prices)
        target = current + direction * height * 0.65
        invalidation = min(prices) if direction == 1 else max(prices)
        return float(target), float(invalidation)
    return None, None


def validate_impulse(window: tuple[WavePivot, ...]) -> WaveCandidate | None:
    prices = pivot_prices(window)
    direction = signed_direction(prices[0], prices[1])
    swings = signed_swings(prices, direction)
    hits: list[str] = []
    misses: list[str] = []

    if not (swings[0] > 0 and swings[1] < 0 and swings[2] > 0 and swings[3] < 0 and swings[4] > 0):
        return None
    hits.append("Alternating 5-wave swing sequence")

    if direction * (prices[2] - prices[0]) <= 0:
        misses.append("Wave 2 retraced more than 100% of wave 1")
        return None
    hits.append("Wave 2 does not fully retrace wave 1")

    actionary = [abs(swings[0]), abs(swings[2]), abs(swings[4])]
    if actionary[1] <= min(actionary[0], actionary[2]):
        misses.append("Wave 3 is the shortest actionary wave")
        return None
    hits.append("Wave 3 is not the shortest actionary wave")

    if direction * (prices[4] - prices[1]) <= 0:
        misses.append("Wave 4 overlaps wave 1 territory")
        return None
    hits.append("Wave 4 avoids wave 1 territory")

    truncated = direction * (prices[5] - prices[3]) <= 0
    if truncated:
        hits.append("Wave 5 truncation detected")
        pattern = "Impulse 1-2-3-4-5 (Truncated 5th)"
    else:
        hits.append("Wave 5 exceeds wave 3")
        pattern = "Impulse 1-2-3-4-5"

    w1, w3, w5 = actionary
    ratios = {
        "Wave 1 extension": safe_div(w1, max(w3, w5)),
        "Wave 3 extension": safe_div(w3, max(w1, w5)),
        "Wave 5 extension": safe_div(w5, max(w1, w3)),
    }
    extension = ""
    best_extension, best_ratio = max(ratios.items(), key=lambda item: item[1])
    if best_ratio >= 1.382:
        extension = best_extension
        hits.append(extension)

    retrace2 = safe_div(abs(swings[1]), w1)
    retrace4 = safe_div(abs(swings[3]), w3)
    fib2_score, fib2 = nearest_fib_score(retrace2, FIB_RETRACEMENTS)
    fib4_score, fib4 = nearest_fib_score(retrace4, FIB_RETRACEMENTS)
    fib3_score, fib3 = nearest_fib_score(safe_div(w3, w1), FIB_EXTENSIONS, tolerance=0.28)
    fib5_score, fib5 = nearest_fib_score(safe_div(w5, w1), FIB_EXTENSIONS, tolerance=0.28)

    score = 62 + 8 * fib2_score + 7 * fib4_score + 7 * fib3_score + 5 * fib5_score
    if extension:
        score += 5
    if truncated:
        score -= 4

    wave2_style = "sharp" if retrace2 >= 0.5 else "sideways"
    wave4_style = "sharp" if retrace4 >= 0.5 else "sideways"
    alternation = "Alternation confirmed" if wave2_style != wave4_style else "Weak alternation"
    if wave2_style != wave4_style:
        score += 4
        hits.append("Wave 2 / Wave 4 alternation")

    target, invalidation = target_from_candidate(pattern, prices, direction)
    summary = (
        f"W2 retrace {retrace2:.2f} near {fib2:.3g}; "
        f"W4 retrace {retrace4:.2f} near {fib4:.3g}; "
        f"W3/W1 {safe_div(w3, w1):.2f} near {fib3:.3g}; "
        f"W5/W1 {safe_div(w5, w1):.2f} near {fib5:.3g}."
    )
    return WaveCandidate(
        pattern=pattern,
        direction=direction_name(direction),
        labels=("0", "1", "2", "3", "4", "5"),
        pivots=window,
        score_raw=clamp(score, 0, 100),
        target=target,
        invalidation=invalidation,
        rule_hits=hits,
        rule_misses=misses,
        extension=extension,
        alternation=alternation,
        summary=summary,
    )


def validate_diagonal(window: tuple[WavePivot, ...]) -> WaveCandidate | None:
    prices = pivot_prices(window)
    direction = signed_direction(prices[0], prices[1])
    swings = signed_swings(prices, direction)
    if not (swings[0] > 0 and swings[1] < 0 and swings[2] > 0 and swings[3] < 0 and swings[4] > 0):
        return None

    hits: list[str] = ["Alternating 5-wave motive sequence"]
    if direction * (prices[2] - prices[0]) <= 0:
        return None
    hits.append("Wave 2 does not fully retrace wave 1")

    actionary = [abs(swings[0]), abs(swings[2]), abs(swings[4])]
    if actionary[1] <= min(actionary[0], actionary[2]):
        return None
    hits.append("Wave 3 is not the shortest actionary wave")

    overlap = direction * (prices[4] - prices[1]) <= 0
    if not overlap:
        return None
    hits.append("Wave 4 overlaps wave 1 territory")

    line13_slope = safe_div(prices[3] - prices[1], max(1, window[3].index - window[1].index))
    line24_slope = safe_div(prices[4] - prices[2], max(1, window[4].index - window[2].index))
    same_slope_direction = np.sign(line13_slope) == np.sign(line24_slope)
    w1, w2, w3, w4, w5 = [abs(item) for item in swings]
    contraction = w3 <= w1 * 1.15 and w5 <= w3 * 1.15 and w4 <= w2 * 1.25
    expansion = w3 >= w1 * 0.85 and w5 >= w3 * 0.85 and w4 >= w2 * 0.75

    score = 58
    if same_slope_direction:
        score += 7
        hits.append("Boundary lines are coherent")
    if contraction:
        score += 12
        diagonal_type = "Contracting"
        hits.append("Contracting diagonal proportions")
    elif expansion:
        score += 6
        diagonal_type = "Expanding"
        hits.append("Expanding diagonal proportions")
    else:
        diagonal_type = "Hybrid"

    if direction * (prices[5] - prices[3]) > 0:
        score += 6
        hits.append("Wave 5 exceeds wave 3")
    else:
        hits.append("Possible fifth-wave shortfall")

    pattern = f"Ending Diagonal ({diagonal_type})"
    target, invalidation = target_from_candidate(pattern, prices, direction)
    summary = "Diagonal count allows wave 4 overlap and emphasizes wedge-like exhaustion behavior."
    return WaveCandidate(
        pattern=pattern,
        direction=direction_name(direction),
        labels=("0", "1", "2", "3", "4", "5"),
        pivots=window,
        score_raw=clamp(score, 0, 100),
        target=target,
        invalidation=invalidation,
        rule_hits=hits,
        extension="",
        alternation="Diagonal overlap",
        summary=summary,
    )


def validate_abc(window: tuple[WavePivot, ...]) -> list[WaveCandidate]:
    prices = pivot_prices(window)
    direction = signed_direction(prices[0], prices[1])
    swings = signed_swings(prices, direction)
    if not (swings[0] > 0 and swings[1] < 0 and swings[2] > 0):
        return []

    candidates: list[WaveCandidate] = []
    a_len = abs(swings[0])
    b_len = abs(swings[1])
    c_len = abs(swings[2])
    b_retrace = safe_div(b_len, a_len)
    c_vs_a = safe_div(c_len, a_len)
    b_beyond_start = direction * (prices[2] - prices[0]) < 0
    c_exceeds_a = direction * (prices[3] - prices[1]) > 0

    zig_score, zig_fib = nearest_fib_score(b_retrace, [0.382, 0.5, 0.618, 0.786], tolerance=0.22)
    c_score, c_fib = nearest_fib_score(c_vs_a, [0.618, 1.0, 1.272, 1.618], tolerance=0.28)
    if 0.2 <= b_retrace <= 0.9 and c_exceeds_a:
        score = 52 + zig_score * 18 + c_score * 16
        target, invalidation = target_from_candidate("ABC Zigzag", prices, direction)
        candidates.append(
            WaveCandidate(
                pattern="ABC Zigzag",
                direction=direction_name(direction),
                labels=("0", "A", "B", "C"),
                pivots=window,
                score_raw=clamp(score, 0, 100),
                target=target,
                invalidation=invalidation,
                rule_hits=["Three-swing corrective sequence", "C wave exceeds A wave extreme"],
                extension="",
                alternation="",
                summary=f"B retrace {b_retrace:.2f} near {zig_fib:.3g}; C/A {c_vs_a:.2f} near {c_fib:.3g}.",
            )
        )

    if b_retrace >= 0.75:
        flat_kind = "Expanded Flat" if b_beyond_start and c_exceeds_a else "Running Flat" if b_beyond_start else "Regular Flat"
        flat_c_score, flat_c_fib = nearest_fib_score(c_vs_a, [0.618, 1.0, 1.236, 1.382, 1.618], tolerance=0.32)
        score = 50 + min(b_retrace, 1.25) * 12 + flat_c_score * 16
        if b_beyond_start:
            score += 5
        target, invalidation = target_from_candidate(f"ABC {flat_kind}", prices, direction)
        candidates.append(
            WaveCandidate(
                pattern=f"ABC {flat_kind}",
                direction=direction_name(direction),
                labels=("0", "A", "B", "C"),
                pivots=window,
                score_raw=clamp(score, 0, 100),
                target=target,
                invalidation=invalidation,
                rule_hits=["Flat-style deep B wave", "Corrective A-B-C sequence"],
                extension="",
                alternation="",
                summary=f"B retrace {b_retrace:.2f}; C/A {c_vs_a:.2f} near {flat_c_fib:.3g}.",
            )
        )
    return candidates


def validate_triangle(window: tuple[WavePivot, ...]) -> WaveCandidate | None:
    prices = pivot_prices(window)
    swings_raw = [prices[i + 1] - prices[i] for i in range(len(prices) - 1)]
    alternating = all(np.sign(swings_raw[i]) != np.sign(swings_raw[i + 1]) for i in range(len(swings_raw) - 1))
    if not alternating:
        return None

    highs = [pivot.price for pivot in window if pivot.kind == "H"]
    lows = [pivot.price for pivot in window if pivot.kind == "L"]
    if len(highs) < 2 or len(lows) < 2:
        return None

    total_range = max(prices) - min(prices)
    last_range = abs(prices[-1] - prices[-2])
    if total_range <= 0:
        return None
    contraction_ratio = 1 - safe_div(last_range, total_range)
    if contraction_ratio < 0.25:
        return None

    first_direction = signed_direction(prices[0], prices[1])
    breakout_direction = -first_direction
    score = 55 + contraction_ratio * 25
    if max(highs[-2:]) <= max(highs[:2]) and min(lows[-2:]) >= min(lows[:2]):
        score += 8
    target, invalidation = target_from_candidate("Triangle A-B-C-D-E", prices, breakout_direction)
    return WaveCandidate(
        pattern="Triangle A-B-C-D-E",
        direction=direction_name(breakout_direction),
        labels=("0", "A", "B", "C", "D", "E"),
        pivots=window,
        score_raw=clamp(score, 0, 100),
        target=target,
        invalidation=invalidation,
        rule_hits=["Five alternating corrective swings", "Contracting range"],
        extension="",
        alternation="Complex correction",
        summary=f"Range contraction score {contraction_ratio:.2f}.",
    )


def validate_combination(window: tuple[WavePivot, ...], triple: bool = False) -> WaveCandidate | None:
    prices = pivot_prices(window)
    swings_raw = [prices[i + 1] - prices[i] for i in range(len(prices) - 1)]
    alternating = all(np.sign(swings_raw[i]) != np.sign(swings_raw[i + 1]) for i in range(len(swings_raw) - 1))
    if not alternating:
        return None
    total_range = max(prices) - min(prices)
    net_move = abs(prices[-1] - prices[0])
    if total_range <= 0:
        return None
    sideways = safe_div(net_move, total_range)
    overlaps = 0
    for i in range(2, len(prices)):
        prev_min = min(prices[i - 2], prices[i - 1])
        prev_max = max(prices[i - 2], prices[i - 1])
        if prev_min <= prices[i] <= prev_max:
            overlaps += 1
    overlap_ratio = safe_div(overlaps, max(1, len(prices) - 2))
    if sideways > 0.85 or overlap_ratio < 0.2:
        return None

    direction = signed_direction(prices[-2], prices[-1])
    score = 48 + (1 - sideways) * 18 + overlap_ratio * 22
    pattern = "Triple Three W-X-Y-X-Z" if triple else "Double Three W-X-Y"
    labels = ("0", "W", "X", "Y", "X2", "Z", "x", "x", "x", "x")[: len(window)] if triple else ("0", "W", "X", "Y", "x", "x", "x", "x")[: len(window)]
    target, invalidation = target_from_candidate(pattern, prices, direction)
    return WaveCandidate(
        pattern=pattern,
        direction=direction_name(direction),
        labels=tuple(labels),
        pivots=window,
        score_raw=clamp(score, 0, 100),
        target=target,
        invalidation=invalidation,
        rule_hits=["Complex corrective combination", "Overlapping sideways structure"],
        extension="Duration extension",
        alternation="Complex correction",
        summary=f"Sideways ratio {sideways:.2f}; overlap ratio {overlap_ratio:.2f}.",
    )


def generate_wave_candidates(pivots: list[WavePivot]) -> list[WaveCandidate]:
    if len(pivots) < 4:
        return []
    candidates: list[WaveCandidate] = []
    recent = pivots[-min(len(pivots), 24) :]

    for i in range(0, max(0, len(recent) - 5)):
        window = tuple(recent[i : i + 6])
        impulse = validate_impulse(window)
        if impulse is not None:
            candidates.append(impulse)
        diagonal = validate_diagonal(window)
        if diagonal is not None:
            candidates.append(diagonal)

    for i in range(0, max(0, len(recent) - 3)):
        candidates.extend(validate_abc(tuple(recent[i : i + 4])))

    for i in range(0, max(0, len(recent) - 5)):
        triangle = validate_triangle(tuple(recent[i : i + 6]))
        if triangle is not None:
            candidates.append(triangle)

    for i in range(0, max(0, len(recent) - 7)):
        combo = validate_combination(tuple(recent[i : i + 8]), triple=False)
        if combo is not None:
            candidates.append(combo)

    for i in range(0, max(0, len(recent) - 9)):
        combo = validate_combination(tuple(recent[i : i + 10]), triple=True)
        if combo is not None:
            candidates.append(combo)

    deduped: dict[tuple[str, tuple[int, ...]], WaveCandidate] = {}
    for candidate in candidates:
        key = (candidate.pattern, tuple(pivot.index for pivot in candidate.pivots))
        if key not in deduped or candidate.score_raw > deduped[key].score_raw:
            deduped[key] = candidate
    return sorted(deduped.values(), key=lambda item: item.score_raw, reverse=True)


# =============================================================================
# Scoring
# =============================================================================


def score_indicator_alignment(df: pd.DataFrame, candidate: WaveCandidate) -> tuple[float, list[str]]:
    if df.empty:
        return 0.0, ["No indicator data"]

    last = df.iloc[-1]
    close = float(last["Close"])
    bullish = candidate.direction == "Bullish"
    bearish = candidate.direction == "Bearish"
    bonus = 0.0
    notes: list[str] = []

    sma50 = last.get("SMA50", np.nan)
    sma200 = last.get("SMA200", np.nan)
    ema21 = last.get("EMA21", np.nan)
    if bullish and pd.notna(sma50) and pd.notna(sma200) and close > sma50 > sma200:
        bonus += 8
        notes.append("Price > SMA50 > SMA200")
    elif bearish and pd.notna(sma50) and pd.notna(sma200) and close < sma50 < sma200:
        bonus += 8
        notes.append("Price < SMA50 < SMA200")
    elif pd.notna(ema21) and ((bullish and close > ema21) or (bearish and close < ema21)):
        bonus += 3
        notes.append("Price aligned with EMA21")

    rsi_last = last.get("RSI14", np.nan)
    if pd.notna(rsi_last):
        if bullish and 45 <= rsi_last <= 72:
            bonus += 5
            notes.append(f"RSI constructive ({rsi_last:.1f})")
        elif bearish and 28 <= rsi_last <= 55:
            bonus += 5
            notes.append(f"RSI bearish/neutral ({rsi_last:.1f})")
        elif rsi_last > 78 or rsi_last < 22:
            bonus -= 4
            notes.append(f"RSI extreme ({rsi_last:.1f})")

    macd_hist = last.get("MACD_HIST", np.nan)
    macd_prev = df["MACD_HIST"].iloc[-5] if "MACD_HIST" in df and len(df) >= 5 else np.nan
    if pd.notna(macd_hist):
        if bullish and macd_hist > 0:
            bonus += 5
            notes.append("MACD histogram positive")
        elif bearish and macd_hist < 0:
            bonus += 5
            notes.append("MACD histogram negative")
        if pd.notna(macd_prev) and ((bullish and macd_hist > macd_prev) or (bearish and macd_hist < macd_prev)):
            bonus += 3
            notes.append("MACD momentum improving")

    adx_last = last.get("ADX14", np.nan)
    if pd.notna(adx_last):
        if adx_last >= 20:
            bonus += 4
            notes.append(f"ADX trend strength {adx_last:.1f}")
        elif "Triangle" in candidate.pattern or "W-X-Y" in candidate.pattern:
            bonus += 2
            notes.append("Low ADX supports corrective/sideways structure")

    rel_volume = last.get("REL_VOLUME", np.nan)
    if pd.notna(rel_volume):
        if rel_volume >= 1.2:
            bonus += 3
            notes.append(f"Relative volume {rel_volume:.2f}x")
        elif rel_volume < 0.65 and ("Triangle" in candidate.pattern or "Diagonal" in candidate.pattern):
            bonus += 2
            notes.append("Volume contraction")

    obv_slope = last.get("OBV_SLOPE", np.nan)
    if pd.notna(obv_slope):
        if bullish and obv_slope > 0:
            bonus += 3
            notes.append("OBV rising")
        elif bearish and obv_slope < 0:
            bonus += 3
            notes.append("OBV falling")

    bb_width = last.get("BB_WIDTH", np.nan)
    if pd.notna(bb_width) and bb_width < df["BB_WIDTH"].tail(120).quantile(0.35):
        if "Triangle" in candidate.pattern or "W-X-Y" in candidate.pattern:
            bonus += 3
            notes.append("Volatility compression")

    if not notes:
        notes.append("No strong indicator confirmation")
    return clamp(bonus, -12, 25), notes


def wave_family(pattern: str) -> str:
    if "Impulse" in pattern:
        return "Impulse"
    if "Diagonal" in pattern:
        return "Diagonal"
    if "Triangle" in pattern:
        return "Triangle"
    if "Flat" in pattern:
        return "Flat"
    if "Zigzag" in pattern:
        return "Zigzag"
    if "W-X-Y" in pattern or "Triple Three" in pattern:
        return "Combination"
    return pattern


def score_trend_context(df: pd.DataFrame, candidate: WaveCandidate) -> tuple[float, list[str]]:
    if df.empty:
        return 0.0, []
    last = df.iloc[-1]
    close = float(last["Close"])
    bullish = candidate.direction == "Bullish"
    bearish = candidate.direction == "Bearish"
    bonus = 0.0
    notes: list[str] = []
    sma20 = last.get("SMA20", np.nan)
    sma50 = last.get("SMA50", np.nan)
    sma200 = last.get("SMA200", np.nan)

    if pd.notna(sma20) and pd.notna(sma50) and pd.notna(sma200):
        if bullish and close > sma20 > sma50 > sma200:
            bonus += 14
            notes.append("장기 이평 정배열")
        elif bearish and close < sma20 < sma50 < sma200:
            bonus += 14
            notes.append("장기 이평 역배열")
        elif bullish and close > sma50 > sma200:
            bonus += 9
            notes.append("가격이 50/200일선 위")
        elif bearish and close < sma50 < sma200:
            bonus += 9
            notes.append("가격이 50/200일선 아래")

        if len(df) >= 80:
            sma50_slope = safe_div(float(df["SMA50"].iloc[-1] - df["SMA50"].iloc[-40]), close)
            sma200_slope = safe_div(float(df["SMA200"].iloc[-1] - df["SMA200"].iloc[-60]), close)
            if bullish and sma50_slope > 0 and sma200_slope > 0:
                bonus += 8
                notes.append("50/200일선 기울기 상승")
            elif bearish and sma50_slope < 0 and sma200_slope < 0:
                bonus += 8
                notes.append("50/200일선 기울기 하락")

        ma_spread = safe_div(max(sma20, sma50, sma200) - min(sma20, sma50, sma200), close)
        if ma_spread <= 0.08 and ("Triangle" in candidate.pattern or "W-X-Y" in candidate.pattern or "Flat" in candidate.pattern):
            bonus += 7
            notes.append("이동평균선 수렴")
        elif ma_spread <= 0.05:
            bonus += 4
            notes.append("이평선 수렴 후 방향 대기")

    duration = candidate.pivots[-1].index - candidate.pivots[0].index
    if duration >= 180:
        bonus += 8
        notes.append("장기 파동 구조")
    elif duration >= 90:
        bonus += 5
        notes.append("중기 이상 파동 구조")

    span = max(pivot.price for pivot in candidate.pivots) - min(pivot.price for pivot in candidate.pivots)
    span_pct = safe_div(span, close)
    if span_pct >= 0.25:
        bonus += 5
        notes.append("충분한 가격 진폭")
    elif span_pct < 0.06:
        bonus -= 5
        notes.append("파동 진폭이 작음")

    return clamp(bonus, -8, 28), notes


def pivot_indicator_value(df: pd.DataFrame, pivot: WavePivot, column: str) -> float | None:
    if column not in df or df.empty:
        return None
    index = int(clamp(pivot.index, 0, len(df) - 1))
    value = df[column].iloc[index]
    if pd.isna(value):
        return None
    return float(value)


def score_convergence_divergence(df: pd.DataFrame, candidate: WaveCandidate) -> tuple[float, list[str]]:
    bonus = 0.0
    notes: list[str] = []
    bullish = candidate.direction == "Bullish"
    bearish = candidate.direction == "Bearish"

    bb_width = df["BB_WIDTH"].iloc[-1] if "BB_WIDTH" in df and not df.empty else np.nan
    if pd.notna(bb_width) and len(df) >= 120:
        bb_quantile = df["BB_WIDTH"].tail(180).quantile(0.30)
        if pd.notna(bb_quantile) and bb_width <= bb_quantile:
            if "Triangle" in candidate.pattern or "W-X-Y" in candidate.pattern or "Diagonal" in candidate.pattern:
                bonus += 8
                notes.append("변동성 수렴")
            else:
                bonus += 4
                notes.append("볼린저 밴드 수축")

    lows = [pivot for pivot in candidate.pivots if pivot.kind == "L"]
    highs = [pivot for pivot in candidate.pivots if pivot.kind == "H"]
    if len(lows) >= 2:
        prev, last = lows[-2], lows[-1]
        prev_rsi = pivot_indicator_value(df, prev, "RSI14")
        last_rsi = pivot_indicator_value(df, last, "RSI14")
        prev_macd = pivot_indicator_value(df, prev, "MACD_HIST")
        last_macd = pivot_indicator_value(df, last, "MACD_HIST")
        if last.price < prev.price and prev_rsi is not None and last_rsi is not None and last_rsi > prev_rsi:
            if bullish:
                bonus += 10
                notes.append("RSI 상승 다이버전스")
            else:
                bonus -= 4
                notes.append("하방 시나리오와 RSI 상승 다이버전스 충돌")
        elif last.price > prev.price and bullish:
            bonus += 4
            notes.append("저점 상승")
        if last.price < prev.price and prev_macd is not None and last_macd is not None and last_macd > prev_macd:
            if bullish:
                bonus += 7
                notes.append("MACD 상승 다이버전스")
            else:
                bonus -= 3

    if len(highs) >= 2:
        prev, last = highs[-2], highs[-1]
        prev_rsi = pivot_indicator_value(df, prev, "RSI14")
        last_rsi = pivot_indicator_value(df, last, "RSI14")
        prev_macd = pivot_indicator_value(df, prev, "MACD_HIST")
        last_macd = pivot_indicator_value(df, last, "MACD_HIST")
        if last.price > prev.price and prev_rsi is not None and last_rsi is not None and last_rsi < prev_rsi:
            if bearish or "Diagonal" in candidate.pattern or "Truncated" in candidate.pattern:
                bonus += 10
                notes.append("RSI 하락 다이버전스")
            elif bullish:
                bonus -= 6
                notes.append("상승 중 RSI 하락 다이버전스 경고")
        elif last.price < prev.price and bearish:
            bonus += 4
            notes.append("고점 하락")
        if last.price > prev.price and prev_macd is not None and last_macd is not None and last_macd < prev_macd:
            if bearish or "Diagonal" in candidate.pattern or "Truncated" in candidate.pattern:
                bonus += 7
                notes.append("MACD 하락 다이버전스")
            elif bullish:
                bonus -= 4

    return clamp(bonus, -12, 22), notes


def score_candidate_consensus(candidate: WaveCandidate, all_candidates: list[WaveCandidate]) -> tuple[float, list[str]]:
    family = wave_family(candidate.pattern)
    last_idx = candidate.pivots[-1].index
    matches = [
        item
        for item in all_candidates
        if wave_family(item.pattern) == family
        and item.direction == candidate.direction
        and abs(item.pivots[-1].index - last_idx) <= 20
    ]
    count = len(matches)
    if count >= 5:
        return 14.0, [f"다중 민감도 합의 {count}개"]
    if count >= 3:
        return 9.0, [f"다중 민감도 합의 {count}개"]
    if count >= 2:
        return 5.0, [f"유사 파동 반복 {count}개"]
    return 0.0, []


def calibrated_probability(confidence_score: float, relative_weight_pct: float) -> float:
    # This is an absolute technical-confidence score, not a normalized sum-to-100 share.
    base = confidence_score * 0.88 + 6.0
    share_bonus = min(relative_weight_pct, 18.0) * 0.35
    return clamp(base + share_bonus, 5.0, 96.0)


def analyze_scenarios(
    df: pd.DataFrame,
    sensitivities: tuple[float, ...] = DEFAULT_SENSITIVITIES,
    atr_multiple: float = DEFAULT_ATR_MULTIPLE,
) -> tuple[list[WavePivot], list[ScenarioScore]]:
    indicators = compute_indicators(df)
    all_candidates: list[WaveCandidate] = []
    primary_pivots = extract_zigzag_pivots(indicators, sensitivities[1], atr_multiple)

    for sensitivity in sensitivities:
        pivots = extract_zigzag_pivots(indicators, sensitivity, atr_multiple)
        all_candidates.extend(generate_wave_candidates(pivots))

    deduped: dict[tuple[str, tuple[int, ...]], WaveCandidate] = {}
    for candidate in all_candidates:
        key = (candidate.pattern, tuple(pivot.index for pivot in candidate.pivots))
        if key not in deduped or candidate.score_raw > deduped[key].score_raw:
            deduped[key] = candidate

    enriched: list[tuple[WaveCandidate, float, list[str], float]] = []
    for candidate in deduped.values():
        indicator_bonus, notes = score_indicator_alignment(indicators, candidate)
        trend_bonus, trend_notes = score_trend_context(indicators, candidate)
        convergence_bonus, convergence_notes = score_convergence_divergence(indicators, candidate)
        consensus_bonus, consensus_notes = score_candidate_consensus(candidate, all_candidates)
        recency_bonus = 0.0
        last_idx = candidate.pivots[-1].index
        bars_since = len(df) - 1 - last_idx
        if bars_since <= 5:
            recency_bonus = 8
        elif bars_since <= 20:
            recency_bonus = 4
        elif bars_since > 120:
            recency_bonus = -8
            notes.append("최근 파동과 거리 있음")
        evidence_bonus = indicator_bonus + trend_bonus + convergence_bonus + consensus_bonus + recency_bonus
        final_score = clamp(candidate.score_raw * 0.70 + evidence_bonus * 0.40 + 8.0, 0, 100)
        evidence_notes = [*notes, *trend_notes, *convergence_notes, *consensus_notes]
        enriched.append((candidate, evidence_bonus, evidence_notes, final_score))

    enriched = sorted(enriched, key=lambda item: item[3], reverse=True)[:TOP_SCENARIOS]
    total_score = sum(max(item[3], 1) for item in enriched)
    scenarios: list[ScenarioScore] = []
    for candidate, indicator_bonus, notes, final_score in enriched:
        relative_weight = 0.0 if total_score <= 0 else max(final_score, 1) / total_score * 100
        probability = calibrated_probability(final_score, relative_weight)
        scenarios.append(
            ScenarioScore(
                candidate=candidate,
                probability_pct=probability,
                indicator_bonus=indicator_bonus,
                indicator_notes=notes,
                confidence_pct=final_score,
                relative_weight_pct=relative_weight,
            )
        )
    return primary_pivots, sorted(scenarios, key=lambda item: item.probability_pct, reverse=True)


def analyze_ticker(ticker: str, df: pd.DataFrame) -> TickerAnalysis:
    started = time.perf_counter()
    errors: list[str] = []
    if len(df) < 80:
        errors.append("At least 80 daily bars are recommended for wave analysis.")
    indicators = compute_indicators(df)
    pivots, scenarios = analyze_scenarios(df)
    elapsed_ms = (time.perf_counter() - started) * 1000
    return TickerAnalysis(
        ticker=ticker,
        df=df,
        indicators=indicators,
        pivots=pivots,
        scenarios=scenarios,
        elapsed_ms=elapsed_ms,
        errors=errors,
    )


# =============================================================================
# Backtest
# =============================================================================


def hit_outcome(df_future: pd.DataFrame, direction: str, target: float | None, invalidation: float | None) -> tuple[str, int, float]:
    if target is None or invalidation is None or df_future.empty:
        return "unresolved", 0, 0.0

    entry = float(df_future["Close"].iloc[0])
    target_reward = abs(target - entry)
    invalid_risk = abs(entry - invalidation)
    risk = max(invalid_risk, 1e-9)

    for offset, (_, row) in enumerate(df_future.iloc[1:].iterrows(), start=1):
        high = float(row["High"])
        low = float(row["Low"])
        if direction == "Bullish":
            target_hit = high >= target
            invalid_hit = low <= invalidation
        elif direction == "Bearish":
            target_hit = low <= target
            invalid_hit = high >= invalidation
        else:
            target_hit = high >= target
            invalid_hit = low <= invalidation

        if target_hit and invalid_hit:
            return "ambiguous", offset, 0.0
        if target_hit:
            return "target", offset, target_reward / risk
        if invalid_hit:
            return "invalid", offset, -1.0

    last_close = float(df_future["Close"].iloc[-1])
    if direction == "Bearish":
        ret_r = (entry - last_close) / risk
        ret_pct = (entry - last_close) / entry * 100
    else:
        ret_r = (last_close - entry) / risk
        ret_pct = (last_close - entry) / entry * 100
    return "unresolved", len(df_future) - 1, float(ret_r if not pd.isna(ret_r) else ret_pct / 100)


def run_walk_forward_backtest(
    df: pd.DataFrame,
    pattern_filter: str = "All",
    sensitivity: float = 0.05,
    horizon: int = 60,
    min_confidence: float = 60,
    step: int = 5,
    max_samples: int = 250,
) -> BacktestStats:
    if len(df) < 180:
        return BacktestStats(pattern_filter, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    outcomes: list[str] = []
    bars_held: list[int] = []
    returns_r: list[float] = []
    returns_pct: list[float] = []
    start = max(140, len(df) - (max_samples * step + horizon))
    end = len(df) - horizon - 1

    for i in range(start, end, step):
        history = df.iloc[: i + 1]
        pivots = extract_zigzag_pivots(history, sensitivity, DEFAULT_ATR_MULTIPLE)
        candidates = generate_wave_candidates(pivots)
        if not candidates:
            continue

        indicators = compute_indicators(history)
        scored: list[ScenarioScore] = []
        total = 0.0
        tmp: list[tuple[WaveCandidate, float, list[str], float]] = []
        for candidate in candidates[:10]:
            if pattern_filter != "All" and pattern_filter not in candidate.pattern:
                continue
            bonus, notes = score_indicator_alignment(indicators, candidate)
            final_score = clamp(candidate.score_raw + bonus, 0, 100)
            if final_score < min_confidence:
                continue
            tmp.append((candidate, bonus, notes, final_score))
            total += max(final_score, 1)
        for candidate, bonus, notes, final_score in tmp:
            scored.append(ScenarioScore(candidate, max(final_score, 1) / total * 100 if total else 0, bonus, notes, final_score))
        if not scored:
            continue

        selected = max(scored, key=lambda item: item.confidence_pct)
        future = df.iloc[i : i + horizon + 1]
        outcome, held, ret_r = hit_outcome(
            future,
            selected.candidate.direction,
            selected.candidate.target,
            selected.candidate.invalidation,
        )
        entry = float(future["Close"].iloc[0])
        exit_close = float(future["Close"].iloc[-1])
        if selected.candidate.direction == "Bearish":
            ret_pct = (entry - exit_close) / entry * 100
        else:
            ret_pct = (exit_close - entry) / entry * 100
        outcomes.append(outcome)
        bars_held.append(held)
        returns_r.append(ret_r)
        returns_pct.append(ret_pct)
        if len(outcomes) >= max_samples:
            break

    samples = len(outcomes)
    if samples == 0:
        return BacktestStats(pattern_filter, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    target_rate = outcomes.count("target") / samples * 100
    invalid_rate = outcomes.count("invalid") / samples * 100
    unresolved_rate = (outcomes.count("unresolved") + outcomes.count("ambiguous")) / samples * 100
    return BacktestStats(
        pattern=pattern_filter,
        samples=samples,
        target_hit_rate=target_rate,
        invalidation_hit_rate=invalid_rate,
        unresolved_rate=unresolved_rate,
        avg_bars_held=float(np.mean(bars_held)) if bars_held else 0.0,
        expectancy_r=float(np.mean(returns_r)) if returns_r else 0.0,
        avg_return_pct=float(np.mean(returns_pct)) if returns_pct else 0.0,
    )


# =============================================================================
# QA
# =============================================================================


def synthetic_ohlcv(points: list[float], bars_per_segment: int = 8) -> pd.DataFrame:
    values: list[float] = []
    for idx in range(len(points) - 1):
        segment = np.linspace(points[idx], points[idx + 1], bars_per_segment, endpoint=False)
        values.extend(segment.tolist())
    values.append(points[-1])
    close = pd.Series(values, dtype=float)
    dates = pd.date_range("2020-01-01", periods=len(close), freq="B")
    high = close + close.abs() * 0.003 + 0.15
    low = close - close.abs() * 0.003 - 0.15
    open_ = close.shift(1).fillna(close.iloc[0])
    volume = np.linspace(1_000_000, 1_300_000, len(close))
    return pd.DataFrame(
        {
            "Open": open_.to_numpy(),
            "High": high.to_numpy(),
            "Low": low.to_numpy(),
            "Close": close.to_numpy(),
            "Volume": volume,
        },
        index=dates,
    )


def make_pivots_from_points(points: list[float]) -> tuple[WavePivot, ...]:
    dates = pd.date_range("2020-01-01", periods=len(points), freq="B")
    pivots: list[WavePivot] = []
    for i, price in enumerate(points):
        if i == 0:
            kind = "L" if points[1] > points[0] else "H"
        elif i == len(points) - 1:
            kind = "H" if points[i] > points[i - 1] else "L"
        else:
            kind = "H" if points[i] > points[i - 1] else "L"
        pivots.append(WavePivot(i, pd.Timestamp(dates[i]), float(price), kind))
    return tuple(pivots)


def run_rule_unit_tests() -> pd.DataFrame:
    tests: list[dict[str, object]] = []

    def record(name: str, passed: bool, detail: str) -> None:
        tests.append({"Test": name, "Result": "PASS" if passed else "FAIL", "Detail": detail})

    impulse = validate_impulse(make_pivots_from_points([100, 120, 110, 150, 135, 162]))
    record("Valid impulse", impulse is not None and "Impulse" in impulse.pattern, impulse.pattern if impulse else "No impulse")

    extended = validate_impulse(make_pivots_from_points([100, 112, 106, 150, 136, 158]))
    record("Wave 3 extension", extended is not None and "Wave 3 extension" in extended.extension, extended.extension if extended else "No impulse")

    truncated = validate_impulse(make_pivots_from_points([100, 120, 110, 150, 135, 146]))
    record("Truncated fifth", truncated is not None and "Truncated" in truncated.pattern, truncated.pattern if truncated else "No truncation")

    diagonal = validate_diagonal(make_pivots_from_points([100, 120, 112, 130, 118, 134]))
    record("Ending diagonal", diagonal is not None and "Diagonal" in diagonal.pattern, diagonal.pattern if diagonal else "No diagonal")

    zigzag = validate_abc(make_pivots_from_points([150, 120, 136, 104]))
    record("ABC zigzag", any("Zigzag" in item.pattern for item in zigzag), ", ".join(item.pattern for item in zigzag) or "No ABC")

    flat = validate_abc(make_pivots_from_points([100, 80, 99, 76]))
    record("ABC flat", any("Flat" in item.pattern for item in flat), ", ".join(item.pattern for item in flat) or "No flat")

    triangle = validate_triangle(make_pivots_from_points([100, 90, 98, 92, 96, 94]))
    record("Triangle", triangle is not None, triangle.pattern if triangle else "No triangle")

    combo = validate_combination(make_pivots_from_points([100, 90, 98, 91, 97, 92, 96, 93]))
    record("W-X-Y combination", combo is not None and "W-X-Y" in combo.pattern, combo.pattern if combo else "No combination")

    invalid_w2 = validate_impulse(make_pivots_from_points([100, 120, 95, 135, 125, 150]))
    record("Invalid impulse: W2 > 100%", invalid_w2 is None, "Rejected" if invalid_w2 is None else invalid_w2.pattern)

    invalid_w3 = validate_impulse(make_pivots_from_points([100, 130, 120, 125, 118, 140]))
    record("Invalid impulse: W3 shortest", invalid_w3 is None, "Rejected" if invalid_w3 is None else invalid_w3.pattern)

    invalid_overlap = validate_impulse(make_pivots_from_points([100, 120, 110, 145, 118, 160]))
    record("Invalid impulse: W4 overlap", invalid_overlap is None, "Rejected" if invalid_overlap is None else invalid_overlap.pattern)

    for name, points, expected in [
        ("Synthetic impulse pipeline", [100, 120, 110, 150, 135, 162], "Impulse"),
        ("Synthetic diagonal pipeline", [100, 120, 112, 130, 118, 134], "Diagonal"),
        ("Synthetic triangle pipeline", [100, 88, 99, 90, 97, 92], "Triangle"),
    ]:
        frame = synthetic_ohlcv(points)
        pivots = extract_zigzag_pivots(frame, 0.03, 0.5)
        candidates = generate_wave_candidates(pivots)
        found = any(expected in candidate.pattern for candidate in candidates)
        record(name, found, ", ".join(candidate.pattern for candidate in candidates[:4]) or "No candidates")

    return pd.DataFrame(tests)


# =============================================================================
# Charting
# =============================================================================


def scenario_table(scenarios: list[ScenarioScore]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for scenario in scenarios:
        candidate = scenario.candidate
        rows.append(
            {
                "Pattern": candidate.pattern,
                "Direction": candidate.direction,
                "종합 확률 %": round(scenario.probability_pct, 1),
                "근거 점수 %": round(scenario.confidence_pct, 1),
                "후보 내 비중 %": round(scenario.relative_weight_pct, 1),
                "근거 보너스": round(scenario.indicator_bonus, 1),
                "Extension": candidate.extension or "-",
                "Alternation": candidate.alternation or "-",
                "Target": candidate.target,
                "Invalidation": candidate.invalidation,
                "Summary": candidate.summary,
            }
        )
    return pd.DataFrame(rows)


def add_fibonacci_levels(fig: go.Figure, candidate: WaveCandidate, transform=lambda value: value, row: int = 1, col: int = 1) -> None:
    prices = pivot_prices(candidate.pivots)
    if len(prices) < 2:
        return
    start = prices[-2]
    end = prices[-1]
    span = end - start
    if span == 0:
        return
    for level in [0.382, 0.5, 0.618, 1.0, 1.618]:
        y = transform(end - span * level)
        fig.add_hline(y=y, line_width=0.8, line_dash="dot", line_color="rgba(120,120,120,0.45)", row=row, col=col)
        fig.add_annotation(xref="paper", x=1.0, y=y, text=f"Fib {level:.3g}", showarrow=False, font=dict(size=10), row=row, col=col)


def transformed_price(value: float | None, base: float, percent_mode: bool) -> float | None:
    if value is None or pd.isna(value):
        return None
    if not percent_mode:
        return float(value)
    if base <= 0 or pd.isna(base):
        return None
    return (float(value) / base - 1) * 100


def auto_zoom_range(plot_df: pd.DataFrame, selected_scenario: ScenarioScore | None, transform, percent_mode: bool) -> list[float] | None:
    if plot_df.empty:
        return None
    focus = plot_df.tail(min(120, len(plot_df)))
    values = [float(focus["Low"].quantile(0.02)), float(focus["High"].quantile(0.98))]
    if selected_scenario is not None:
        focus_start = focus.index[0]
        for pivot in selected_scenario.candidate.pivots:
            if pivot.date >= focus_start:
                pivot_value = transform(pivot.price)
                if pivot_value is not None:
                    values.append(float(pivot_value))
        current = float(plot_df["Close"].iloc[-1])
        base_span = max(abs(values[1] - values[0]), 1e-9)
        for raw_level in [selected_scenario.candidate.target, selected_scenario.candidate.invalidation]:
            level = transform(raw_level)
            if level is not None and abs(float(level) - current) <= base_span * (2.2 if percent_mode else 1.6):
                values.append(float(level))
    low = min(values)
    high = max(values)
    if high <= low:
        return None
    pad = (high - low) * 0.10
    return [low - pad, high + pad]


def make_wave_chart(
    analysis: TickerAnalysis,
    selected_scenario: ScenarioScore | None,
    lookback: int = 260,
    display_mode: str = "최근 구간 자동 줌",
) -> go.Figure:
    df = analysis.indicators.tail(lookback).copy()
    percent_mode = display_mode == "수익률(%)"
    log_mode = display_mode == "로그 가격"
    auto_zoom = display_mode == "최근 구간 자동 줌"
    base_price = float(df["Close"].iloc[0]) if not df.empty else 1.0
    transform = lambda value: transformed_price(value, base_price, percent_mode)
    plot_df = df.copy()
    if percent_mode:
        for column in ["Open", "High", "Low", "Close", "SMA20", "SMA50", "SMA200", "EMA21"]:
            if column in plot_df:
                plot_df[column] = (plot_df[column] / base_price - 1) * 100

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.62, 0.18, 0.20],
        vertical_spacing=0.03,
        specs=[[{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": False}]],
    )
    fig.add_trace(
        go.Candlestick(
            x=plot_df.index,
            open=plot_df["Open"],
            high=plot_df["High"],
            low=plot_df["Low"],
            close=plot_df["Close"],
            name="Daily OHLC",
        ),
        row=1,
        col=1,
    )
    for column, color in [("SMA20", "#1f77b4"), ("SMA50", "#ff7f0e"), ("SMA200", "#2ca02c")]:
        if column in plot_df:
            fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df[column], mode="lines", name=column, line=dict(width=1.3, color=color)), row=1, col=1)

    visible_pivots = [pivot for pivot in analysis.pivots if pivot.date >= df.index[0]]
    if visible_pivots:
        fig.add_trace(
            go.Scatter(
                x=[pivot.date for pivot in visible_pivots],
                y=[transform(pivot.price) for pivot in visible_pivots],
                mode="lines+markers+text",
                text=[pivot.kind for pivot in visible_pivots],
                textposition="top center",
                name="ZigZag pivots",
                line=dict(color="#7b1fa2", width=1.4),
                marker=dict(size=7),
            ),
            row=1,
            col=1,
        )

    if selected_scenario is not None:
        candidate = selected_scenario.candidate
        scenario_pivots = [pivot for pivot in candidate.pivots if pivot.date >= df.index[0]]
        if scenario_pivots:
            fig.add_trace(
                go.Scatter(
                    x=[pivot.date for pivot in scenario_pivots],
                    y=[transform(pivot.price) for pivot in scenario_pivots],
                    mode="lines+markers+text",
                    text=list(candidate.labels[-len(scenario_pivots) :]),
                    textposition="bottom center",
                    name=candidate.pattern,
                    line=dict(color="#d62728", width=2.4),
                    marker=dict(size=10, symbol="diamond"),
                ),
                row=1,
                col=1,
            )
            add_fibonacci_levels(fig, candidate, transform)
        target_y = transform(candidate.target)
        if target_y is not None:
            fig.add_hline(y=target_y, line_width=1.4, line_dash="dash", line_color="#2e7d32", row=1, col=1)
            fig.add_annotation(xref="paper", x=0.0, y=target_y, text="목표가", showarrow=False, font=dict(color="#2e7d32"), row=1, col=1)
        invalid_y = transform(candidate.invalidation)
        if invalid_y is not None:
            fig.add_hline(y=invalid_y, line_width=1.4, line_dash="dash", line_color="#c62828", row=1, col=1)
            fig.add_annotation(xref="paper", x=0.0, y=invalid_y, text="무효화", showarrow=False, font=dict(color="#c62828"), row=1, col=1)

    colors = np.where(df["Close"] >= df["Open"], "#2e7d32", "#c62828")
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], marker_color=colors, name="Volume"), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["RSI14"], mode="lines", name="RSI14", line=dict(color="#455a64")), row=3, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="rgba(180,0,0,.45)", row=3, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="rgba(0,120,0,.45)", row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD_HIST"], mode="lines", name="MACD Hist", line=dict(color="#6a1b9a")), row=3, col=1)

    fig.update_layout(
        height=780,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=35, b=20),
    )
    fig.update_yaxes(title_text="수익률 %" if percent_mode else "가격", row=1, col=1)
    fig.update_yaxes(title_text="거래량", row=2, col=1)
    fig.update_yaxes(title_text="RSI / MACD", row=3, col=1)
    if log_mode and float(plot_df["Low"].min()) > 0:
        fig.update_yaxes(type="log", row=1, col=1)
    if auto_zoom:
        y_range = auto_zoom_range(plot_df, selected_scenario, transform, percent_mode)
        if y_range is not None:
            fig.update_yaxes(range=y_range, row=1, col=1)
    return fig


# =============================================================================
# Streamlit UI
# =============================================================================


def render_summary_metrics(analysis: TickerAnalysis) -> None:
    latest = analysis.indicators.iloc[-1]
    top = analysis.scenarios[0] if analysis.scenarios else None
    cols = st.columns(5)
    cols[0].metric("현재가", format_price(float(latest["Close"])))
    cols[1].metric("대표 파동", pattern_name_ko(top.candidate.pattern) if top else "없음")
    cols[2].metric("종합 확률", f"{top.probability_pct:.1f}%" if top else "-")
    cols[3].metric("근거 점수", f"{top.confidence_pct:.1f}%" if top else "-")
    cols[4].metric("분석 시간", f"{analysis.elapsed_ms:.0f} ms")


def render_korean_guide(analysis: TickerAnalysis, scenario: ScenarioScore) -> None:
    guide = build_korean_guide(analysis, scenario)
    candidate = scenario.candidate
    current = float(analysis.indicators["Close"].iloc[-1])
    move_pct = price_move_pct(current, candidate.target, candidate.direction)
    upside_label = "예상 상승/이동폭"
    if candidate.direction == "Bearish":
        upside_label = "예상 하락/이동폭"

    st.subheader("한글 해설 가이드")
    with st.container(border=True):
        st.markdown(f"**{guide['headline']}**")
        st.write(guide["state"])
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("현재가", format_price(current))
        m2.metric(upside_label, format_price(candidate.target), f"{move_pct:.1f}%" if move_pct is not None else None)
        m3.metric("무효화 가격", format_price(candidate.invalidation))
        m4.metric("방향", direction_ko(candidate.direction))
        st.markdown(
            "\n".join(
                [
                    f"- {guide['target']}",
                    f"- {guide['risk']}",
                    f"- {guide['wave']}",
                    f"- {guide['extension']}",
                    f"- {guide['indicator']}",
                ]
            )
        )
        st.caption(
            "가이드는 규칙 기반 확률 해석입니다. 목표가는 가장 가까운 기술적 목표 구간이며, "
            "실제 매매 판단은 거래량, 시장 환경, 손절 기준과 함께 확인해야 합니다."
        )


def render_analysis_tab(analysis: TickerAnalysis) -> ScenarioScore | None:
    render_summary_metrics(analysis)
    if analysis.errors:
        for error in analysis.errors:
            st.warning(error)
    if not analysis.scenarios:
        st.warning("No valid Elliott Wave scenario was found at the current sensitivity. Try a longer period or lower reversal threshold.")
        return None

    selected_index = 0
    table = scenario_table(analysis.scenarios)
    event = st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        height=420,
        on_select="rerun",
        selection_mode="single-row",
        column_config={
            "종합 확률 %": st.column_config.ProgressColumn("종합 확률 %", min_value=0, max_value=100, format="%.1f%%"),
            "근거 점수 %": st.column_config.ProgressColumn("근거 점수 %", min_value=0, max_value=100, format="%.1f%%"),
            "후보 내 비중 %": st.column_config.NumberColumn("후보 내 비중 %", format="%.1f"),
            "Target": st.column_config.NumberColumn("Target", format="%.2f"),
            "Invalidation": st.column_config.NumberColumn("Invalidation", format="%.2f"),
        },
    )
    selected_index = event.selection.rows[0] if event.selection.rows else 0
    selected = analysis.scenarios[selected_index]

    render_korean_guide(analysis, selected)

    st.subheader("선택 시나리오 상세")
    c1, c2 = st.columns([0.55, 0.45])
    with c1:
        st.write(
            {
                "파동": pattern_name_ko(selected.candidate.pattern),
                "원본 패턴": selected.candidate.pattern,
                "방향": direction_ko(selected.candidate.direction),
                "라벨": " -> ".join(selected.candidate.labels),
                "목표가": selected.candidate.target,
                "무효화": selected.candidate.invalidation,
                "연장": selected.candidate.extension or None,
                "교대 법칙": selected.candidate.alternation or None,
            }
        )
        st.caption(selected.candidate.summary)
    with c2:
        st.markdown("**통과한 파동 규칙**")
        st.write(selected.candidate.rule_hits)
        if selected.candidate.rule_misses:
            st.markdown("**약한 규칙 / 실패 규칙**")
            st.write(selected.candidate.rule_misses)
        st.markdown("**보조지표 확인**")
        st.write(selected.indicator_notes)
    return selected


def render_backtest_tab(df: pd.DataFrame, top_pattern: str | None) -> None:
    st.caption("Walk-forward validation uses only data available at each historical decision point.")
    patterns = ["All"]
    if top_pattern:
        patterns.append(top_pattern)
    patterns.extend(["Impulse", "Diagonal", "ABC", "Triangle", "W-X-Y"])
    c1, c2, c3, c4 = st.columns(4)
    pattern_filter = c1.selectbox("Pattern filter", list(dict.fromkeys(patterns)), index=1 if top_pattern else 0)
    horizon = c2.slider("Horizon bars", 20, 120, 60, step=10)
    min_confidence = c3.slider("Min confidence", 40, 85, 60, step=5)
    step = c4.slider("Sampling step", 1, 20, 5, step=1)

    started = time.perf_counter()
    stats = run_walk_forward_backtest(
        df,
        pattern_filter=pattern_filter,
        horizon=int(horizon),
        min_confidence=float(min_confidence),
        step=int(step),
    )
    elapsed = (time.perf_counter() - started) * 1000

    cols = st.columns(6)
    cols[0].metric("Samples", stats.samples)
    cols[1].metric("Target First", f"{stats.target_hit_rate:.1f}%")
    cols[2].metric("Invalid First", f"{stats.invalidation_hit_rate:.1f}%")
    cols[3].metric("Unresolved", f"{stats.unresolved_rate:.1f}%")
    cols[4].metric("Expectancy", f"{stats.expectancy_r:.2f} R")
    cols[5].metric("Runtime", f"{elapsed:.0f} ms")
    st.write(
        pd.DataFrame(
            [
                {
                    "Pattern": stats.pattern,
                    "Samples": stats.samples,
                    "Target Hit Rate %": stats.target_hit_rate,
                    "Invalidation Hit Rate %": stats.invalidation_hit_rate,
                    "Unresolved %": stats.unresolved_rate,
                    "Avg Bars Held": stats.avg_bars_held,
                    "Expectancy R": stats.expectancy_r,
                    "Avg Horizon Return %": stats.avg_return_pct,
                }
            ]
        )
    )


def render_qa_tab(load_elapsed_ms: float, analyses: dict[str, TickerAnalysis], warnings: list[str]) -> None:
    st.subheader("QA Rule Tests")
    qa_started = time.perf_counter()
    qa_df = run_rule_unit_tests()
    qa_elapsed = (time.perf_counter() - qa_started) * 1000
    st.dataframe(qa_df, hide_index=True, width="stretch")
    pass_count = int((qa_df["Result"] == "PASS").sum())
    st.metric("QA Pass Rate", f"{pass_count}/{len(qa_df)}")

    st.subheader("Performance Diagnostics")
    perf_rows = []
    for ticker, analysis in analyses.items():
        perf_rows.append(
            {
                "Ticker": ticker,
                "Bars": len(analysis.df),
                "Pivots": len(analysis.pivots),
                "Scenarios": len(analysis.scenarios),
                "Analysis ms": analysis.elapsed_ms,
            }
        )
    st.dataframe(pd.DataFrame(perf_rows), hide_index=True, width="stretch")
    st.write(
        {
            "data_load_ms": round(load_elapsed_ms, 1),
            "qa_runtime_ms": round(qa_elapsed, 1),
            "total_analysis_ms": round(sum(item.elapsed_ms for item in analyses.values()), 1),
            "warnings": warnings,
            "performance_target": "Cached single ticker near 1s; 20 tickers / 5y daily near 8s on a typical laptop.",
        }
    )


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption(DISCLAIMER)

    with st.sidebar:
        st.header("Data")
        ticker_text = st.text_area("Tickers", value=DEFAULT_TICKERS, height=92)
        period = st.selectbox("Daily history", ["1y", "2y", "5y", "10y", "max"], index=3)
        refresh = st.button("Refresh yfinance data", width="stretch")
        st.divider()
        st.header("Engine")
        st.caption("10년 일봉 중심으로 파동, 장기 이평, 수렴/발산, 다중 민감도 합의를 함께 봅니다.")
        st.write({"sensitivities": [f"{item:.0%}" for item in DEFAULT_SENSITIVITIES], "atr_multiple": DEFAULT_ATR_MULTIPLE})

    refresh_nonce = int(time.time()) if refresh else 0
    with st.status("Loading daily OHLCV from yfinance", expanded=False) as status:
        prices, warnings, load_elapsed_ms = load_price_data(ticker_text, period, refresh_nonce)
        status.update(label=f"Loaded {len(prices)} ticker(s) in {load_elapsed_ms:.0f} ms", state="complete")

    if warnings:
        for warning in warnings:
            st.warning(warning)

    if not prices:
        st.error("No data loaded. Check ticker symbols and network access.")
        return

    analysis_started = time.perf_counter()
    analyses = {ticker: analyze_ticker(ticker, frame) for ticker, frame in prices.items()}
    total_analysis_ms = (time.perf_counter() - analysis_started) * 1000

    tickers = list(analyses.keys())
    selected_ticker = st.sidebar.selectbox("Selected ticker", tickers)
    analysis = analyses[selected_ticker]

    tab_analysis, tab_chart, tab_backtest, tab_qa = st.tabs(["분석", "차트", "백테스트", "QA / 성능"])

    selected_scenario: ScenarioScore | None = analysis.scenarios[0] if analysis.scenarios else None
    with tab_analysis:
        selected_scenario = render_analysis_tab(analysis)

    with tab_chart:
        c1, c2 = st.columns([0.35, 0.65])
        lookback = c1.slider("차트 표시 봉 수", 80, min(MAX_ANALYSIS_BARS, max(81, len(analysis.df))), min(220, len(analysis.df)), step=20)
        display_mode = c2.radio(
            "차트 표시 방식",
            ["최근 구간 자동 줌", "일반 가격", "로그 가격", "수익률(%)"],
            horizontal=True,
            help="큰 상승폭 때문에 캔들이 눌려 보이면 '최근 구간 자동 줌'이나 '로그 가격'을 사용하세요.",
        )
        st.plotly_chart(make_wave_chart(analysis, selected_scenario, lookback=int(lookback), display_mode=display_mode), width="stretch")
        if selected_scenario:
            current = float(analysis.indicators["Close"].iloc[-1])
            move_pct = price_move_pct(current, selected_scenario.candidate.target, selected_scenario.candidate.direction)
            move_text = "-" if move_pct is None else f"{move_pct:.1f}%"
            st.info(
                f"대표 시나리오: {pattern_name_ko(selected_scenario.candidate.pattern)} | "
                f"종합 확률 {selected_scenario.probability_pct:.1f}% | "
                f"예상 이동폭 {move_text} | "
                f"목표 {format_price(selected_scenario.candidate.target)} | "
                f"무효화 {format_price(selected_scenario.candidate.invalidation)}"
            )

    with tab_backtest:
        top_pattern = selected_scenario.candidate.pattern if selected_scenario else None
        render_backtest_tab(analysis.df, top_pattern)

    with tab_qa:
        render_qa_tab(load_elapsed_ms + total_analysis_ms, analyses, warnings)


if __name__ == "__main__":
    main()
