import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

MIN_ROWS_REQUIRED = 60

st.set_page_config(page_title="10대 지표 컨센서스 타점 분석기 Pro", layout="wide")

DEFAULT_WEIGHTS = {
    "SMA_Cross": 1.5,
    "RSI": 1.5,
    "MACD": 1.5,
    "PSAR": 1.5,
    "Bollinger": 1.0,
    "Stochastic": 1.0,
    "CCI": 1.0,
    "MFI": 1.0,
    "Envelope": 1.0,
    "Momentum": 1.0,
}
INDICATOR_WEIGHTS = dict(DEFAULT_WEIGHTS)


def format_price(v, is_krw=False):
    """주식/코인 가격 표시용 포맷터."""
    if v is None or pd.isna(v):
        return "산출 불가"
    if is_krw:
        return f"{v:,.0f}원"
    v = float(v)
    if abs(v) >= 1:
        return f"${v:,.2f}"
    elif abs(v) >= 0.01:
        return f"${v:.4f}"
    else:
        return f"${v:.6f}"


# ---------------------------------------------------------
# 1. 데이터 수집 및 캐싱
# ---------------------------------------------------------
@st.cache_data(ttl=300)
def load_stock_data(ticker_symbol):
    try:
        df = yf.download(ticker_symbol, period="1y", progress=False)
        if df.empty:
            return df

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        required_cols = ["Close", "High", "Low", "Open", "Volume"]
        available_cols = [c for c in required_cols if c in df.columns]
        df = df.dropna(subset=available_cols)
        return df
    except Exception as e:
        st.error(f"데이터 수집 중 오류 발생: {e}")
        return pd.DataFrame()


# ---------------------------------------------------------
# 2. 정통 Parabolic SAR 수식 엔진
# ---------------------------------------------------------
def calculate_psar(df, af_start=0.02, af_inc=0.02, af_max=0.2):
    high, low, close = df["High"].values, df["Low"].values, df["Close"].values
    n = len(df)
    psar = np.zeros(n)
    psarbull = [True] * n

    if n < 3:
        return pd.Series(close, index=df.index), pd.Series(True, index=df.index)

    bull = close[1] >= close[0]
    af = af_start
    hp = max(high[0], high[1]) if bull else min(low[0], low[1])
    lp = min(low[0], low[1]) if bull else max(high[0], high[1])
    psar[0] = low[0] if bull else high[0]
    psar[1] = low[0] if bull else high[0]
    psarbull[0] = bull
    psarbull[1] = bull

    for i in range(2, n):
        prev_psar = psar[i - 1]
        if bull:
            psar[i] = prev_psar + af * (hp - prev_psar)
            psar[i] = min(psar[i], low[i - 1], low[i - 2])
            if low[i] < psar[i]:
                bull = False
                psar[i] = hp
                lp = low[i]
                af = af_start
            else:
                if high[i] > hp:
                    hp = high[i]
                    af = min(af + af_inc, af_max)
        else:
            psar[i] = prev_psar + af * (lp - prev_psar)
            psar[i] = max(psar[i], high[i - 1], high[i - 2])
            if high[i] > psar[i]:
                bull = True
                psar[i] = lp
                hp = high[i]
                af = af_start
            else:
                if low[i] < lp:
                    lp = low[i]
                    af = min(af + af_inc, af_max)
        psarbull[i] = bull

    return pd.Series(psar, index=df.index), pd.Series(psarbull, index=df.index)


# ---------------------------------------------------------
# 3. 전체 기간 지표 시계열 계산
# ---------------------------------------------------------
def compute_full_series(df):
    high, low, close, volume = df["High"], df["Low"], df["Close"], df["Volume"]
    ind = pd.DataFrame(index=df.index)
    ind["open"] = df["Open"]
    ind["high"] = high
    ind["low"] = low
    ind["close"] = close

    with np.errstate(divide="ignore", invalid="ignore"):
        sma20 = close.rolling(20).mean()
        sma60 = close.rolling(60).mean()
        ind["sma20"] = sma20
        ind["sma60"] = sma60

        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        rsi = rsi.mask((loss == 0) & (gain > 0), 100.0)
        ind["rsi"] = rsi

        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        macd_signal = macd.ewm(span=9, adjust=False).mean()
        ind["macd"] = macd
        ind["macd_signal"] = macd_signal

        psar_series, psar_bull = calculate_psar(df)
        ind["psar"] = psar_series
        ind["psar_bull"] = psar_bull

        std20 = close.rolling(20).std()
        ind["bb_upper"] = sma20 + std20 * 2
        ind["bb_lower"] = sma20 - std20 * 2

        low14 = low.rolling(14).min()
        high14 = high.rolling(14).max()
        stoch_range = (high14 - low14).replace(0, np.nan)
        k = 100 * ((close - low14) / stoch_range)
        d = k.rolling(3).mean()
        ind["stoch_k"] = k
        ind["stoch_d"] = d

        tp = (high + low + close) / 3
        cci = (tp - tp.rolling(20).mean()) / (
            0.015 * tp.rolling(20).std().replace(0, np.nan)
        )
        ind["cci"] = cci

        raw_mf = tp * volume
        pos_flow = np.where(tp > tp.shift(1), raw_mf, 0)
        neg_flow = np.where(tp < tp.shift(1), raw_mf, 0)
        pos_sum = pd.Series(pos_flow, index=df.index).rolling(14).sum()
        neg_sum = pd.Series(neg_flow, index=df.index).rolling(14).sum()
        mfi_ratio = pos_sum / neg_sum.replace(0, np.nan)
        mfi = 100 - (100 / (1 + mfi_ratio))
        mfi = mfi.mask((neg_sum == 0) & (pos_sum > 0), 100.0)
        ind["mfi"] = mfi

        ind["env_upper"] = sma20 * 1.05
        ind["env_lower"] = sma20 * 0.95

        ind["momentum"] = close - close.shift(10)

        prev_close = close.shift(1)
        tr = pd.concat(
            [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
            axis=1,
        ).max(axis=1)
        ind["atr14"] = tr.rolling(14).mean()

    return ind


@st.cache_data(ttl=300)
def get_indicator_series(ticker_symbol):
    df = load_stock_data(ticker_symbol)
    if df.empty:
        return df, pd.DataFrame()
    ind = compute_full_series(df)
    return df, ind


# ---------------------------------------------------------
# 4. 시점별 신호 추출
# ---------------------------------------------------------
def build_signal_row(ind, i):
    row = ind.iloc[i]
    signals = {}
    na_indicators = []

    def set_signal(name, condition_up, condition_down, *check_values):
        if any(pd.isna(v) for v in check_values):
            signals[name] = np.nan
            na_indicators.append(name)
        else:
            signals[name] = 1 if condition_up else (-1 if condition_down else 0)

    if i >= 1:
        prev = ind.iloc[i - 1]
        vals = (row["sma20"], row["sma60"], prev["sma20"], prev["sma60"])
        if any(pd.isna(v) for v in vals):
            signals["SMA_Cross"] = np.nan
            na_indicators.append("SMA_Cross")
        else:
            is_golden = (prev["sma20"] <= prev["sma60"]) and (
                row["sma20"] > row["sma60"]
            )
            is_dead = (prev["sma20"] >= prev["sma60"]) and (
                row["sma20"] < row["sma60"]
            )
            if is_golden:
                signals["SMA_Cross"] = 1
            elif is_dead:
                signals["SMA_Cross"] = -1
            else:
                signals["SMA_Cross"] = 1 if row["sma20"] > row["sma60"] else -1
    else:
        signals["SMA_Cross"] = np.nan
        na_indicators.append("SMA_Cross")

    set_signal("RSI", row["rsi"] < 35, row["rsi"] > 65, row["rsi"])
    set_signal(
        "MACD",
        row["macd"] > row["macd_signal"],
        row["macd"] < row["macd_signal"],
        row["macd"],
        row["macd_signal"],
    )
    set_signal(
        "PSAR",
        bool(row["psar_bull"]),
        not bool(row["psar_bull"]),
        row["psar"],
    )
    set_signal(
        "Bollinger",
        row["close"] <= row["bb_lower"],
        row["close"] >= row["bb_upper"],
        row["bb_lower"],
        row["bb_upper"],
    )
    set_signal(
        "Stochastic",
        (row["stoch_k"] < 20) and (row["stoch_k"] > row["stoch_d"]),
        (row["stoch_k"] > 80) and (row["stoch_k"] < row["stoch_d"]),
        row["stoch_k"],
        row["stoch_d"],
    )
    set_signal("CCI", row["cci"] < -100, row["cci"] > 100, row["cci"])
    set_signal("MFI", row["mfi"] < 20, row["mfi"] > 80, row["mfi"])
    set_signal(
        "Envelope",
        row["close"] <= row["env_lower"],
        row["close"] >= row["env_upper"],
        row["env_lower"],
        row["env_upper"],
    )
    set_signal(
        "Momentum", row["momentum"] > 0, row["momentum"] < 0, row["momentum"]
    )

    return signals, row["close"], na_indicators, row["atr14"]


# ---------------------------------------------------------
# 5. TradingView 다크스타일 시각화 모듈
# ---------------------------------------------------------
def create_interactive_chart(
    df,
    target_price=None,
    stop_price=None,
    buy_signals_dates=None,
    is_krw=False,
    rsi_series=None,
):
    buy_signals_dates = buy_signals_dates or []
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.60, 0.15, 0.25],
        vertical_spacing=0.03,
    )

    # TradingView 캔들 색상 (양봉: #26a69a, 음봉: #ef5350)
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="주가",
            increasing_line_color="#26a69a",
            increasing_fillcolor="#26a69a",
            decreasing_line_color="#ef5350",
            decreasing_fillcolor="#ef5350",
        ),
        row=1,
        col=1,
    )

    sma20 = df["Close"].rolling(20).mean()
    sma60 = df["Close"].rolling(60).mean()
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=sma20,
            name="SMA 20",
            line=dict(color="#f39c12", width=1.2),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=sma60,
            name="SMA 60",
            line=dict(color="#ab47bc", width=1.2),
        ),
        row=1,
        col=1,
    )

    if buy_signals_dates:
        valid_dates = [d for d in buy_signals_dates if d in df.index]
        if valid_dates:
            buy_lows = df.loc[valid_dates, "Low"] * 0.985
            fig.add_trace(
                go.Scatter(
                    x=valid_dates,
                    y=buy_lows,
                    mode="markers+text",
                    name="매수 타점",
                    marker=dict(symbol="triangle-up", size=12, color="#00e676"),
                    text=["▲매수"] * len(valid_dates),
                    textposition="bottom center",
                    textfont=dict(color="#00e676", size=10),
                ),
                row=1,
                col=1,
            )

    if target_price is not None and not np.isnan(target_price):
        price_str = format_price(target_price, is_krw)
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=[target_price] * len(df.index),
                mode="lines",
                line=dict(color="#00e676", dash="dash", width=1.5),
                name=f"목표가(ATR) {price_str}",
                hovertemplate=f"목표가(ATR): {price_str}<extra></extra>",
            ),
            row=1,
            col=1,
        )
    if stop_price is not None and not np.isnan(stop_price):
        price_str = format_price(stop_price, is_krw)
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=[stop_price] * len(df.index),
                mode="lines",
                line=dict(color="#ff5252", dash="dash", width=1.5),
                name=f"손절가(ATR) {price_str}",
                hovertemplate=f"손절가(ATR): {price_str}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    # 거래량
    colors = [
        "#26a69a" if c >= o else "#ef5350"
        for c, o in zip(df["Close"], df["Open"])
    ]
    fig.add_trace(
        go.Bar(
            x=df.index, y=df["Volume"], name="거래량", marker_color=colors, opacity=0.7
        ),
        row=2,
        col=1,
    )

    # RSI
    if rsi_series is not None:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=rsi_series,
                name="RSI(14)",
                line=dict(color="#29b6f6", width=1.5),
                hovertemplate="RSI: %{y:.1f}<extra></extra>",
            ),
            row=3,
            col=1,
        )
        for level, color in [(30, "#ff5252"), (50, "#78909c"), (70, "#00e676")]:
            fig.add_hline(
                y=level,
                line_dash="dot",
                line_color=color,
                line_width=1,
                annotation_text=str(level),
                annotation_position="right",
                row=3,
                col=1,
            )
        fig.update_yaxes(range=[0, 100], row=3, col=1, title_text="RSI")

    # TradingView 다크 테마 레이아웃 설정
    fig.update_layout(
        height=720,
        margin=dict(l=10, r=10, t=30, b=10),
        template="plotly_dark",
        paper_bgcolor="#131722",
        plot_bgcolor="#131722",
        hovermode="x unified",
        xaxis_rangeslider_visible=True,  # 하단 슬라이더 활성화
        xaxis_rangeslider_thickness=0.05,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
        ),
    )

    # 마우스 스크롤 줌 및 오토 스케일 활성화
    fig.update_xaxes(
        gridcolor="#2a2e39",
        zerolinecolor="#2a2e39",
        rangeslider_visible=False,  # 메인 축에만 슬라이더 비활성화 설정
    )
    fig.update_yaxes(gridcolor="#2a2e39", zerolinecolor="#2a2e39", fixedrange=False)

    return fig


def create_gauge_chart(buy_ratio):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=buy_ratio,
            title={"text": "가중 매수 컨센서스 강도 (%)", "font": {"color": "#ffffff"}},
            number={"font": {"color": "#ffffff"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#ffffff"},
                "bar": {
                    "color": "#2ecc71"
                    if buy_ratio >= 70
                    else ("#e74c3c" if buy_ratio <= 30 else "#f39c12")
                },
                "steps": [
                    {"range": [0, 30], "color": "#3c1818"},
                    {"range": [30, 70], "color": "#3c3318"},
                    {"range": [70, 100], "color": "#183c24"},
                ],
            },
        )
    )
    fig.update_layout(
        height=240,
        margin=dict(l=20, r=20, t=20, b=20),
        template="plotly_dark",
        paper_bgcolor="#1e222d",
        plot_bgcolor="#1e222d",
    )
    return fig


def create_indicator_status_chart(signals):
    valid_sigs = {k: v for k, v in signals.items() if not pd.isna(v)}
    names = list(valid_sigs.keys())
    vals = list(valid_sigs.values())
    colors = [
        "#2ecc71" if v == 1 else ("#e74c3c" if v == -1 else "#78909c")
        for v in vals
    ]

    fig = go.Figure(
        go.Bar(
            x=vals,
            y=names,
            orientation="h",
            marker=dict(color=colors),
            text=["매수" if v == 1 else ("매도" if v == -1 else "중립") for v in vals],
            textposition="auto",
        )
    )
    fig.update_layout(
        title="지표별 시그널 상태 (1: 매수 / -1: 매도)",
        xaxis=dict(
            range=[-1.2, 1.2],
            tickvals=[-1, 0, 1],
            ticktext=["매도(-1)", "중립(0)", "매수(+1)"],
            gridcolor="#2a2e39",
        ),
        yaxis=dict(gridcolor="#2a2e39"),
        height=320,
        margin=dict(l=10, r=10, t=40, b=10),
        template="plotly_dark",
        paper_bgcolor="#1e222d",
        plot_bgcolor="#1e222d",
    )
    return fig


# ---------------------------------------------------------
# 6. 백테스팅 엔진
# ---------------------------------------------------------
def run_backtest(ind, hold_days=5, threshold_ratio=0.7):
    trades = []
    buy_signals_dates = []
    skip_until = -1

    max_idx = len(ind) - hold_days - 2
    if max_idx < MIN_ROWS_REQUIRED:
        return None, []

    for i in range(MIN_ROWS_REQUIRED, max_idx + 1):
        signals, _, _, _ = build_signal_row(ind, i)

        weighted_buy, total_weight = 0.0, 0.0
        for name, sig in signals.items():
            if not pd.isna(sig):
                w = INDICATOR_WEIGHTS.get(name, 1.0)
                total_weight += w
                if sig == 1:
                    weighted_buy += w

        if total_weight == 0:
            continue

        buy_ratio = weighted_buy / total_weight

        if buy_ratio >= threshold_ratio:
            buy_signals_dates.append(ind.index[i])

            if i < skip_until:
                continue

            entry_price = ind["open"].iloc[i + 1]
            exit_price = ind["open"].iloc[i + hold_days + 1]
            ret = (exit_price - entry_price) / entry_price
            trades.append({"entry_date": ind.index[i + 1], "return": ret})

            skip_until = i + hold_days

    if not trades:
        return None, buy_signals_dates

    tdf = pd.DataFrame(trades)
    win_rate = (tdf["return"] > 0).mean() * 100
    cum_returns = (1 + tdf["return"]).cumprod()
    cum_return = (cum_returns.iloc[-1] - 1) * 100
    avg_return = tdf["return"].mean() * 100

    peak = cum_returns.cummax()
    drawdown = (cum_returns - peak) / peak
    mdd = drawdown.min() * 100

    gains = tdf[tdf["return"] > 0]["return"].sum()
    losses = abs(tdf[tdf["return"] < 0]["return"].sum())
    profit_factor = (
        (gains / losses) if losses > 0 else (gains if gains > 0 else 0)
    )

    return {
        "total_trades": len(tdf),
        "win_rate": win_rate,
        "cum_return": cum_return,
        "avg_return": avg_return,
        "mdd": mdd,
        "profit_factor": profit_factor,
    }, buy_signals_dates


# ---------------------------------------------------------
# 7. 메인 UI 및 화면 구성
# ---------------------------------------------------------
st.title("🎯 10대 지표 컨센서스 타점 분석기 Pro")
st.caption(
    "TradingView 다크 모드 스타일이 적용되었습니다. 차트 영역 위에서 마우스 휠을 스크롤하여 확대/축소가 가능합니다."
)

st.sidebar.header("⚙️ 지표별 가중치 설정")
for _name, _default_w in DEFAULT_WEIGHTS.items():
    INDICATOR_WEIGHTS[_name] = st.sidebar.slider(
        f"{_name} 가중치", 0.0, 3.0, _default_w, 0.1
    )

if "analyzed" not in st.session_state:
    st.session_state.analyzed = False
if "ticker" not in st.session_state:
    st.session_state.ticker = "TSLA"
if "recent_tickers" not in st.session_state:
    st.session_state.recent_tickers = []


def _set_active_ticker(t):
    t = t.strip().upper()
    if not t:
        return
    st.session_state.ticker = t
    st.session_state.analyzed = True
    recents = [r for r in st.session_state.recent_tickers if r != t]
    recents.insert(0, t)
    st.session_state.recent_tickers = recents[:8]


col_input, col_btn = st.columns([4, 1])
with col_input:
    input_ticker = st.text_input(
        "종목 티커 입력 (예: TSLA, 005930.KS, BTC-USD, ETH-USD)",
        st.session_state.ticker,
    )
with col_btn:
    st.write(" ")
    st.write(" ")
    if st.button("타점 분석 및 검증 개시", use_container_width=True):
        _set_active_ticker(input_ticker)

if st.session_state.recent_tickers:
    st.caption("최근 검색")
    recent_cols = st.columns(len(st.session_state.recent_tickers))
    for col, t in zip(recent_cols, st.session_state.recent_tickers):
        with col:
            if st.button(t, key=f"recent_{t}", use_container_width=True):
                _set_active_ticker(t)
                st.rerun()

if st.session_state.analyzed:
    ticker = st.session_state.ticker
    df, ind = get_indicator_series(ticker)

    if df.empty:
        st.error("데이터를 불러오지 못했습니다. 티커명을 확인해주세요.")
    else:
        if len(df) < MIN_ROWS_REQUIRED:
            st.warning(
                f"⚠️ 데이터 수량({len(df)}건)이 부족하여 일부 지표 계산의 정확도가 낮아질 수 있습니다."
            )

        signals, current_price, na_indicators, atr14 = build_signal_row(
            ind, len(ind) - 1
        )

        weighted_buy, weighted_sell, total_weight = 0.0, 0.0, 0.0
        for k, v in signals.items():
            if not pd.isna(v):
                w = INDICATOR_WEIGHTS.get(k, 1.0)
                total_weight += w
                if v == 1:
                    weighted_buy += w
                elif v == -1:
                    weighted_sell += w

        is_krw = ticker.endswith(".KS") or ticker.endswith(".KQ")

        if total_weight == 0:
            st.error("계산 가능한 지표가 없습니다.")
        else:
            buy_ratio = (weighted_buy / total_weight) * 100
            sell_ratio = (weighted_sell / total_weight) * 100

            col1, col2, col3 = st.columns(3)
            price_display = format_price(current_price, is_krw)
            col1.metric("현재가", price_display)
            col2.metric("가중 매수 비율", f"{buy_ratio:.0f}%")
            col3.metric("가중 매도 비율", f"{sell_ratio:.0f}%")

            if na_indicators:
                st.caption(f"※ 데이터 부족 제외 지표: {', '.join(na_indicators)}")

            target_price = (
                current_price + (atr14 * 2) if not pd.isna(atr14) else None
            )
            stop_price = (
                current_price - (atr14 * 1.2) if not pd.isna(atr14) else None
            )
            support_price = (
                current_price - (atr14 * 2) if not pd.isna(atr14) else None
            )

            def fmt(v):
                return format_price(v, is_krw)

            bt_res_default, buy_signals_dates = run_backtest(
                ind, hold_days=5, threshold_ratio=0.7
            )

            main_tab1, main_tab2, main_tab3 = st.tabs(
                [
                    "📈 실시간 분석 & 차트",
                    "🧪 백테스트 (MDD/손익비)",
                    "🔍 관심종목 멀티 스캐너",
                ]
            )

            with main_tab2:
                st.subheader("🧪 과거 1년 가중 시그널(≥70%) 성과 검증")
                hold_period = st.slider("보유 기간 (일)", 1, 20, 5)

                if hold_period == 5:
                    bt_res = bt_res_default
                else:
                    bt_res, _ = run_backtest(
                        ind, hold_days=hold_period, threshold_ratio=0.7
                    )

                if bt_res:
                    b1, b2, b3, b4, b5, b6 = st.columns(6)
                    b1.metric("총 매수 신호", f"{bt_res['total_trades']}회")
                    b2.metric("승률", f"{bt_res['win_rate']:.1f}%")
                    b3.metric("평균 수익률", f"{bt_res['avg_return']:.2f}%")
                    b4.metric("누적 수익률", f"{bt_res['cum_return']:.1f}%")
                    b5.metric("최대 낙폭 (MDD)", f"{bt_res['mdd']:.1f}%")
                    b6.metric("손익비 (Profit Factor)", f"{bt_res['profit_factor']:.2f}")
                else:
                    st.info(
                        "과거 1년 내 '70% 이상 매수 컨센서스'가 발생한 타점이 없습니다."
                    )

            with main_tab1:
                st.subheader("📌 컨센서스 진단")
                gauge_col, msg_col = st.columns([1, 2])
                with gauge_col:
                    st.plotly_chart(
                        create_gauge_chart(buy_ratio), use_container_width=True
                    )
                with msg_col:
                    if buy_ratio >= 70:
                        st.success(
                            f"🟢 **강한 매수 우위** ({buy_ratio:.0f}%) — 주요 지표들이 일치된 상방을 나타냅니다."
                        )
                    elif sell_ratio >= 70:
                        st.error(
                            f"🔴 **강한 매도 우위** ({sell_ratio:.0f}%) — 하방 압력이 높습니다."
                        )
                    else:
                        st.warning(
                            f"🟡 **관망 구간** (매수 {buy_ratio:.0f}%, 매도 {sell_ratio:.0f}%)"
                        )

                    st.markdown(
                        f"**현재가:** {fmt(current_price)} · "
                        f"**목표가(ATR×2):** {fmt(target_price)} · "
                        f"**손절가(ATR×1.2):** {fmt(stop_price)} · "
                        f"**지지선(ATR×2):** {fmt(support_price)}"
                    )

                st.write("---")
                st.subheader("📊 차트 및 지표 현황")
                
                # Plotly 마우스 스크롤 줌 및 트레이딩뷰 컨트롤 옵션 적용
                st.plotly_chart(
                    create_interactive_chart(
                        df,
                        target_price,
                        stop_price,
                        buy_signals_dates,
                        is_krw,
                        rsi_series=ind["rsi"],
                    ),
                    use_container_width=True,
                    config={
                        "scrollZoom": True,  # 마우스 휠 스크롤 줌 활성화
                        "displayModeBar": True,
                        "modeBarButtonsToAdd": ["drawline", "eraseshape"],
                    },
                )
                st.plotly_chart(
                    create_indicator_status_chart(signals),
                    use_container_width=True,
                )

            with main_tab3:
                st.subheader("🔍 관심종목 멀티 스캐너")
                watchlist_input = st.text_area(
                    "티커 목록 (쉼표로 구분)",
                    "TSLA, NVDA, AAPL, MSFT, 005930.KS, BTC-USD, ETH-USD",
                )
                if st.button("멀티 스캔 실행", use_container_width=True):
                    tickers = [
                        t.strip().upper()
                        for t in watchlist_input.split(",")
                        if t.strip()
                    ]
                    scan_results = []
                    progress_bar = st.progress(0)
                    for idx, t_sym in enumerate(tickers):
                        t_df, t_ind = get_indicator_series(t_sym)
                        if not t_df.empty and len(t_ind) >= MIN_ROWS_REQUIRED:
                            t_sigs, t_close, _, _ = build_signal_row(
                                t_ind, len(t_ind) - 1
                            )
                            w_buy, w_total = 0.0, 0.0
                            for k_s, v_s in t_sigs.items():
                                if not pd.isna(v_s):
                                    w = INDICATOR_WEIGHTS.get(k_s, 1.0)
                                    w_total += w
                                    if v_s == 1:
                                        w_buy += w

                            if w_total > 0:
                                b_ratio = w_buy / w_total * 100
                                status = (
                                    "🟢 매수"
                                    if b_ratio >= 70
                                    else ("🔴 매도" if b_ratio <= 30 else "🟡 중립")
                                )
                                ratio_display = round(b_ratio, 1)
                            else:
                                ratio_display = np.nan
                                status = "⚪ 데이터 부족"

                            scan_results.append(
                                {
                                    "티커": t_sym,
                                    "현재가": format_price(
                                        t_close,
                                        t_sym.endswith(".KS")
                                        or t_sym.endswith(".KQ"),
                                    ),
                                    "가중 매수 강도 (%)": ratio_display,
                                    "상태": status,
                                }
                            )
                        progress_bar.progress((idx + 1) / len(tickers))

                    st.session_state.scan_results = scan_results
                    progress_bar.empty()

                if (
                    "scan_results" in st.session_state
                    and st.session_state.scan_results
                ):
                    res_df = pd.DataFrame(
                        st.session_state.scan_results
                    ).sort_values(
                        by="가중 매수 강도 (%)", ascending=False, na_position="last"
                    )
                    st.dataframe(res_df, use_container_width=True)
