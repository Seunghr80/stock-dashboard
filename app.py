import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
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
# 실제 계산에 쓰이는 가중치. 사이드바에서 사용자가 조절하면 이 값이 갱신된다.
# (build_signal_row/run_backtest는 이 전역 딕셔너리를 실행 시점에 조회한다)
INDICATOR_WEIGHTS = dict(DEFAULT_WEIGHTS)
 
 
def format_price(v, is_krw=False):
    """주식/코인 가격 표시용 포맷터. 코인은 $1 미만 종목이 흔해서 소수점 자리를 자동으로 늘린다."""
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
        # yfinance MultiIndex 컬럼 단일화
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
 
        # 필수 컬럼 존재 여부 확인 및 결측치 제거 (일부 티커는 Volume이 없을 수 있음)
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
 
    # 초기 추세 판단 (첫 2개 봉 기준)
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
            # SAR 범위 제한 (이전 2개 봉의 Low보다 높을 수 없음) — 반전 판단 전에 먼저 적용
            psar[i] = min(psar[i], low[i - 1], low[i - 2])
            if low[i] < psar[i]:  # 추세 반전 (상승 -> 하강)
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
            # SAR 범위 제한 (이전 2개 봉의 High보다 낮을 수 없음) — 반전 판단 전에 먼저 적용
            psar[i] = max(psar[i], high[i - 1], high[i - 2])
            if high[i] > psar[i]:  # 추세 반전 (하강 -> 상승)
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
# 3. 전체 기간 지표 시계열 벡터화 계산
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
        cci = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).std().replace(0, np.nan))
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
            [(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
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
 
    # 1. SMA Cross
    if i >= 1:
        prev = ind.iloc[i - 1]
        vals = (row["sma20"], row["sma60"], prev["sma20"], prev["sma60"])
        if any(pd.isna(v) for v in vals):
            signals["SMA_Cross"] = np.nan
            na_indicators.append("SMA_Cross")
        else:
            is_golden = (prev["sma20"] <= prev["sma60"]) and (row["sma20"] > row["sma60"])
            is_dead = (prev["sma20"] >= prev["sma60"]) and (row["sma20"] < row["sma60"])
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
    set_signal("MACD", row["macd"] > row["macd_signal"], row["macd"] < row["macd_signal"],
                row["macd"], row["macd_signal"])
    set_signal("PSAR", bool(row["psar_bull"]), not bool(row["psar_bull"]), row["psar"])
    set_signal("Bollinger", row["close"] <= row["bb_lower"], row["close"] >= row["bb_upper"],
                row["bb_lower"], row["bb_upper"])
    set_signal(
        "Stochastic",
        (row["stoch_k"] < 20) and (row["stoch_k"] > row["stoch_d"]),
        (row["stoch_k"] > 80) and (row["stoch_k"] < row["stoch_d"]),
        row["stoch_k"], row["stoch_d"],
    )
    set_signal("CCI", row["cci"] < -100, row["cci"] > 100, row["cci"])
    set_signal("MFI", row["mfi"] < 20, row["mfi"] > 80, row["mfi"])
    set_signal("Envelope", row["close"] <= row["env_lower"], row["close"] >= row["env_upper"],
                row["env_lower"], row["env_upper"])
    set_signal("Momentum", row["momentum"] > 0, row["momentum"] < 0, row["momentum"])
 
    return signals, row["close"], na_indicators, row["atr14"]
 
 
# ---------------------------------------------------------
# 5. 시각화 모듈
# ---------------------------------------------------------
def create_interactive_chart(df, target_price=None, stop_price=None, buy_signals_dates=None, is_krw=False):
    buy_signals_dates = buy_signals_dates or []
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.03
    )
 
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="주가"
    ), row=1, col=1)
 
    sma20 = df["Close"].rolling(20).mean()
    sma60 = df["Close"].rolling(60).mean()
    fig.add_trace(go.Scatter(x=df.index, y=sma20, name="SMA 20", line=dict(color="orange", width=1.2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=sma60, name="SMA 60", line=dict(color="purple", width=1.2)), row=1, col=1)
 
    if buy_signals_dates:
        valid_dates = [d for d in buy_signals_dates if d in df.index]
        if valid_dates:
            buy_lows = df.loc[valid_dates, "Low"] * 0.985
            fig.add_trace(go.Scatter(
                x=valid_dates, y=buy_lows, mode="markers+text", name="매수 타점",
                marker=dict(symbol="triangle-up", size=11, color="#2ecc71"),
                text=["▲매수"] * len(valid_dates), textposition="bottom center",
                textfont=dict(color="#2ecc71", size=10),
            ), row=1, col=1)
 
    # add_hline은 배경 도형이라 마우스를 올려도 값이 안 뜨므로, 호버 툴팁이 뜨도록
    # 실제 데이터 트레이스(가로선)로 그린다.
    if target_price is not None and not np.isnan(target_price):
        price_str = format_price(target_price, is_krw)
        fig.add_trace(go.Scatter(
            x=[df.index.min(), df.index.max()], y=[target_price, target_price],
            mode="lines", line=dict(color="green", dash="dash", width=1.5),
            name=f"목표가(ATR) {price_str}",
            hovertemplate=f"목표가(ATR): {price_str}<extra></extra>",
        ), row=1, col=1)
    if stop_price is not None and not np.isnan(stop_price):
        price_str = format_price(stop_price, is_krw)
        fig.add_trace(go.Scatter(
            x=[df.index.min(), df.index.max()], y=[stop_price, stop_price],
            mode="lines", line=dict(color="red", dash="dash", width=1.5),
            name=f"손절가(ATR) {price_str}",
            hovertemplate=f"손절가(ATR): {price_str}<extra></extra>",
        ), row=1, col=1)
 
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="거래량", marker_color="gray"), row=2, col=1)
 
    fig.update_layout(
        height=520, margin=dict(l=10, r=10, t=30, b=10), xaxis_rangeslider_visible=False,
        template="plotly_white", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig
 
 
def create_gauge_chart(buy_ratio):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=buy_ratio,
        title={"text": "가중 매수 컨센서스 강도 (%)"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#2ecc71" if buy_ratio >= 70 else ("#e74c3c" if buy_ratio <= 30 else "#f39c12")},
            "steps": [
                {"range": [0, 30], "color": "#fadbd8"},
                {"range": [30, 70], "color": "#fdebd0"},
                {"range": [70, 100], "color": "#d4efdf"},
            ],
        },
    ))
    fig.update_layout(height=240, margin=dict(l=20, r=20, t=20, b=20))
    return fig
 
 
def create_indicator_status_chart(signals):
    valid_sigs = {k: v for k, v in signals.items() if not pd.isna(v)}
    names = list(valid_sigs.keys())
    vals = list(valid_sigs.values())
    colors = ["#2ecc71" if v == 1 else ("#e74c3c" if v == -1 else "#bdc3c7") for v in vals]
 
    fig = go.Figure(go.Bar(
        x=vals, y=names, orientation="h", marker=dict(color=colors),
        text=["매수" if v == 1 else ("매도" if v == -1 else "중립") for v in vals],
        textposition="auto",
    ))
    fig.update_layout(
        title="지표별 시그널 상태 (1: 매수 / -1: 매도)",
        xaxis=dict(range=[-1.2, 1.2], tickvals=[-1, 0, 1], ticktext=["매도(-1)", "중립(0)", "매수(+1)"]),
        height=320, margin=dict(l=10, r=10, t=40, b=10), template="plotly_white",
    )
    return fig
 
 
# ---------------------------------------------------------
# 6. 백테스팅 엔진 (중복 보유 스킵 로직 적용, 단일 포지션 가정)
# ---------------------------------------------------------
def run_backtest(ind, hold_days=5, threshold_ratio=0.7):
    trades = []
    buy_signals_dates = []
    skip_until = -1  # 중복 포지션 체결 방지 쿨다운 인덱스
 
    # 진입(i+1일 시가) ~ 청산(i+hold_days+1일 시가)까지 정확히 hold_days일을 보유하도록
    # 인덱스를 계산한다. 청산 인덱스가 len(ind)-1을 넘지 않으려면 i <= len(ind)-hold_days-2 여야 한다.
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
                continue  # 이전 포지션 보유 중이면 백테스트 거래 집계에서는 중복 진입 스킵
 
            entry_price = ind["open"].iloc[i + 1]
            # 청산: 진입일로부터 정확히 hold_days일 후의 시가로 청산 (종가 대신 시가로 통일)
            exit_price = ind["open"].iloc[i + hold_days + 1]
            ret = (exit_price - entry_price) / entry_price
            trades.append({"entry_date": ind.index[i + 1], "return": ret})
 
            skip_until = i + hold_days  # 청산 시까지 다음 진입 방지
 
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
    profit_factor = (gains / losses) if losses > 0 else (gains if gains > 0 else 0)
 
    return {
        "total_trades": len(tdf), "win_rate": win_rate, "cum_return": cum_return,
        "avg_return": avg_return, "mdd": mdd, "profit_factor": profit_factor,
    }, buy_signals_dates
 
 
# ---------------------------------------------------------
# 7. 메인 UI 및 화면 구성 (Session State로 결과 유지)
# ---------------------------------------------------------
st.title("🎯 10대 지표 컨센서스 타점 분석기 Pro")
st.caption("가중 지표 컨센서스, PSAR 정밀 계산, 차트 타점 마커 및 백테스트 리스크 지표가 통합된 전문 대시보드입니다. 주식뿐 아니라 BTC-USD, ETH-USD 같은 암호화폐 티커도 지원합니다.")
 
# 사이드바: 사용자가 지표별 가중치 직접 조절
st.sidebar.header("⚙️ 지표별 가중치 설정")
for _name, _default_w in DEFAULT_WEIGHTS.items():
    INDICATOR_WEIGHTS[_name] = st.sidebar.slider(f"{_name} 가중치", 0.0, 3.0, _default_w, 0.1)
 
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
    # 최근 검색 목록 갱신: 중복 제거 후 맨 앞에 추가, 최대 8개 유지
    recents = [r for r in st.session_state.recent_tickers if r != t]
    recents.insert(0, t)
    st.session_state.recent_tickers = recents[:8]
 
 
col_input, col_btn = st.columns([4, 1])
with col_input:
    input_ticker = st.text_input("종목 티커 입력 (예: TSLA, 005930.KS, BTC-USD, ETH-USD)", st.session_state.ticker)
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
            st.warning(f"⚠️ 데이터 수량({len(df)}건)이 부족하여 일부 지표 계산의 정확도가 낮아질 수 있습니다.")
 
        signals, current_price, na_indicators, atr14 = build_signal_row(ind, len(ind) - 1)
 
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
 
            target_price = current_price + (atr14 * 2) if not pd.isna(atr14) else None
            stop_price = current_price - (atr14 * 1.2) if not pd.isna(atr14) else None
            support_price = current_price - (atr14 * 2) if not pd.isna(atr14) else None
 
            def fmt(v):
                return format_price(v, is_krw)
 
            # 탭 분리 전에 기본값(5일 보유)으로 백테스트를 먼저 실행해 차트 마커(buy_signals_dates)를
            # 확보한다. 탭2의 슬라이더를 조작해도 탭1 차트의 마커는 이 기본값 기준으로 안정적으로 유지된다.
            bt_res_default, buy_signals_dates = run_backtest(ind, hold_days=5, threshold_ratio=0.7)
 
            main_tab1, main_tab2, main_tab3 = st.tabs([
                "📈 실시간 분석 & 차트",
                "🧪 백테스트 (MDD/손익비)",
                "🔍 관심종목 멀티 스캐너",
            ])
 
            with main_tab2:
                st.subheader("🧪 과거 1년 가중 시그널(≥70%) 성과 검증")
                hold_period = st.slider("보유 기간 (일)", 1, 20, 5)
 
                if hold_period == 5:
                    bt_res = bt_res_default  # 위에서 이미 계산한 기본값 결과 재사용
                else:
                    bt_res, _ = run_backtest(ind, hold_days=hold_period, threshold_ratio=0.7)
 
                if bt_res:
                    b1, b2, b3, b4, b5, b6 = st.columns(6)
                    b1.metric("총 매수 신호", f"{bt_res['total_trades']}회")
                    b2.metric("승률", f"{bt_res['win_rate']:.1f}%")
                    b3.metric("평균 수익률", f"{bt_res['avg_return']:.2f}%")
                    b4.metric("누적 수익률", f"{bt_res['cum_return']:.1f}%")
                    b5.metric("최대 낙폭 (MDD)", f"{bt_res['mdd']:.1f}%")
                    b6.metric("손익비 (Profit Factor)", f"{bt_res['profit_factor']:.2f}")
                    st.caption("※ 청산 전 중복 매수 신호는 스킵되며, 단일 포지션 보유 기준으로 백테스트되었습니다. 수수료·슬리피지는 반영되지 않았습니다.")
                else:
                    st.info("과거 1년 내 '70% 이상 매수 컨센서스'가 발생한 타점이 없습니다.")
 
            with main_tab1:
                st.subheader("📌 컨센서스 진단")
                gauge_col, msg_col = st.columns([1, 2])
                with gauge_col:
                    st.plotly_chart(create_gauge_chart(buy_ratio), use_container_width=True)
                with msg_col:
                    if buy_ratio >= 70:
                        st.success(f"🟢 **강한 매수 우위** ({buy_ratio:.0f}%) — 주요 지표들이 일치된 상방을 나타냅니다.")
                    elif sell_ratio >= 70:
                        st.error(f"🔴 **강한 매도 우위** ({sell_ratio:.0f}%) — 하방 압력이 높습니다. 진입 보류 및 비중 축소를 권장합니다.")
                    else:
                        st.warning(f"🟡 **관망 구간** (매수 {buy_ratio:.0f}%, 매도 {sell_ratio:.0f}%)")
                        st.info("💡 뚜렷한 추세가 나타날 때까지 대기하는 것이 안전합니다.")
 
                    st.markdown(
                        f"**현재가:** {fmt(current_price)} · "
                        f"**목표가(ATR×2):** {fmt(target_price)} · "
                        f"**손절가(ATR×1.2):** {fmt(stop_price)} · "
                        f"**지지선(ATR×2):** {fmt(support_price)}"
                    )
 
                st.write("---")
                st.subheader("📊 차트 및 지표 현황")
                st.plotly_chart(create_interactive_chart(df, target_price, stop_price, buy_signals_dates, is_krw), use_container_width=True)
                st.plotly_chart(create_indicator_status_chart(signals), use_container_width=True)
 
            with main_tab3:
                st.subheader("🔍 관심종목 멀티 스캐너")
                st.write("여러 관심 종목을 입력하여 현재 10대 지표 매수 컨센서스 순위를 한눈에 비교하세요.")
 
                if "scan_results" not in st.session_state:
                    st.session_state.scan_results = None  # None = 아직 스캔 전
 
                watchlist_input = st.text_area("티커 목록 (쉼표로 구분)", "TSLA, NVDA, AAPL, MSFT, 005930.KS, BTC-USD, ETH-USD")
                if st.button("멀티 스캔 실행", use_container_width=True):
                    tickers = [t.strip().upper() for t in watchlist_input.split(",") if t.strip()]
                    scan_results = []
 
                    progress_bar = st.progress(0)
                    for idx, t_sym in enumerate(tickers):
                        t_df, t_ind = get_indicator_series(t_sym)
                        if not t_df.empty and len(t_ind) >= MIN_ROWS_REQUIRED:
                            t_sigs, t_close, _, _ = build_signal_row(t_ind, len(t_ind) - 1)
                            w_buy, w_total = 0.0, 0.0
                            for k_s, v_s in t_sigs.items():
                                if not pd.isna(v_s):
                                    w = INDICATOR_WEIGHTS.get(k_s, 1.0)
                                    w_total += w
                                    if v_s == 1:
                                        w_buy += w
 
                            if w_total > 0:
                                b_ratio = w_buy / w_total * 100
                                status = "🟢 매수" if b_ratio >= 70 else ("🔴 매도" if b_ratio <= 30 else "🟡 중립")
                                ratio_display = round(b_ratio, 1)
                            else:
                                ratio_display = np.nan  # 지표 계산 불가 — 매도가 아니라 판단 불가
                                status = "⚪ 데이터 부족"
 
                            scan_results.append({
                                "티커": t_sym,
                                "현재가": format_price(t_close, t_sym.endswith(".KS") or t_sym.endswith(".KQ")),
                                "가중 매수 강도 (%)": ratio_display,
                                "상태": status,
                            })
                        progress_bar.progress((idx + 1) / len(tickers))
 
                    st.session_state.scan_results = scan_results  # 빈 리스트여도 "스캔은 했음"으로 저장
                    progress_bar.empty()
 
                if st.session_state.scan_results is not None:
                    if st.session_state.scan_results:
                        res_df = pd.DataFrame(st.session_state.scan_results).sort_values(
                            by="가중 매수 강도 (%)", ascending=False, na_position="last"
                        )
                        st.dataframe(res_df, use_container_width=True)
 
                        chart_df = res_df.dropna(subset=["가중 매수 강도 (%)"])
                        if not chart_df.empty:
                            fig_scan = go.Figure(go.Bar(
                                x=chart_df["가중 매수 강도 (%)"],
                                y=chart_df["티커"],
                                orientation="h",
                                marker=dict(color="#2ecc71"),
                                text=chart_df["가중 매수 강도 (%)"].astype(str) + "%",
                                textposition="auto",
                            ))
                            fig_scan.update_layout(
                                title="관심 종목 매수 컨센서스 순위",
                                xaxis=dict(range=[0, 100]),
                                yaxis=dict(autorange="reversed"),  # 고순위 종목이 차트 상단에 오도록 설정
                                height=350,
                                template="plotly_white",
                            )
                            st.plotly_chart(fig_scan, use_container_width=True)
                    else:
                        st.warning("스캔 결과를 가져올 수 있는 종목이 없습니다.")
 
            st.caption("※ 본 대시보드의 결과는 참고용이며, 주식 매매 시 발생하는 최종 손익 책임은 투자자 본인에게 있습니다.")
