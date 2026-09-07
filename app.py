import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# -----------------------------------------------------------------------------
# 1. 페이지 및 기본 설정
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="주식/가상자산 기술적 분석 대시보드",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

MIN_ROWS_REQUIRED = 30

INDICATOR_WEIGHTS = {
    "RSI": 1.5,
    "MACD": 1.5,
    "SMA_Cross": 1.0,
    "Stochastic": 1.0,
    "Bollinger": 1.0,
    "CCI": 1.0,
    "Williams_R": 1.0,
    "PSAR": 1.0
}

# -----------------------------------------------------------------------------
# 2. 데이터 수집 및 정교한 지표 계산
# -----------------------------------------------------------------------------
@st.cache_data(ttl=300)
def fetch_stock_data(ticker_symbol, period="1y"):
    """yfinance 시세 데이터 수집"""
    try:
        df = yf.download(ticker_symbol, period=period, progress=False)
        if df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.dropna()
        return df
    except Exception:
        return pd.DataFrame()


def calculate_technical_indicators(df):
    """8가지 주요 기술적 지표 정상 계산"""
    if df.empty or len(df) < MIN_ROWS_REQUIRED:
        return pd.DataFrame()

    ind_df = df.copy()

    # 1. 이동평균선 (SMA 20, 60)
    ind_df['SMA20'] = ind_df['Close'].rolling(window=20).mean()
    ind_df['SMA60'] = ind_df['Close'].rolling(window=60).mean()

    # 2. RSI (14)
    delta = ind_df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan)
    ind_df['RSI'] = 100 - (100 / (1 + rs))

    # 3. MACD
    ema12 = ind_df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = ind_df['Close'].ewm(span=26, adjust=False).mean()
    ind_df['MACD'] = ema12 - ema26
    ind_df['MACD_Signal'] = ind_df['MACD'].ewm(span=9, adjust=False).mean()
    ind_df['MACD_Hist'] = ind_df['MACD'] - ind_df['MACD_Signal']

    # 4. 스토캐스틱 (%K, %D)
    low14 = ind_df['Low'].rolling(window=14).min()
    high14 = ind_df['High'].rolling(window=14).max()
    ind_df['Stoch_K'] = (ind_df['Close'] - low14) / (high14 - low14).replace(0, np.nan) * 100
    ind_df['Stoch_D'] = ind_df['Stoch_K'].rolling(window=3).mean()

    # 5. 볼린저 밴드
    std20 = ind_df['Close'].rolling(window=20).std()
    ind_df['BB_Upper'] = ind_df['SMA20'] + (std20 * 2)
    ind_df['BB_Lower'] = ind_df['SMA20'] - (std20 * 2)

    # 6. CCI (20)
    tp = (ind_df['High'] + ind_df['Low'] + ind_df['Close']) / 3
    sma_tp = tp.rolling(window=20).mean()
    mad = tp.rolling(window=20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    ind_df['CCI'] = (tp - sma_tp) / (0.015 * mad.replace(0, np.nan))

    # 7. Williams %R (14)
    ind_df['Williams_R'] = ((high14 - ind_df['Close']) / (high14 - low14).replace(0, np.nan)) * -100

    # 8. ATR (변동성 계산용)
    tr1 = ind_df['High'] - ind_df['Low']
    tr2 = (ind_df['High'] - ind_df['Close'].shift(1)).abs()
    tr3 = (ind_df['Low'] - ind_df['Close'].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    ind_df['ATR'] = tr.rolling(window=14).mean()

    # 9. Parabolic SAR
    highs = ind_df['High'].values
    lows = ind_df['Low'].values
    closes = ind_df['Close'].values
    psar = closes.copy()
    af = 0.02
    max_af = 0.2
    bull = True
    hp = highs[0]
    lp = lows[0]

    for i in range(2, len(closes)):
        if bull:
            psar[i] = psar[i - 1] + af * (hp - psar[i - 1])
            psar[i] = min(psar[i], lows[i - 1], lows[i - 2])
            if lows[i] < psar[i]:
                bull = False
                psar[i] = hp
                lp = lows[i]
                af = 0.02
            else:
                if highs[i] > hp:
                    hp = highs[i]
                    af = min(af + 0.02, max_af)
        else:
            psar[i] = psar[i - 1] + af * (lp - psar[i - 1])
            psar[i] = max(psar[i], highs[i - 1], highs[i - 2])
            if highs[i] > psar[i]:
                bull = True
                psar[i] = lp
                hp = highs[i]
                af = 0.02
            else:
                if lows[i] < lp:
                    lp = lows[i]
                    af = min(af + 0.02, max_af)

    ind_df['PSAR'] = psar

    return ind_df


def build_signal_row(ind_df, idx_pos):
    """현실적인 구간 및 추세 기반 8대 지표 신호 판정"""
    if idx_pos < 1 or idx_pos >= len(ind_df):
        return {}, np.nan, np.nan, np.nan

    curr = ind_df.iloc[idx_pos]
    signals = {}

    # 1. RSI
    if not pd.isna(curr['RSI']):
        if curr['RSI'] <= 40:
            signals['RSI'] = 1
        elif curr['RSI'] >= 60:
            signals['RSI'] = -1
        else:
            signals['RSI'] = 0

    # 2. MACD
    if not (pd.isna(curr['MACD']) or pd.isna(curr['MACD_Signal'])):
        if curr['MACD'] > curr['MACD_Signal']:
            signals['MACD'] = 1
        elif curr['MACD'] < curr['MACD_Signal']:
            signals['MACD'] = -1
        else:
            signals['MACD'] = 0

    # 3. SMA Cross
    if not (pd.isna(curr['SMA20']) or pd.isna(curr['SMA60'])):
        if curr['SMA20'] > curr['SMA60']:
            signals['SMA_Cross'] = 1
        elif curr['SMA20'] < curr['SMA60']:
            signals['SMA_Cross'] = -1
        else:
            signals['SMA_Cross'] = 0

    # 4. Stochastic
    if not (pd.isna(curr['Stoch_K']) or pd.isna(curr['Stoch_D'])):
        if curr['Stoch_K'] > curr['Stoch_D'] and curr['Stoch_K'] < 80:
            signals['Stochastic'] = 1
        elif curr['Stoch_K'] < curr['Stoch_D'] and curr['Stoch_K'] > 20:
            signals['Stochastic'] = -1
        else:
            signals['Stochastic'] = 0

    # 5. Bollinger
    if not (pd.isna(curr['BB_Lower']) or pd.isna(curr['BB_Upper']) or pd.isna(curr['SMA20'])):
        if curr['Close'] < curr['SMA20']:
            signals['Bollinger'] = 1
        elif curr['Close'] > curr['BB_Upper']:
            signals['Bollinger'] = -1
        else:
            signals['Bollinger'] = 0

    # 6. CCI
    if not pd.isna(curr['CCI']):
        if curr['CCI'] < -50:
            signals['CCI'] = 1
        elif curr['CCI'] > 50:
            signals['CCI'] = -1
        else:
            signals['CCI'] = 0

    # 7. Williams %R
    if not pd.isna(curr['Williams_R']):
        if curr['Williams_R'] < -50:
            signals['Williams_R'] = 1
        elif curr['Williams_R'] > -20:
            signals['Williams_R'] = -1
        else:
            signals['Williams_R'] = 0

    # 8. PSAR
    if not pd.isna(curr['PSAR']):
        if curr['Close'] > curr['PSAR']:
            signals['PSAR'] = 1
        elif curr['Close'] < curr['PSAR']:
            signals['PSAR'] = -1
        else:
            signals['PSAR'] = 0

    return signals, curr['Close'], curr['SMA20'], curr['SMA60']


def get_indicator_series(ticker_symbol, period="1y"):
    df = fetch_stock_data(ticker_symbol, period)
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()
    ind_df = calculate_technical_indicators(df)
    return df, ind_df


def format_price(val, is_krx=False):
    if pd.isna(val):
        return "-"
    if is_krx:
        return f"{int(round(val)):,}원"
    return f"${val:,.2f}"


def _set_active_ticker(ticker_name):
    st.session_state.active_ticker = ticker_name


# -----------------------------------------------------------------------------
# 3. 대화형 차트 (마우스 스크롤 줌 및 드래그 팬 활성화)
# -----------------------------------------------------------------------------
def create_interactive_chart(df, ticker_symbol, target_price=None, stop_loss=None, show_signals=True):
    if df.empty:
        return go.Figure()

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.6, 0.2, 0.2],
        specs=[[{"secondary_y": True}], [{"secondary_y": False}], [{"secondary_y": False}]]
    )

    # Row 1: 캔들스틱 & 이동평균선
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'], high=df['High'],
            low=df['Low'], close=df['Close'],
            name="주가",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350"
        ),
        row=1, col=1, secondary_y=False
    )

    if 'SMA20' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['SMA20'], name="SMA 20", line=dict(color="#2962ff", width=1.5)),
            row=1, col=1, secondary_y=False
        )
    if 'SMA60' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['SMA60'], name="SMA 60", line=dict(color="#ff6d00", width=1.5)),
            row=1, col=1, secondary_y=False
        )

    # 과거 패턴 기반 상승/하락 예상 시점 예측 마커
    if show_signals:
        buy_x, buy_y = [], []
        sell_x, sell_y = [], []

        for i in range(1, len(df)):
            macd_gold = (df['MACD'].iloc[i - 1] < df['MACD_Signal'].iloc[i - 1]) and (df['MACD'].iloc[i] >= df['MACD_Signal'].iloc[i])
            macd_dead = (df['MACD'].iloc[i - 1] > df['MACD_Signal'].iloc[i - 1]) and (df['MACD'].iloc[i] <= df['MACD_Signal'].iloc[i])

            rsi_buy = (df['RSI'].iloc[i - 1] <= 35) and (df['RSI'].iloc[i] > 35)
            rsi_sell = (df['RSI'].iloc[i - 1] >= 65) and (df['RSI'].iloc[i] < 65)

            if macd_gold or rsi_buy:
                buy_x.append(df.index[i])
                buy_y.append(df['Low'].iloc[i] * 0.985)

            if macd_dead or rsi_sell:
                sell_x.append(df.index[i])
                sell_y.append(df['High'].iloc[i] * 1.015)

        if buy_x:
            fig.add_trace(
                go.Scatter(
                    x=buy_x, y=buy_y,
                    mode="markers",
                    name="상승예상 (▲)",
                    marker=dict(symbol="triangle-up", size=10, color="#00e676"),
                    showlegend=True
                ),
                row=1, col=1
            )

        if sell_x:
            fig.add_trace(
                go.Scatter(
                    x=sell_x, y=sell_y,
                    mode="markers",
                    name="하락예상 (▼)",
                    marker=dict(symbol="triangle-down", size=10, color="#ff1744"),
                    showlegend=True
                ),
                row=1, col=1
            )

    # 거래량
    if 'Volume' in df.columns:
        fig.add_trace(
            go.Bar(
                x=df.index, y=df['Volume'], name="거래량",
                marker_color="rgba(120, 123, 134, 0.25)",
                showlegend=False
            ),
            row=1, col=1, secondary_y=True
        )

    # 목표가 / 손절가
    if target_price:
        fig.add_hline(
            y=target_price, line_dash="dash", line_color="#00e676", line_width=2,
            annotation_text=f"🎯 목표가 ({target_price:,.1f})",
            annotation_position="top right", row=1, col=1
        )
    if stop_loss:
        fig.add_hline(
            y=stop_loss, line_dash="dash", line_color="#ff1744", line_width=2,
            annotation_text=f"🛑 손절가 ({stop_loss:,.1f})",
            annotation_position="bottom right", row=1, col=1
        )

    # Row 2: RSI
    if 'RSI' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['RSI'], name="RSI", line=dict(color="#ab47bc", width=1.5)),
            row=2, col=1
        )
        fig.add_hline(y=70, line_dash="dash", line_color="#ef5350", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="#26a69a", row=2, col=1)

    # Row 3: MACD
    if 'MACD' in df.columns and 'MACD_Signal' in df.columns and 'MACD_Hist' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['MACD'], name="MACD", line=dict(color="#2962ff", width=1.5)),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['MACD_Signal'], name="Signal", line=dict(color="#ff6d00", width=1.5)),
            row=3, col=1
        )
        hist_colors = ["#26a69a" if val >= 0 else "#ef5350" for val in df['MACD_Hist']]
        fig.add_trace(
            go.Bar(x=df.index, y=df['MACD_Hist'], name="Histogram", marker_color=hist_colors, showlegend=False),
            row=3, col=1
        )

    # 마우스 줌/팬 인터랙션 강화 설정
    fig.update_layout(
        title=f"📊 {ticker_symbol} 기술적 분석 및 추세 예측 차트 (마우스 스크롤 확대/축소 가능)",
        height=800,
        margin=dict(l=10, r=10, t=50, b=10),
        template="plotly_dark",
        paper_bgcolor="#131722",
        plot_bgcolor="#131722",
        hovermode="x unified",
        dragmode="pan",  # 기본 드래그 동작을 이동(Pan)으로 설정
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    fig.update_xaxes(
        gridcolor="#2a2e39",
        zerolinecolor="#2a2e39",
        rangeslider_visible=True,
        rangeslider_thickness=0.06,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1개월", step="month", stepmode="backward"),
                dict(count=3, label="3개월", step="month", stepmode="backward"),
                dict(count=6, label="6개월", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(step="all", label="전체/오늘(리셋)")
            ]),
            bgcolor="#2a2e39", activecolor="#26a69a",
            font=dict(color="#ffffff", size=11), x=0, y=1.12
        )
    )

    fig.update_yaxes(gridcolor="#2a2e39", zerolinecolor="#2a2e39", fixedrange=False, row=1, col=1)
    fig.update_yaxes(showgrid=False, secondary_y=True, row=1, col=1)
    fig.update_yaxes(gridcolor="#2a2e39", range=[0, 100], row=2, col=1)
    fig.update_yaxes(gridcolor="#2a2e39", row=3, col=1)

    return fig


# -----------------------------------------------------------------------------
# 4. Streamlit 메인 화면
# -----------------------------------------------------------------------------
def main():
    st.title("📈 주식/가상자산 종합 기술적 분석 대시보드")

    if "active_ticker" not in st.session_state:
        st.session_state.active_ticker = "TSLA"

    st.sidebar.header("⚙️ 분석 설정")
    input_ticker = st.sidebar.text_input(
        "종목 코드 (티커)",
        value=st.session_state.active_ticker
    ).strip().upper()

    if input_ticker != st.session_state.active_ticker:
        st.session_state.active_ticker = input_ticker

    selected_period = st.sidebar.selectbox(
        "데이터 조회 기간",
        options=["3m", "6m", "1y", "2y", "5y"],
        index=2
    )

    toggle_signals = st.sidebar.toggle("상승/하락 예상 시점 마커 표시", value=True)

    current_ticker = st.session_state.active_ticker
    is_krx = current_ticker.endswith(".KS") or current_ticker.endswith(".KQ")

    raw_df, ind_df = get_indicator_series(current_ticker, period=selected_period)

    main_tab1, main_tab2, main_tab3 = st.tabs([
        "📊 대화형 차트 & 추세 예측",
        "⚙️ 8대 지표별 매수/매도 신호",
        "🔍 멀티 종목 스캐너"
    ])

    # TAB 1: 대화형 차트
    with main_tab1:
        if ind_df.empty:
            st.error(f"'{current_ticker}'의 시세 데이터를 불러올 수 없거나 데이터가 부족합니다.")
        else:
            latest = ind_df.iloc[-1]
            prev = ind_df.iloc[-2]
            chg = latest['Close'] - prev['Close']
            chg_pct = (chg / prev['Close']) * 100

            atr_val = latest['ATR'] if not pd.isna(latest['ATR']) else latest['Close'] * 0.03
            target_p = latest['Close'] + (atr_val * 2.0)
            stop_l = latest['Close'] - (atr_val * 1.5)

            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("현재가", format_price(latest['Close'], is_krx), f"{chg_pct:+.2f}%")
            col2.metric("🎯 ATR 목표가", format_price(target_p, is_krx))
            col3.metric("🛑 ATR 손절가", format_price(stop_l, is_krx))
            col4.metric("RSI (14)", f"{latest['RSI']:.1f}" if not pd.isna(latest['RSI']) else "-")
            col5.metric("CCI (20)", f"{latest['CCI']:.1f}" if not pd.isna(latest['CCI']) else "-")

            chart_fig = create_interactive_chart(
                ind_df, current_ticker, target_price=target_p, stop_loss=stop_l, show_signals=toggle_signals
            )

            # config 옵션에 scrollZoom=True를 추가하여 마우스 휠 확대를 활성화
            st.plotly_chart(
                chart_fig,
                use_container_width=True,
                config={"scrollZoom": True, "displayModeBar": True}
            )

    # TAB 2: 8대 상세 지표 매수/매도 분석
    with main_tab2:
        if ind_df.empty or len(ind_df) < MIN_ROWS_REQUIRED:
            st.warning("분석에 필요한 충분한 데이터가 없습니다.")
        else:
            st.subheader(f"🔍 {current_ticker} 8대 기술적 지표 종합 판정")
            signals, cur_close, cur_sma20, cur_sma60 = build_signal_row(ind_df, len(ind_df) - 1)

            w_buy, w_total = 0.0, 0.0
            sig_details = []

            for sig_name, sig_val in signals.items():
                w = INDICATOR_WEIGHTS.get(sig_name, 1.0)
                if not pd.isna(sig_val):
                    w_total += w
                    if sig_val == 1:
                        w_buy += w
                        txt = "🟢 매수"
                    elif sig_val == -1:
                        txt = "🔴 매도"
                    else:
                        txt = "🟡 중립"
                else:
                    txt = "⚪ 측정 불가"

                sig_details.append({
                    "지표": sig_name,
                    "신호": txt,
                    "가중치": w
                })

            if w_total > 0:
                buy_ratio = (w_buy / w_total) * 100
                st.progress(int(buy_ratio))
                st.write(f"**가중 매수 강도:** `{buy_ratio:.1f}%`")

            st.table(pd.DataFrame(sig_details))

    # TAB 3: 멀티 스캐너
    with main_tab3:
        st.subheader("🔍 관심종목 멀티 스캐너")
        watchlist_input = st.text_area(
            "티커 목록 (쉼표로 구분)",
            "TSLA, NVDA, AAPL, MSFT, 005930.KS, BTC-USD, ETH-USD",
        )
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
                        ratio_display = np.nan
                        status = "⚪ 데이터 부족"

                    scan_results.append({
                        "티커": t_sym,
                        "현재가": format_price(t_close, t_sym.endswith(".KS") or t_sym.endswith(".KQ")),
                        "가중 매수 강도 (%)": ratio_display,
                        "상태": status,
                    })
                progress_bar.progress((idx + 1) / len(tickers))

            st.session_state.scan_results = scan_results
            progress_bar.empty()

        if "scan_results" in st.session_state and st.session_state.scan_results:
            res_df = pd.DataFrame(st.session_state.scan_results).sort_values(
                by="가중 매수 강도 (%)", ascending=False, na_position="last"
            ).reset_index(drop=True)

            event = st.dataframe(
                res_df,
                use_container_width=True,
                on_select="rerun",
                selection_mode="single-row",
                key="scanner_dataframe"
            )

            selected_rows = event.selection.get("rows", [])
            if selected_rows:
                selected_index = selected_rows[0]
                selected_ticker = res_df.iloc[selected_index]["티커"]

                st.write("---")
                st.info(f"선택된 종목: **{selected_ticker}**")

                if st.button(
                    f"🔍 {selected_ticker} 상세 분석 차트로 이동",
                    key="btn_go_selected",
                    use_container_width=True,
                    type="primary"
                ):
                    _set_active_ticker(selected_ticker)
                    st.rerun()


if __name__ == "__main__":
    main()
