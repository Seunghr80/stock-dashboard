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
def fetch_stock_data(ticker_symbol, interval="1d", period="1y"):
    """yfinance 시세 데이터 수집"""
    try:
        df = yf.download(ticker_symbol, period=period, interval=interval, progress=False)
        if df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.dropna()
        return df
    except Exception:
        return pd.DataFrame()


def calculate_technical_indicators(df):
    """8가지 주요 기술적 지표 계산"""
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

    # 8. ATR (변동성)
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
    """지표별 매수/매도 판정"""
    if idx_pos < 1 or idx_pos >= len(ind_df):
        return {}, np.nan, np.nan, np.nan

    curr = ind_df.iloc[idx_pos]
    signals = {}

    if not pd.isna(curr['RSI']):
        signals['RSI'] = 1 if curr['RSI'] <= 40 else (-1 if curr['RSI'] >= 60 else 0)

    if not (pd.isna(curr['MACD']) or pd.isna(curr['MACD_Signal'])):
        signals['MACD'] = 1 if curr['MACD'] > curr['MACD_Signal'] else (-1 if curr['MACD'] < curr['MACD_Signal'] else 0)

    if not (pd.isna(curr['SMA20']) or pd.isna(curr['SMA60'])):
        signals['SMA_Cross'] = 1 if curr['SMA20'] > curr['SMA60'] else (-1 if curr['SMA20'] < curr['SMA60'] else 0)

    if not (pd.isna(curr['Stoch_K']) or pd.isna(curr['Stoch_D'])):
        if curr['Stoch_K'] > curr['Stoch_D'] and curr['Stoch_K'] < 80:
            signals['Stochastic'] = 1
        elif curr['Stoch_K'] < curr['Stoch_D'] and curr['Stoch_K'] > 20:
            signals['Stochastic'] = -1
        else:
            signals['Stochastic'] = 0

    if not (pd.isna(curr['BB_Lower']) or pd.isna(curr['BB_Upper']) or pd.isna(curr['SMA20'])):
        signals['Bollinger'] = 1 if curr['Close'] < curr['SMA20'] else (-1 if curr['Close'] > curr['BB_Upper'] else 0)

    if not pd.isna(curr['CCI']):
        signals['CCI'] = 1 if curr['CCI'] < -50 else (-1 if curr['CCI'] > 50 else 0)

    if not pd.isna(curr['Williams_R']):
        signals['Williams_R'] = 1 if curr['Williams_R'] < -50 else (-1 if curr['Williams_R'] > -20 else 0)

    if not pd.isna(curr['PSAR']):
        signals['PSAR'] = 1 if curr['Close'] > curr['PSAR'] else (-1 if curr['Close'] < curr['PSAR'] else 0)

    return signals, curr['Close'], curr['SMA20'], curr['SMA60']


def get_indicator_series(ticker_symbol, interval="1d", period="1y"):
    df = fetch_stock_data(ticker_symbol, interval=interval, period=period)
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
# 3. 대화형 차트 (가로축 넓게 펼침 및 초기 표시 범위 조정 반영)
# -----------------------------------------------------------------------------
def create_interactive_chart(df, ticker_symbol, interval_label="일봉", target_price=None, stop_loss=None, show_signals=True, cooldown_bars=4, marker_size=7, visible_bars=45):
    if df.empty:
        return go.Figure()

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.65, 0.175, 0.175],
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

    # 중복 시그널 방지 로직
    if show_signals:
        buy_x, buy_y = [], []
        sell_x, sell_y = [], []
        
        last_buy_idx = -cooldown_bars
        last_sell_idx = -cooldown_bars

        for i in range(1, len(df)):
            macd_gold = (df['MACD'].iloc[i - 1] < df['MACD_Signal'].iloc[i - 1]) and (df['MACD'].iloc[i] >= df['MACD_Signal'].iloc[i])
            macd_dead = (df['MACD'].iloc[i - 1] > df['MACD_Signal'].iloc[i - 1]) and (df['MACD'].iloc[i] <= df['MACD_Signal'].iloc[i])

            rsi_buy = (df['RSI'].iloc[i - 1] <= 30) and (df['RSI'].iloc[i] > 30)
            rsi_sell = (df['RSI'].iloc[i - 1] >= 70) and (df['RSI'].iloc[i] < 70)

            if (macd_gold or rsi_buy) and (i - last_buy_idx >= cooldown_bars):
                buy_x.append(df.index[i])
                buy_y.append(df['Low'].iloc[i] * 0.98)
                last_buy_idx = i

            if (macd_dead or rsi_sell) and (i - last_sell_idx >= cooldown_bars):
                sell_x.append(df.index[i])
                sell_y.append(df['High'].iloc[i] * 1.02)
                last_sell_idx = i

        if buy_x:
            fig.add_trace(
                go.Scatter(
                    x=buy_x, y=buy_y,
                    mode="markers",
                    name="상승예상 (▲)",
                    marker=dict(symbol="triangle-up", size=marker_size, color="#00e676"),
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
                    marker=dict(symbol="triangle-down", size=marker_size, color="#ff1744"),
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
            y=target_price, line_dash="dash", line_color="#00e676", line_width=1.5,
            annotation_text=f"🎯 목표가 ({target_price:,.1f})",
            annotation_position="top right", row=1, col=1
        )
    if stop_loss:
        fig.add_hline(
            y=stop_loss, line_dash="dash", line_color="#ff1744", line_width=1.5,
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

    # 가로축(X축) 확대 범위 자동 지정: 최근 N개 봉만 가로로 크게 펼쳐서 보여줌
    if len(df) > visible_bars:
        x_min = df.index[-visible_bars]
        x_max = df.index[-1]
    else:
        x_min = df.index[0]
        x_max = df.index[-1]

    fig.update_layout(
        title=f"📊 {ticker_symbol} ({interval_label}) 기술적 분석 차트",
        height=950,  # 세로 크기 확충
        margin=dict(l=10, r=10, t=50, b=10),
        template="plotly_dark",
        paper_bgcolor="#131722",
        plot_bgcolor="#131722",
        hovermode="x unified",
        dragmode="pan",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    # X축 간격 및 스크롤 설정
    fig.update_xaxes(
        gridcolor="#2a2e39",
        zerolinecolor="#2a2e39",
        range=[x_min, x_max],  # 초기 가로 표시 범위 설정 (자동 넓힘)
        rangeslider_visible=True,
        rangeslider_thickness=0.05,
        type="date"
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

    interval_type = st.sidebar.radio(
        "차트 봉 주기 (Interval)",
        options=["일봉 (1D)", "4시간봉 (4H)"],
        index=0,
        horizontal=True
    )
    interval_code = "1d" if "일봉" in interval_type else "4h"

    selected_period = st.sidebar.selectbox(
        "데이터 조회 기간",
        options=["1m", "3m", "6m", "1y", "2y", "5y"],
        index=3
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("📐 차트 가시성 & 시그널 설정")
    
    # 한 번에 보여줄 봉 개수 지정 슬라이더 (가로 간격 조절용)
    visible_bars_val = st.sidebar.slider("한 화면에 볼 봉 개수 (가로 간격)", min_value=15, max_value=120, value=40, step=5)
    
    toggle_signals = st.sidebar.toggle("상승/하락 마커 표시", value=True)
    cooldown_val = st.sidebar.slider("시그널 발생 최소 간격 (봉 개수)", min_value=1, max_value=15, value=5)
    marker_size_val = st.sidebar.slider("마커 크기", min_value=4, max_value=12, value=7)

    current_ticker = st.session_state.active_ticker
    is_krx = current_ticker.endswith(".KS") or current_ticker.endswith(".KQ")

    raw_df, ind_df = get_indicator_series(current_ticker, interval=interval_code, period=selected_period)

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
                ind_df, 
                current_ticker, 
                interval_label="4시간봉" if interval_code == "4h" else "일봉",
                target_price=target_p, 
                stop_loss=stop_l, 
                show_signals=toggle_signals,
                cooldown_bars=cooldown_val,
                marker_size=marker_size_val,
                visible_bars=visible_bars_val
            )

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
            st.subheader(f"🔍 {current_ticker} ({'4시간봉' if interval_code == '4h' else '일봉'}) 8대 기술적 지표 종합 판정")
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
                t_df, t_ind = get_indicator_series(t_sym, interval=interval_code, period=selected_period)
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
