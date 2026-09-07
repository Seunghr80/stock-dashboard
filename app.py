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
    "Bollinger": 1.0
}

# -----------------------------------------------------------------------------
# 2. 데이터 수집 및 지표 계산 함수
# -----------------------------------------------------------------------------
@st.cache_data(ttl=300)
def fetch_stock_data(ticker_symbol, period="1y"):
    """yfinance를 이용한 시세 데이터 수집"""
    try:
        df = yf.download(ticker_symbol, period=period, progress=False)
        if df.empty:
            return pd.DataFrame()
        
        # MultiIndex 컬럼 단일화 처리
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df = df.dropna()
        return df
    except Exception:
        return pd.DataFrame()


def calculate_technical_indicators(df):
    """주요 기술적 지표 계산 (SMA, RSI, MACD, Stochastic, Bollinger)"""
    if df.empty or len(df) < MIN_ROWS_REQUIRED:
        return pd.DataFrame()

    ind_df = df.copy()

    # 이동평균선
    ind_df['SMA20'] = ind_df['Close'].rolling(window=20).mean()
    ind_df['SMA60'] = ind_df['Close'].rolling(window=60).mean()

    # RSI
    delta = ind_df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan)
    ind_df['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = ind_df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = ind_df['Close'].ewm(span=26, adjust=False).mean()
    ind_df['MACD'] = ema12 - ema26
    ind_df['MACD_Signal'] = ind_df['MACD'].ewm(span=9, adjust=False).mean()
    ind_df['MACD_Hist'] = ind_df['MACD'] - ind_df['MACD_Signal']

    # 스토캐스틱 (Fast %K, %D)
    low14 = ind_df['Low'].rolling(window=14).min()
    high14 = ind_df['High'].rolling(window=14).max()
    ind_df['Stoch_K'] = (ind_df['Close'] - low14) / (high14 - low14).replace(0, np.nan) * 100
    ind_df['Stoch_D'] = ind_df['Stoch_K'].rolling(window=3).mean()

    # 볼린저 밴드
    std20 = ind_df['Close'].rolling(window=20).std()
    ind_df['BB_Upper'] = ind_df['SMA20'] + (std20 * 2)
    ind_df['BB_Lower'] = ind_df['SMA20'] - (std20 * 2)

    return ind_df


def build_signal_row(ind_df, idx_pos):
    """특정 시점(idx_pos)의 신호 판정 로직"""
    if idx_pos < 1 or idx_pos >= len(ind_df):
        return {}, np.nan, np.nan, np.nan

    curr = ind_df.iloc[idx_pos]
    prev = ind_df.iloc[idx_pos - 1]

    signals = {}

    # 1. RSI (30 이하 매수, 70 이상 매도)
    if not pd.isna(curr['RSI']):
        if curr['RSI'] <= 30:
            signals['RSI'] = 1
        elif curr['RSI'] >= 70:
            signals['RSI'] = -1
        else:
            signals['RSI'] = 0

    # 2. MACD 크로스
    if not (pd.isna(curr['MACD']) or pd.isna(curr['MACD_Signal']) or pd.isna(prev['MACD']) or pd.isna(prev['MACD_Signal'])):
        if prev['MACD'] < prev['MACD_Signal'] and curr['MACD'] >= curr['MACD_Signal']:
            signals['MACD'] = 1
        elif prev['MACD'] > prev['MACD_Signal'] and curr['MACD'] <= curr['MACD_Signal']:
            signals['MACD'] = -1
        else:
            signals['MACD'] = 0

    # 3. 이동평균선(SMA20/60) 크로스
    if not (pd.isna(curr['SMA20']) or pd.isna(curr['SMA60']) or pd.isna(prev['SMA20']) or pd.isna(prev['SMA60'])):
        if prev['SMA20'] < prev['SMA60'] and curr['SMA20'] >= curr['SMA60']:
            signals['SMA_Cross'] = 1
        elif prev['SMA20'] > prev['SMA60'] and curr['SMA20'] <= prev['SMA60']:
            signals['SMA_Cross'] = -1
        else:
            signals['SMA_Cross'] = 0

    # 4. 스토캐스틱 크로스
    if not (pd.isna(curr['Stoch_K']) or pd.isna(curr['Stoch_D']) or pd.isna(prev['Stoch_K']) or pd.isna(prev['Stoch_D'])):
        if prev['Stoch_K'] < prev['Stoch_D'] and curr['Stoch_K'] >= curr['Stoch_D'] and curr['Stoch_K'] <= 20:
            signals['Stochastic'] = 1
        elif prev['Stoch_K'] > prev['Stoch_D'] and curr['Stoch_K'] <= curr['Stoch_D'] and curr['Stoch_K'] >= 80:
            signals['Stochastic'] = -1
        else:
            signals['Stochastic'] = 0

    # 5. 볼린저 밴드
    if not (pd.isna(curr['BB_Lower']) or pd.isna(curr['BB_Upper'])):
        if curr['Close'] <= curr['BB_Lower']:
            signals['Bollinger'] = 1
        elif curr['Close'] >= curr['BB_Upper']:
            signals['Bollinger'] = -1
        else:
            signals['Bollinger'] = 0

    return signals, curr['Close'], curr['SMA20'], curr['SMA60']


def get_indicator_series(ticker_symbol, period="1y"):
    """종목코드로 수집 및 지표 계산을 한 번에 실행"""
    df = fetch_stock_data(ticker_symbol, period)
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()
    ind_df = calculate_technical_indicators(df)
    return df, ind_df


def format_price(val, is_krx=False):
    """가격 표기 포맷팅"""
    if pd.isna(val):
        return "-"
    if is_krx:
        return f"{int(round(val)):,}원"
    return f"${val:,.2f}"


def _set_active_ticker(ticker_name):
    """활성 티커 세션 변경 및 갱신"""
    st.session_state.active_ticker = ticker_name


# -----------------------------------------------------------------------------
# 3. 대화형 차트 생성 함수 (Plotly - Rangeslider & Rangeselector 적용)
# -----------------------------------------------------------------------------
def create_interactive_chart(df, ticker_symbol):
    """
    캔들차트, 이동평균선(SMA 20/60), 거래량, RSI, MACD를 포함하고
    하단 슬라이더 및 기간 선택 버튼이 적용된 Plotly 대화형 차트 생성 함수
    """
    if df.empty:
        return go.Figure()

    # 서브플롯 구성 (Row 1: 메인 차트, Row 2: RSI, Row 3: MACD)
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.6, 0.2, 0.2],
        specs=[[{"secondary_y": True}], [{"secondary_y": False}], [{"secondary_y": False}]]
    )

    # --- [Row 1: 캔들스틱 차트 & 이동평균선] ---
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'], high=df['High'],
            low=df['Low'], close=df['Close'],
            name="주가",
            increasing_line_color="#26a69a",  # 상승 (초록/양봉)
            decreasing_line_color="#ef5350"   # 하락 (빨강/음봉)
        ),
        row=1, col=1, secondary_y=False
    )

    # 20일 / 60일 이동평균선
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

    # 거래량 (우측 Y축 / 투명 바)
    if 'Volume' in df.columns:
        fig.add_trace(
            go.Bar(
                x=df.index, y=df['Volume'], name="거래량",
                marker_color="rgba(120, 123, 134, 0.3)",
                showlegend=False
            ),
            row=1, col=1, secondary_y=True
        )

    # --- [Row 2: RSI 지표] ---
    if 'RSI' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['RSI'], name="RSI", line=dict(color="#ab47bc", width=1.5)),
            row=2, col=1
        )
        fig.add_hline(y=70, line_dash="dash", line_color="#ef5350", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="#26a69a", row=2, col=1)

    # --- [Row 3: MACD 지표] ---
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

    # --- [TradingView 다크 테마 및 전체 레이아웃 설정] ---
    fig.update_layout(
        title=f"📊 {ticker_symbol} 대화형 차트 분석",
        height=780,  # 하단 미니 슬라이더 바 공간을 위해 높이 확보
        margin=dict(l=10, r=10, t=50, b=10),
        template="plotly_dark",
        paper_bgcolor="#131722",
        plot_bgcolor="#131722",
        hovermode="x unified",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
        ),
    )

    # --- [X축: 미니 슬라이더(Rangeslider) 및 원클릭 기간 선택 버튼(Rangeselector)] ---
    fig.update_xaxes(
        gridcolor="#2a2e39",
        zerolinecolor="#2a2e39",
        rangeslider_visible=True,        # 차트 하단 미니 슬라이더 바 활성화 (드래그로 과거 날짜 이동)
        rangeslider_thickness=0.08,      # 슬라이더 바 두께
        rangeselector=dict(              # 좌측 상단 원클릭 기간 이동 단축 버튼
            buttons=list([
                dict(count=1, label="1개월", step="month", stepmode="backward"),
                dict(count=3, label="3개월", step="month", stepmode="backward"),
                dict(count=6, label="6개월", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(step="all", label="전체/오늘(리셋)") # 클릭 시 원래 최신(오늘) 화면으로 리셋
            ]),
            bgcolor="#2a2e39",
            activecolor="#26a69a",
            font=dict(color="#ffffff", size=11),
            x=0,
            y=1.12
        )
    )

    # Y축 레이아웃 설정
    fig.update_yaxes(gridcolor="#2a2e39", zerolinecolor="#2a2e39", fixedrange=False, row=1, col=1)
    fig.update_yaxes(showgrid=False, secondary_y=True, row=1, col=1)
    fig.update_yaxes(gridcolor="#2a2e39", range=[0, 100], row=2, col=1)
    fig.update_yaxes(gridcolor="#2a2e39", row=3, col=1)

    return fig


# -----------------------------------------------------------------------------
# 4. Streamlit UI 메인 화면 구성
# -----------------------------------------------------------------------------
def main():
    st.title("📈 주식/가상자산 종합 기술적 분석 대시보드")

    if "active_ticker" not in st.session_state:
        st.session_state.active_ticker = "TSLA"

    # --- 사이드바 ---
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

    current_ticker = st.session_state.active_ticker
    is_krx = current_ticker.endswith(".KS") or current_ticker.endswith(".KQ")

    # 데이터 로드
    raw_df, ind_df = get_indicator_series(current_ticker, period=selected_period)

    # --- 메인 탭 구획 ---
    main_tab1, main_tab2, main_tab3 = st.tabs([
        "📊 대화형 차트 & 상판 지표",
        "⚙️ 지표별 매수/매도 신호",
        "🔍 멀티 종목 스캐너"
    ])

    # -------------------------------------------------------------------------
    # TAB 1: 대화형 차트
    # -------------------------------------------------------------------------
    with main_tab1:
        if ind_df.empty:
            st.error(f"'{current_ticker}'의 시세 데이터를 불러올 수 없거나 데이터가 부족합니다.")
        else:
            # 상단 요약 지표 카드
            latest = ind_df.iloc[-1]
            prev = ind_df.iloc[-2]
            chg = latest['Close'] - prev['Close']
            chg_pct = (chg / prev['Close']) * 100

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("현재가", format_price(latest['Close'], is_krx), f"{chg_pct:+.2f}%")
            col2.metric("RSI (14)", f"{latest['RSI']:.1f}" if not pd.isna(latest['RSI']) else "-")
            col3.metric("MACD Hist", f"{latest['MACD_Hist']:.2f}" if not pd.isna(latest['MACD_Hist']) else "-")
            col4.metric("SMA 20", format_price(latest['SMA20'], is_krx))

            # 대화형 차트 출력
            chart_fig = create_interactive_chart(ind_df, current_ticker)
            st.plotly_chart(chart_fig, use_container_width=True)

    # -------------------------------------------------------------------------
    # TAB 2: 상세 지표 신호 분석
    # -------------------------------------------------------------------------
    with main_tab2:
        if ind_df.empty or len(ind_df) < MIN_ROWS_REQUIRED:
            st.warning("분석에 필요한 충분한 데이터가 없습니다.")
        else:
            st.subheader(f"🔍 {current_ticker} 기술적 신호 종합 판정")
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

    # -------------------------------------------------------------------------
    # TAB 3: 관심종목 멀티 스캐너 (클릭 선택 연동 반영)
    # -------------------------------------------------------------------------
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
            # 정렬 후 인덱스를 재설정하여 iloc 위치 참조 오류 방지
            res_df = pd.DataFrame(
                st.session_state.scan_results
            ).sort_values(
                by="가중 매수 강도 (%)", ascending=False, na_position="last"
            ).reset_index(drop=True)

            # 1. 행 클릭 선택(Selection) 이벤트 처리
            event = st.dataframe(
                res_df,
                use_container_width=True,
                on_select="rerun",
                selection_mode="single-row",
                key="scanner_dataframe"
            )

            # 2. 클릭된 행이 있을 경우 전환 버튼 노출
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
