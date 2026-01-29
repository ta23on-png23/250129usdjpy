import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from datetime import timedelta

st.set_page_config(page_title="USDJPY 確率予測", layout="wide")

st.title("📈 USD/JPY 5分足 到達確率予測")

# サイドバー設定
st.sidebar.header("分析設定")
lookback = st.sidebar.select_slider("分析対象件数 (過去)", options=[256, 512, 1024], value=512)
horizon = st.sidebar.slider("予測期間 (5分足の本数)", 6, 48, 12)
update_btn = st.sidebar.button("最新価格で予測更新")

@st.cache_data(ttl=300)
def get_fx_data(n):
    try:
        df = yf.download("USDJPY=X", interval="5m", period="5d")
        if df.empty: return None, None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.iloc[:-1].tail(n)
        return df, float(df['Close'].iloc[-1])
    except:
        return None, None

if update_btn:
    with st.spinner("計算中..."):
        df, price = get_fx_data(lookback)
        if df is not None:
            # 日本時間 (UTC+9)
            last_time_jst = df.index[-1] + timedelta(hours=9)
            
            # ボラティリティ計算
            returns = np.log(df['Close'] / df['Close'].shift(1)).dropna()
            vol = returns.std() * np.sqrt(horizon)
            
            # 到達確率関数 (タッチ確率近似)
            def calc_prob(pips):
                # ログリターン空間でのターゲット
                dist = np.log((price + (pips * 0.01)) / price)
                # 標準化スコア
                z = abs(dist) / vol
                # タッチ確率は通常の到達確率(1-CDF)の約2倍になる性質を利用
                prob = 2 * (1 - norm.cdf(z)) * 100
                return min(round(prob, 1), 99.9)

            p15_u = calc_prob(15)
            p10_u = calc_prob(10)
            p10_d = calc_prob(-10)
            p15_d = calc_prob(-15)

            st.success(f"現在価格: {price:.3f} | 更新時刻: {df.index[-1].strftime('%H:%M')} (日本時間: {last_time_jst.strftime('%H:%M')})")
            
            # 棒グラフ表示
            st.subheader("🎯 ターゲット到達確率")
            fig_bar = go.Figure(data=[go.Bar(
                x=['+15 pips', '+10 pips', '-10 pips', '-15 pips'],
                y=[p15_u, p10_u, p10_d, p15_d],
                marker_color=['#00cc66', '#00cc66', '#ff3300', '#ff3300'],
                text=[f"{x}%" for x in [p15_u, p10_u, p10_d, p15_d]],
                textposition='auto'
            )])
            fig_bar.update_layout(template="plotly_dark", yaxis=dict(title="確率 (%)", range=[0, 100]), height=400)
            st.plotly_chart(fig_bar, use_container_width=True)

            # チャート表示
            st.subheader("📊 価格チャート")
            fig_chart = go.Figure()
            fig_chart.add_trace(go.Scatter(x=df.index, y=df['Close'], name="実績", line=dict(color="#00fbff")))
            for p, c, d in [(0.15, "#00cc66", "dot"), (0.1, "#00cc66", "dash"), (-0.1, "#ff3300", "dash"), (-0.15, "#ff3300", "dot")]:
                fig_chart.add_hline(y=price + p, line_dash=d, line_color=c)
            fig_chart.update_layout(template="plotly_dark", height=500, xaxis_rangeslider_visible=False)
            st.plotly_chart(fig_chart, use_container_width=True)
