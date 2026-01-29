import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from datetime import timedelta

st.set_page_config(page_title="USDJPY 方向予測", layout="wide")

st.title("📈 USD/JPY 5分足 方向予測 (急変対応版)")
st.markdown("直近の値動きを重視するEWMAモデルにより、急激なボラティリティの変化を確率に反映します。")

# サイドバー設定
st.sidebar.header("分析設定")
lookback = st.sidebar.select_slider("分析対象件数", options=[256, 512, 1024], value=512)
horizon = st.sidebar.slider("予測期間 (5分足の本数)", 6, 48, 12)

# 感度設定の追加
span = st.sidebar.slider("直近感度 (数値が小さいほど急変に敏感)", 10, 100, 30, help="直近何本分の動きを重点的に見るかの指標です。")

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
            last_time_jst = df.index[-1] + timedelta(hours=9)
            
            # --- 直近重視のボラティリティ計算 (EWMA) ---
            returns = np.log(df['Close'] / df['Close'].shift(1)).dropna()
            
            # 直近の動きに重みを置く (指数平滑分散)
            # ewm().var() を使うことで、直近の急変が分散に即座に反映される
            latest_vol = np.sqrt(returns.ewm(span=span).var().iloc[-1])
            
            # 予測期間(horizon)に換算
            h_vol = latest_vol * np.sqrt(horizon)
            
            # 方向確率の算出 (累積分布関数 CDF を使用)
            # 上昇確率 (現在価格より高くなる確率)
            prob_up_base = (1 - norm.cdf(0, loc=0, scale=h_vol)) * 100
            prob_down_base = 100 - prob_up_base

            # ターゲット別勝率 (100%を上昇・下落・レンジで分ける考え方)
            def calc_target_win_rate(pips):
                target_ret = np.log((price + (pips * 0.01)) / price)
                # 上昇側: そのpips以上で終わる確率
                if pips > 0:
                    return (1 - norm.cdf(target_ret, loc=0, scale=h_vol)) * 100
                # 下降側: そのpips以下で終わる確率
                else:
                    return norm.cdf(target_ret, loc=0, scale=h_vol) * 100

            p15_u = round(calc_target_win_rate(15), 1)
            p10_u = round(calc_target_win_rate(10), 1)
            p10_d = round(calc_target_win_rate(-10), 1)
            p15_d = round(calc_target_win_rate(-15), 1)

            # --- UI表示 ---
            st.success(f"現在価格: {price:.3f} | 日本時間: {last_time_jst.strftime('%H:%M')}")
            
            st.subheader(f"方向予測 ({horizon*5}分後の着地確率)")
            c1, c2 = st.columns(2)
            c1.metric("上昇する確率", f"{prob_up_base:.1f}%")
            c2.metric("下落する確率", f"{prob_down_base:.1f}%")

            # 勝率棒グラフ
            st.subheader("🎯 ターゲット到達勝率 (期間終了時)")
            fig_bar = go.Figure(data=[go.Bar(
                x=['+15 pips 以上', '+10 pips 以上', '-10 pips 以下', '-15 pips 以下'],
                y=[p15_u, p10_u, p10_d, p15_d],
                marker_color=['#00cc66', '#00cc66', '#ff3300', '#ff3300'],
                text=[f"{x}%" for x in [p15_u, p10_u, p10_d, p15_d]],
                textposition='auto'
            )])
            fig_bar.update_layout(template="plotly_dark", yaxis=dict(title="勝率 (%)", range=[0, 100]), height=400)
            st.plotly_chart(fig_bar, use_container_width=True)

            # チャート
            st.subheader("📊 価格推移とターゲットライン")
            fig_chart = go.Figure()
            fig_chart.add_trace(go.Scatter(x=df.index, y=df['Close'], name="実績", line=dict(color="#00fbff")))
            for p, c, d in [(0.15, "#00cc66", "dot"), (0.1, "#00cc66", "dash"), (-0.1, "#ff3300", "dash"), (-0.15, "#ff3300", "dot")]:
                fig_chart.add_hline(y=price + p, line_dash=d, line_color=c)
            fig_chart.update_layout(template="plotly_dark", height=500, xaxis_rangeslider_visible=False)
            st.plotly_chart(fig_chart, use_container_width=True)
