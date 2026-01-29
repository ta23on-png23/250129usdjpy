import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from datetime import timedelta

# 1. ページ設定
st.set_page_config(page_title="USDJPY 確率予測", layout="wide")

st.title("📈 USD/JPY 5分足 到達確率予測")
st.markdown("ボラティリティに基づき、指定時間内にターゲットへ到達する確率を算出します。")

# 2. サイドバー設定
st.sidebar.header("分析設定")
lookback = st.sidebar.select_slider("分析対象件数 (過去)", options=[256, 512, 1024], value=512)
horizon = st.sidebar.slider("予測期間 (5分足の本数)", 6, 48, 12)
update_btn = st.sidebar.button("最新価格で予測更新")

# 3. データ取得関数
@st.cache_data(ttl=300)
def get_fx_data(n):
    try:
        df = yf.download("USDJPY=X", interval="5m", period="5d")
        if df.empty: return None, None
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df = df.iloc[:-1].tail(n)
        latest_price = float(df['Close'].iloc[-1])
        return df, latest_price
    except Exception as e:
        st.error(f"データ取得エラー: {e}")
        return None, None

# 4. メイン処理
if update_btn:
    with st.spinner("計算中..."):
        df, price = get_fx_data(lookback)
        
        if df is not None:
            # 日本時間への変換 (UTC+9)
            last_time_utc = df.index[-1]
            last_time_jst = last_time_utc + timedelta(hours=9)
            
            # 統計的確率計算
            returns = np.log(df['Close'] / df['Close'].shift(1)).dropna()
            vol = returns.std()
            h_vol = vol * np.sqrt(horizon)
            
            def calc_reach_probs(pips):
                target_diff = pips * 0.01
                # 上昇ターゲットのログリターン
                target_ret_up = np.log((price + target_diff) / price)
                prob_up = (1 - norm.cdf(target_ret_up, loc=0, scale=h_vol)) * 100
                # 下降ターゲット（対称と仮定）
                prob_down = prob_up 
                return round(prob_up, 1), round(prob_down, 1)

            p10_up, p10_down = calc_reach_probs(10)
            p15_up, p15_down = calc_reach_probs(15)

            # --- 表示セクション ---
            st.success(f"現在価格: {price:.3f} | 更新時刻: {last_time_utc.strftime('%H:%M')} (日本時間: {last_time_jst.strftime('%H:%M')})")
            
            # 確率の棒グラフ表示
            st.subheader("🎯 ターゲット到達確率")
            
            labels = ['+15 pips', '+10 pips', '-10 pips', '-15 pips']
            probs = [p15_up, p10_up, p10_down, p15_down]
            colors = ['#00cc66', '#00cc66', '#ff3300', '#ff3300'] # 上昇:緑, 下降:赤

            fig_prob = go.Figure(data=[go.Bar(
                x=labels, 
                y=probs,
                marker_color=colors,
                text=[f"{p}%" for p in probs],
                textposition='auto',
            )])
            
            fig_prob.update_layout(
                template="plotly_dark",
                yaxis=dict(title="確率 (%)", range=[0, 100]),
                height=400,
                margin=dict(l=20, r=20, t=30, b=20)
            )
            st.plotly_chart(fig_prob, use_container_width=True)

            # --- チャート表示 ---
            st.subheader("📊 価格チャートとターゲットライン")
            fig_chart = go.Figure()
            fig_chart.add_trace(go.Scatter(x=df.index, y=df['Close'], name="実績値", line=dict(color="#00fbff")))
            
            # 10pips, 15pipsのライン追加
            fig_chart.add_hline(y=price + 0.15, line_dash="dot", line_color="#00cc66", annotation_text="+15pips")
            fig_chart.add_hline(y=price + 0.10, line_dash="dash", line_color="#00cc66", annotation_text="+10pips")
            fig_chart.add_hline(y=price - 0.10, line_dash="dash", line_color="#ff3300", annotation_text="-10pips")
            fig_chart.add_hline(y=price - 0.15, line_dash="dot", line_color="#ff3300", annotation_text="-15pips")
            
            fig_chart.update_layout(
                template="plotly_dark", 
                height=500, 
                xaxis_rangeslider_visible=False,
                margin=dict(l=20, r=20, t=30, b=20)
            )
            st.plotly_chart(fig_chart, use_container_width=True)
            
        else:
            st.warning("最新データの取得に失敗しました。")
else:
    st.info("左側の「予測更新」ボタンを押すと解析を開始します。")
