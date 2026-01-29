import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm

# 1. ページ設定
st.set_page_config(page_title="USDJPY 確率予測", layout="wide")

st.title("📈 USD/JPY 5分足 到達確率予測 (統計・安定版)")
st.markdown("サーバー負荷を抑えるため、統計的ボラティリティに基づき確率を算出します。")

# 2. サイドバー設定
st.sidebar.header("分析設定")
lookback = st.sidebar.select_slider("分析対象件数 (過去)", options=[256, 512, 1024], value=512)
horizon = st.sidebar.slider("予測期間 (5分足の本数)", 6, 48, 12)
update_btn = st.sidebar.button("最新価格で予測更新")

# 3. データ取得関数
@st.cache_data(ttl=300)
def get_fx_data(n):
    try:
        # Yahoo Financeから取得
        df = yf.download("USDJPY=X", interval="5m", period="5d")
        if df.empty: return None, None
        
        # マルチインデックス対策
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        # 確定足のみ抽出
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
            # --- 統計的確率計算 (ボラティリティ・アプローチ) ---
            # 5分足ごとのリターンを算出
            returns = np.log(df['Close'] / df['Close'].shift(1)).dropna()
            vol = returns.std() # 標準偏差
            
            def calc_prob(pips):
                target_diff = pips * 0.01
                # 期間(horizon)を考慮したボラティリティ
                h_vol = vol * np.sqrt(horizon)
                # ターゲットへの到達確率 (累積分布関数を使用)
                target_ret = np.log((price + target_diff) / price)
                prob_up = (1 - norm.cdf(target_ret, loc=0, scale=h_vol)) * 100
                return round(prob_up, 1)

            p10 = calc_prob(10)
            p15 = calc_prob(15)

            # --- 表示セクション ---
            st.success(f"現在価格: {price:.3f} (更新時刻: {df.index[-1].strftime('%H:%M')})")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("10 Pips 到達確率", f"{p10}%")
            with col2:
                st.metric("15 Pips 到達確率", f"{p15}%")

            # --- チャート表示 ---
            st.subheader("📊 チャートとターゲットライン")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="実績値", line=dict(color="#00fbff")))
            
            # ターゲットライン
            fig.add_hline(y=price + 0.1, line_dash="dash", line_color="orange", annotation_text="+10pips")
            fig.add_hline(y=price - 0.1, line_dash="dash", line_color="orange", annotation_text="-10pips")
            
            fig.update_layout(
                template="plotly_dark", 
                height=500, 
                xaxis_rangeslider_visible=False,
                margin=dict(l=20, r=20, t=30, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.warning("最新データの取得に失敗しました。数秒待ってから再度お試しください。")
else:
    st.info("左側の「予測更新」ボタンを押すと解析を開始します。")
