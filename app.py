import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from prophet import Prophet
from scipy.stats import norm
import plotly.graph_objs as go
from datetime import timedelta

# --- ページ設定 ---
st.set_page_config(page_title="USD/JPY 短期決着予測", layout="wide")

st.title("⚡ USD/JPY 5分足 短期決着予測")
st.markdown("10〜15pipsの利益を狙うための、最大30分後までの超短期予測モデルです。")

# --- サイドバー設定 ---
st.sidebar.header("スキャルピング設定")
# 期間を5分〜30分に限定
predict_minutes = st.sidebar.slider("予測完了までの時間 (分後)", min_value=5, max_value=30, value=15, step=5)
horizon = predict_minutes // 5

# 短期決着なので、直近の動きへの感度を高く設定可能に
trend_sensitivity = st.sidebar.slider("トレンド追従感度", 0.05, 0.50, 0.25, step=0.05)
entry_threshold = st.sidebar.radio("エントリー基準勝率 (%)", [60, 65, 70], index=1, horizontal=True)

update_btn = st.sidebar.button("最新の勢いを解析")

# --- データ取得 ---
@st.cache_data(ttl=60) # 短期なのでキャッシュ時間を1分に短縮
def get_short_term_data(n=300):
    try:
        # 短期予測には直近数日分あれば十分
        df = yf.download("USDJPY=X", interval="5m", period="2d", progress=False)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.reset_index().rename(columns={'Datetime': 'ds', 'Close': 'y'})
        df['ds'] = pd.to_datetime(df['ds']).dt.tz_convert('Asia/Tokyo').dt.tz_localize(None)
        return df.tail(n)
    except:
        return None

# --- メイン処理 ---
if update_btn:
    with st.spinner(f'{predict_minutes}分以内の決着ポイントを計算中...'):
        df = get_short_term_data()
        if df is not None:
            # 1. Prophetによる短期学習
            # changepoint_prior_scale を高くして直近の動きに敏感に反応させる
            m = Prophet(changepoint_prior_scale=trend_sensitivity, daily_seasonality=True)
            m.fit(df[['ds', 'y']])
            
            future = m.make_future_dataframe(periods=horizon + 2, freq='5min')
            forecast = m.predict(future)
            
            current_price = float(df['y'].iloc[-1])
            last_time = df['ds'].iloc[-1]
            predicted_price = float(forecast.iloc[-1]['yhat'])
            
            # 2. 超短期ボラティリティ (直近20本=100分に集中)
            recent_returns = np.log(df['y'] / df['y'].shift(1)).dropna().tail(20)
            vol = recent_returns.std()
            h_vol = vol * np.sqrt(horizon)
            
            # 3. 勝率(期待方向)の計算
            target_ret = np.log(predicted_price / current_price)
            prob_up = (1 - norm.cdf(0, loc=target_ret, scale=h_vol)) * 100
            prob_down = 100 - prob_up
            
            # --- UI表示 ---
            jst_now = last_time + timedelta(hours=0)
            st.success(f"現在値: {current_price:.3f} | 日本時間: {jst_now.strftime('%H:%M')}")
            
            st.subheader(f"🎯 {predict_minutes}分後の着地期待度")
            col1, col2 = st.columns(2)
            
            status_up = "🚀 BUY CHANCE" if prob_up >= entry_threshold else ""
            status_down = "📉 SELL CHANCE" if prob_down >= entry_threshold else ""
            col1.metric("上昇勝率", f"{prob_up:.1f}%", status_up)
            col2.metric("下落勝率", f"{prob_down:.1f}%", status_down)

            # ターゲット勝率グラフ
            st.markdown(f"#### {predict_minutes}分以内に10〜15pips圏内へ到達する確率")
            t_pips = [15, 10, -10, -15]
            t_labels = ["+15pips", "+10pips", "-10pips", "-15pips"]
            t_probs = []
            for tp in t_pips:
                t_ret = np.log((current_price + (tp * 0.01)) / current_price)
                # 分布の中心(loc)にAI予測の勢いを含める
                p = (1 - norm.cdf(t_ret, loc=target_ret, scale=h_vol)) * 100
                t_probs.append(p if tp > 0 else 100 - p)

            fig_bar = go.Figure(go.Bar(
                x=t_labels, y=t_probs,
                marker_color=['#00cc96', '#00cc96', '#ff4b4b', '#ff4b4b'],
                text=[f"{p:.1f}%" for p in t_probs], textposition='auto'
            ))
            fig_bar.update_layout(template="plotly_dark", height=350, yaxis=dict(range=[0, 100]))
            st.plotly_chart(fig_bar, use_container_width=True)

            # チャート表示
            fig_chart = go.Figure()
            # 表示範囲を直近2時間分に絞って見やすく
            display_df = df.tail(24) 
            fig_chart.add_trace(go.Scatter(x=display_df['ds'], y=display_df['y'], name="実績", line=dict(color="#00fbff")))
            # AIの予測軌道
            pred_future = forecast[forecast['ds'] >= last_time].head(horizon + 1)
            fig_chart.add_trace(go.Scatter(x=pred_future['ds'], y=pred_future['yhat'], name="AI推論パス", line=dict(color="yellow", dash="dot")))
            
            # ターゲットライン
            for tp, color in [(0.10, "#00cc96"), (-0.10, "#ff4b4b")]:
                fig_chart.add_hline(y=current_price + tp, line_dash="dash", line_color=color, opacity=0.5)

            fig_chart.update_layout(template="plotly_dark", height=450, xaxis_rangeslider_visible=False)
            st.plotly_chart(fig_chart, use_container_width=True)
            
        else:
            st.error("データの取得に失敗しました。")
else:
    st.info("「最新の勢いを解析」ボタンを押して、超短期の勝機を判定します。")
