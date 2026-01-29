import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import torch
from chronos import ChronosPipeline
from datetime import datetime

# --- ページ設定 ---
st.set_page_config(page_title="USD/JPY AI予測", layout="wide")

# --- タイトル ---
st.title("📈 USD/JPY 5分足 高低確率予測 (AI Ensemble)")
st.markdown("Amazon Chronosを使用して未来の軌道をシミュレーションし、指定pipsへの到達確率を算出します。")

# --- サイドバー設定 ---
st.sidebar.header("分析パラメータ")

lookback = st.sidebar.select_slider(
    "分析対象データ件数 (Lookback Window)",
    options=[256, 384, 512, 640, 768, 896, 1024],
    value=512,
    help="推奨値は512件です。"
)

horizon = st.sidebar.slider(
    "予測期間 (5分足本数)",
    min_value=6, max_value=48, value=12,
    help="10pips狙いなら12本(1h)、15pips狙いなら24本(2h)を推奨"
)

update_btn = st.sidebar.button("最新確定足を取得して予測更新")

# --- データ取得関数 ---
@st.cache_data(ttl=300)
def get_data(lookback_count):
    ticker = "USDJPY=X"
    # 5分足を取得
    data = yf.download(ticker, interval="5m", period="5d")
    
    if data.empty:
        return None, None, None

    # Yahoo Financeのマルチインデックス対策
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    # 最新の未確定足を除外し、指定件数分を取得
    confirmed_data = data.iloc[:-1].tail(lookback_count)
    
    # 確実に数値型(float)として抽出
    latest_price = float(confirmed_data['Close'].iloc[-1])
    last_time = confirmed_data.index[-1]
    
    return confirmed_data, latest_price, last_time

# --- AI推論関数 ---
def run_chronos_inference(context_data, prediction_length, num_samples=250):
    # 軽量モデルをCPUで実行
    pipeline = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-small",
        device_map="cpu",
        torch_dtype=torch.float32,
    )
    # 価格データをTensorに変換
    context = torch.tensor(context_data['Close'].values, dtype=torch.float32).unsqueeze(0)
    forecast = pipeline.predict(context, prediction_length, num_samples=num_samples)
    return forecast[0].numpy() # (num_samples, prediction_length)

# --- 確率計算関数 ---
def calculate_probs(current_price, samples, pips):
    target_val = pips * 0.01
    up_target = current_price + target_val
    down_target = current_price - target_val
    
    # 期間内に一度でもターゲットに触れたパスをカウント
    up_hits = np.any(samples >= up_target, axis=1).sum()
    down_hits = np.any(samples <= down_target, axis=1).sum()
    
    total = samples.shape[0]
    return (up_hits / total) * 100, (down_hits / total) * 100

# --- メインロジック ---
if update_btn:
    with st.status("分析実行中...", expanded=True) as status:
        st.write("📡 データ取得中...")
        df, current_price, last_time = get_data(lookback)
        
        if df is not None:
            st.write(f"✅ 確定足取得完了: {last_time} (価格: {current_price:.3f})")
            
            st.write("🤖 AI推論実行中 (Amazon Chronos)...")
            samples = run_chronos_inference(df, horizon)
            
            st.write("🧮 確率計算中...")
            p_up_10, p_down_10 = calculate_probs(current_price, samples, 10)
            p_up_15, p_down_15 = calculate_probs(current_price, samples, 15)
            
            status.update(label="分析完了", state="complete")
            
            # 結果表示
            st.divider()
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("🎯 10 Pips 到達率")
                st.metric("上昇", f"{p_up_10:.1f}%")
                st.metric("下落", f"{p_down_10:.1f}%")
            with c2:
                st.subheader("🎯 15 Pips 到達率")
                st.metric("上昇", f"{p_up_15:.1f}%")
                st.metric("下落", f"{p_down_15:.1f}%")

            # チャート表示
            st.subheader("📊 予測パスの可視化")
            future_index = [last_time + pd.Timedelta(minutes=5*i) for i in range(1, horizon+1)]
            median = np.median(samples, axis=0)
            
            fig = go.Figure()
            # 過去の足
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="実績", line=dict(color="gray")))
            # AI予測の中心
            fig.add_trace(go.Scatter(x=future_index, y=median, name="AI予測平均", line=dict(color="red", width=3)))
            # ターゲットライン
            fig.add_hline(y=current_price + 0.1, line_dash="dash", line_color="orange", annotation_text="+10pips")
            fig.add_hline(y=current_price - 0.1, line_dash="dash", line_color="orange", annotation_text="-10pips")
            
            fig.update_layout(template="plotly_dark", height=500, xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("データの取得に失敗しました。時間をおいて再度お試しください。")
else:
    st.info("サイドバーの「最新確定足を取得して予測更新」ボタンを押してください。")
