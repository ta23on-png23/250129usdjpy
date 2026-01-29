import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import torch
from chronos import ChronosPipeline
from datetime import datetime

# --- ページ設定 ---
st.set_page_config(page_title="USD/JPY AI予測ダッシュボード", layout="wide")

# --- タイトル・説明 ---
st.title("📈 USD/JPY 5分足 高低確率予測 (AI Ensemble)")
st.markdown("Amazon Chronos を使用して未来の軌道をシミュレーションし、指定pipsへの到達確率を算出します。")

# --- サイドバー設定 ---
st.sidebar.header("分析パラメータ")

# モデル選択（今回はChronosをメインに、将来的にLag-Llama等を追加可能）
model_option = st.sidebar.selectbox("予測モデル", ["Amazon Chronos (T5-Small)"])

# 過去データ件数
lookback = st.sidebar.select_slider(
    "分析対象データ件数 (Lookback Window)",
    options=[256, 384, 512, 640, 768, 896, 1024],
    value=512,
    help="推奨値は512件です。最新の確定足から遡る件数を指定します。"
)

# 予測期間
horizon = st.sidebar.slider(
    "予測期間 (Prediction Horizon / 5分足本数)",
    min_value=6, max_value=48, value=12,
    help="10pips狙いなら12本(1h)、15pips狙いなら24本(2h)を推奨"
)

# 更新ボタン
update_btn = st.sidebar.button("最新確定足を取得して予測更新")

# --- 関数定義 ---

@st.cache_data(ttl=300) # 5分間キャッシュ
def get_data(lookback_count):
    ticker = "USDJPY=X"
    data = yf.download(ticker, interval="5m", period="5d")
    if data.empty: return None, None, None
    confirmed_data = data.iloc[:-1].tail(lookback_count) # 最新の未確定足を除外
    latest_price = confirmed_data['Close'].iloc[-1]
    last_time = confirmed_data.index[-1]
    return confirmed_data, latest_price, last_time

def run_chronos_inference(context_data, prediction_length, num_samples=250):
    # 軽量版モデルを使用
    pipeline = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-small",
        device_map="cpu", # サーバー環境に合わせて調整
        torch_dtype=torch.float32,
    )
    context = torch.tensor(context_data['Close'].values).unsqueeze(0)
    # 未来のパスを生成
    forecast = pipeline.predict(context, prediction_length, num_samples=num_samples)
    return forecast[0].numpy() # (num_samples, prediction_length)

def calculate_probs(current_price, samples, pips):
    target_val = pips * 0.01
    up_target = current_price + target_val
    down_target = current_price - target_val
    
    # 各パスが期間内にターゲットに先に触れたかを判定
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
            st.write(f"✅ 確定足取得: {last_time} ({current_price:.3f})")
            
            st.write("🤖 AI推論実行中 (Chronos)...")
            # AI予測実行
            samples = run_chronos_inference(df, horizon)
            
            st.write("🧮 確率計算中...")
            p_up_10, p_down_10 = calculate_probs(current_price, samples, 10)
            p_up_15, p_down_15 = calculate_probs(current_price, samples, 15)
            
            status.update(label="分析完了", state="complete")
            
            # --- 結果表示 ---
            st.divider()
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("10 Pips 上昇確率", f"{p_up_10:.1f}%")
                st.metric("10 Pips 下落確率", f"{p_down_10:.1f}%", delta_color="inverse")
            with col_b:
                st.metric("15 Pips 上昇確率", f"{p_up_15:.1f}%")
                st.metric("15 Pips 下落確率", f"{p_down_15:.1f}%", delta_color="inverse")

            # --- チャート描画 ---
            st.subheader("📊 予測パスの可視化")
            future_index = [last_time + pd.Timedelta(minutes=5*i) for i in range(1, horizon+1)]
            median = np.median(samples, axis=0)
            
            fig = go.Figure()
            # 過去の足
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="実績値", line=dict(color="gray")))
            # 予測の中心
            fig.add_trace(go.Scatter(x=future_index, y=median, name="AI予測平均", line=dict(color="red", width=3)))
            # ターゲットライン
            fig.add_hline(y=current_price + 0.1, line_dash="dash", line_color="orange", annotation_text="+10 pips")
            fig.add_hline(y=current_price - 0.1, line_dash="dash", line_color="orange", annotation_text="-10 pips")
            
            fig.update_layout(template="plotly_dark", height=500, xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.error("データが取得できませんでした。")
else:
    st.info("サイドバーの「最新価格で予測を更新」ボタンを押してください。")
