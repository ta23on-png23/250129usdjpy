import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import torch
from chronos import ChronosPipeline

# --- ページ設定 ---
st.set_page_config(page_title="USD/JPY AI予測", layout="wide")

# --- メモリ管理: モデルをキャッシュして1度だけ読み込む ---
@st.cache_resource
def load_model():
    # tinyモデルに変更してメモリ消費を大幅に削減
    return ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-tiny", 
        device_map="cpu", 
        torch_dtype=torch.float32
    )

st.title("📈 USD/JPY 5分足 高低確率予測")

# --- サイドバー ---
st.sidebar.header("分析パラメータ")
lookback = st.sidebar.select_slider("分析対象件数", options=[256, 384, 512], value=512)
horizon = st.sidebar.slider("予測期間 (5分足本数)", 6, 24, 12)
update_btn = st.sidebar.button("最新確定足を取得して予測更新")

# --- データ取得 ---
@st.cache_data(ttl=300)
def get_data(lookback_count):
    try:
        data = yf.download("USDJPY=X", interval="5m", period="5d")
        if data.empty: return None, None, None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        confirmed_data = data.iloc[:-1].tail(lookback_count)
        latest_price = float(confirmed_data['Close'].iloc[-1])
        last_time = confirmed_data.index[-1]
        return confirmed_data, latest_price, last_time
    except:
        return None, None, None

# --- メインロジック ---
if update_btn:
    with st.spinner("AIが解析中... (30秒ほどかかります)"):
        df, current_price, last_time = get_data(lookback)
        
        if df is not None:
            # モデル読み込み
            pipeline = load_model()
            
            # 推論実行 (サンプル数を減らして高速化)
            context = torch.tensor(df['Close'].values, dtype=torch.float32).unsqueeze(0)
            forecast = pipeline.predict(context, horizon, num_samples=100)
            samples = forecast[0].numpy()

            # 確率計算
            def get_p(pips):
                val = pips * 0.01
                u, d = (np.any(samples >= current_price + val, axis=1).sum(), 
                        np.any(samples <= current_price - val, axis=1).sum())
                return (u/100)*100, (d/100)*100

            p10_u, p10_d = get_p(10)
            p15_u, p15_d = get_p(15)

            # 表示
            st.success(f"データ取得完了: {last_time} ({current_price:.3f})")
            c1, c2 = st.columns(2)
            c1.metric("10Pips 上昇確率", f"{p10_u:.1f}%")
            c1.metric("10Pips 下落確率", f"{p10_d:.1f}%")
            c2.metric("15Pips 上昇確率", f"{p15_u:.1f}%")
            c2.metric("15Pips 下落確率", f"{p15_d:.1f}%")

            # チャート
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="実績", line=dict(color="gray")))
            future_idx = [last_time + pd.Timedelta(minutes=5*i) for i in range(1, horizon+1)]
            fig.add_trace(go.Scatter(x=future_idx, y=np.median(samples, axis=0), name="AI予測", line=dict(color="red")))
            fig.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("データ取得エラー。再度ボタンを押してください。")
