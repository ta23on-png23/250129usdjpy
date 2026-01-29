import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import torch
from chronos import ChronosPipeline

# ページ設定
st.set_page_config(page_title="USDJPY AI", layout="wide")

# モデル読み込み（メモリ節約のため一度だけ実行）
@st.cache_resource
def load_tiny_model():
    return ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-tiny",
        device_map="cpu",
        torch_dtype=torch.float32
    )

st.title("📈 USD/JPY 5分足 予測")

# サイドバー
st.sidebar.header("設定")
lookback = st.sidebar.select_slider("分析件数", options=[256, 512], value=512)
horizon = st.sidebar.slider("予測期間", 6, 20, 12)
update_btn = st.sidebar.button("予測更新")

# データ取得
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
    with st.spinner("解析中..."):
        df, price = get_fx_data(lookback)
        if df is not None:
            # 推論
            model = load_tiny_model()
            context = torch.tensor(df['Close'].values, dtype=torch.float32).unsqueeze(0)
            # サンプル数を最小限(50)にしてメモリ消費を抑制
            forecast = model.predict(context, horizon, num_samples=50)
            samples = forecast[0].numpy()

            # 確率計算
            def calc(pips):
                v = pips * 0.01
                u = np.any(samples >= price + v, axis=1).mean() * 100
                d = np.any(samples <= price - v, axis=1).mean() * 100
                return u, d

            u10, d10 = calc(10)
            u15, d15 = calc(15)

            # 結果表示
            st.success(f"現在価格: {price:.3f}")
            col1, col2 = st.columns(2)
            col1.metric("10Pips 上昇", f"{u10:.1f}%")
            col1.metric("10Pips 下落", f"{d10:.1f}%")
            col2.metric("15Pips 上昇", f"{u15:.1f}%")
            col2.metric("15Pips 下落", f"{d15:.1f}%")

            # チャート
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=df['Close'], name="実績", line=dict(color="cyan")))
            # 予測の平均線を実績の最後につなげる
            pred_mean = np.median(samples, axis=0)
            fig.add_trace(go.Scatter(x=list(range(len(df), len(df)+horizon)), y=pred_mean, name="AI予測", line=dict(color="red")))
            fig.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("データ取得失敗")
