
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from arch import arch_model

# Configuração da página
st.set_page_config(page_title="Amplitude Dólar Futuro", layout="centered")

st.markdown("<h1 style='text-align: center;'>📊 Amplitude Esperada do Dólar Futuro</h1>", unsafe_allow_html=True)

# Busca de dados
data = yf.download("BRL=X", period="60d")
data["Return"] = np.log(data["Close"] / data["Close"].shift(1)).dropna()

# Volatilidade Histórica
vol_hist = data["Return"].std()

# Volatilidade via GARCH(1,1)
model = arch_model(data["Return"].dropna() * 100, vol="Garch", p=1, q=1)
res = model.fit(disp="off")
sigma_garch = np.sqrt(res.forecast(horizon=1).variance.values[-1][0]) / 100

# Cálculos de amplitude
preco = data["Close"].iloc[-1]
amp_hist_rs = preco * vol_hist
amp_garch_rs = preco * sigma_garch
amp_hist_pts = amp_hist_rs * 1000
amp_garch_pts = amp_garch_rs * 1000

# Exibição
st.metric("📈 Preço atual (R$)", f"{preco:.4f}")
st.metric("📉 Volatilidade diária (histórica)", f"{vol_hist*100:.2f}%")
st.metric("📉 Volatilidade diária (GARCH)", f"{sigma_garch*100:.2f}%")
st.metric("🎯 Amplitude esperada (histórica)", f"{amp_hist_pts:.2f} pts")
st.metric("🎯 Amplitude esperada (GARCH)", f"{amp_garch_pts:.2f} pts")

st.caption("Últimos dados com base em BRL=X (Yahoo Finance).")
