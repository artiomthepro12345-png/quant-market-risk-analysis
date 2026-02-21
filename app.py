# ============================================
# Quant Market Risk Analysis Dashboard
# Built with Streamlit + Plotly + yFinance
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from scipy import stats
from datetime import datetime, timedelta

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="Quant Market Risk Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------
# Custom CSS
# ----------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #00d4aa, #7b61ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #888;
        margin-top: 0;
    }
    .metric-card {
        background: #1a1f2e;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #00d4aa;
    }
</style>
"", unsafe_allow_html=True)

# ----------------------------
# Sidebar Controls
# ----------------------------
st.sidebar.title("⚙️ Dashboard Controls")

# Asset Selection
default_assets = ["SPY", "QQQ", "VOO", "NVDA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "ARKQ", "SPMO", "BTC-USD", "ETH-USD", "GLD", "TLT"]
selected_assets = st.sidebar.multiselect(
    "Select Assets",
    options=default_assets,
    default=["SPY", "AAPL", "TSLA", "NVDA"]
)

# Date Range
col1, col2 = st.sidebar.columns(2)
start_date = col1.date_input("Start Date", datetime(2020, 1, 1))
end_date = col2.date_input("End Date", datetime.today())

# Risk-Free Rate
risk_free_rate = st.sidebar.slider("Risk-Free Rate (%)", 0.0, 10.0, 2.0, 0.1) / 100

# Confidence Level for VaR
confidence_level = st.sidebar.slider("VaR Confidence Level (%)", 90, 99, 95) / 100

# Monte Carlo Simulations
num_simulations = st.sidebar.select_slider(
    "Monte Carlo Simulations",
    options=[1000, 5000, 10000, 25000, 50000],
    value=10000
)

st.sidebar.markdown("---")
st.sidebar.markdown("Built by **artiomthepro12345-png**")
st.sidebar.markdown("Data: Yahoo Finance API")

# ----------------------------
# Data Loading (Cached)
# ----------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def load_data(tickers, start, end):
    """Download price data from yfinance, handling both old and new API formats."""
    raw = yf.download(tickers, start=start, end=end)

    # Handle MultiIndex columns (newer yfinance versions)
    if isinstance(raw.columns, pd.MultiIndex):
        # Try "Adj Close" first, fall back to "Close"
        if "Adj Close" in raw.columns.get_level_values(0):
            data = raw["Adj Close"]
        elif "Close" in raw.columns.get_level_values(0):
            data = raw["Close"]
        else:
            # Use whatever price column is available
            price_col = raw.columns.get_level_values(0)[0]
            data = raw[price_col]
    else:
        # Single ticker returns a simple DataFrame
        if "Adj Close" in raw.columns:
            data = raw[["Adj Close"]]
            data.columns = [tickers[0]] if isinstance(tickers, list) else [tickers]
        elif "Close" in raw.columns:
            data = raw[["Close"]]
            data.columns = [tickers[0]] if isinstance(tickers, list) else [tickers]
        else:
            data = raw

    # Ensure we have a DataFrame (not a Series)
    if isinstance(data, pd.Series):
        data = data.to_frame(name=tickers[0] if isinstance(tickers, list) else tickers)

    # Flatten MultiIndex columns if still present
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(-1)

    return data

# ----------------------------
# Main Header
# ----------------------------
st.markdown('<p class="main-header">📊 Quant Market Risk Analysis</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Real-time quantitative analysis of asset returns, volatility, and risk-adjusted performance</p>', unsafe_allow_html=True)
st.markdown("---")

if len(selected_assets) < 2:
    st.warning("⚠️ Please select at least 2 assets from the sidebar.")
    st.stop()

# Load data
with st.spinner("📡 Fetching market data..."):
    data = load_data(selected_assets, start_date, end_date)

if data.empty:
    st.error("❌ No data returned. Check your asset symbols and date range.")
    st.stop()

# Drop any columns or rows that are entirely NaN
data = data.dropna(how="all", axis=1).dropna(how="all", axis=0)

returns = data.pct_change().dropna()

# ============================================
# TAB LAYOUT
# ============================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Price Analysis",
    "📊 Return Analysis",
    "⚠️ Risk Metrics",
    "🎲 Monte Carlo",
    "📐 Optimization",
    "📋 Raw Data"
])

# ============================================
# TAB 1: PRICE ANALYSIS
# ============================================
with tab1:
    st.subheader("📈 Adjusted Close Prices")

    normalize = st.checkbox("Normalize prices (base = 100)", value=True)

    if normalize:
        plot_data = (data / data.iloc[0]) * 100
        y_label = "Normalized Price (Base = 100)"
    else:
        plot_data = data
        y_label = "Price (USD)"

    fig = px.line(
        plot_data, x=plot_data.index, y=plot_data.columns,
        title="Asset Price Performance",
        labels={"value": y_label, "variable": "Asset"},
    )
    fig.update_layout(
        template="plotly_dark",
        height=500,
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.15)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📉 Moving Averages")
    ma_asset = st.selectbox("Select asset for MA analysis", selected_assets)
    ma_short = st.slider("Short MA window", 5, 50, 20)
    ma_long = st.slider("Long MA window", 50, 200, 50)

    ma_data = pd.DataFrame({
        "Price": data[ma_asset],
        f"MA{ma_short}": data[ma_asset].rolling(ma_short).mean(),
        f"MA{ma_long}": data[ma_asset].rolling(ma_long).mean(),
    })

    fig_ma = px.line(ma_data, title=f"{ma_asset} — Moving Averages")
    fig_ma.update_layout(template="plotly_dark", height=450)
    st.plotly_chart(fig_ma, use_container_width=True)

# ============================================
# TAB 2: RETURN ANALYSIS
# ============================================
with tab2:
    st.subheader("📊 Daily Return Distribution")

    hist_asset = st.selectbox("Select asset for histogram", selected_assets, key="hist")

    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=returns[hist_asset],
        nbinsx=100,
        name=hist_asset,
        marker_color="#00d4aa",
        opacity=0.8
    ))

    x_range = np.linspace(returns[hist_asset].min(), returns[hist_asset].max(), 100)
    normal_curve = stats.norm.pdf(x_range, returns[hist_asset].mean(), returns[hist_asset].std())
    fig_hist.add_trace(go.Scatter(
        x=x_range,
        y=normal_curve * len(returns[hist_asset]) * (returns[hist_asset].max() - returns[hist_asset].min()) / 100,
        mode="lines",
        name="Normal Distribution",
        line=dict(color="red", width=2)
    ))

    fig_hist.update_layout(
        template="plotly_dark",
        title=f"{hist_asset} — Daily Return Distribution",
        xaxis_title="Daily Return",
        yaxis_title="Frequency",
        height=450
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    st.subheader("🔗 Correlation Matrix")
    corr = returns.corr()

    fig_corr = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        title="Asset Return Correlations",
        zmin=-1, zmax=1
    )
    fig_corr.update_layout(template="plotly_dark", height=500)
    st.plotly_chart(fig_corr, use_container_width=True)

    st.subheader("📈 Cumulative Returns")
    cum_returns = (1 + returns).cumprod() - 1

    fig_cum = px.line(
        cum_returns, x=cum_returns.index, y=cum_returns.columns,
        title="Cumulative Returns Over Time",
        labels={"value": "Cumulative Return", "variable": "Asset"}
    )
    fig_cum.update_layout(template="plotly_dark", height=450, hovermode="x unified")
    st.plotly_chart(fig_cum, use_container_width=True)

# ============================================
# TAB 3: RISK METRICS
# ============================================
with tab3:
    st.subheader("⚠️ Comprehensive Risk Metrics")

    trading_days = 252

    annual_return = returns.mean() * trading_days
    annual_volatility = returns.std() * np.sqrt(trading_days)
    sharpe = (annual_return - risk_free_rate) / annual_volatility

    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() * np.sqrt(trading_days)
    sortino = (annual_return - risk_free_rate) / downside_std

    z_score = stats.norm.ppf(1 - confidence_level)
    var_parametric = returns.mean() + z_score * returns.std()

    cvar = returns[returns <= var_parametric].mean()

    cum_returns_dd = (1 + returns).cumprod()
    rolling_max = cum_returns_dd.cummax()
    drawdown = (cum_returns_dd - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    skewness = returns.skew()
    kurtosis = returns.kurtosis()

    metrics_df = pd.DataFrame({
        "Annual Return": (annual_return * 100).round(2).astype(str) + "%",
        "Annual Volatility": (annual_volatility * 100).round(2).astype(str) + "%",
        "Sharpe Ratio": sharpe.round(3),
        "Sortino Ratio": sortino.round(3),
        f"VaR ({int(confidence_level*100)}%)": (var_parametric * 100).round(3).astype(str) + "%",
        f"CVaR ({int(confidence_level*100)}%)": (cvar * 100).round(3).astype(str) + "%",
        "Max Drawdown": (max_drawdown * 100).round(2).astype(str) + "%",
        "Skewness": skewness.round(3),
        "Excess Kurtosis": kurtosis.round(3),
    })

    st.dataframe(metrics_df, use_container_width=True)

    st.subheader("🏆 Best Performers")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Highest Return", annual_return.idxmax(), f"{annual_return.max()*100:.1f}%")
    kpi2.metric("Lowest Volatility", annual_volatility.idxmin(), f"{annual_volatility.min()*100:.1f}%")
    kpi3.metric("Best Sharpe", sharpe.idxmax(), f"{sharpe.max():.3f}")
    kpi4.metric("Best Sortino", sortino.idxmax(), f"{sortino.max():.3f}")

    st.subheader("📉 Drawdown Over Time")
    fig_dd = px.area(
        drawdown, x=drawdown.index, y=drawdown.columns,
        title="Portfolio Drawdown",
        labels={"value": "Drawdown", "variable": "Asset"}
    )
    fig_dd.update_layout(template="plotly_dark", height=400)
    st.plotly_chart(fig_dd, use_container_width=True)

    st.subheader("🌊 30-Day Rolling Volatility")
    rolling_vol = returns.rolling(30).std() * np.sqrt(trading_days)

    fig_rv = px.line(
        rolling_vol, x=rolling_vol.index, y=rolling_vol.columns,
        title="30-Day Rolling Annualized Volatility"
    )
    fig_rv.update_layout(template="plotly_dark", height=400, hovermode="x unified")
    st.plotly_chart(fig_rv, use_container_width=True)

# ============================================
# TAB 4: MONTE CARLO SIMULATION
# ============================================
with tab4:
    st.subheader("🎲 Monte Carlo Portfolio Simulation")
    st.markdown(f"Running **{num_simulations:,}** random portfolio allocations...")

    np.random.seed(42)
    num_assets = len(selected_assets)
    results = np.zeros((3, num_simulations))
    weights_record = np.zeros((num_simulations, num_assets))

    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    progress_bar = st.progress(0)
    for i in range(num_simulations):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        weights_record[i, :] = weights

        port_return = np.sum(mean_returns * weights) * trading_days
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * trading_days, weights)))
        port_sharpe = (port_return - risk_free_rate) / port_vol

        results[0, i] = port_return
        results[1, i] = port_vol
        results[2, i] = port_sharpe

        if i % max(1, num_simulations // 20) == 0:
            progress_bar.progress(i / num_simulations)

    progress_bar.progress(1.0)

    sim_df = pd.DataFrame({
        "Return": results[0],
        "Volatility": results[1],
        "Sharpe Ratio": results[2]
    })

    fig_mc = px.scatter(
        sim_df, x="Volatility", y="Return",
        color="Sharpe Ratio",
        color_continuous_scale="viridis",
        title="Monte Carlo Efficient Frontier",
        labels={"Return": "Annual Return", "Volatility": "Annual Volatility"}
    )

    max_sharpe_idx = results[2].argmax()
    min_vol_idx = results[1].argmin()

    fig_mc.add_trace(go.Scatter(
        x=[results[1, max_sharpe_idx]], y=[results[0, max_sharpe_idx]],
        mode="markers", marker=dict(color="red", size=15, symbol="star"),
        name="Max Sharpe"
    ))
    fig_mc.add_trace(go.Scatter(
        x=[results[1, min_vol_idx]], y=[results[0, min_vol_idx]],
        mode="markers", marker=dict(color="blue", size=15, symbol="diamond"),
        name="Min Volatility"
    ))

    fig_mc.update_layout(template="plotly_dark", height=600)
    st.plotly_chart(fig_mc, use_container_width=True)

# ============================================
# TAB 5: PORTFOLIO OPTIMIZATION RESULTS
# ============================================
with tab5:
    st.subheader("📐 Optimal Portfolio Allocations")

    col_opt1, col_opt2 = st.columns(2)

    with col_opt1:
        st.markdown("### 🔴 Maximum Sharpe Ratio Portfolio")
        st.metric("Expected Return", f"{results[0, max_sharpe_idx]*100:.2f}%")
        st.metric("Volatility", f"{results[1, max_sharpe_idx]*100:.2f}%")
        st.metric("Sharpe Ratio", f"{results[2, max_sharpe_idx]:.3f}")

        sharpe_weights = pd.DataFrame({
            "Asset": selected_assets,
            "Weight": weights_record[max_sharpe_idx]
        })
        fig_sw = px.pie(sharpe_weights, values="Weight", names="Asset",
                        title="Max Sharpe Allocation", hole=0.4,
                        color_discrete_sequence=px.colors.qualitative.Set2)
        fig_sw.update_layout(template="plotly_dark")
        st.plotly_chart(fig_sw, use_container_width=True)

    with col_opt2:
        st.markdown("### 🔵 Minimum Volatility Portfolio")
        st.metric("Expected Return", f"{results[0, min_vol_idx]*100:.2f}%")
        st.metric("Volatility", f"{results[1, min_vol_idx]*100:.2f}%")
        st.metric("Sharpe Ratio", f"{results[2, min_vol_idx]:.3f}")

        vol_weights = pd.DataFrame({
            "Asset": selected_assets,
            "Weight": weights_record[min_vol_idx]
        })
        fig_vw = px.pie(vol_weights, values="Weight", names="Asset",
                        title="Min Volatility Allocation", hole=0.4,
                        color_discrete_sequence=px.colors.qualitative.Pastel)
        fig_vw.update_layout(template="plotly_dark")
        st.plotly_chart(fig_vw, use_container_width=True)

    st.subheader("📊 Weight Comparison")
    compare_df = pd.DataFrame({
        "Asset": selected_assets,
        "Max Sharpe": weights_record[max_sharpe_idx],
        "Min Volatility": weights_record[min_vol_idx]
    })
    fig_comp = px.bar(
        compare_df.melt(id_vars="Asset"),
        x="Asset", y="value", color="variable",
        barmode="group",
        title="Portfolio Weight Comparison",
        labels={"value": "Weight", "variable": "Strategy"}
    )
    fig_comp.update_layout(template="plotly_dark", height=400)
    st.plotly_chart(fig_comp, use_container_width=True)

# ============================================
# TAB 6: RAW DATA
# ============================================
with tab6:
    st.subheader("📋 Raw Price Data")
    st.dataframe(data.tail(50), use_container_width=True)

    st.subheader("📋 Daily Returns")
    st.dataframe(returns.tail(50), use_container_width=True)

    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        csv_prices = data.to_csv()
        st.download_button("⬇️ Download Prices CSV", csv_prices, "prices.csv", "text/csv")
    with col_dl2:
        csv_returns = returns.to_csv()
        st.download_button("⬇️ Download Returns CSV", csv_returns, "returns.csv", "text/csv")

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>📊 <strong>Quant Market Risk Analysis Dashboard</strong></p>
    <p>Built with Streamlit • Plotly • yFinance • Python</p>
    <p>Data sourced from Yahoo Finance | For educational purposes</p>
</div>
"", unsafe_allow_html=True)