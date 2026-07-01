import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import cvxpy as cp
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Wasserstein Portfolio Optimizer", layout="wide")
st.title("Portfolio Optimizer — Wasserstein Duality")
st.caption("Distributionally Robust Optimization vs. Classical Markowitz · with Walk-Forward Backtest")

tab1, tab2 = st.tabs(["Optimizer", "Walk-Forward Backtest"])

# ── SIDEBAR ────────────────────────────────────────────────────────────────────
st.sidebar.header("Settings")
default_tickers = "AAPL, MSFT, JPM, GS, XOM, JNJ"
ticker_input = st.sidebar.text_input("Tickers (comma-separated)", default_tickers)
tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
period      = st.sidebar.selectbox("Historical Period", ["2y","3y","5y"], index=1)
epsilon     = st.sidebar.slider("ε — Wasserstein Uncertainty Radius", 0.0, 0.5, 0.1, step=0.01,
                                 help="0 = Markowitz. Higher = more robust, more diversified.")
risk_aversion = st.sidebar.slider("λ — Risk Aversion", 0.5, 10.0, 2.0, step=0.5)
allow_short = st.sidebar.checkbox("Allow Short Selling", value=False)

# Backtest params (used in tab2)
st.sidebar.divider()
st.sidebar.subheader("Backtest Settings")
train_months = st.sidebar.slider("Training Window (months)", 6, 24, 12)
test_months  = st.sidebar.slider("Test Window (months)", 1, 6, 3)

# ── DATA ───────────────────────────────────────────────────────────────────────
with st.spinner("Fetching data..."):
    raw    = yf.download(tickers, period=period, auto_adjust=True, progress=False)
    prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0])
    prices  = prices.dropna(axis=1)
    tickers = list(prices.columns)
    returns = prices.pct_change().dropna()

n  = len(tickers)
R  = returns.values
mu = R.mean(axis=0)
Sigma = np.cov(R.T)

# ── OPTIMIZATION FUNCTIONS ─────────────────────────────────────────────────────
def markowitz(mu, Sigma, lam, allow_short=False):
    x = cp.Variable(len(mu))
    obj = cp.Maximize(mu @ x - (lam/2) * cp.quad_form(x, Sigma))
    cons = [cp.sum(x) == 1]
    if not allow_short: cons.append(x >= 0)
    cp.Problem(obj, cons).solve(solver=cp.OSQP)
    return x.value

def wasserstein_dro(mu, Sigma, lam, epsilon, allow_short=False):
    x = cp.Variable(len(mu))
    obj = cp.Maximize(mu @ x - (lam/2)*cp.quad_form(x, Sigma) - epsilon*cp.norm(x,2))
    cons = [cp.sum(x) == 1]
    if not allow_short: cons.append(x >= 0)
    cp.Problem(obj, cons).solve(solver=cp.SCS)
    return x.value

def port_stats(w, R_data):
    dr = R_data @ w
    ann_ret = dr.mean() * 252
    ann_vol = dr.std()  * np.sqrt(252)
    sharpe  = ann_ret / ann_vol if ann_vol > 0 else 0
    cum     = np.cumprod(1 + dr)
    max_dd  = ((np.maximum.accumulate(cum) - cum) / np.maximum.accumulate(cum)).max()
    return ann_ret, ann_vol, sharpe, max_dd

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OPTIMIZER
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    w_mk  = markowitz(mu, Sigma, risk_aversion, allow_short)
    w_dro = wasserstein_dro(mu, Sigma, risk_aversion, epsilon, allow_short)

    if w_mk is None or w_dro is None:
        st.error("Optimization failed — adjust parameters or tickers.")
        st.stop()

    mk_ret,mk_vol,mk_sr,mk_dd = port_stats(w_mk,  R)
    dr_ret,dr_vol,dr_sr,dr_dd = port_stats(w_dro, R)

    st.subheader("Portfolio Comparison (In-Sample)")
    st.caption(f"ε={epsilon} · λ={risk_aversion} · {period} data · {'Long/Short' if allow_short else 'Long-Only'}")
    c1,c2,c3,c4 = st.columns(4)
    for col,m,mv,dv in zip([c1,c2,c3,c4],
                            ["Ann. Return","Ann. Vol","Sharpe","Max Drawdown"],
                            [f"{mk_ret:.2%}",f"{mk_vol:.2%}",f"{mk_sr:.3f}",f"{mk_dd:.2%}"],
                            [f"{dr_ret:.2%}",f"{dr_vol:.2%}",f"{dr_sr:.3f}",f"{dr_dd:.2%}"]):
        col.markdown(f"**{m}**")
        col.markdown(f"Markowitz: `{mv}`")
        col.markdown(f"DRO (ε={epsilon}): `{dv}`")

    st.divider()
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Efficient Frontier")
        lambdas = np.linspace(0.1, 15, 50)
        mk_f, dro_f = [], []
        for lam in lambdas:
            wm = markowitz(mu, Sigma, lam, allow_short)
            wd = wasserstein_dro(mu, Sigma, lam, epsilon, allow_short)
            if wm  is not None: mk_f.append(port_stats(wm,  R)[:2])
            if wd  is not None: dro_f.append(port_stats(wd, R)[:2])
        fig = go.Figure()
        if mk_f:
            v,r = zip(*mk_f)
            fig.add_trace(go.Scatter(x=v,y=r,mode="lines",name="Markowitz",
                                      line=dict(color="#4C78A8",width=2)))
        if dro_f:
            v,r = zip(*dro_f)
            fig.add_trace(go.Scatter(x=v,y=r,mode="lines",name=f"DRO (ε={epsilon})",
                                      line=dict(color="#F58518",width=2,dash="dash")))
        fig.add_trace(go.Scatter(x=[mk_vol],y=[mk_ret],mode="markers",name="Markowitz ★",
                                  marker=dict(size=12,color="#4C78A8",symbol="star")))
        fig.add_trace(go.Scatter(x=[dr_vol],y=[dr_ret],mode="markers",name="DRO ★",
                                  marker=dict(size=12,color="#F58518",symbol="star")))
        fig.update_layout(xaxis_title="Ann. Vol",yaxis_title="Ann. Return",
                           xaxis_tickformat=".1%",yaxis_tickformat=".1%",
                           margin=dict(t=20,b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.subheader("Weight Allocation")
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(name="Markowitz",x=tickers,y=w_mk,marker_color="#4C78A8"))
        fig2.add_trace(go.Bar(name=f"DRO (ε={epsilon})",x=tickers,y=w_dro,marker_color="#F58518"))
        fig2.update_layout(barmode="group",yaxis_tickformat=".1%",margin=dict(t=20,b=20))
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    st.subheader("Cumulative Returns")
    cum_mk  = np.cumprod(1+R@w_mk)-1
    cum_dro = np.cumprod(1+R@w_dro)-1
    cum_ew  = np.cumprod(1+R@np.ones(n)/n)-1
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=returns.index,y=cum_mk,  name="Markowitz",line=dict(color="#4C78A8")))
    fig3.add_trace(go.Scatter(x=returns.index,y=cum_dro, name=f"DRO (ε={epsilon})",
                               line=dict(color="#F58518",dash="dash")))
    fig3.add_trace(go.Scatter(x=returns.index,y=cum_ew,  name="Equal Weight",
                               line=dict(color="#72B7B2",dash="dot")))
    fig3.update_layout(yaxis_tickformat=".1%",margin=dict(t=20,b=20))
    st.plotly_chart(fig3, use_container_width=True)

    st.divider()
    st.subheader("ε Sensitivity — How Robustness Shifts Weights")
    epsilons = np.linspace(0, 0.5, 30)
    wpaths   = {t: [] for t in tickers}
    for e in epsilons:
        w = wasserstein_dro(mu, Sigma, risk_aversion, e, allow_short)
        for i,t in enumerate(tickers):
            wpaths[t].append(w[i] if w is not None else np.nan)
    fig4 = go.Figure()
    colors = px.colors.qualitative.Plotly
    for i,t in enumerate(tickers):
        fig4.add_trace(go.Scatter(x=epsilons,y=wpaths[t],mode="lines",name=t,
                                   line=dict(color=colors[i%len(colors)],width=2)))
    fig4.add_vline(x=epsilon,line_dash="dash",line_color="white",
                   annotation_text=f"Current ε={epsilon}")
    fig4.update_layout(xaxis_title="ε",yaxis_title="Weight",
                        yaxis_tickformat=".1%",margin=dict(t=20,b=20))
    st.plotly_chart(fig4, use_container_width=True)
    st.caption("ε=0 → Markowitz. As ε increases, weights converge toward equal allocation.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — WALK-FORWARD BACKTEST
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Walk-Forward Out-of-Sample Backtest")
    st.markdown(
        "Trains each optimizer on a rolling window of historical data, "
        "then applies the resulting weights to the **next unseen period**. "
        "This is the only honest way to evaluate a portfolio optimizer — "
        "in-sample performance tells you nothing about real-world behavior."
    )

    train_days = train_months * 21
    test_days  = test_months  * 21

    if len(R) < train_days + test_days:
        st.warning("Not enough data for these backtest settings. Increase the historical period or reduce window sizes.")
    else:
        with st.spinner("Running walk-forward backtest..."):
            mk_oos, dro_oos, ew_oos = [], [], []
            dates_oos = []

            i = 0
            while i + train_days + test_days <= len(R):
                R_train = R[i : i + train_days]
                R_test  = R[i + train_days : i + train_days + test_days]
                dates_test = returns.index[i + train_days : i + train_days + test_days]

                mu_t  = R_train.mean(axis=0)
                Sig_t = np.cov(R_train.T)

                wm = markowitz(mu_t, Sig_t, risk_aversion, allow_short)
                wd = wasserstein_dro(mu_t, Sig_t, risk_aversion, epsilon, allow_short)
                we = np.ones(n) / n

                if wm is not None and wd is not None:
                    mk_oos.extend(R_test @ wm)
                    dro_oos.extend(R_test @ wd)
                    ew_oos.extend(R_test @ we)
                    dates_oos.extend(dates_test)

                i += test_days

        if mk_oos:
            mk_oos  = np.array(mk_oos)
            dro_oos = np.array(dro_oos)
            ew_oos  = np.array(ew_oos)

            cum_mk_oos  = np.cumprod(1+mk_oos)-1
            cum_dro_oos = np.cumprod(1+dro_oos)-1
            cum_ew_oos  = np.cumprod(1+ew_oos)-1

            # OOS stats
            def oos_stats(r):
                ann_r = r.mean()*252
                ann_v = r.std()*np.sqrt(252)
                sh    = ann_r/ann_v if ann_v>0 else 0
                cum   = np.cumprod(1+r)
                dd    = ((np.maximum.accumulate(cum)-cum)/np.maximum.accumulate(cum)).max()
                return ann_r, ann_v, sh, dd

            mk_s  = oos_stats(mk_oos)
            dro_s = oos_stats(dro_oos)
            ew_s  = oos_stats(ew_oos)

            st.subheader("Out-of-Sample Performance")
            df_stats = pd.DataFrame({
                "Ann. Return": [f"{s[0]:.2%}" for s in [mk_s,dro_s,ew_s]],
                "Ann. Vol":    [f"{s[1]:.2%}" for s in [mk_s,dro_s,ew_s]],
                "Sharpe":      [f"{s[2]:.3f}" for s in [mk_s,dro_s,ew_s]],
                "Max Drawdown":[f"{s[3]:.2%}" for s in [mk_s,dro_s,ew_s]],
            }, index=["Markowitz",f"DRO (ε={epsilon})","Equal Weight"])
            st.dataframe(df_stats, use_container_width=True)

            fig5 = go.Figure()
            fig5.add_trace(go.Scatter(x=dates_oos,y=cum_mk_oos, name="Markowitz",
                                       line=dict(color="#4C78A8")))
            fig5.add_trace(go.Scatter(x=dates_oos,y=cum_dro_oos,name=f"DRO (ε={epsilon})",
                                       line=dict(color="#F58518",dash="dash")))
            fig5.add_trace(go.Scatter(x=dates_oos,y=cum_ew_oos, name="Equal Weight",
                                       line=dict(color="#72B7B2",dash="dot")))
            fig5.update_layout(
                title=f"OOS Cumulative Return — Train: {train_months}mo · Test: {test_months}mo rolling",
                yaxis_tickformat=".1%", margin=dict(t=40,b=20)
            )
            st.plotly_chart(fig5, use_container_width=True)

            n_windows = len(mk_oos)//test_days
            st.caption(f"{n_windows} test windows · {len(mk_oos)} total out-of-sample trading days")

            st.info(
                "**What to look for:** DRO should outperform Markowitz in high-volatility periods "
                "because its ε penalty explicitly optimizes for distributional uncertainty. "
                "If Markowitz beats DRO consistently, try decreasing ε — the uncertainty radius "
                "may be overcorrecting. This is the model parameterization debate in action."
            )

st.caption("Built with Python · cvxpy · yfinance · Streamlit · Plotly")
