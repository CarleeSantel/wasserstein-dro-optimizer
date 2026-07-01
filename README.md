# Wasserstein DRO Optimizer
## Distributionally robust portfolio optimization with walk-forward backtesting

Standard Markowitz optimization assumes your historical return distribution is the true one. This tool doesn't. It uses Wasserstein distributionally robust optimization (DRO) to find portfolios that perform well even if the true distribution is somewhere near — but not exactly — what history shows. The epsilon parameter controls how far from history you're willing to plan for: at zero you get Markowitz, at high values you approach equal weight. A walk-forward backtester validates out-of-sample performance so you can see whether the robustness actually pays off on unseen data.

---

## Installation (run it yourself)

Requires Python 3.9+.

```bash
git clone https://github.com/CarleeSantel/wasserstein-dro-optimizer.git
cd wasserstein-dro-optimizer
pip install streamlit yfinance pandas numpy cvxpy plotly
streamlit run wasserstein_optimizer.py
```

Note: cvxpy must be installed with the SCS solver backend. SCS is included in the default cvxpy install — no extra step needed.

---

## Installation (contribute)

```bash
git clone https://github.com/CarleeSantel/wasserstein-dro-optimizer.git
cd wasserstein-dro-optimizer
pip install streamlit yfinance pandas numpy cvxpy plotly
```

Run `streamlit run wasserstein_optimizer.py` to test locally. If you change the optimization formulation, note that the Wasserstein DRO problem is a second-order cone program (SOCP) — it requires the SCS solver, not OSQP. Swapping solvers will cause a silent failure.

---

## Contributing

Open an issue before submitting a pull request. If you're changing the optimization formulation, include a brief note explaining why the new form is still a valid DRO relaxation.

---

## Known issues

- Efficient frontier computation can be slow for large asset universes (15+ assets) — frontier resolution is currently fixed at 50 points
- Walk-forward backtest assumes fixed rebalancing windows; no support for dynamic rebalancing yet
- No transaction cost model in the backtest
