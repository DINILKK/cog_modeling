"""
================================================
Bayesian Prospect Theory on choices13k Dataset
================================================
Author: Dinil
Dataset: choices13k (Peterson et al., 2021, Science)

Goal: Fit a hierarchical Bayesian Prospect Theory model using PyMC
to understand how humans distort probabilities and values when making
risky decisions.

Model Parameters:
  - alpha  : value function curvature (risk attitudes for gains)
  - gamma  : probability weighting function curvature
  - delta  : probability weighting function elevation
  - beta   : decision noise (softmax temperature)
  - kappa  : Beta-likelihood concentration
"""

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import matplotlib
matplotlib.rcParams['figure.dpi'] = 120
import matplotlib.pyplot as plt
import arviz as az


# ─────────────────────────────────────────────
# PROSPECT THEORY HELPER FUNCTIONS
# (defined at module level so Windows can pickle them)
# ─────────────────────────────────────────────

def value_function(x, alpha):
    """
    Power value function (Tversky & Kahneman 1992).
    v(x) = x^alpha       for gains (x >= 0)
    v(x) = -(-x)^alpha   for losses (x < 0)
    """
    return pt.sign(x) * pt.abs(x + 1e-8) ** alpha


def prob_weighting(p, gamma, delta):
    """
    Prelec (1998) two-parameter probability weighting function.
    w(p) = exp(-delta * (-log(p))^gamma)
    """
    p_safe = pt.clip(p, 1e-6, 1 - 1e-6)
    return pt.exp(-delta * (-pt.log(p_safe)) ** gamma)


def prospect_value(H, L, pH, alpha, gamma, delta):
    """
    Prospect theory value of a two-outcome gamble.
    """
    pL = 1.0 - pH
    wH = prob_weighting(pH, gamma, delta)
    wL = prob_weighting(pL, gamma, delta)
    vH = value_function(H, alpha)
    vL = value_function(L, alpha)
    return wH * vH + wL * vL


# ─────────────────────────────────────────────
# MAIN — required on Windows to avoid multiprocessing crash
# ─────────────────────────────────────────────

if __name__ == "__main__":

    # ── 1. LOAD DATA ──────────────────────────
    print("Loading choices13k dataset...")
    df = pd.read_csv("choices13k/c13k_selections.csv")
    print(f"Dataset shape: {df.shape}")
    print(df.head())

    # ── 2. PREPARE DATA ───────────────────────
    df_clean = df.dropna(subset=["Ha", "La", "pHa", "Hb", "Lb", "pHb", "bRate"]).copy()

    N_PROBLEMS = 500
    df_model = df_clean.sample(N_PROBLEMS, random_state=42).reset_index(drop=True)

    Ha       = df_model["Ha"].values.astype(float)
    La       = df_model["La"].values.astype(float)
    pHa      = df_model["pHa"].values.astype(float)
    Hb       = df_model["Hb"].values.astype(float)
    Lb       = df_model["Lb"].values.astype(float)
    pHb      = df_model["pHb"].values.astype(float)
    rate_obs = df_model["bRate"].values.astype(float)

    print(f"\nUsing {N_PROBLEMS} choice problems for model fitting.")
    print(f"Mean choice rate for B: {rate_obs.mean():.3f}")

    # ── 3. BUILD & SAMPLE MODEL ───────────────
    print("\nBuilding PyMC model...")

    with pm.Model() as pt_model:

        alpha = pm.Beta("alpha", alpha=2, beta=2)
        gamma = pm.Beta("gamma", alpha=2, beta=2)
        delta = pm.HalfNormal("delta", sigma=1.0)
        beta  = pm.HalfNormal("beta",  sigma=2.0)
        kappa = pm.HalfNormal("kappa", sigma=10.0)

        VA = prospect_value(Ha, La, pHa, alpha, gamma, delta)
        VB = prospect_value(Hb, Lb, pHb, alpha, gamma, delta)

        p_choose_B = pm.math.sigmoid(beta * (VB - VA))
        p_choose_B = pt.clip(p_choose_B, 1e-6, 1 - 1e-6)

        alpha_beta = p_choose_B * kappa
        beta_beta  = (1 - p_choose_B) * kappa

        obs = pm.Beta("obs", alpha=alpha_beta, beta=beta_beta, observed=rate_obs)

        print("Starting MCMC sampling (this may take a few minutes)...")
        trace = pm.sample(
            draws=1000,
            tune=1000,
            chains=2,
            cores=1,           # use 1 core to avoid Windows multiprocessing issues
            target_accept=0.9,
            return_inferencedata=True,
            progressbar=True,
            random_seed=42
        )

    print("\nSampling complete!")

    # ── 4. POSTERIOR SUMMARY ──────────────────
    print("\n── Posterior Summary ──────────────────────────")
    summary = az.summary(trace, var_names=["alpha", "gamma", "delta", "beta", "kappa"])
    print(summary)

    # ── 5. PLOT: Posterior Distributions ──────
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle("Posterior Distributions - Bayesian Prospect Theory", fontsize=14, fontweight='bold')

    params = ["alpha", "gamma", "delta", "beta", "kappa"]
    colors = ["#2196F3", "#E91E63", "#FF9800", "#4CAF50", "#9C27B0"]

    for i, (param, color) in enumerate(zip(params, colors)):
        ax = axes[i // 3][i % 3]
        az.plot_posterior(trace, var_names=[param], ax=ax, color=color)
        ax.set_title(f"Posterior: {param}", fontweight='bold')

    axes[1][2].axis('off')
    plt.tight_layout()
    plt.savefig("posterior_distributions.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: posterior_distributions.png")

    # ── 6. PLOT: Probability Weighting Function ─
    gamma_mean = float(trace.posterior["gamma"].mean())
    delta_mean = float(trace.posterior["delta"].mean())

    p_range = np.linspace(0.01, 0.99, 200)
    w_mean  = np.exp(-delta_mean * (-np.log(p_range)) ** gamma_mean)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(p_range, p_range, 'k--', alpha=0.4, label="Linear (rational)")
    ax.plot(p_range, w_mean, color="#E91E63", lw=2.5,
            label=f"Fitted w(p)\n(gamma={gamma_mean:.2f}, delta={delta_mean:.2f})")

    gamma_samples = trace.posterior["gamma"].values.flatten()[:200]
    delta_samples = trace.posterior["delta"].values.flatten()[:200]
    for g, d in zip(gamma_samples, delta_samples):
        w = np.exp(-d * (-np.log(p_range)) ** g)
        ax.plot(p_range, w, color="#E91E63", alpha=0.03, lw=1)

    ax.set_xlabel("Objective Probability p", fontsize=12)
    ax.set_ylabel("Weighted Probability w(p)", fontsize=12)
    ax.set_title("Probability Weighting Function (Prelec 1998)", fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("probability_weighting.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: probability_weighting.png")

    # ── 7. PLOT: Posterior Predictive Check ────
    with pt_model:
        ppc = pm.sample_posterior_predictive(trace, random_seed=42)

    predicted_rates = ppc.posterior_predictive["obs"].mean(("chain", "draw")).values

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(rate_obs, predicted_rates, alpha=0.4, s=20, color="#2196F3", edgecolors='none')
    ax.plot([0, 1], [0, 1], 'r--', lw=1.5, label="Perfect prediction")
    ax.set_xlabel("Observed Choice Rate (B)", fontsize=12)
    ax.set_ylabel("Predicted Choice Rate (B)", fontsize=12)
    ax.set_title("Posterior Predictive Check", fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    corr = np.corrcoef(rate_obs, predicted_rates)[0, 1]
    ax.text(0.05, 0.92, f"r = {corr:.3f}", transform=ax.transAxes, fontsize=11,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig("posterior_predictive_check.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: posterior_predictive_check.png")

    print("\nAll done! Check the 3 saved plots.")
    print("\nKey findings:")
    print(f"  alpha (value curvature)      : {float(trace.posterior['alpha'].mean()):.3f}")
    print(f"  gamma (prob weighting curve) : {float(trace.posterior['gamma'].mean()):.3f}")
    print(f"  delta (prob weighting elev.) : {float(trace.posterior['delta'].mean()):.3f}")
    print(f"  beta  (decision noise)       : {float(trace.posterior['beta'].mean()):.3f}")