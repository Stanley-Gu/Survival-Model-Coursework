"""Survival Models coursework: Question 1, parts (a)-(d).

Put cwdat.csv beside this script, then run: python task1.py
Dependencies: numpy pandas scipy matplotlib lifelines
Install with: python -m pip install numpy pandas scipy matplotlib lifelines
Verified with Python 3.12 and lifelines 0.30.3.

The comments explain the calculations used in the report. Figures 1 and 2
are saved as task1c_survival.png and task1d_cumulative_hazard.png. Numerical
estimates and the nonparametric curves are also saved as CSV files.
All calculations use unrounded estimates; rounding is only for display.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp


# ============================================================
# Load and check the data
# ============================================================

folder = Path(__file__).resolve().parent
df = pd.read_csv(folder / "cwdat.csv")

# One finite-value check also detects missing values. Keep only the checks
# needed for the likelihood: positive times and a binary event indicator.
if not np.isfinite(df[["T", "Delta"]].to_numpy()).all():
    raise ValueError("T and Delta must contain finite values with no missing data.")
if not df["T"].gt(0).all() or not df["Delta"].isin([0, 1]).all():
    raise ValueError("T must be positive and Delta must be 0 or 1.")

# Store observed times as t and event indicators as event (1 = event,
# 0 = right-censored). Store the sample size as n, the number of observed
# events as r, and the sum of ALL observed times as total_time.
t = df["T"].to_numpy(dtype=float)
event = df["Delta"].to_numpy(dtype=int)

n = len(t)
r = int(event.sum())
total_time = t.sum()

if r == 0:
    raise ValueError("At least one observed event is needed.")

print(f"Number of observations: {n}")
print(f"Number of observed events: {r}")
print(f"Number of censored observations: {n - r}")
print(f"Total observed time: {total_time:.3f}")


# ============================================================
# Part (a): Exponential maximum likelihood estimate
# S(t) = exp(-rate * t)
# Each event contributes log(rate) - rate*t to the log-likelihood;
# a censored observation contributes -rate*t. Thus
# ell(rate) = r*log(rate) - rate*total_time and rate_hat = r/total_time.
# ============================================================

rate_hat = r / total_time

print(f"\nExponential rate estimate: {rate_hat:.5f}")


# ============================================================
# Part (b): Weibull maximum likelihood estimates
# S(t) = exp(-lambda * t**eta)
#
# For a fixed eta:
# lambda_hat(eta) = r / sum(t**eta)
# Substitute this expression into the log-likelihood, leaving a function
# of eta alone (the profile log-likelihood). Estimate eta numerically,
# then substitute eta_hat into lambda_hat(eta).
# ============================================================

# log_t stores log times for every observation; event_log_time is the
# sum of log times for observed events only.
log_t = np.log(t)
event_log_time = log_t[event == 1].sum()


def negative_profile_loglik(log_eta):
    """Negative profile log-likelihood, parameterised by log(eta)."""
    # Optimising log_eta ensures eta > 0. logsumexp computes
    # log(sum(t**eta)) stably, without directly raising large powers.
    eta = np.exp(log_eta)
    log_sum_power = logsumexp(eta * log_t)
    log_lambda = np.log(r) - log_sum_power

    loglik = (
        r * log_lambda
        + r * log_eta
        + (eta - 1) * event_log_time
        - r
    )
    return -loglik


# Brent's scalar minimisation combines parabolic interpolation with
# golden-section steps. Minimising -ell is equivalent to maximising ell;
# derivatives are not required. result.x is the estimated log(eta).
result = minimize_scalar(
    negative_profile_loglik,
    method="brent",
)

if not result.success:
    raise RuntimeError("Weibull optimisation failed.")

eta_hat = np.exp(result.x)
lambda_hat = np.exp(
    np.log(r) - logsumexp(eta_hat * log_t)
)

print(f"Weibull shape estimate (eta): {eta_hat:.5f}")
print(f"Weibull coefficient (lambda): {lambda_hat:.5f}")

pd.DataFrame({
    "Quantity": ["Observations", "Events", "Censored", "Total observed time",
                 "Exponential rate", "Weibull eta", "Weibull lambda"],
    "Value": [n, r, n - r, total_time, rate_hat, eta_hat, lambda_hat],
}).to_csv(folder / "task1_estimates.csv", index=False)


# ============================================================
# Common plotting settings
# ============================================================

EXP_COLOUR = "tab:orange"
WEIBULL_COLOUR = "tab:green"
EMPIRICAL_COLOUR = "black"

plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 13,
    "legend.fontsize": 11,
    "savefig.dpi": 300,
})

time_grid = np.linspace(0, t.max(), 1000)


# ============================================================
# Part (c): Kaplan–Meier and fitted survival functions
# At each event time t_j, let d_j be the number of events and n_j the
# number at risk immediately before t_j. The Kaplan-Meier estimate is
# S_KM(t) = product_{t_j <= t}(1 - d_j/n_j). Censored observations remain
# at risk up to their censoring times, but do not cause a survival drop.
#
# The grey shading gives pointwise 95% confidence intervals for S_KM,
# calculated by lifelines using Greenwood's formula and a log-log
# transformation. For 0 < S_KM(t) < 1, define
#   G(t) = sum_{t_j <= t} d_j / [n_j*(n_j-d_j)],
#   g(t) = log(-log(S_KM(t))),
#   se_g(t) = sqrt(G(t)) / abs(log(S_KM(t))).
# Transform g +/- 1.96*se_g back to the survival scale. Since the inverse
# exp(-exp(g)) is decreasing, the lower/upper endpoints are respectively
# exp(-exp(g + 1.96*se_g)) and exp(-exp(g - 1.96*se_g)). The package handles
# the endpoints S_KM=1 and S_KM=0. These are pointwise intervals, not a
# simultaneous confidence band or intervals for the fitted model curves.
# ============================================================

km = KaplanMeierFitter(alpha=0.05)
km.fit(
    durations=t,
    event_observed=event,
    label="Kaplan–Meier",
)
km.survival_function_.join(km.confidence_interval_).to_csv(
    folder / "task1c_kaplan_meier.csv", index_label="Time")

fig, ax = plt.subplots(figsize=(10, 6))

km.plot_survival_function(
    ax=ax,
    color=EMPIRICAL_COLOUR,
    linewidth=2,
    ci_show=True,
    ci_alpha=0.20,  # Shading opacity; alpha=0.05 above sets the 95% level.
    show_censors=False,
)

ax.plot(
    time_grid,
    np.exp(-rate_hat * time_grid),
    color=EXP_COLOUR,
    linestyle="-",
    linewidth=2,
    label="Fitted exponential",
)

ax.plot(
    time_grid,
    np.exp(-lambda_hat * time_grid**eta_hat),
    color=WEIBULL_COLOUR,
    linestyle="-",
    linewidth=2,
    label="Fitted Weibull",
)

ax.set(
    xlabel="Time",
    ylabel="Survival probability",
    xlim=(0, t.max()),
    ylim=(0, 1.02),
)

ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig(folder / "task1c_survival.png")


# ============================================================
# Part (d): Nelson–Aalen cumulative hazard estimate
# H_NA(t) = sum(d_j / n_j)
# Compare this step function with H_exp(t)=rate_hat*t and
# H_Weibull(t)=lambda_hat*t**eta_hat. On log-log axes the fitted curves
# are straight lines, with slopes 1 (exponential) and eta_hat (Weibull).
# Exclude time zero when taking logarithms.
# ============================================================

# Distinct ordered event times and corresponding event counts
event_times, d_j = np.unique(
    t[event == 1],
    return_counts=True,
)

# Number at risk immediately before each event time.
# Individuals censored at that time are included.
sorted_t = np.sort(t)
n_j = n - np.searchsorted(
    sorted_t,
    event_times,
    side="left",
)

H_na = np.cumsum(d_j / n_j)
pd.DataFrame({"Time": event_times, "Events": d_j, "At_risk": n_j,
              "Nelson_Aalen": H_na}).to_csv(
    folder / "task1d_nelson_aalen.csv", index=False)

# Include the origin and extend the final horizontal step
step_times = np.concatenate(([0], event_times, [t.max()]))
step_hazard = np.concatenate(([0], H_na, [H_na[-1]]))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))


# ------------------------------------------------------------
# Left panel: cumulative hazard against time
# ------------------------------------------------------------

ax = axes[0]

ax.step(
    step_times,
    step_hazard,
    where="post",
    color=EMPIRICAL_COLOUR,
    linewidth=2,
    label="Nelson–Aalen",
)

ax.plot(
    time_grid,
    rate_hat * time_grid,
    color=EXP_COLOUR,
    linestyle="-",
    linewidth=2,
    label="Fitted exponential",
)

ax.plot(
    time_grid,
    lambda_hat * time_grid**eta_hat,
    color=WEIBULL_COLOUR,
    linestyle="-",
    linewidth=2,
    label="Fitted Weibull",
)

ax.set(
    xlabel="Time",
    ylabel="Cumulative hazard",
)

ax.legend(loc="upper left")


# ------------------------------------------------------------
# Right panel: log cumulative hazard against log time
# Only positive event times and cumulative hazards are used.
# ------------------------------------------------------------

ax = axes[1]

log_time_grid = np.linspace(
    np.log(event_times.min()),
    np.log(event_times.max()),
    1000,
)

ax.step(
    np.log(event_times),
    np.log(H_na),
    where="post",
    color=EMPIRICAL_COLOUR,
    linewidth=2,
    label="Nelson–Aalen",
)

ax.plot(
    log_time_grid,
    np.log(rate_hat) + log_time_grid,
    color=EXP_COLOUR,
    linestyle="-",
    linewidth=2,
    label="Fitted exponential",
)

ax.plot(
    log_time_grid,
    np.log(lambda_hat) + eta_hat * log_time_grid,
    color=WEIBULL_COLOUR,
    linestyle="-",
    linewidth=2,
    label="Fitted Weibull",
)

ax.set(
    xlabel="log(time)",
    ylabel="log(cumulative hazard)",
)

ax.legend(loc="upper left")

fig.tight_layout()
fig.savefig(folder / "task1d_cumulative_hazard.png")

plt.show()

