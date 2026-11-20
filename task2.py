"""
Survival Models coursework: Question 2
Data source: UCI Heart Failure Clinical Records, DOI 10.24432/C5Z89R
(CC BY 4.0).
"""

from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter, WeibullAFTFitter
from scipy.stats import norm, chi2
from scipy.optimize import minimize


PREDICTORS = ["age_c", "ef_c", "ln_cr_c"]
REFERENCE_AGE = 60.0
REFERENCE_EF = 38.0
REFERENCE_CR = 1.1
FORMULA = "age_c + ef_c + ln_cr_c"
GROUPS = ["EF <= 30%", "30% < EF <= 40%", "EF > 40%"]
COLOURS = ["tab:orange", "tab:green", "tab:blue"]
# EF values for the three fitted profiles.
PROFILE_EF = [25.0, 38.0, 50.0]


def transform_covariates(data):
    """Centre age and EF, and store ln(creatinine/1.1)."""
    return pd.DataFrame({
        "age_c": data["age"] - REFERENCE_AGE,
        "ef_c": data["ejection_fraction"] - REFERENCE_EF,
        "ln_cr_c": np.log(data["serum_creatinine"] / REFERENCE_CR),
    }, index=data.index)


def save_figure(fig, output, name):
    fig.tight_layout()
    fig.savefig(output / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def wald_effect(log_ph, standard_error):
    """Compute the hazard ratio, exponentiated normal CI and Wald p-value."""
    z = log_ph / standard_error
    margin = norm.ppf(0.975) * standard_error
    return {"log_PH": log_ph, "SE_log_PH": standard_error,
            "PH": np.exp(log_ph), "lower95": np.exp(log_ph - margin),
            "upper95": np.exp(log_ph + margin), "Wald_chi2": z * z,
            "p": chi2.sf(z * z, df=1)}


def breslow_baseline(analysis, beta):
    """Compute the Breslow baseline cumulative hazard at z=0.

    Include patients censored at each event time in the risk set.
    """
    scores = np.exp(analysis[PREDICTORS].to_numpy() @ beta)
    times = analysis["time"].to_numpy()
    events = analysis["DEATH_EVENT"].to_numpy()
    rows = []
    for t in np.sort(np.unique(times)):
        deaths = int(events[times == t].sum())
        denominator = scores[times >= t].sum()
        rows.append((t, deaths, denominator, deaths / denominator))
    result = pd.DataFrame(rows, columns=["Time", "Deaths", "Risk_score_sum", "dH0"])
    result = result.set_index("Time")
    result["H0"] = result["dH0"].cumsum()
    result["S0"] = np.exp(-result["H0"])
    return result


def weibull_ph_inference(analysis, scale, shape, beta_aft):
    """Estimate Weibull PH coefficients and their covariance.

    Use BFGS with the AFT estimates as starting values. Invert the full
    observed information, then extract the PH coefficient block so
    the standard errors include uncertainty in the shape parameter.
    """
    X = np.column_stack([np.ones(len(analysis)), analysis[PREDICTORS]])
    log_t = np.log(analysis["time"].to_numpy())
    event = analysis["DEATH_EVENT"].to_numpy()
    start = np.r_[-shape * np.log(scale), shape * beta_aft, np.log(shape)]

    def objective(p):
        eta = np.exp(p[-1])
        log_H = X @ p[:-1] + eta * log_t
        H = np.exp(log_H)
        residual = event - H
        # log f = log h - H; censored observations contribute -H.
        nll = np.sum(H - event * (p[-1] + log_H - log_t))
        gradient = -np.r_[X.T @ residual,
                          np.sum(event + eta * log_t * residual)]
        return nll, gradient

    fit = minimize(objective, start, jac=True, method="BFGS",
                   options={"gtol": 1e-8, "maxiter": 2000})
    if np.max(np.abs(objective(fit.x)[1])) > 1e-5:
        raise RuntimeError(f"Weibull PH optimisation did not converge: {fit.message}")
    p = fit.x
    eta = np.exp(p[-1])
    H = np.exp(X @ p[:-1] + eta * log_t)
    u = eta * log_t
    information = np.zeros((len(p), len(p)))
    information[:-1, :-1] = X.T @ (H[:, None] * X)
    information[:-1, -1] = X.T @ (H * u)
    information[-1, :-1] = information[:-1, -1]
    information[-1, -1] = np.sum(H * u**2 - (event - H) * u)
    covariance = np.linalg.inv(information)
    return p[1:-1], covariance[1:-1, 1:-1], -fit.fun


def main():
    folder = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, help="Path to the UCI CSV")
    args = parser.parse_args()

    if args.data is not None:
        data_path = args.data.expanduser().resolve()
    else:
        data_path = folder / "heart_failure_clinical_records_dataset.csv"
        if not data_path.is_file():
            raise FileNotFoundError(
                "Put the heart failure CSV beside task2.py, or use --data PATH."
            )

    # ========================================================
    # Part (a): Import data and compute summary statistics
    # ========================================================
    raw = pd.read_csv(data_path)
    required = ["time", "DEATH_EVENT", "age", "ejection_fraction",
                "serum_creatinine"]
    # Select follow-up time, death indicator and the three covariates.
    # DEATH_EVENT is 1 for death and 0 for right-censoring.
    data = raw[required].copy()
    if not np.isfinite(data.to_numpy()).all():
        raise ValueError("Selected columns contain missing or non-finite values.")
    if not data["DEATH_EVENT"].isin([0, 1]).all():
        raise ValueError("DEATH_EVENT must be 1 (death) or 0 (censored).")
    if not (data[["time", "serum_creatinine"]] > 0).all().all():
        raise ValueError("Time and serum creatinine must be positive for logarithms.")
    data["DEATH_EVENT"] = data["DEATH_EVENT"].astype(int)

    output = folder / "task2_output"
    output.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 11, "axes.labelsize": 12})

    # Store the patient count, deaths, censored count and death proportion.
    counts = pd.DataFrame({
        "Quantity": ["Patients", "Observed deaths", "Right-censored", "Death proportion"],
        "Value": [len(data), data.DEATH_EVENT.sum(),
                  (data.DEATH_EVENT == 0).sum(), data.DEATH_EVENT.mean()],
    })
    counts.to_csv(output / "task2a_sample_counts.csv", index=False)

    labels = {
        "age": "Age (years)",
        "ejection_fraction": "Ejection fraction (%)",
        "serum_creatinine": "Serum creatinine (mg/dL)",
        "time": "Observed follow-up (days)",
    }
    # Compute descriptive statistics for each selected variable.
    rows = []
    for variable, label in labels.items():
        x = data[variable]
        rows.append({"Variable": label, "N": x.size,
                     "Mean": x.mean(), "SD": x.std(), "Median": x.median(),
                     "Q1": x.quantile(0.25), "Q3": x.quantile(0.75),
                     "Minimum": x.min(), "Maximum": x.max()})
    descriptive = pd.DataFrame(rows)
    descriptive.to_csv(output / "task2a_descriptive.csv", index=False)

    print("\nQUESTION 2(a): SAMPLE")
    print(f"Missing values in selected columns: {data.isna().sum().sum()}")
    print(counts.to_string(index=False))
    print(descriptive.round(3).to_string(index=False))

    # ========================================================
    # Part (b): Fit the Cox and Weibull models
    # ========================================================
    # Store the transformed covariates with follow-up time and event status.
    analysis = data[["time", "DEATH_EVENT"]].join(transform_covariates(data))
    analysis.to_csv(output / "task2_analysis_data.csv", index=False)

    # Fit the Cox model; lifelines uses Efron's method for tied deaths.
    cox = CoxPHFitter(alpha=0.05)
    cox.fit(analysis, duration_col="time", event_col="DEATH_EVENT",
            formula=FORMULA)

    # Fit the Weibull AFT model with a common shape parameter.
    aft = WeibullAFTFitter(alpha=0.05, model_ancillary=False)
    aft.fit(analysis, duration_col="time", event_col="DEATH_EVENT",
            formula=FORMULA)

    cox.summary.to_csv(output / "task2b_cox_full_summary.csv")
    aft.summary.to_csv(output / "task2b_weibull_full_summary.csv")

    # Extract shape and scale, and reverse the software's AFT coefficient
    # signs to use the lecture notes' convention.
    shape = float(np.exp(aft.params_.loc[("rho_", "Intercept")]))
    scale = float(np.exp(aft.params_.loc[("lambda_", "Intercept")]))
    beta_aft = -aft.params_.loc["lambda_", PREDICTORS].to_numpy()
    beta_cox = cox.params_.loc[PREDICTORS].to_numpy()
    # Compute Weibull PH coefficients and their covariance for inference.
    theta, covariance_ph, _ = weibull_ph_inference(
        analysis, scale, shape, beta_aft)
    pd.DataFrame(covariance_ph, index=PREDICTORS, columns=PREDICTORS).to_csv(
        output / "task2b_weibull_PH_covariance.csv")
    aft.variance_matrix_.to_csv(output / "task2b_weibull_software_covariance.csv")

    contrasts = ["Age: +1 year", "Ejection fraction: +1 percentage point",
                 "Serum creatinine: +10%"]
    # Use changes of one year, one EF percentage point and 10% creatinine.
    # The last change is ln(1.1) on the log-creatinine scale.
    changes = np.array([1.0, 1.0, np.log(1.1)])
    effects, cox_rows, weibull_rows = [], [], []
    for j, (variable, contrast, change) in enumerate(zip(PREDICTORS, contrasts, changes)):
        c = wald_effect(change * beta_cox[j],
                        change * np.sqrt(cox.variance_matrix_.loc[variable, variable]))
        w = wald_effect(change * theta[j], change * np.sqrt(covariance_ph[j, j]))
        cox_rows.append({"Contrast": contrast, **c})
        weibull_rows.append({"Contrast": contrast, **w})
        effects.append({"Covariate": variable, "Contrast": contrast,
                        **{"Cox_" + key: value for key, value in c.items()},
                        **{"Weibull_" + key: value for key, value in w.items()},
                        "Weibull_TR": np.exp(-change * beta_aft[j])})
    effects = pd.DataFrame(effects)
    effects.to_csv(output / "task2b_model_comparison.csv", index=False)
    cox_table = pd.DataFrame(cox_rows)
    weibull_table = pd.DataFrame(weibull_rows)
    cox_table.to_csv(output / "task2b_cox_PH.csv", index=False)
    weibull_table.to_csv(output / "task2b_weibull_PH.csv", index=False)
    coefficients = pd.DataFrame({"Covariate": PREDICTORS, "Cox_beta": beta_cox,
                                 "Weibull_AFT_beta_notes": beta_aft,
                                 "Weibull_PH_eta_beta": theta})
    coefficients.to_csv(output / "task2b_coefficients.csv", index=False)
    pd.DataFrame({"Parameter": ["alpha (baseline scale, days)", "eta (shape)"],
                  "Estimate": [scale, shape]}).to_csv(
                      output / "task2b_weibull_parameters.csv", index=False)

    # Save formatted LaTeX rows for the two hazard-ratio tables.
    for name, table in [("cox", cox_table), ("weibull", weibull_table)]:
        lines = []
        for row in table.itertuples(index=False):
            mantissa, exponent = f"{row.p:.2e}".split("e")
            p_latex = rf"${mantissa}\times10^{{{int(exponent)}}}$"
            label = row.Contrast.replace("%", r"\%")
            lines.append(f"{label} & {row.PH:.3f} & "
                         f"$({row.lower95:.3f}, {row.upper95:.3f})$ & {p_latex} " + r"\\")
        (output / f"task2b_{name}_table_rows.tex").write_text(
            "\n".join(lines) + "\n", encoding="utf-8")

    print("\nQUESTION 2(b): COEFFICIENTS (notes' AFT sign convention)")
    print(coefficients.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print(f"Weibull alpha = {scale:.3f} days; eta = {shape:.5f}")
    print("\nCOX HAZARD RATIOS (labelled PH in the report)")
    print(cox_table.to_string(index=False, float_format=lambda x: f"{x:.6g}"))
    print("\nWEIBULL HAZARD RATIOS: exp(eta * beta dot change)")
    print(weibull_table.to_string(index=False, float_format=lambda x: f"{x:.6g}"))
    print("Both tables use two-sided Wald tests and exponentiated normal intervals.")

    # ========================================================
    # Part (c): Compute and plot the survival curves
    # ========================================================
    # Divide patients into three EF groups and count their event outcomes.
    data["EF_group"] = pd.cut(data["ejection_fraction"],
                              bins=[-np.inf, 30, 40, np.inf], labels=GROUPS)
    group_summary = data.groupby("EF_group", observed=True).agg(
        Patients=("time", "size"), Deaths=("DEATH_EVENT", "sum"),
        Median_EF=("ejection_fraction", "median"), Max_followup=("time", "max"),
    ).reindex(GROUPS)
    group_summary["Censored"] = group_summary.Patients - group_summary.Deaths
    group_summary.to_csv(output / "task2c_group_summary.csv")

    # Set EF to 25%, 38% and 50%, keeping age and creatinine fixed.
    profiles = pd.DataFrame({
        "age": REFERENCE_AGE,
        "ejection_fraction": PROFILE_EF,
        "serum_creatinine": REFERENCE_CR,
    }, index=pd.Index(GROUPS, name="EF_group"))
    profiles.to_csv(output / "task2c_profiles.csv")
    encoded_profiles = transform_covariates(profiles)

    # Predict survival at the reference profile z=0.
    reference = pd.DataFrame({"age": [REFERENCE_AGE],
                              "ejection_fraction": [REFERENCE_EF],
                              "serum_creatinine": [REFERENCE_CR]})
    reference_z = transform_covariates(reference)
    reference_curves = pd.DataFrame({
        "Cox": cox.predict_survival_function(reference_z).iloc[:, 0],
        "Weibull": aft.predict_survival_function(reference_z).iloc[:, 0],
    })
    reference_curves.loc[0.0] = 1.0
    reference_curves.sort_index().to_csv(
        output / "task2b_reference_survival.csv", index_label="Time")
    pd.DataFrame({
        "Model": ["Cox", "Weibull AFT"],
        "Log_likelihood": [cox.log_likelihood_, aft.log_likelihood_],
        "Likelihood_type": ["partial", "full"],
        "Weibull_shape": [np.nan, shape],
    }).to_csv(output / "task2b_fit_details.csv", index=False)

    # Compute the baseline at z=0 and obtain survival for each profile.
    baseline = breslow_baseline(analysis, beta_cox)
    baseline.to_csv(output / "task2c_breslow_baseline.csv", index_label="Time")
    relative_hazard = np.exp(encoded_profiles[PREDICTORS].to_numpy() @ beta_cox)
    predicted = pd.DataFrame(
        np.exp(-np.outer(baseline["H0"], relative_hazard)),
        index=baseline.index, columns=profiles.index)
    predicted.loc[0.0] = 1.0
    predicted = predicted.sort_index()
    predicted.to_csv(output / "task2c_cox_survival_predictions.csv", index_label="Time")

    # For day 90 use the last step at or before 90, not linear interpolation.
    day90 = profiles.copy()
    day90["Cox_S90"] = predicted.loc[predicted.index <= 90].iloc[-1]
    day90["Group_at_risk_day90"] = [
        int(((data.EF_group == g) & (data.time >= 90)).sum()) for g in GROUPS
    ]
    day90.to_csv(output / "task2c_day90_survival.csv")

    # Plot each group's Kaplan–Meier estimate with pointwise 95% CIs
    # (Greenwood with a log-log transform), alongside its fitted profile.
    # Stop each fitted curve at the group's maximum observed follow-up.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True, sharey=True)
    for ax, group, colour in zip(axes, GROUPS, COLOURS):
        subset = data.loc[data.EF_group == group]
        km = KaplanMeierFitter(alpha=0.05)
        km.fit(subset.time, event_observed=subset.DEATH_EVENT, label="Kaplan–Meier")
        km_table = km.survival_function_.join(km.confidence_interval_)
        km_table.to_csv(output / f"task2c_KM_group{GROUPS.index(group) + 1}.csv",
                        index_label="Time")
        km.plot_survival_function(ax=ax, color="black", linewidth=1.8,
                                  ci_show=True, ci_alpha=0.15, show_censors=False)
        curve = predicted.loc[predicted.index <= subset.time.max(), group]
        ax.step(curve.index, curve, where="post", color=colour,
                linewidth=2, label="Cox fitted profile")
        ax.set_title(f"{group}\nn = {len(subset)}, deaths = {subset.DEATH_EVENT.sum()}")
        ax.set(xlabel="Time (days)", xlim=(0, data.time.max()), ylim=(0, 1.02))
        ax.legend(loc="lower left", fontsize=9)
    axes[0].set_ylabel("Survival probability")
    save_figure(fig, output, "task2c_survival_profiles")

    print("\nQUESTION 2(c): GROUPS AND PROFILES")
    print(group_summary.to_string())
    print(day90.round(3).to_string())
    print("\nKM summarises each actual group; the fitted curve describes one profile.")
    print(f"\nAll tables and figures saved to: {output}")


if __name__ == "__main__":
    main()
