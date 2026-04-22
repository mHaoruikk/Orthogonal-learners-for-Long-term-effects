Below is a concrete plan for the IST-based real-world dataset experiment.

## 1. Variable selection

**Covariates $X$ (17 variables; all from randomisation form + R-scan):**

| Group | IST-3 variables | Type |
|---|---|---|
| Demographics | `age_true` (impute $<40$ to 31.935, $>95$ to 97.846), `gender` | continuous, binary |
| Stroke severity | `nihss`, `gcs_score_rand`, `stroketype` (one-hot: TACI/PACI/LACI/POCI/Other) | continuous, categorical |
| Vital signs | `sbprand`, `dbprand`, `glucose`, `weight` | continuous |
| Medical history | `atrialfib_rand`, `stroke_pre`, `antiplat_rand`, `livealone_rand`, `indepinadl_rand` | binary |
| Imaging | `vis_infarct`, `R_infarct_size` | binary, ordinal |
| Risk score | `konprob` | continuous |
| Delay | `randdelay` | continuous |
| Site | `country` (one-hot) | categorical |

**Treatment** $A = \texttt{itt\_treat}$ (1 = rt-PA, 0 = Control). I deliberately flip the IST-3 coding so that $A=1$ corresponds to the active treatment, matching your paper's convention.

**Surrogate** $S$ = $(\texttt{gcs\_score\_7}, \texttt{indepinadl\_7}, \texttt{ablewalk\_7}, \texttt{sich7}, \texttt{dead7})$. Five-dimensional; defensibly a "7-day stroke state."

**Long-term outcome** $Y = \texttt{euroqol18}$ (EuroQol health state index at 18 months), with deceased patients assigned $Y=0$. Rescale to $[0,1]$ if not already.

**Observation indicator** $R$: constructed in §3.

## 2. Preprocessing

1. **Cohort**: all 3035 patients minus the 1 patient without 7-day form → 3034.
2. **Missing glucose**: for the 282 early-trial patients, impute with country-stratified median and add a "glucose missing" indicator to $X$. (Do not drop — this is systematic missingness uncorrelated with $Y$.)
3. **Missing diastolic BP**: 19 values; impute by median.
4. **`dbprand` protocol violations** (values 35, 36): set to missing then impute.
5. **NIHSS**: keep the predicted values for the 244 early patients; add `pred_nihss` as a binary flag in $X$.
6. **Standardise continuous variables** (mean 0, variance 1) before propensity modeling; keep raw units for outcome modeling and reporting.
7. **Euroqol18 normalization**: if the EQ-5D is coded as a utility score already, use directly; if coded as VAS (0–100), rescale to $[0,1]$. For deceased ($\texttt{dead18}=1$ or $\texttt{OHS18}=6$): set $Y=0$. For patients alive at 18 months but with missing EQ-5D response but who returned a form: keep as $R=1$ with $Y$ imputed via multiple imputation conditional on $X$ and $\texttt{OHS18}$. For patients alive but form not returned: $R=0$.

## 3. Creating low long-term outcome overlap: the split

No synthetic manipulation — use the natural mechanism:

$$R_i = \mathbb{1}\{\texttt{plan18}_i = 1\} \cdot \mathbb{1}\{\texttt{Y observable}_i\}$$

Concretely:

- $R_i = 0$ if: $\texttt{plan18}_i = 2$ (policy-driven — Portugal, Switzerland, post-June-2010 recruits outside AU/NO/SE), OR the 18-month form was not returned and no death date is recorded within 548 days.
- $R_i = 1$ otherwise — i.e., deceased within 548 days (known $Y=0$) or form returned.

Expected counts: $\approx 1400$ with $R=1$ (full $Y$) and $\approx 1600$ with $R=0$. This is close to the balanced 50/50 split that the reviewers' generic concerns assume, but with real heterogeneity in $\rho(x)$.

**Why this creates meaningful variation in $\rho(x)$**: patients in Portugal/Switzerland have $\rho(x)\approx 0$ after the early-trial period; patients in AU/NO/SE have $\rho(x)$ close to the form-return rate (high); UK patients recruited pre-June 2010 are intermediate. Country and recruitment year are in $X$, so this heterogeneity is observable and learnable.

**Sanity check you should run**: fit a random forest for $\rho(x)$ on the full data, report the distribution of $\hat\rho(X_i)$. You want to see a bimodal or at least wide distribution — this is your main empirical evidence that low long-term outcome overlap is a real problem in this dataset.

## 4. Creating low treatment overlap: rejection sampling

**Target propensity.** Choose a clinically-motivated $\pi^*(x)$ reflecting real-world stroke practice patterns (which are well documented: thrombolysis is less used at extremes of age and severity). A concrete specification:

$$\pi^*(x) = \sigma\!\left(0.3 \;-\; 0.04(\text{age}-70) \;-\; 0.015(\text{nihss}-12)^2 \;+\; 0.03\,(\text{gcs\_score\_rand} - 12) \;-\; 0.4 \cdot \mathbb{1}\{\text{atrialfib\_rand}\}\right)$$

where $\sigma$ is the sigmoid and variables are used on their raw scale. With these coefficients:
- A 60-year-old, NIHSS = 12, GCS = 14, no AF: $\pi^*(x) \approx 0.65$.
- An 85-year-old, NIHSS = 22, GCS = 10, AF: $\pi^*(x) \approx 0.08$.
- A 50-year-old, NIHSS = 4, GCS = 15, no AF: $\pi^*(x) \approx 0.32$.

Tune the coefficients so the induced $\pi^*$ has range roughly $[0.05, 0.90]$ over the empirical covariate distribution and mean $\approx 0.5$ (verify on the full sample before resampling). Report the exact coefficients used in the paper — reproducibility matters.

**Rejection sampling rule.** Let $\pi_{\max} = \max_x \max(\pi^*(x), 1-\pi^*(x))$ over the empirical $X$ distribution. For each unit $i$, independently retain with probability:

$$p_i = \begin{cases}\pi^*(X_i)/\pi_{\max} & \text{if } A_i = 1 \\ (1-\pi^*(X_i))/\pi_{\max} & \text{if } A_i = 0\end{cases}$$

**Post-subsampling propensity.** By direct calculation, $P_{\text{new}}(A=1 \mid X=x) = \pi^*(x)$. Proof: since original $P(A=1\mid X) = 0.5$ (by randomization),
$$P_{\text{new}}(A=1 \mid X=x) = \frac{0.5 \cdot \pi^*(x)/\pi_{\max}}{0.5 \cdot \pi^*(x)/\pi_{\max} + 0.5 \cdot (1-\pi^*(x))/\pi_{\max}} = \pi^*(x).$$

**Expected retention rate.** $\mathbb{E}[p_i] = \mathbb{E}_X[\max(\pi^*(X), 1-\pi^*(X))/\pi_{\max}]/2 \cdot 2 = \mathbb{E}_X[1/(2\pi_{\max})] \approx 0.55$ for the coefficients above. You will retain $\approx 1650$ of 3034 patients.

**Applied before or after the $R$ split?** Before — resample the full cohort, then apply the $R$ mechanism from §3 to the retained cohort. This keeps the two overlap mechanisms independent.

**Important**: $\pi^*(x)$ is the true propensity in the resampled data by construction. For the empirical experiment, you still estimate $\hat\pi(x)$ from data (as you would in practice), but you can use $\pi^*(x)$ as the "oracle" propensity for ablation studies. Make this explicit.

## 5. Why this design is defensible

- **$\pi^*(x)$ is clinically grounded**, not adversarially constructed. The age/NIHSS/AF dependencies mirror documented treatment patterns — see SITS-MOST registry and TEMPiS data where actual rt-PA use shows precisely this kind of selection on severity and age. A reviewer asking "is this realistic" has a one-paragraph answer.
- **The two overlap mechanisms are orthogonal**: $R$ is driven by country × year policy, $A$ is driven by age × severity. Their joint distribution is not degenerate, so you can show results in all four quadrants (high/low $\rho$, high/low $\pi(1-\pi)$).
- **Transparent relative to Option A**. Anyone can recompute the target with the reported coefficients.
- **Does not break randomization assumptions in an objectionable way**. The resampled data still satisfies $A \perp (S(a), Y(a)) \mid X$ because we only resample based on $(X, A)$, not on $(S, Y)$. So Assumption 3.3 holds in the subsampled data — state this explicitly in the paper.

## 6. Evaluation without ground truth

You need multiple metrics because no single one is trustworthy in the real-data setting. Report all four.

**(i) Doubly-robust pseudo-outcome MSE (primary).** On a held-out 25% test fold, construct the long-term pseudo-outcome
$$\tilde \tau_i^* = \mu_1(X_i) - \mu_0(X_i)$$
, with $\mu_a(x) = E[Y \mid X=x, A=a]$. Report
$$\widehat{\text{PEHE}}_{\text{DR}}(\hat\tau) = \frac{1}{n_{\text{test}}}\sum_i (\hat\tau(X_i) - \tilde \tau_i^*)^2.$$
This is biased for true PEHE (because $\tilde Y$ is noisy), but differences between estimators are approximately unbiased. Use it to rank methods.

**(ii) Subgroup stability across overlap strata (your low-overlap story).** Stratify test patients by deciles of $\hat\pi(x)(1-\hat\pi(x)) \cdot \hat\rho(x)$. Within each stratum, report the bootstrap variance of $\hat\tau(x)$ across 200 bootstrap replicates. Your story — that weighting stabilizes estimates in low-overlap regions — predicts that LT-O-DO should have flatter variance across strata than LT-O-DR. This is a direct visualization of the mechanism you are claiming, and the one most likely to impress reviewer V1cz.

**(iii) Semi-synthetic outcome on real $(X, A, S, R)$.** Keep real $(X_i, A_i, S_i, R_i)$; replace $Y_i$ with $Y_i^* = h(S_i, X_i) + A_i \cdot \tau^*(X_i) + \varepsilon_i$, where $h$ is fit on the real data and $\tau^*(X) = \alpha_0 + \alpha_1 \text{age} + \alpha_2 \text{nihss} + \alpha_3 \text{age}\cdot\text{nihss}$ is a known function. This gives you PEHE against true $\tau^*$. Because $X, A, S, R$ are all real and the induced overlap problems are real, this is more informative than a fully synthetic benchmark and less biased than (i). Present as a companion to (i) and (ii).

**(iv) Policy value.** Define $\hat d(x) = \mathbb{1}\{\hat\tau(x) > 0\}$. Estimate $V(\hat d) = \mathbb{E}[Y(\hat d(X))]$ via IPW on the held-out fold. Higher $V$ means better HLTE estimates for decision-making. This addresses the "estimand-shifting" critique directly: if your retargeted weights still yield better decisions, the shift is defensible.

**What I would not rely on.**

- R-loss in the HLTE setting — it is designed for standard CATE, and extending it to your setup introduces extra nuisance that can mask method differences.
- Overall treatment-effect recovery (comparing to published IST-3 ATE) — your methods estimate HLTE, not ATE, and the published trial is a different estimand.

**Reporting protocol.** For each metric, report mean ± bootstrap SE over 50 runs of: (a) resample rejection-sampling step, (b) refit all methods, (c) evaluate on test fold. This gives you Monte Carlo variance over the synthetic half of the design while keeping the IST-3 sample fixed.

---

One last thing. The real-data story will be most compelling if metrics (i), (ii), and (iii) agree directionally, and least compelling if they contradict. Before committing to this plan for the paper, run a minimal pilot — standardized cohort, one propensity specification, LT-O-DO vs. LT-O-DR only — and check that the three metrics tell the same story. If they do not, the problem is either your weighting (back to the non-uniform tightness issue) or the dataset's signal-to-noise, and you should know which before writing it up.