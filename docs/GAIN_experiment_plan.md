# GAIN Empirical Validation Plan — LT-O-Learners for HLTE

**Purpose.** This document specifies the real-data empirical protocol for the LT-O-learner paper on the GAIN (Greater Avenues for Independence) dataset. The protocol is designed to (i) align with the existing Athey et al. (2025) and Kallus & Mao (2024) GAIN experiments so that results are interpretable against known ATE benchmarks, and (ii) answer the two specific reviewer critiques the current draft is vulnerable to: *retargeting trivializes the low-overlap problem* (Reviewer iqo7) and *surrogacy violations dominate any overlap-weighting gains* (Reviewers PBfF, tr9r).

A methodological caveat is stated upfront and must be reiterated wherever results are reported: no ground-truth heterogeneous long-term treatment effect exists on GAIN. All PEHE-type numbers are **pseudo-PEHE** against a high-capacity oracle CATE fit on full Riverside; relative comparisons across methods are meaningful, absolute magnitudes are not.

---

## 1. Setting

The paper's canonical HLTE setting fuses a short-term randomized dataset and a long-term observational dataset:

- **D1** (short-term, randomized): contains `(X, A, S)`. No long-term outcome `Y`.
- **D2** (long-term, observational): contains `(X, S, Y)`. No treatment `A`.

D1 and D2 share the covariate space and surrogate space but differ in which variables are observed. The GAIN data naturally instantiate this fusion structure if we play the Riverside site as D1 and the other three California sites (Alameda, Los Angeles, San Diego) as D2 — the same split used by Athey et al. (2025).

The target estimand is the conditional average long-term treatment effect
$$\tau(x) \;=\; \mathbb{E}[Y(1) - Y(0) \mid X = x],$$
identified under unconfoundedness in D1, surrogacy (`Y ⊥ A ∣ X, S`), comparability between D1 and D2, and the usual overlap conditions.

---

## 2. Notation

### 2.1 Random variables

| Symbol | Meaning |
|---|---|
| `X ∈ ℝ^d` | Pre-treatment covariates |
| `A ∈ {0, 1}` | Binary treatment (GAIN program assignment) |
| `S ∈ ℝ^{d_S}` | Surrogate vector (short-term post-treatment outcomes) |
| `R ∈ {0, 1}` | Dataset indicator; `R = 1` if unit is in D2 (long-term observed), `R = 0` if unit is in D1 (long-term missing) |
| `Y ∈ ℝ` | Long-term outcome. Observed only when `R = 1`. |

### 2.2 Nuisance functions

| Symbol | Definition |
|---|---|
| `π(x)` | `P(A = 1 ∣ X = x, R = 0)` — treatment propensity in D1 |
| `ρ(x)` | `P(R = 1 ∣ X = x)` — marginal long-term-data propensity |
| `ρ(x, a, s)` | `P(R = 1 ∣ X = x, A = a, S = s)` — conditional long-term-data propensity |
| `μ(a, x)` | `E[Y(a) ∣ X = x]` — conditional mean outcome |
| `μ̃(a, x, s)` | `E[Y ∣ A = a, X = x, S = s, R = 1]` — surrogate-index outcome model |

### 2.3 Overlap weights

| Symbol | Definition | Associated learner |
|---|---|---|
| `ω₁(x) = 1` | No retargeting | LT-O-DR (baseline) |
| `ω_TO(x) = π(x)(1 − π(x))` | Treatment overlap only | LT-O-TO |
| `ω_LO(x) = ρ(x)` | Long-term overlap only | LT-O-LO |
| `ω_DO(x) = π(x)(1 − π(x)) ρ(x)` | Doubly overlap | LT-O-DO |

### 2.4 Estimand variants

Let `f_X` denote the covariate density. The retargeted estimand under weight `ω` is
$$\tau_\omega^\star \;=\; \arg\min_{\tau} \; \mathbb{E}\!\left[\omega(X)\,(\tau(X) - \tau^\star(X))^2\right].$$
At the population level, `τ_ω^⋆(x) = τ(x)` for almost every `x` such that `ω(x) > 0`. The weight changes the *loss used to select among candidate `τ`-functions in finite samples*, not the pointwise target.

### 2.5 Learners

The five baselines and three LT-O variants evaluated are:

| Learner | Description |
|---|---|
| LT-T | Plug-in T-learner: regress `Y` on `X` separately under `A = 0, 1` in D2 via surrogate index |
| LT-RA | Regression-adjustment learner |
| LT-IPW | Inverse-propensity-weighting learner |
| LT-O-DR | Orthogonal DR-learner (no overlap weight; treated as baseline) |
| LT-O-TO | Orthogonal + treatment-overlap weight |
| LT-O-LO | Orthogonal + long-term-overlap weight |
| LT-O-DO | Orthogonal + doubly-overlap weight |

All learners use the same neural architecture and hyperparameters as in the synthetic and IST-3 experiments (Appendix F of the paper) for a clean head-to-head comparison.

### 2.6 GAIN variable mapping

| Notation | GAIN variable(s) | Description |
|---|---|---|
| `A` | `e` | Treatment indicator (GAIN assignment) |
| `X` | `age`, `agesq`, `xsexf`, `xhsdip`, `x1chld`, `xchld05`, `single`, `dumkids`, `white`, `hisp`, `black`, `grde911`, `grade12`, `grd1315`, `grade16`, `grd1720`, `tcprn1`–`tcprn10`, `paid1`–`paid4`, `adcpc1`–`adcpc4`, `padcpc1`–`padcpc4`, `grew1`, `gepop1` | Demographics, pre-RA earnings/AFDC history, macro conditions |
| `S` | `tcedd1`–`tcedd6`, `aid1`–`aid6`, and `1{tcedd_k > 0}` for `k = 1..6` | First 6 post-RA quarters of earnings, AFDC receipt, derived employment indicators |
| `Y` | `mean(tcedd13, …, tcedd36)` | Average quarterly earnings over quarters 13–36 (years 4–9 post-RA) |
| `R` | Constructed | `1` if unit ∈ D2, `0` if unit ∈ D1 |

Rationale for `Y`: temporally disjoint from `S`, so surrogacy is non-trivial to satisfy. Rationale for `S = 6 quarters`: Athey et al. (2025) establishes six quarters as a sufficient horizon to recover the long-run ATE in GAIN; keeping this fixed as the default allows a direct ATE benchmark comparison.

---

## 3. Dataset construction pipeline

### 3.1 Raw source

The Hotz et al. (2006) GAIN release contains four California sites with 36 post-RA quarters of outcomes. After standard filtering (non-missing treatment, non-missing covariates in the `X` list), approximate sample sizes are:

| Site | `N` | `N(A=1)` | `N(A=0)` |
|---|---|---|---|
| Riverside | 5,445 | 4,405 | 1,040 |
| Alameda + Los Angeles + San Diego | ≈ 13,725 | — (discarded) | — (discarded) |

### 3.2 D1 / D2 split (Athey-style)

- **D1 (`R = 0`)**: Riverside. Retain `(X, A, S)`; discard `Y`.
- **D2 (`R = 1`)**: Alameda ∪ Los Angeles ∪ San Diego. Retain `(X, S, Y)`; discard `A` and the site label.

This split induces `ρ(x)` heterogeneity naturally: Riverside and the other three sites differ substantially in pre-treatment covariate distribution (Athey et al. 2025, online Appendix Table 8 documents this). No synthetic labeling mechanism is needed — this is the empirical answer to Reviewer tr9r's concern that `ρ(x)` lacks real-world motivation.

### 3.3 Induction of low treatment overlap

Riverside is randomized, so without intervention `π(x) ≈ 0.81` everywhere — treatment overlap is excessive. To study low-`π` behavior, apply covariate-dependent rejection sampling within D1:

- Keep a treated unit with probability `m(x)`.
- Keep a control unit with probability `1 − m(x)`.

Where
$$m(x) \;=\; \operatorname{trim}_{0.01}\!\big( \sigma(\gamma_\pi \cdot (\beta_{\text{earn}}\, x_{\text{earn}} + \beta_{\text{age}}\, x_{\text{age}})) \big),$$
with `x_earn = standardize(mean(tcprn1, …, tcprn10))`, `x_age = standardize(age)`, and `(β_earn, β_age) = (0.7, 0.3)`. The parameter `γ_π ∈ {0, 1, 2, 4}` sweeps from full overlap to severe low-overlap. The resulting post-filter `π(x)` is given by
$$\pi(x) \;=\; \frac{m(x)\,\pi_0}{m(x)\,\pi_0 + (1 - m(x))(1 - \pi_0)},\qquad \pi_0 \approx 0.81.$$
This mirrors the IST-3 construction in Appendix D.2 of the paper (eq. 164) — same recipe, different covariates.

### 3.4 Train/test partition

Split **Riverside only** at the unit level, stratified on `A`, with an 80/20 train/test ratio:

- Training pool: 80% of Riverside (after the rejection sampling in §3.3) → forms D1; combined with all of D2.
- Test pool: 20% of Riverside, held out entirely. Used only for pseudo-PEHE computation (§4).

Do not touch D2 with the train/test split — D2 is used in full during training, matching the canonical HLTE setting.

### 3.5 Monte Carlo repetitions

Repeat the full pipeline (rejection sampling + 80/20 split + training + evaluation) under `B = 50` independent random seeds. Report mean ± standard deviation across seeds, matching the existing Tables 2–3 of the paper.

---

## 4. Ground-truth pseudo-CATE construction

Since Riverside is randomized, `A ⊥ (Y(0), Y(1)) ∣ X`, and the CATE is identified from `(X, A, Y)` directly without surrogates. Construct the pseudo-oracle `τ̂⋆(x)` as follows:

1. Using the **full, unfiltered Riverside sample** (all 5,445 units), fit a DR-learner with 5-fold cross-fitting.
2. Nuisances: gradient-boosted trees (XGBoost, 500 rounds, depth 4) for `μ(0, x)`, `μ(1, x)`; logistic regression with `x_earn` and `x_age` for `π_oracle(x)` (which under true randomization is near-constant — include these regressors only as sanity).
3. Compute the DR pseudo-outcome for each unit `i`, and fit the final-stage model (gradient-boosted regressor) on these pseudo-outcomes.
4. Cache `{τ̂⋆(x_i)}` for all Riverside units.

This is done **once**, outside the MC loop. The LT-O learners are never shown `Y` for any Riverside unit, so there is no leakage at training time. At evaluation, `τ̂⋆(x_i)` for `i` in the test pool serves as the benchmark.

A second caveat: `τ̂⋆` is itself a noisy estimator. Relative comparisons across learners are meaningful; absolute pseudo-PEHE values are not.

---

## 5. Experiments

### 5.1 Experiment A — Stratified PEHE by overlap

**Purpose.** Directly answer Reviewer iqo7's concern: if LT-O-DO wins globally only by deprioritizing low-overlap regions, stratified PEHE will show it loses in the bottom quintile. This experiment either validates or falsifies the robustness claim.

**Protocol.**

1. For each test unit, compute the estimated doubly-overlap weight `ω̂(x_i) = π̂(x_i)(1 − π̂(x_i))\, ρ̂(x_i)` using a **single, fixed** nuisance estimator held constant across learners (e.g., a separate cross-fitted random-forest estimator not used for any of the evaluated learners). This prevents overlap strata from drifting with each learner's own nuisances.
2. Partition the test set into five quintiles `Q1, …, Q5` by `ω̂`, with `Q1` = lowest overlap.
3. For each learner and each quintile, compute:
   - Pseudo-PEHE: `(1/|Q_k|) Σ_{i ∈ Q_k} (τ̂(x_i) − τ̂⋆(x_i))²`.
   - Bias component: `(1/|Q_k|) Σ_{i ∈ Q_k} (τ̂(x_i) − τ̂⋆(x_i))`.
   - Variance component: `Var_MC(τ̂(x_i))` averaged over `i ∈ Q_k`.
4. Report all three quantities — reporting only PEHE would repeat the original critique.

**What to look for.** If LT-O-DO has lower variance but higher bias in Q1 than LT-O-DR, and the variance reduction dominates, PEHE in Q1 still goes down — and that *is* a substantive defense. If LT-O-DO has *both* higher bias and comparable variance in Q1, the retargeting critique holds.

**Configuration.** Use `γ_π = 2` (moderate low-overlap regime), `T = 6`. One configuration is sufficient for A; the full `γ_π` sweep is in Experiment B.

### 5.2 Experiment B — Variance-over-overlap curve

**Purpose.** Real-data analog of synthetic Figure 2. Demonstrate the stability claim: LT-O-DO maintains low variance as overall overlap shrinks, while LT-O-DR's variance blows up.

**Protocol.**

1. For `γ_π ∈ {0, 1, 2, 4}`, run the full pipeline `B = 50` times.
2. For each `γ_π`, compute the test-set-averaged variance
$$V(\gamma_\pi) \;=\; \mathbb{E}_X\big[\operatorname{Var}_\text{MC}(\hat\tau(X))\big].$$
3. Plot `V(γ_π)` against the empirical average `Ē_X[π̂(X)(1 − π̂(X))]` for each learner.

**Configuration.** `T = 6`. Evaluate all eight learners.

**What to look for.** LT-O-DO's `V(γ_π)` should remain approximately flat as overlap shrinks; LT-O-DR's should grow super-linearly. If both grow comparably, overlap weighting is not delivering on the stability claim on real data.

### 5.3 Experiment C — Surrogacy sensitivity via `T`-sweep

**Purpose.** Address Reviewer PBfF and tr9r's concern that surrogacy violations (the Kallus & Mao 2024 critique) dominate any overlap-weighting gains. Smaller `T` implies weaker surrogacy; Athey et al. (2025) show that for `T < 5` the surrogacy bias is large.

**Protocol.**

1. For `T ∈ {2, 4, 6, 12}`, reconstruct the surrogate `S = (tcedd_{1..T}, aid_{1..T}, 1{tcedd_k > 0}_{k=1..T})`.
2. Run the full pipeline at `γ_π = 2` (fixed moderate low-overlap).
3. Report pseudo-PEHE and ATE bias vs. Athey's Riverside benchmark for each learner at each `T`.

**What to look for.** All learners should degrade as `T` shrinks (this is a surrogacy, not an overlap, effect). The test is whether LT-O-DO degrades **no worse than** baselines under surrogacy violation. If it degrades strictly worse, the paper must narrow its scope claims (e.g., condition on surrogacy approximately holding). If all learners degrade at comparable rates, overlap-weighting gains are orthogonal to surrogacy concerns — a defensible and desirable result.

**Configuration.** Fix `γ_π = 2`; evaluate all eight learners.

### 5.4 Experiment D — Subgroup-stratified heterogeneity (interpretive)

**Purpose.** Substantive interpretability check. Demonstrate that LT-O-DO recovers known qualitative heterogeneity patterns in the GAIN program (e.g., stronger effects for younger, lower-skill, or lower-prior-earnings subgroups), matching what labor economists report and what the pseudo-oracle recovers on full Riverside.

This experiment is framed as **interpretive**, not as a PEHE-based benchmark. It answers the question: *if a practitioner deployed the LT-O-DO learner, would the subgroup-level predictions be qualitatively sensible?*

**Subgroups.** Define substantively meaningful partitions on the test set:

| Subgroup | Definition | Rationale |
|---|---|---|
| Age | `age < 30`, `30 ≤ age < 40`, `40 ≤ age < 50`, `age ≥ 50` | Returns to training typically decline with age |
| Education | `xhsdip = 1` (HS diploma) vs `xhsdip = 0` | Prior schooling interacts with program effectiveness |
| Prior labor attachment | `mean(tcprn1..tcprn10) > 0` (ever earned) vs `= 0` (never earned) | Job-first programs may differ in effect by prior attachment |
| Ethnicity | `white`, `hisp`, `black` | Documented heterogeneity in Hotz et al. (2006) |

**Protocol.**

1. For each subgroup `G ⊂ Test_R`, compute:
   - Subgroup size `|G|`.
   - Mean predicted CATE: `τ̄_learner(G) = (1/|G|) Σ_{i ∈ G} τ̂_learner(x_i)`.
   - Mean pseudo-oracle CATE: `τ̄⋆(G) = (1/|G|) Σ_{i ∈ G} τ̂⋆(x_i)`.
   - Subgroup bias: `τ̄_learner(G) − τ̄⋆(G)`.
   - MC standard deviation of `τ̄_learner(G)` across the `B = 50` repetitions.
2. Report as a table with rows = subgroups, columns = (learner-mean CATE, oracle-mean CATE, bias, MC std) × {LT-O-DR, LT-O-DO, LT-RA}.

**Configuration.** `γ_π = 2`, `T = 6`.

**What to look for.** Qualitative agreement between the LT-O-DO learner and the pseudo-oracle across subgroups — same sign, similar ordering, comparable magnitude. This is about defensibility of deployment decisions, not statistical dominance. If the learner inverts the sign of the subgroup effect relative to the oracle in any substantively important subgroup, that is a red flag worth discussing.

**Important framing for the paper.** Do not compute a single aggregate score from this table. The value is the table itself, not a summary statistic. Positioning it as interpretive (not a claim of quantitative superiority) avoids giving reviewers a new hostage to fortune.

---

## 6. Consolidated metrics table

| Experiment | Primary metric | Sweep axis | Fixed config | Learners |
|---|---|---|---|---|
| A | Pseudo-PEHE, bias, MC variance — stratified by `ω̂` quintile | none (single config) | `γ_π = 2`, `T = 6` | all 8 |
| B | `V(γ_π) = E_X[Var_MC(τ̂)]` | `γ_π ∈ {0, 1, 2, 4}` | `T = 6` | all 8 |
| C | Pseudo-PEHE, ATE bias vs. Athey benchmark | `T ∈ {2, 4, 6, 12}` | `γ_π = 2` | all 8 |
| D | Subgroup CATE: learner mean, oracle mean, bias, MC std | none (table) | `γ_π = 2`, `T = 6` | LT-O-DR, LT-O-DO, LT-RA |

---

## 7. Implementation details

### 7.1 Cross-fitting

All nuisance functions — `π̂`, `ρ̂`, `μ̂`, `μ̃` — are estimated via 5-fold cross-fitting within training data. The final-stage `τ̂` is fit on the full training data using out-of-fold pseudo-outcomes, following the standard two-stage orthogonal learner recipe.

### 7.2 Nuisance models

Identical to the paper's synthetic and IST-3 experiments: feed-forward neural networks with the architecture specified in Appendix F. Using a consistent architecture across learners isolates the effect of the loss/weighting choice.

### 7.3 Fixed overlap estimator for Experiment A

For stratification in Experiment A only, fit a separate pair of random-forest models for `π̂_A(x)` and `ρ̂_A(x)` with 5-fold cross-fitting on the full training pool. Use these to compute `ω̂_A(x)` for quintile definition. This prevents any learner from "marking its own homework" when strata are formed.

### 7.4 Pseudo-oracle caching

Fit `τ̂⋆` once on the full Riverside sample (5,445 units, pre-rejection-sampling) with 5-fold cross-fitting. Cache `{(x_i, τ̂⋆(x_i))}_{i ∈ Riverside}` to disk. All downstream evaluation reads from this cache.

### 7.5 Compute estimate

Per seed, per configuration, training wall-time is dominated by the neural nuisance fits (≈ 5–10 min on a single GPU). Full matrix:

- Experiment A: 1 config × 50 seeds × 8 learners ≈ 400 runs.
- Experiment B: 4 configs × 50 seeds × 8 learners ≈ 1,600 runs.
- Experiment C: 4 configs × 50 seeds × 8 learners ≈ 1,600 runs.
- Experiment D: shares runs with A.

Total ≈ 3,600 training runs. Budget ≈ 300–600 GPU-hours. Can be reduced by lowering `B` to 25 for A, C, D and keeping `B = 50` only for B.

### 7.6 Reproducibility

Seed all stochastic operations: rejection-sampling RNG, train/test split RNG, neural initialization, cross-fitting fold assignment. Log seeds in results.

---

## 8. Caveats to disclose in the paper

Four items must be stated openly in the empirical section. Omitting them invites reviewer pushback.

1. **Pseudo-PEHE is not PEHE.** The "ground truth" is a DR-oracle estimate. Absolute magnitudes are meaningless; only relative comparisons across learners are.
2. **Comparability between Riverside and the other sites is imperfect.** Athey et al. (2025) flag this in §7.3. Unfavorable results in Experiments A or C may partly reflect comparability violation rather than the LT-O methodology itself. Experiment C partially addresses this (as surrogacy strength varies, comparability-driven bias changes character), but it does not fully disentangle.
3. **Randomness in the rejection-sampling induction.** The induced `π(x)` depends on the specific choice of `(β_earn, β_age)` and `γ_π`. Robustness to alternative inductions (e.g., `x_age` only, or different weight vectors) should be included in the appendix.
4. **Experiment A decides the robustness claim.** If stratified PEHE in Q1 (low-overlap) is unfavorable to LT-O-DO, the paper must either narrow its scope claims or include the result honestly. Cherry-picking aggregate PEHE while the tail loses is reviewer-visible and will backfire.

---

## 9. Open decisions

Two decisions should be made before implementation begins:

1. **Whether to include the within-Riverside (Kallus-Mao-style) ablation.** Adds an MCAR split as a second path to `ρ(x)`, which sharpens comparison to Kallus & Mao's variance results. Not required for the four experiments above. Recommend deferring to appendix.
2. **Whether to target ATE or CATE in Experiment C's secondary metric.** Current plan reports both pseudo-PEHE (CATE) and ATE-bias against Athey's benchmark. If ATE-recovery fails across all learners at small `T`, the CATE-level story is harder to isolate; flagging ATE-recovery as a sanity check is worth doing regardless.
