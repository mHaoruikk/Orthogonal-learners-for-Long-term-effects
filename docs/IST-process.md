# IST-3 real-world preprocessing

Documentation of [`notebooks/ist-real.py`](../notebooks/ist-real.py), which implements the experiment plan in [`IST-realword.md`](IST-realword.md) and produces [`src/data/ist_real.csv`](../src/data/ist_real.csv).

## Inputs / outputs

- **Input:** `C:\Users\ma\Research\Long-term-effects\IST-3-Dataset\datashare_aug2015.sas7bdat` (3035 patients, 266 columns).
- **Output:** `src/data/ist_real.csv` (3034 patients, 33 columns).
- **Seed:** `42` (used for both the Ridge imputer and rejection sampling).

The CSV contains the full cohort; two binary columns, `R` and `retained`, encode the overlap mechanisms so downstream code can select subsets without re-running the pipeline.

## Column substitutions vs. the plan

Two names from `IST-realword.md` §1 do not exist in the raw SAS file; the closest available constructions are used:

| Plan name | Action |
|---|---|
| `gcs_score_7` | Constructed as `gcs_eye_7 + gcs_motor_7 + gcs_verbal_7` (mirrors how `gcs_score_rand` is built). Component code `10` ("died — question not relevant") is mapped to `0`; codes `20/30/40` ("question not answered / form not returned / not asked") are treated as missing. |
| `dead18` | Derived from `ohs18 == 6` (OHS level 6 = died before 18 months, per the IST-3 data description). |

Additional mappings worth flagging:
- `itt_treat` is coded **0 = rt-PA, 1 = Placebo** in IST-3. The script flips this so that `A = 1` means active treatment: `A = (itt_treat == 0).astype(int)`.
- `age` (rather than `age_true`) is used because `age` already has the anonymisation means 31.935 and 97.846 substituted for the <40 and >95 groups, making it identical to the plan's imputation rule.
- `euroqol18` is a VAS on the **0–100** scale in the raw data; the script rescales by dividing by 100 to land in [0, 1].

## Cohort

- Start from all 3035 patients.
- Drop the 1 patient with no 7-day form, identified by any of `gcs_eye_7 / gcs_motor_7 / gcs_verbal_7 == 30` ("Form not returned").
- Final cohort: **3034 patients**, retained intact through the rest of the pipeline.

## Variables

### Covariates `X` (21 columns)

Continuous: `age`, `nihss`, `gcs_score_rand`, `sbprand`, `dbprand`, `glucose`, `weight`, `R_infarct_size`, `konprob`, `randdelay`.

Binary (0/1 after recoding from IST-3 `1=Yes / 2=No`): `gender` (1 = female), `pred_nihss`, `atrialfib_rand`, `stroke_pre`, `antiplat_rand`, `livealone_rand`, `indepinadl_rand`, `vis_infarct`, `glucose_missing`.

Raw categoricals (kept unencoded for one-hot at modeling time): `stroketype` (1–5), `country` (9 levels).

Continuous variables are stored at **raw scale** — standardisation is deferred to the modeling step.

### Surrogate `S` (5 columns)

`gcs_score_7` (constructed, continuous 0–15), `indepinadl_7` (binary), `ablewalk_7` (binary), `sich7` (binary), `dead7` (binary). For `indepinadl_7` / `ablewalk_7`, death code `10` is mapped to `0` (not independent / unable to walk).

### Treatment `A`

`A = 1` means rt-PA, `A = 0` means Control.

### Outcome `Y`

Start from `euroqol18 / 100` on [0, 1]. Set `Y = 0` when the patient is known to be deceased — either `ohs18 == 6` or (`censor18 == 0` AND `surv18 <= 548`). For alive, form-returned patients with a missing EQ-5D response (116 cases), `Y` is regression-imputed via scikit-learn `Ridge(alpha=1.0, random_state=42)` conditional on all covariates in `X` (with `country` and `stroketype` one-hot encoded) plus `ohs18`; predictions are clipped to [0, 1]. For rows with `R = 0`, `Y` is left as `NaN` (an assertion enforces this at the end).

### Observation indicator `R`

Per §3 of the plan:

- `R = 0` if `plan18 == 2` **or** (18-month form not returned **and** no death date within 548 days).
- `R = 1` otherwise — i.e., form returned or death recorded within 548 days.

Implemented as `R = (plan18 != 2) & (receighteen == 1 | (censor18 == 0 & surv18 <= 548))`. Note the quirky variable name `receighteen` (extra `h`).

## Missing-data handling

| Variable | Treatment |
|---|---|
| `glucose` (282 early-trial NaN) | Country-stratified median; add `glucose_missing` flag. Any residual NaN (unknown country) filled with global median. |
| `dbprand` (19 NaN + 35/36 protocol violations) | Values `35` and `36` set to NaN; then filled with the median. |
| `R_infarct_size` (18 NaN) | Median imputation. |
| `konprob`, `randdelay`, `weight` | Median imputation (defensive — few or no NaN in practice). |
| YNDQ binaries with missing/unknown codes | Filled with the mode. |
| 7-day categorical S variables | Filled with the mode. |
| `gcs_score_7` (NaN from 20/40 codes) | Median imputation. |
| `euroqol18` (alive, form returned, missing) | Ridge regression on `X + ohs18`, clipped to [0, 1]. |
| `nihss` | Kept as-is, including the 244 predicted values; `pred_nihss` is retained as an indicator. |

## Rejection sampling for low treatment overlap

Target propensity (raw scale, coefficients exactly as in plan §4):

$$
\pi^*(x) = \sigma\!\big(0.3 - 0.04\,(\text{age}-70) - 0.015\,(\text{nihss}-12)^2 + 0.03\,(\text{gcs\_score\_rand}-12) - 0.4\cdot\mathbb{1}\{\text{atrialfib\_rand}\}\big).
$$

Stored as the column `pi_star` for use as an oracle propensity in ablations.

Let $\pi_{\max} = \max_i \max(\pi^*_i,\, 1-\pi^*_i)$. Each unit is independently retained with probability

$$p_i = \begin{cases}\pi^*_i / \pi_{\max} & \text{if } A_i = 1\\ (1-\pi^*_i)/\pi_{\max} & \text{if } A_i = 0\end{cases}$$

A uniform draw decides retention; the result is stored as `retained ∈ {0, 1}`. Rejection sampling is applied to the **full 3034-patient cohort before the R split**, per §4 of the plan, so that the two overlap mechanisms stay independent.

## Diagnostics from the reference run (seed 42)

```
Cohort: 3034
R=1: 2238, R=0: 796
retained: 1488 / 3034
pi_star: min=0.000, mean=0.349, max=0.870, pi_max=1.000
retained A=1 share: 0.341
retained R=1 count: 1098
Regression-imputed EQ-5D: 116 alive-form-returned missing cases
```

**Notes.**
- `R=1 = 2238` is higher than the plan's round estimate of ~1400 — but this matches the IST-3 description exactly: 1417 form-returned + 822 deaths within plan18==1 = 2239.
- `retained = 1488` is slightly below the plan's ~1650 because the very-high-NIHSS tail drives `pi_max` to ~1 via the quadratic term. The plan flags that coefficients may need tuning to land in [0.05, 0.90]; they are left as specified for reproducibility.
- The script asserts that every row with `Y` missing has `R = 0`.

## Output schema

| Column | Type | Notes |
|---|---|---|
| `age`, `nihss`, `gcs_score_rand`, `sbprand`, `dbprand`, `glucose`, `weight`, `R_infarct_size`, `konprob`, `randdelay` | float | Raw scale |
| `gender`, `pred_nihss`, `atrialfib_rand`, `stroke_pre`, `antiplat_rand`, `livealone_rand`, `indepinadl_rand`, `vis_infarct`, `glucose_missing` | int | 0/1 |
| `stroketype` | int | 1..5 (TACI/PACI/LACI/POCI/OTHER) |
| `country` | str | 9 levels |
| `gcs_score_7` | float | 0..15 |
| `indepinadl_7`, `ablewalk_7`, `sich7`, `dead7` | int | 0/1 |
| `A` | int | 1 = rt-PA |
| `Y` | float | [0,1]; NaN iff `R = 0` |
| `R` | int | Long-term outcome observation indicator |
| `retained` | int | Rejection-sampling retention indicator |
| `pi_star` | float | Oracle propensity used for rejection sampling |

## Reproducing

```bash
~/AppData/Local/miniconda3/envs/lte/python.exe notebooks/ist-real.py
```

Requires `pyreadstat`, `pandas`, `numpy`, `scikit-learn` — all installed in the `lte` conda environment.
