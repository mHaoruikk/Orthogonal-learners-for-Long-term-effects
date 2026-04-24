"""Generate notebooks/real-world-DO-interpret.ipynb (outputs cleared)."""
from __future__ import annotations
import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
cells = []


def md(src: str):
    cells.append(nbf.v4.new_markdown_cell(src.strip("\n")))


def code(src: str):
    cells.append(nbf.v4.new_code_cell(src.strip("\n")))


# ---------------------------------------------------------------- Cell 0 (md)
md(r"""
# LTO-learner family interpretation on real IST-3

Fit four LTO learners — DR, TO, LO, DO (`weight_type` ∈ {identity, to, lo, dual}) — on the
IST-3 retained cohort and interpret their CATE estimates against the IST-3 18-month
follow-up findings (Sandercock et al. 2013, Lancet). Key benchmarks:

- **ATE on EQ-5D utility:** mean difference **+0.060** in favor of alteplase (p=0.019).
- **Age × treatment:** significant interaction, age **>80 favored** (p=0.032).
- **Delay (0–3h / 3–4.5h / 4.5–6h):** **null** (no significant interaction).
- **NIHSS severity:** **null** (no significant interaction).

For each learner we run 20 seeded replicates, average per-individual `tau_hat`, and report
on the **in-sample (train)** and **held-out (test)** splits.
""")

# ---------------------------------------------------------------- Cell 1
code(r"""
import sys, os
from pathlib import Path

# Make ../src and ../scripts importable from notebooks/
REPO_ROOT = Path.cwd().parent
for p in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import warnings, logging
warnings.filterwarnings("ignore")
logging.getLogger("sklearn").setLevel(logging.ERROR)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

from src.data.real_world import RealWorldIST
from src.data.base_dataset import TwoSampleDataSplit
from src.data.utils import load_config
from src.model.lto_learner import LTO_Learner

# Seed-threading helper (reused from src.eval_utils)
from src.eval_utils import seed_model_cfg

RETAIN_SEED = 42
S_SEEDS = 20
LEARNERS = ["DR-learner", "TO-learner", "LO-learner", "DO-learner"]
SHORT = {"DR-learner": "DR", "TO-learner": "TO", "LO-learner": "LO", "DO-learner": "DO"}
COLORS = {"DR": "#1f77b4", "TO": "#2ca02c", "LO": "#ff7f0e", "DO": "#d62728"}

# Load one config to materialize the dataset; per-learner cfgs loaded later.
cfg0 = load_config(dataset='real-world-ist', model='DO-learner', trainer='default',
                   config_root=REPO_ROOT / 'config')
cfg0.dataset.data_path = str((REPO_ROOT / cfg0.dataset.data_path).resolve())
ds = RealWorldIST(cfg0.dataset)
ds.resample_retained(RETAIN_SEED)

X, A, S, R, Y = ds.X, ds.A, ds.S, ds.R, ds.Y
feature_names = ds.feature_names
print(f"Retained cohort: n={len(X)}  (R=0: {(R==0).sum()}, R=1: {(R==1).sum()})")
print(f"dim_x={ds.dim_x}  S-dim={S.shape[1]}")
print(f"age idx={feature_names.index('age')}  "
      f"randdelay idx={feature_names.index('randdelay')}  "
      f"nihss idx={feature_names.index('nihss')}")
""")

# ---------------------------------------------------------------- Cell 2
code(r"""
# Train / held-out test split: stratified 80/20 within each of R=0 and R=1.
# We still split per-R because (i) TwoSampleDataSplit needs separate X_e (R=0) and
# X_o (R=1) inputs, and (ii) baseline ATE estimators (diff-in-means, IPW) are
# only well-defined on R=1 (Y is unobserved on R=0).
rng = np.random.default_rng(42)

def _split_80_20(idx):
    idx = idx.copy()
    rng.shuffle(idx)
    n_test = int(round(0.2 * len(idx)))
    return idx[n_test:], idx[:n_test]

r0_idx = np.where(R == 0)[0]
r1_idx = np.where(R == 1)[0]
r0_train, r0_test = _split_80_20(r0_idx)
r1_train, r1_test = _split_80_20(r1_idx)

# Aggregated train / test indices (used everywhere downstream).
train_idx = np.concatenate([r0_train, r1_train])
test_idx  = np.concatenate([r0_test,  r1_test])

assert len(set(train_idx) & set(test_idx)) == 0
print(f"R=0  train={len(r0_train)}  test={len(r0_test)}")
print(f"R=1  train={len(r1_train)}  test={len(r1_test)}")
print(f"Combined  train={len(train_idx)}  test={len(test_idx)}")
""")

# ---------------------------------------------------------------- Cell 3
code(r"""
# Fit each of {DR, TO, LO, DO} x S_SEEDS on training split; predict on train + test.
train_split = TwoSampleDataSplit(
    X_e=X[r0_train], A_e=A[r0_train], S_e=S[r0_train],
    X_o=X[r1_train], S_o=S[r1_train], Y_o=Y[r1_train],
)

# Per-learner per-seed prediction matrices, sized (n_split, S_SEEDS).
tau_train = {l: np.empty((len(train_idx), S_SEEDS)) for l in LEARNERS}
tau_test  = {l: np.empty((len(test_idx),  S_SEEDS)) for l in LEARNERS}

for lname in LEARNERS:
    cfg_l = load_config(dataset='real-world-ist', model=lname, trainer='default',
                        config_root=REPO_ROOT / 'config')
    print(f"[{lname}] weight_type={cfg_l.model.weight_type}")
    for s in range(S_SEEDS):
        seeded_cfg = seed_model_cfg(cfg_l.model, seed=s)
        learner = LTO_Learner(seeded_cfg)
        learner.fit(train_split)
        tau_train[lname][:, s] = learner.predict_cate(X[train_idx])
        tau_test[lname][:,  s] = learner.predict_cate(X[test_idx])
    print(f"  done {S_SEEDS} fits")

# Per-individual seed averages and seed std (model uncertainty).
tau_train_bar = {l: tau_train[l].mean(axis=1) for l in LEARNERS}
tau_test_bar  = {l: tau_test[l].mean(axis=1)  for l in LEARNERS}
for l in LEARNERS:
    assert np.isfinite(tau_train_bar[l]).all() and np.isfinite(tau_test_bar[l]).all()
    print(f"{l}: train mean={tau_train_bar[l].mean():+.4f}  "
          f"test mean={tau_test_bar[l].mean():+.4f}")
""")

# ---------------------------------------------------------------- Cell 4 md
md(r"""
## Average treatment effect

Per-learner ATE on the combined train and test cohorts, with two RCT baselines
(diff-in-means and IPW) restricted to the R=1 portion (where Y is observed).
Reference: IST-3 2013 EQ-5D mean difference of **+0.060**.
""")

# ---------------------------------------------------------------- Cell 4 code
code(r"""
def bootstrap_mean(vals, B=1000, seed=0):
    vals = np.asarray(vals)
    rng = np.random.default_rng(seed)
    n = len(vals)
    if n == 0:
        return np.nan, (np.nan, np.nan)
    samples = rng.integers(0, n, size=(B, n))
    means = vals[samples].mean(axis=1)
    return float(vals.mean()), (float(np.quantile(means, 0.025)),
                                float(np.quantile(means, 0.975)))

def seed_std(tau_matrix):
    # std across the S_SEEDS per-seed population means
    return float(tau_matrix.mean(axis=0).std(ddof=1))

LIT = 0.060

print(f"{'Learner':<14} {'Split':<6} {'ATE':>8}  {'seed std':>9}  {'95% boot CI':>22}  {'n':>5}")
print("-" * 72)
for lname in LEARNERS:
    for split_name, tau_bar, tau_mat, n in [
        ("train", tau_train_bar[lname], tau_train[lname], len(train_idx)),
        ("test",  tau_test_bar[lname],  tau_test[lname],  len(test_idx)),
    ]:
        m, ci = bootstrap_mean(tau_bar)
        sd = seed_std(tau_mat)
        print(f"{lname:<14} {split_name:<6} {m:+8.4f}  {sd:9.4f}  "
              f"({ci[0]:+.3f}, {ci[1]:+.3f})   {n:5d}")

print()
print(f"{'Baseline (R=1)':<14} {'Split':<6} {'ATE':>8}  {'seed std':>9}  {'95% boot CI':>22}  {'n':>5}")
print("-" * 72)
for split_name, idx in [("train", r1_train), ("test", r1_test)]:
    Y_r1 = Y[idx]; A_r1 = A[idx]
    Y1 = Y_r1[A_r1 == 1]; Y0 = Y_r1[A_r1 == 0]
    dim = float(Y1.mean() - Y0.mean())
    rngb = np.random.default_rng(0)
    dim_boot = np.empty(1000)
    for b in range(1000):
        s1 = rngb.integers(0, len(Y1), size=len(Y1))
        s0 = rngb.integers(0, len(Y0), size=len(Y0))
        dim_boot[b] = Y1[s1].mean() - Y0[s0].mean()
    dim_ci = (float(np.quantile(dim_boot, 0.025)), float(np.quantile(dim_boot, 0.975)))
    print(f"{'Diff-in-means':<14} {split_name:<6} {dim:+8.4f}  {'  -  ':>9}  "
          f"({dim_ci[0]:+.3f}, {dim_ci[1]:+.3f})   {len(idx):5d}")

# IPW: fit pi on r1_train, evaluate on r1_train and r1_test
pi_clf = LogisticRegression(penalty='l2', C=1.0, max_iter=2000, tol=1e-4)
pi_clf.fit(X[r1_train], A[r1_train])
for split_name, idx in [("train", r1_train), ("test", r1_test)]:
    pi_hat = np.clip(pi_clf.predict_proba(X[idx])[:, 1], 1e-3, 1 - 1e-3)
    Y_r1 = Y[idx]; A_r1 = A[idx]
    ipw_terms = A_r1 * Y_r1 / pi_hat - (1 - A_r1) * Y_r1 / (1 - pi_hat)
    m, ci = bootstrap_mean(ipw_terms)
    print(f"{'IPW':<14} {split_name:<6} {m:+8.4f}  {'  -  ':>9}  "
          f"({ci[0]:+.3f}, {ci[1]:+.3f})   {len(idx):5d}")

print(f"\nLiterature (IST-3 2013, EQ-5D): +{LIT:.3f}")
""")

# ---------------------------------------------------------------- Cell 5 md
md(r"""
## Subgroup analysis — Age (cutoff 80)

IST-3 2013 reported a significant age × treatment interaction, with patients **older than 80**
showing a **larger benefit** (p=0.032).
""")

# ---------------------------------------------------------------- Cell 5 code
code(r"""
def subgroup_stats(tau_bar, masks):
    stats = []
    for label, mask in masks:
        vals = tau_bar[mask]
        if len(vals) == 0:
            stats.append((label, np.nan, (np.nan, np.nan), 0))
        else:
            m, ci = bootstrap_mean(vals)
            stats.append((label, m, ci, int(mask.sum())))
    return stats

def collect_per_learner(tau_bar_dict, masks):
    return {SHORT[l]: subgroup_stats(tau_bar_dict[l], masks) for l in LEARNERS}

def _plot_subgroup(per_learner_train, per_learner_test, title, xlabels, bin_sizes):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), sharey=True)
    n_bins = len(xlabels)
    x = np.arange(n_bins)
    n_l = len(per_learner_train)
    width = 0.8 / n_l
    for ax, per_l, panel_title, ns in [
        (axes[0], per_learner_train, "Train (in-sample)", bin_sizes['train']),
        (axes[1], per_learner_test,  "Test (held-out)",   bin_sizes['test']),
    ]:
        for i, (lname, stats) in enumerate(per_l.items()):
            offset = (i - (n_l - 1) / 2) * width
            means = [s[1] for s in stats]
            lows  = [max(0, s[1] - s[2][0]) for s in stats]
            highs = [max(0, s[2][1] - s[1]) for s in stats]
            ax.bar(x + offset, means, width, label=lname, color=COLORS[lname],
                   yerr=[lows, highs], capsize=3, edgecolor='black')
        ax.axhline(0, color='black', lw=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{xl}\n(n={n})" for xl, n in zip(xlabels, ns)])
        ax.set_title(panel_title)
        ax.grid(axis='y', linestyle=':', alpha=0.5)
    axes[0].set_ylabel(r"mean $\hat\tau(x)$ (EQ-5D scale)")
    axes[1].legend(loc='best', frameon=True, ncol=2, fontsize=9)
    fig.suptitle(title)
    plt.tight_layout(); plt.show()

AGE_IDX = feature_names.index('age')
age_train = X[train_idx][:, AGE_IDX]
age_test  = X[test_idx][:,  AGE_IDX]
age_masks_train = [("age <= 80", age_train <= 80), ("age > 80", age_train > 80)]
age_masks_test  = [("age <= 80", age_test  <= 80), ("age > 80", age_test  > 80)]
xlabels_age = [r"age $\leq$ 80", r"age > 80"]

per_train = collect_per_learner(tau_train_bar, age_masks_train)
per_test  = collect_per_learner(tau_test_bar,  age_masks_test)
bin_sizes = {
    'train': [int(m.sum()) for _, m in age_masks_train],
    'test':  [int(m.sum()) for _, m in age_masks_test],
}
print("Train bin sizes:", bin_sizes['train'])
print("Test  bin sizes:", bin_sizes['test'])
_plot_subgroup(per_train, per_test,
               "CATE by age (IST-3 2013: age >80 favored, p=0.032)",
               xlabels_age, bin_sizes)
""")

# ---------------------------------------------------------------- Cell 6 md
md(r"""
## Subgroup analysis — Randomisation delay (0–3h, 3–4.5h, 4.5–6h)

IST-3 2013 reported **no significant interaction** between treatment effect and
time-to-treatment windows on OHS at 18 months.
""")

# ---------------------------------------------------------------- Cell 6 code
code(r"""
DELAY_IDX = feature_names.index('randdelay')

def delay_masks(d):
    return [("0-3h",   (d >= 0)   & (d < 3)),
            ("3-4.5h", (d >= 3)   & (d < 4.5)),
            ("4.5-6h", (d >= 4.5) & (d < 6))]

d_train = X[train_idx][:, DELAY_IDX]
d_test  = X[test_idx][:,  DELAY_IDX]
masks_train = delay_masks(d_train)
masks_test  = delay_masks(d_test)
xlabels_delay = ["0-3h", "3-4.5h", "4.5-6h"]

per_train = collect_per_learner(tau_train_bar, masks_train)
per_test  = collect_per_learner(tau_test_bar,  masks_test)
bin_sizes = {
    'train': [int(m.sum()) for _, m in masks_train],
    'test':  [int(m.sum()) for _, m in masks_test],
}
print("Train bin sizes:", bin_sizes['train'])
print("Test  bin sizes:", bin_sizes['test'])
_plot_subgroup(per_train, per_test,
               "CATE by delay (IST-3 2013: null interaction)",
               xlabels_delay, bin_sizes)
""")

# ---------------------------------------------------------------- Cell 7 md
md(r"""
## Subgroup analysis — NIHSS severity (mild / moderate / severe)

Standard cutoffs: mild 0–5, moderate 6–14, severe ≥15. IST-3 2013 reported
**no significant interaction** between treatment effect and NIHSS severity.
""")

# ---------------------------------------------------------------- Cell 7 code
code(r"""
NIHSS_IDX = feature_names.index('nihss')

def nihss_masks(v):
    return [("mild 0-5",      (v >= 0) & (v <= 5)),
            ("moderate 6-14", (v >= 6) & (v <= 14)),
            ("severe >=15",   v >= 15)]

n_train = X[train_idx][:, NIHSS_IDX]
n_test  = X[test_idx][:,  NIHSS_IDX]
masks_train = nihss_masks(n_train)
masks_test  = nihss_masks(n_test)
xlabels_n = [r"mild 0-5", r"moderate 6-14", r"severe $\geq$ 15"]

per_train = collect_per_learner(tau_train_bar, masks_train)
per_test  = collect_per_learner(tau_test_bar,  masks_test)
bin_sizes = {
    'train': [int(m.sum()) for _, m in masks_train],
    'test':  [int(m.sum()) for _, m in masks_test],
}
print("Train bin sizes:", bin_sizes['train'])
print("Test  bin sizes:", bin_sizes['test'])
_plot_subgroup(per_train, per_test,
               "CATE by NIHSS severity (IST-3 2013: null interaction)",
               xlabels_n, bin_sizes)
""")

# ---------------------------------------------------------------- Cell 8 md
md(r"""
## Summary

| Finding | IST-3 2013 | LTO learners (train / test) |
|---|---|---|
| ATE (EQ-5D) | +0.060 (p=0.019) | See cell 4. Compare 4 learners against diff-in-means / IPW baselines. |
| Age × treatment | Age >80 favored (p=0.032) | See cell 5. |
| Delay × treatment | Null | See cell 6. |
| NIHSS × treatment | Null | See cell 7. |

### Caveats
- The held-out test set is small (~298 patients combined; ~78 R=0 and ~220 R=1);
  subgroup splits yield a handful per cell, so bars are noisy.
- LTO learners' CATE is on the EQ-5D scale. The literature headline outcome is OHS
  (ordinal); EQ-5D is secondary. Direction-of-effect comparison is still valid.
- The 2013 paper tests association via regression **interactions**, not per-patient
  CATE point estimates — so the comparison is qualitative.
- Diff-in-means and IPW use only the R=1 portion of each split (Y is unobserved on R=0).
""")

# ---------------------------------------------------------------- write
nb['cells'] = cells
out = Path('notebooks/real-world-DO-interpret.ipynb')
out.parent.mkdir(parents=True, exist_ok=True)
with out.open('w', encoding='utf-8') as f:
    nbf.write(nb, f)
print(f"wrote {out} ({out.stat().st_size} bytes, {len(cells)} cells)")
