"""Surgically replace cells 5 and 6 (the plot cells) of syn_variance_overlap.ipynb.

Leaves all other cells (including the MC-loop cell-4 with its logged output)
untouched. Run after editing _build_variance_overlap_nb.py if you only want to
refresh the plotting logic without rerunning the heavy MC loop.
"""
import json
import os

NB = os.path.join(os.path.dirname(__file__), "syn_variance_overlap.ipynb")


def src_lines(src: str):
    lines = src.splitlines(keepends=False)
    return [l + "\n" for l in lines[:-1]] + ([lines[-1]] if lines else [])


PLOT_LINEAR = """\
import matplotlib as mpl

# LaTeX rendering for paper-ready labels (mirrors plot_variance_overlap.ipynb).
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "text.latex.preamble": r"\\usepackage{amsmath}",
})

# Plotting selection: drop linear-weight TO/DO; show the squared variants
# under the canonical TO-learner / DO-learner names.
PLOT_MODELS = [
    "T-learner",
    "RA-learner",
    "IPW-learner",
    "DR-learner",
    "LO-learner",
    "TO-learner-2",   # displayed as TO-learner
    "DO-learner-2",   # displayed as DO-learner
]

# x-axis: gamma in [0, 5]; drop the gamma=6 point.
GAMMA_MAX = 5.0
XLABEL = r"Overlap Strength $\\gamma\\;\\longrightarrow$ \\textit{(lower overlap)}"

# Three-tier green for the LTO family (light -> dark, by weighting strength):
#   LO (rho) < TO-2 ((pi(1-pi))^2) < DO-2 ((pi(1-pi))^2 * rho)
_GREEN_LEVELS  = {"LO-learner": 0.55, "TO-learner-2": 0.78, "DO-learner-2": 1.00}
_GREEN_MARKERS = {"LO-learner": "s",  "TO-learner-2": "v",  "DO-learner-2": "P"}
_greens = mpl.colormaps["Greens"]
_lto_styles = {m: dict(marker=_GREEN_MARKERS[m], color=_greens(lvl))
               for m, lvl in _GREEN_LEVELS.items()}

STYLE = {
    "T-learner":    dict(marker="o", color="blue"),
    "RA-learner":   dict(marker="x", color="purple"),
    "IPW-learner":  dict(marker="d", color="red"),
    "DR-learner":   dict(marker="^", color="orange"),
    **_lto_styles,
}

LABEL = {
    "T-learner":    "LT-T-learner",
    "RA-learner":   "LT-RA-learner",
    "IPW-learner":  "LT-IPW-learner",
    "DR-learner":   "LT-O-DR-learner (ours)",
    "LO-learner":   "LT-O-LO-learner (ours)",
    "TO-learner-2": "LT-O-TO-learner (ours)",   # relabeled
    "DO-learner-2": "LT-O-DO-learner (ours)",   # relabeled
}

fig, ax = plt.subplots(figsize=(9, 6))
for m in PLOT_MODELS:
    gammas_m = np.asarray(results[m]["gamma"])
    Vs       = np.asarray(results[m]["V"])
    mask = gammas_m <= GAMMA_MAX
    gammas_m, Vs = gammas_m[mask], Vs[mask]
    order = np.argsort(gammas_m)
    ax.plot(
        gammas_m[order], Vs[order],
        label=LABEL.get(m, m), lw=2, ms=10, **STYLE.get(m, {}),
    )

ax.set_xlabel(XLABEL, fontsize=15)
ax.set_ylabel(r"Average variance of estimator $\\operatorname{Var}(\\hat{\\tau}(X))$", fontsize=14)
legend = ax.legend(frameon=True, borderpad=1.2, labelspacing=0.6, handlelength=2.5, fontsize=12, ncol=2)
legend.get_frame().set_alpha(0.9)
legend.get_frame().set_linewidth(1.2)
ax.grid(alpha=0.3)
plt.tight_layout()
OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUTPUT_PDF, bbox_inches="tight", dpi=300)
plt.show()"""

PLOT_LOG = """\
fig, ax = plt.subplots(figsize=(9, 6))
for m in PLOT_MODELS:
    gammas_m = np.asarray(results[m]["gamma"])
    Vs       = np.asarray(results[m]["V"])
    mask = gammas_m <= GAMMA_MAX
    gammas_m, Vs = gammas_m[mask], Vs[mask]
    order = np.argsort(gammas_m)
    ax.plot(
        gammas_m[order], Vs[order],
        label=LABEL.get(m, m), lw=2, ms=10, **STYLE.get(m, {}),
    )

ax.set_yscale("log")
ax.set_xlabel(XLABEL, fontsize=15)
ax.set_ylabel(r"$V$ (log scale)", fontsize=14)
legend = ax.legend(frameon=True, borderpad=1.2, labelspacing=0.6, handlelength=2.5, fontsize=12, ncol=2)
legend.get_frame().set_alpha(0.9)
legend.get_frame().set_linewidth(1.2)
ax.grid(alpha=0.3, which="both")
plt.tight_layout()
plt.savefig(OUTPUT_PDF.with_name("syn_variance_overlap_log.pdf"), bbox_inches="tight", dpi=300)
plt.show()"""


with open(NB, encoding="utf-8") as f:
    nb = json.load(f)

# Locate the linear and log plot cells by content (the user may have inserted
# extra cells, so we cannot rely on fixed indices).
def _src(cell):
    return "".join(cell.get("source", []))

linear_idx = None
log_idx = None
for i, cell in enumerate(nb["cells"]):
    if cell.get("cell_type") != "code":
        continue
    src = _src(cell)
    if "ax.set_yscale(\"log\")" in src and "fig, ax = plt.subplots" in src:
        log_idx = i
    elif "fig, ax = plt.subplots(figsize=(9, 6))" in src and "OUTPUT_PDF" in src \
            and "ax.set_yscale(\"log\")" not in src:
        linear_idx = i

if linear_idx is None or log_idx is None:
    raise RuntimeError(
        f"could not locate plot cells (linear_idx={linear_idx}, log_idx={log_idx})"
    )

print(f"Patching linear plot cell at index {linear_idx}")
nb["cells"][linear_idx]["source"] = src_lines(PLOT_LINEAR)
nb["cells"][linear_idx]["outputs"] = []
nb["cells"][linear_idx]["execution_count"] = None

print(f"Patching log plot cell at index {log_idx}")
nb["cells"][log_idx]["source"] = src_lines(PLOT_LOG)
nb["cells"][log_idx]["outputs"] = []
nb["cells"][log_idx]["execution_count"] = None

with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Patched {NB}")
