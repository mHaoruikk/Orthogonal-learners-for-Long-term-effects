Implementation of an LT-learner for long-term heterogeneous treatment effects by combining an experimental cohort (A,S,X, no Y) with an observational cohort (S,X,Y, no A). Includes synthetic and semi-synthetic data generators and nuisance-model cross-fitting.

## Quick start
1) Create an environment (Python ≥3.10) and install deps:
   pip install numpy pandas scipy scikit-learn xgboost torch hydra-core omegaconf matplotlib
   # or add them to requirements.txt and pip install -r requirements.txt
2) Run the R-learner baseline:
   python -m scripts.train_r_learner dataset=setupA model=R-learner trainer=default
3) Other options:
   - DR learner: python -m scripts.train_dr_learner dataset=setupA model=DR-learner
   - LT learner: python -m scripts.train_hlte dataset=setupA model=HLTE-learner
   - Semi-synthetic data: dataset=semi-synthetic

## Configuration
Hydra config lives in `config/`. Defaults are set in `config/config.yaml`; override pieces via CLI, e.g. `dataset=setupB model=T-learner`. Key files:
- `config/dataset/*.yaml` – data generators (synthetic surrogate, semi-synthetic)
- `config/model/*.yaml` – learner + nuisance settings (logistic/xgboost, overlap weights, cross-fitting)
- `config/trainer/*.yaml` – train/val/test split fractions.

## Repository map
- `src/data/` data classes and generators
- `src/model/` learners (T, DR, overlap R) and nuisance builders
- `scripts/` train/eval entry points
- `docs/` paper draft
- `notebooks/` quick experiments
- `outputs/` Hydra run artifacts
