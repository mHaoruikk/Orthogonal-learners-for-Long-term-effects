import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

# Allow imports from the project root when running as a script or via pytest
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.synthetic import NieWagerSyntheticDataset
from src.model.nuisances import NuisanceFactory


def _accuracy(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Binary accuracy from predicted probabilities."""
    preds = (proba >= 0.5).astype(int)
    return float(np.mean(preds == y_true))


def _mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def _make_synthetic_data():
    cfg = SimpleNamespace(
        name="debug-synth",
        type="synthetic_surrogate",
        propensity_E="sin",
        propensity_O="exp",
        dim_x=6,
        n_e=450,
        n_o=700,
        sigma_s=0.2,
        sigma_y=0.2,
        seed=2025,
    )
    dataset = NieWagerSyntheticDataset(cfg)
    data, _ = dataset.sample()
    return data


def _make_model_cfg(K: int = 3):
    nuisances = {
        "pi_x": SimpleNamespace(type="logistic", parameters={"max_iter": 200}),
        "pi_s_x": SimpleNamespace(type="logistic", parameters={"max_iter": 200}),
        "rho_x": SimpleNamespace(type="logistic", parameters={"max_iter": 200}),
        "rho_s_x": SimpleNamespace(type="logistic", parameters={"max_iter": 200}),
        "h": SimpleNamespace(type="linear", parameters={"fit_intercept": True}),
        "mu_0": SimpleNamespace(type="linear", parameters={"fit_intercept": True}),
        "mu_1": SimpleNamespace(type="linear", parameters={"fit_intercept": True}),
    }
    return SimpleNamespace(num_crossfit=K, nuisances=nuisances)


def test_crossfit_nuisance_metrics():
    """
    Fit nuisances with cross-fitting and report train vs hold-out metrics per fold.
    """
    data = _make_synthetic_data()
    model_cfg = _make_model_cfg(K=3)
    factory = NuisanceFactory(model_cfg)
    cf = factory.crossfit_nuisance(data)

    n_e, n_o = data.X_e.shape[0], data.X_o.shape[0]
    assert len(cf.folds) == model_cfg.num_crossfit
    assert cf.fold_id_e.shape[0] == n_e
    assert cf.fold_id_o.shape[0] == n_o
    assert set(cf.fold_id_e) == set(range(model_cfg.num_crossfit))
    assert set(cf.fold_id_o) == set(range(model_cfg.num_crossfit))

    SX_e = np.column_stack([data.S_e, data.X_e])
    SX_o = np.column_stack([data.S_o, data.X_o])
    Z_e = np.column_stack([data.S_e, data.X_e])

    # Cross-fitted pseudo-outcome mean for mu_0, mu_1 targets
    h_hat_e = sum(fold.h.predict(SX_e) for fold in cf.folds) / model_cfg.num_crossfit

    reports = []
    for k in range(model_cfg.num_crossfit):
        nm = cf.folds[k]
        train_e = cf.fold_id_e != k
        hold_e = ~train_e
        train_o = cf.fold_id_o != k
        hold_o = ~train_o

        report = {"fold": k}

        report["pi_x_train_acc"] = _accuracy(data.A_e[train_e], nm.pi_x.predict(data.X_e[train_e]))
        report["pi_x_hold_acc"] = _accuracy(data.A_e[hold_e], nm.pi_x.predict(data.X_e[hold_e]))

        report["pi_s_x_train_acc"] = _accuracy(data.A_e[train_e], nm.pi_s_x.predict(Z_e[train_e]))
        report["pi_s_x_hold_acc"] = _accuracy(data.A_e[hold_e], nm.pi_s_x.predict(Z_e[hold_e]))

        rho_train_X = np.vstack([data.X_e[train_e], data.X_o[train_o]])
        rho_train_y = np.concatenate([np.zeros(train_e.sum(), dtype=int), np.ones(train_o.sum(), dtype=int)])
        rho_hold_X = np.vstack([data.X_e[hold_e], data.X_o[hold_o]])
        rho_hold_y = np.concatenate([np.zeros(hold_e.sum(), dtype=int), np.ones(hold_o.sum(), dtype=int)])
        report["rho_x_train_acc"] = _accuracy(rho_train_y, nm.rho_x.predict(rho_train_X))
        report["rho_x_hold_acc"] = _accuracy(rho_hold_y, nm.rho_x.predict(rho_hold_X))

        rho_s_train_X = np.vstack([SX_e[train_e], SX_o[train_o]])
        rho_s_hold_X = np.vstack([SX_e[hold_e], SX_o[hold_o]])
        report["rho_s_x_train_acc"] = _accuracy(rho_train_y, nm.rho_s_x.predict(rho_s_train_X))
        report["rho_s_x_hold_acc"] = _accuracy(rho_hold_y, nm.rho_s_x.predict(rho_s_hold_X))

        report["h_train_mse"] = _mse(data.Y_o[train_o], nm.h.predict(SX_o[train_o]))
        report["h_hold_mse"] = _mse(data.Y_o[hold_o], nm.h.predict(SX_o[hold_o]))

        mu0_train_mask = train_e & (data.A_e == 0)
        mu0_hold_mask = hold_e & (data.A_e == 0)
        report["mu0_train_mse"] = (
            _mse(h_hat_e[mu0_train_mask], nm.mu_0.predict(data.X_e[mu0_train_mask]))
            if mu0_train_mask.any()
            else np.nan
        )
        report["mu0_hold_mse"] = (
            _mse(h_hat_e[mu0_hold_mask], nm.mu_0.predict(data.X_e[mu0_hold_mask]))
            if mu0_hold_mask.any()
            else np.nan
        )

        mu1_train_mask = train_e & (data.A_e == 1)
        mu1_hold_mask = hold_e & (data.A_e == 1)
        report["mu1_train_mse"] = (
            _mse(h_hat_e[mu1_train_mask], nm.mu_1.predict(data.X_e[mu1_train_mask]))
            if mu1_train_mask.any()
            else np.nan
        )
        report["mu1_hold_mse"] = (
            _mse(h_hat_e[mu1_hold_mask], nm.mu_1.predict(data.X_e[mu1_hold_mask]))
            if mu1_hold_mask.any()
            else np.nan
        )

        reports.append(report)

    # Human-readable metrics for debugging cross-fitting behaviour
    for rep in reports:
        metrics_str = ", ".join(
            f"{k}={v:.3f}" if isinstance(v, float) and not np.isnan(v) else f"{k}=nan"
            for k, v in rep.items()
            if k != "fold"
        )
        print(f"Fold {rep['fold']}: {metrics_str}")
