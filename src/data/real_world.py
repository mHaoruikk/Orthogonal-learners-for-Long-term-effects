import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from src.data.base_dataset import BaseDataset, TwoSampleDataSplit, GroundTruth

logger = logging.getLogger(__name__)


S_COLS = ["gcs_score_7", "indepinadl_7", "ablewalk_7", "sich7", "dead7"]


class RealWorldIST(BaseDataset):
    """
    Real-world IST-3 dataset.

    Loads the preprocessed CSV produced by `notebooks/ist-real.py`. Treatment
    overlap was already created via rejection sampling (column `retained`); this
    class filters to `retained == 1` and partitions on `R`:
      - R == 0 (Y missing)    -> experimental partition (X_e, A_e, S_e)
      - R == 1 (Y observable) -> observational partition (X_o, S_o, Y_o)
    """

    def __init__(self, config):
        self.config = config
        self.seed = config.seed

        self.data_path = Path(config.data_path)
        self.numerical_cols = list(config.numerical_cols)
        self.binary_cols = list(config.binary_cols)
        self.categorical_cols = list(config.categorical_cols)

        self.df_full = self._read_csv(self.data_path)
        self.pi_star_full = self.df_full["pi_star"].to_numpy(dtype=float)
        self.A_full = self.df_full["A"].to_numpy(dtype=int)
        self.retained_mask = self.df_full["retained"].to_numpy(dtype=int).astype(bool)

        self._refresh_retained_view()

    def _refresh_retained_view(self):
        self.df = self.df_full.loc[self.retained_mask].reset_index(drop=True)
        logger.info(
            "RealWorldIST: %d rows after filtering retained==1 (full=%d)",
            len(self.df), len(self.df_full),
        )

        X_df = self._build_design_matrix(self.df)
        self.feature_names = list(X_df.columns)
        self.dim_x = X_df.shape[1]

        self.X = X_df.to_numpy(dtype=float)
        self.A = self.df["A"].to_numpy(dtype=int)
        self.S = self.df[S_COLS].to_numpy(dtype=float)
        self.R = self.df["R"].to_numpy(dtype=int)
        self.Y = self.df["Y"].to_numpy(dtype=float)
        self.pi_star = self.df["pi_star"].to_numpy(dtype=float)

        self.X_e = self.A_e = self.S_e = None
        self.X_o = self.A_o = self.S_o = self.Y_o = None
        self.pi_star_e = self.pi_star_o = None

    def resample_retained(self, seed: int):
        """Redraw the retained mask from pi_star with a new seed.

        Reproduces the rejection-sampling rule from notebooks/ist-real.py:
            p_keep = pi_star / pi_max          if A == 1
                     (1 - pi_star) / pi_max    otherwise
        where pi_max = max(pi_star, 1 - pi_star) over the full cohort.
        """
        pi = self.pi_star_full
        pi_max = float(max(pi.max(), (1.0 - pi).max()))
        p_keep = np.where(self.A_full == 1, pi / pi_max, (1.0 - pi) / pi_max)
        rng = np.random.default_rng(seed)
        self.retained_mask = rng.random(len(pi)) < p_keep
        self._refresh_retained_view()
        return self

    @staticmethod
    def _read_csv(path: Path) -> pd.DataFrame:
        df = pd.read_csv(path)
        required = {"A", "R", "Y", "retained", "pi_star", *S_COLS}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"CSV {path} missing required columns: {sorted(missing)}")
        return df

    def _build_design_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        cont = df[self.numerical_cols].astype(float)
        binary = df[self.binary_cols].astype(int)
        cats = pd.get_dummies(
            df[self.categorical_cols].astype("category"),
            columns=self.categorical_cols,
            drop_first=True,
            dtype=int,
        )
        return pd.concat([cont, binary, cats], axis=1)

    def sample(self) -> Tuple[TwoSampleDataSplit, GroundTruth]:
        e_mask = self.R == 0
        o_mask = self.R == 1

        X_e = self.X[e_mask]
        A_e = self.A[e_mask]
        S_e = self.S[e_mask]

        X_o = self.X[o_mask]
        S_o = self.S[o_mask]
        Y_o = self.Y[o_mask]

        if np.isnan(Y_o).any():
            raise ValueError("Y contains NaN among R==1 rows; check ist-real.py output.")

        self.X_e, self.A_e, self.S_e = X_e, A_e, S_e
        self.X_o, self.A_o, self.S_o, self.Y_o = X_o, self.A[o_mask], S_o, Y_o
        self.pi_star_e = self.pi_star[e_mask]
        self.pi_star_o = self.pi_star[o_mask]

        logger.info(
            "RealWorldIST.sample: n_e=%d (R=0), n_o=%d (R=1), dim_x=%d",
            X_e.shape[0], X_o.shape[0], self.dim_x,
        )

        data = TwoSampleDataSplit(
            X_e=X_e.copy(), A_e=A_e.copy(), S_e=S_e.copy(),
            X_o=X_o.copy(), S_o=S_o.copy(), Y_o=Y_o.copy(),
        )
        gt = GroundTruth(
            tau=lambda X: np.full(X.shape[0], np.nan),
            pi_E=None,
            e_O=None,
        )
        return data, gt
