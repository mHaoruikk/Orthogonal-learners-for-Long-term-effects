from typing import Tuple
from src.data.base_dataset import TwoSampleDataSplit
from sklearn.model_selection import train_test_split

def split_two_sample_data(
    data: TwoSampleDataSplit,
    val_fraction: float,
    test_fraction: float,
    random_state: int = 42
) -> Tuple[TwoSampleDataSplit, TwoSampleDataSplit, TwoSampleDataSplit]:
    """
    Split a TwoSampleDataSplit into train, validation, and test sets.

    Args:
        data: The original TwoSampleDataSplit dataset.
        val_fraction: Fraction of data to use for validation.
        test_fraction: Fraction of data to use for testing.
        random_state: Random seed for reproducibility.
    """

    if not 0 <= val_fraction < 1:
        raise ValueError("val_fraction must be in [0, 1).")
    if not 0 <= test_fraction < 1:
        raise ValueError("test_fraction must be in [0, 1).")
    if val_fraction + test_fraction >= 1:
        raise ValueError("val_fraction + test_fraction must be < 1.")

    def _empty_like(arr):
        return arr[:0].copy()

    def _split(arrs):
        n = arrs[0].shape[0]
        if any(arr.shape[0] != n for arr in arrs):
            raise ValueError("All arrays must have the same first dimension.")

        holdout = val_fraction + test_fraction

        if val_fraction > 0 and test_fraction > 0:
            split = train_test_split(*arrs, test_size=holdout, random_state=random_state, shuffle=True)
            train = [split[2 * i] for i in range(len(arrs))]
            temp = [split[2 * i + 1] for i in range(len(arrs))]

            temp_split = train_test_split(
                *temp,
                test_size=test_fraction / holdout,
                random_state=random_state,
                shuffle=True,
            )
            val = [temp_split[2 * i] for i in range(len(arrs))]
            test = [temp_split[2 * i + 1] for i in range(len(arrs))]
            return train, val, test

        if val_fraction > 0:
            split = train_test_split(*arrs, test_size=val_fraction, random_state=random_state, shuffle=True)
            train = [split[2 * i] for i in range(len(arrs))]
            val = [split[2 * i + 1] for i in range(len(arrs))]
            test = [_empty_like(arr) for arr in arrs]
            return train, val, test

        split = train_test_split(*arrs, test_size=test_fraction, random_state=random_state, shuffle=True)
        train = [split[2 * i] for i in range(len(arrs))]
        test = [split[2 * i + 1] for i in range(len(arrs))]
        val = [_empty_like(arr) for arr in arrs]
        return train, val, test

    idx_e, val_e, test_e = _split([data.X_e, data.A_e, data.S_e])
    idx_o, val_o, test_o = _split([data.X_o, data.S_o, data.Y_o])

    train_split = TwoSampleDataSplit(
        X_e=idx_e[0],
        A_e=idx_e[1],
        S_e=idx_e[2],
        X_o=idx_o[0],
        S_o=idx_o[1],
        Y_o=idx_o[2],
    )

    val_split = TwoSampleDataSplit(
        X_e=val_e[0],
        A_e=val_e[1],
        S_e=val_e[2],
        X_o=val_o[0],
        S_o=val_o[1],
        Y_o=val_o[2],
    )

    test_split = TwoSampleDataSplit(
        X_e=test_e[0],
        A_e=test_e[1],
        S_e=test_e[2],
        X_o=test_o[0],
        S_o=test_o[1],
        Y_o=test_o[2],
    )

    return train_split, val_split, test_split
