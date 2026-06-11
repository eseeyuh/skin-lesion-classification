import numpy as np
import pandas as pd
import pytest

from src.data import group_split


@pytest.fixture
def synthetic_df():
    """200 lesions, 1-4 images each, 7 classes — enough to split meaningfully."""
    rng = np.random.default_rng(0)
    rows = []
    classes = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
    for lesion in range(200):
        dx = classes[lesion % 7]
        for img in range(rng.integers(1, 5)):
            rows.append({"lesion_id": f"L{lesion:04d}",
                         "image_id": f"L{lesion:04d}_{img}",
                         "dx": dx, "label": classes.index(dx)})
    return pd.DataFrame(rows)


def test_no_lesion_leakage_across_splits(synthetic_df, cfg):
    """The same lesion must never appear in more than one split."""
    df_train, df_val, df_test = group_split(synthetic_df, cfg)
    train, val, test = (set(d["lesion_id"]) for d in (df_train, df_val, df_test))
    assert not (train & val)
    assert not (train & test)
    assert not (val & test)


def test_split_covers_every_row_once(synthetic_df, cfg):
    df_train, df_val, df_test = group_split(synthetic_df, cfg)
    total = len(df_train) + len(df_val) + len(df_test)
    assert total == len(synthetic_df)
    ids = set(df_train["image_id"]) | set(df_val["image_id"]) | set(df_test["image_id"])
    assert ids == set(synthetic_df["image_id"])


def test_split_is_deterministic(synthetic_df, cfg):
    a = group_split(synthetic_df, cfg, seed=123)
    b = group_split(synthetic_df, cfg, seed=123)
    for da, db in zip(a, b):
        assert list(da["image_id"]) == list(db["image_id"])
