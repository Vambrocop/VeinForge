"""Stress phenotyping from vein traits (P2-b).

Two tiers, both operating on the trait table VeinForge already produces:
  - `compare_groups`  — NO training. Per-trait group means + a significance test
    (control vs stress). Pure statistics; needs only the core deps.
  - `train_stress_classifier` — a lightweight RandomForest that learns to predict
    the stress label from vein traits. Needs the [stress] extra (scikit-learn).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats

FEATURE_COLUMNS = [
    "vein_density", "mean_vein_width_um", "free_ending_density",
    "areole_count", "areole_mean_area_um2", "interveinal_distance_um",
    "vein_area_fraction",
]


def _feature_cols(df: pd.DataFrame, features) -> list[str]:
    return [c for c in (features or FEATURE_COLUMNS) if c in df.columns]


def compare_groups(df: pd.DataFrame, label_col: str = "treatment", features=None) -> pd.DataFrame:
    """No-training analysis: per-trait group means, difference, Mann-Whitney p.

    Expects exactly two groups in `label_col` (e.g. control vs heat).
    """
    cols = _feature_cols(df, features)
    groups = list(pd.Series(df[label_col]).dropna().unique())
    if len(groups) != 2:
        raise ValueError(f"compare_groups needs exactly 2 groups in '{label_col}', got {groups}")
    g0, g1 = groups
    a, b = df[df[label_col] == g0], df[df[label_col] == g1]
    rows = []
    for c in cols:
        x, y = a[c].dropna(), b[c].dropna()
        p = stats.mannwhitneyu(x, y, alternative="two-sided").pvalue if len(x) and len(y) else np.nan
        rows.append({"trait": c, f"{g0}_mean": x.mean(), f"{g1}_mean": y.mean(),
                     "difference": y.mean() - x.mean(), "p_value": p})
    return pd.DataFrame(rows)


def train_stress_classifier(df: pd.DataFrame, label_col: str = "treatment", features=None,
                            n_estimators: int = 300, cv: int = 5, random_state: int = 0):
    """Fit a RandomForest predicting the stress label; return (model, metrics)."""
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score
    except ImportError as e:
        raise RuntimeError('scikit-learn not installed. Run: pip install -e ".[stress]"') from e
    cols = _feature_cols(df, features)
    X, y = df[cols].to_numpy(), df[label_col].to_numpy()
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    if len(set(y)) > 1:
        n_splits = max(2, min(cv, int(pd.Series(y).value_counts().min())))
        scores = cross_val_score(model, X, y, cv=n_splits)
    else:
        scores = np.array([np.nan])
    model.fit(X, y)
    metrics = {
        "cv_accuracy_mean": float(np.nanmean(scores)),
        "cv_accuracy_std": float(np.nanstd(scores)),
        "n_samples": int(len(y)),
        "features": cols,
        "feature_importances": dict(zip(cols, model.feature_importances_)),
    }
    return model, metrics


def predict_stress(model, df: pd.DataFrame, features=None) -> np.ndarray:
    return model.predict(df[_feature_cols(df, features)].to_numpy())


def save_model(model, path) -> None:
    import joblib
    joblib.dump(model, path)


def load_model(path):
    import joblib
    return joblib.load(path)
