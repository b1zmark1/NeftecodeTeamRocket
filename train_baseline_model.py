from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import MultiTaskElasticNetCV, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = ROOT / "artifacts"
MODELS_DIR = ROOT / "models"

TRAIN_NPZ = ARTIFACTS_DIR / "train_flat_features_v1.npz"
TEST_NPZ = ARTIFACTS_DIR / "test_flat_features_v1.npz"

N_SPLITS = 5
N_REPEATS = 5


VISC_SCALE = 50.0


def transform_y(y: np.ndarray) -> np.ndarray:
    out = np.empty_like(y, dtype=np.float64)
    out[:, 0] = np.arcsinh(y[:, 0] / VISC_SCALE)
    out[:, 1] = np.log1p(np.clip(y[:, 1], a_min=0.0, a_max=None))
    return out


def inverse_transform_y(y: np.ndarray) -> np.ndarray:
    out = np.empty_like(y, dtype=np.float64)
    out[:, 0] = VISC_SCALE * np.sinh(y[:, 0])
    out[:, 1] = np.expm1(y[:, 1])
    return out


@dataclass
class Dataset:
    X: np.ndarray
    y: np.ndarray | None
    feature_names: List[str]
    scenario_ids: List[str]


def load_dataset(path: Path, with_targets: bool) -> Dataset:
    data = np.load(path, allow_pickle=True)
    return Dataset(
        X=data["X"].astype(np.float64),
        y=data["y"].astype(np.float64) if with_targets else None,
        feature_names=data["feature_names"].tolist(),
        scenario_ids=data["scenario_ids"].tolist(),
    )


def unsupervised_feature_filter(X: np.ndarray, feature_names: List[str]) -> Tuple[np.ndarray, List[str], np.ndarray]:
    missing_frac = np.isnan(X).mean(axis=0)
    keep_missing = missing_frac <= 0.85

    keep_variance = np.zeros(X.shape[1], dtype=bool)
    for j in range(X.shape[1]):
        col = X[:, j]
        observed = col[~np.isnan(col)]
        if observed.size >= 3 and np.nanstd(observed) > 0:
            keep_variance[j] = True

    keep = keep_missing & keep_variance
    filtered_X = X[:, keep]
    filtered_names = [name for name, flag in zip(feature_names, keep) if flag]
    return filtered_X, filtered_names, keep


def make_models() -> Dict[str, Pipeline]:
    return {
        "ridge_cv": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("scaler", RobustScaler()),
                (
                    "model",
                    MultiOutputRegressor(
                        RidgeCV(alphas=np.array([0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0], dtype=np.float64))
                    ),
                ),
            ]
        ),
        "elastic_cv": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("scaler", RobustScaler()),
                (
                    "model",
                    MultiTaskElasticNetCV(
                        l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
                        alphas=np.array([1e-2, 3e-2, 1e-1, 3e-1], dtype=np.float64),
                        cv=5,
                        max_iter=50000,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }


def metrics_dict(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae_visc = mean_absolute_error(y_true[:, 0], y_pred[:, 0])
    mae_oxid = mean_absolute_error(y_true[:, 1], y_pred[:, 1])
    rmse_visc = math.sqrt(mean_squared_error(y_true[:, 0], y_pred[:, 0]))
    rmse_oxid = math.sqrt(mean_squared_error(y_true[:, 1], y_pred[:, 1]))
    r2_visc = r2_score(y_true[:, 0], y_pred[:, 0])
    r2_oxid = r2_score(y_true[:, 1], y_pred[:, 1])
    iqr_visc = np.subtract(*np.percentile(y_true[:, 0], [75, 25]))
    iqr_oxid = np.subtract(*np.percentile(y_true[:, 1], [75, 25]))
    nmae_visc = mae_visc / iqr_visc if iqr_visc else np.nan
    nmae_oxid = mae_oxid / iqr_oxid if iqr_oxid else np.nan
    return {
        "mae_viscosity": float(mae_visc),
        "mae_oxidation": float(mae_oxid),
        "rmse_viscosity": float(rmse_visc),
        "rmse_oxidation": float(rmse_oxid),
        "r2_viscosity": float(r2_visc),
        "r2_oxidation": float(r2_oxid),
        "nmae_viscosity_iqr": float(nmae_visc),
        "nmae_oxidation_iqr": float(nmae_oxid),
        "mean_nmae_iqr": float(np.nanmean([nmae_visc, nmae_oxid])),
    }


def evaluate_model(name: str, estimator: Pipeline, X: np.ndarray, y: np.ndarray, scenario_ids: List[str]) -> Tuple[dict, pd.DataFrame]:
    rkf = RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=42)
    fold_rows = []
    oof = np.zeros_like(y, dtype=np.float64)
    counts = np.zeros(y.shape[0], dtype=np.int32)

    for fold_idx, (train_idx, valid_idx) in enumerate(rkf.split(X), start=1):
        model = clone(estimator)
        y_train_t = transform_y(y[train_idx])
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            model.fit(X[train_idx], y_train_t)
        pred_valid = inverse_transform_y(model.predict(X[valid_idx]))
        oof[valid_idx] += pred_valid
        counts[valid_idx] += 1

        fold_metric = metrics_dict(y[valid_idx], pred_valid)
        fold_metric["fold"] = fold_idx
        fold_rows.append(fold_metric)

    counts = np.where(counts == 0, 1, counts)
    oof /= counts[:, None]
    summary = metrics_dict(y, oof)
    summary["model_name"] = name
    summary["n_features_input"] = int(X.shape[1])
    summary["n_folds"] = len(fold_rows)

    oof_df = pd.DataFrame(
        {
            "scenario_id": scenario_ids,
            "y_viscosity_true": y[:, 0],
            "y_oxidation_true": y[:, 1],
            "y_viscosity_oof": oof[:, 0],
            "y_oxidation_oof": oof[:, 1],
        }
    )
    return {"summary": summary, "folds": fold_rows}, oof_df


def fit_full_model(estimator: Pipeline, X: np.ndarray, y: np.ndarray):
    model = clone(estimator)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        model.fit(X, transform_y(y))
    return model


def predict_original_scale(model, X: np.ndarray) -> np.ndarray:
    return inverse_transform_y(model.predict(X))


def extract_model_details(fitted_pipeline: Pipeline) -> dict:
    model = fitted_pipeline.named_steps["model"]
    details: dict = {}
    if isinstance(model, MultiOutputRegressor):
        alphas = [float(est.alpha_) for est in model.estimators_ if hasattr(est, "alpha_")]
        details["ridge_alphas_per_target"] = alphas
    elif isinstance(model, MultiTaskElasticNetCV):
        details["elastic_alpha"] = float(model.alpha_)
        details["elastic_l1_ratio"] = float(model.l1_ratio_)
        details["nonzero_coefficients"] = int(np.count_nonzero(model.coef_))
    return details


def main() -> None:
    MODELS_DIR.mkdir(exist_ok=True)

    train = load_dataset(TRAIN_NPZ, with_targets=True)
    test = load_dataset(TEST_NPZ, with_targets=False)

    X_train, filtered_names, keep_mask = unsupervised_feature_filter(train.X, train.feature_names)
    X_test = test.X[:, keep_mask]
    y_train = train.y

    models = make_models()
    summary_rows = []
    model_reports = {}
    oof_paths = {}

    for name, estimator in models.items():
        report, oof_df = evaluate_model(name, estimator, X_train, y_train, train.scenario_ids)
        summary_rows.append(report["summary"])
        model_reports[name] = report
        oof_path = MODELS_DIR / f"{name}_oof_predictions.csv"
        oof_df.to_csv(oof_path, index=False, encoding="utf-8-sig")
        oof_paths[name] = oof_path.name

    summary_df = pd.DataFrame(summary_rows).sort_values(["mean_nmae_iqr", "mae_oxidation", "mae_viscosity"]).reset_index(drop=True)
    best_name = summary_df.iloc[0]["model_name"]
    best_estimator = models[best_name]
    fitted = fit_full_model(best_estimator, X_train, y_train)

    test_pred = predict_original_scale(fitted, X_test)
    test_pred_df = pd.DataFrame(
        {
            "scenario_id": test.scenario_ids,
            "prediction_viscosity_delta_pct": test_pred[:, 0],
            "prediction_oxidation_acm": test_pred[:, 1],
        }
    )
    test_pred_path = MODELS_DIR / "baseline_test_predictions.csv"
    test_pred_df.to_csv(test_pred_path, index=False, encoding="utf-8-sig")

    model_path = MODELS_DIR / "baseline_best_model.joblib"
    joblib.dump(
        {
            "model_name": best_name,
            "feature_names": filtered_names,
            "keep_mask": keep_mask,
            "pipeline": fitted,
        },
        model_path,
    )

    summary_path = MODELS_DIR / "baseline_cv_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    report = {
        "best_model": best_name,
        "selected_feature_count": len(filtered_names),
        "dropped_feature_count": int(len(train.feature_names) - len(filtered_names)),
        "cv_summary": summary_df.to_dict(orient="records"),
        "best_model_details": extract_model_details(fitted),
        "artifacts": {
            "summary_csv": summary_path.name,
            "best_model_joblib": model_path.name,
            "test_predictions_csv": test_pred_path.name,
            "oof_prediction_files": oof_paths,
        },
    }
    (MODELS_DIR / "baseline_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(summary_df.to_string(index=False))
    print()
    print(f"best_model={best_name}")
    print(f"selected_features={len(filtered_names)}")
    print(MODELS_DIR)


if __name__ == "__main__":
    main()
