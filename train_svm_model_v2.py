from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import LinearSVR, SVR


ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = ROOT / "artifacts"
MODELS_DIR = ROOT / "models"

TRAIN_CSV = ARTIFACTS_DIR / "train_flat_features_v2.csv"
TEST_CSV = ARTIFACTS_DIR / "test_flat_features_v2.csv"

N_SPLITS = 5
N_REPEATS = 5
VISC_SCALE = 50.0
VISC_TRANSFORM_CLIP = (-6.0, 6.0)
OXID_TRANSFORM_CLIP = (0.0, 6.0)


def transform_y(y: np.ndarray) -> np.ndarray:
    out = np.empty_like(y, dtype=np.float64)
    out[:, 0] = np.arcsinh(y[:, 0] / VISC_SCALE)
    out[:, 1] = np.log1p(np.clip(y[:, 1], a_min=0.0, a_max=None))
    return out


def inverse_transform_y(y: np.ndarray) -> np.ndarray:
    out = np.empty_like(y, dtype=np.float64)
    visc_t = np.clip(y[:, 0], VISC_TRANSFORM_CLIP[0], VISC_TRANSFORM_CLIP[1])
    oxid_t = np.clip(y[:, 1], OXID_TRANSFORM_CLIP[0], OXID_TRANSFORM_CLIP[1])
    out[:, 0] = VISC_SCALE * np.sinh(visc_t)
    out[:, 1] = np.expm1(oxid_t)
    return out


@dataclass
class Dataset:
    X: np.ndarray
    y: np.ndarray | None
    feature_names: List[str]
    scenario_ids: List[str]


def load_csv_dataset(path: Path, with_targets: bool) -> Dataset:
    df = pd.read_csv(path)
    scenario_ids = df["scenario_id"].astype(str).tolist()

    target_cols = ["target_viscosity_delta_pct", "target_oxidation_acm"]
    feature_cols = [col for col in df.columns if col != "scenario_id" and col not in target_cols]
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)

    y = None
    if with_targets:
        y = df[target_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)

    return Dataset(X=X, y=y, feature_names=feature_cols, scenario_ids=scenario_ids)


def unsupervised_feature_filter(
    X: np.ndarray,
    feature_names: List[str],
    max_missing_frac: float = 0.85,
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    missing_frac = np.isnan(X).mean(axis=0)
    keep_missing = missing_frac <= max_missing_frac

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
        "linear_svr": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("scaler", RobustScaler()),
                (
                    "model",
                    MultiOutputRegressor(
                        LinearSVR(
                            C=1.0,
                            epsilon=0.1,
                            loss="epsilon_insensitive",
                            dual="auto",
                            max_iter=20000,
                            random_state=42,
                        )
                    ),
                ),
            ]
        ),
        "rbf_svr": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("scaler", RobustScaler()),
                (
                    "model",
                    MultiOutputRegressor(
                        SVR(
                            kernel="rbf",
                            C=3.0,
                            epsilon=0.1,
                            gamma="scale",
                        )
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


def fit_full_model(estimator: Pipeline, X: np.ndarray, y: np.ndarray) -> Pipeline:
    model = clone(estimator)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        model.fit(X, transform_y(y))
    return model


def predict_original_scale(model: Pipeline, X: np.ndarray) -> np.ndarray:
    return inverse_transform_y(model.predict(X))


def extract_model_details(fitted_pipeline: Pipeline) -> dict:
    base = fitted_pipeline.named_steps["model"]
    details: dict = {"base_estimator_type": type(base.estimator).__name__}

    if hasattr(base.estimator, "kernel"):
        details["kernel"] = base.estimator.kernel
        details["C"] = float(base.estimator.C)
        details["epsilon"] = float(base.estimator.epsilon)
    elif isinstance(base.estimator, LinearSVR):
        details["C"] = float(base.estimator.C)
        details["epsilon"] = float(base.estimator.epsilon)

    return details


def main() -> None:
    MODELS_DIR.mkdir(exist_ok=True)

    train = load_csv_dataset(TRAIN_CSV, with_targets=True)
    test = load_csv_dataset(TEST_CSV, with_targets=False)

    X_train, filtered_names, keep_mask = unsupervised_feature_filter(train.X, train.feature_names)
    X_test = test.X[:, keep_mask]
    y_train = train.y
    if y_train is None:
        raise ValueError("Targets are required for training.")

    models = make_models()
    summary_rows = []
    model_reports = {}
    oof_paths = {}

    for name, estimator in models.items():
        print(f"[svm] evaluating {name} on {X_train.shape[0]} rows x {X_train.shape[1]} features")
        report, oof_df = evaluate_model(name, estimator, X_train, y_train, train.scenario_ids)
        summary_rows.append(report["summary"])
        model_reports[name] = report
        oof_path = MODELS_DIR / f"{name}_v2_oof_predictions.csv"
        oof_df.to_csv(oof_path, index=False, encoding="utf-8-sig")
        oof_paths[name] = oof_path.name

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["mean_nmae_iqr", "mae_oxidation", "mae_viscosity"]
    ).reset_index(drop=True)
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
    test_pred_path = MODELS_DIR / "svm_v2_test_predictions.csv"
    test_pred_df.to_csv(test_pred_path, index=False, encoding="utf-8-sig")

    model_path = MODELS_DIR / "svm_v2_best_model.joblib"
    joblib.dump(
        {
            "model_name": best_name,
            "feature_names": filtered_names,
            "keep_mask": keep_mask,
            "pipeline": fitted,
        },
        model_path,
    )

    summary_path = MODELS_DIR / "svm_v2_cv_summary.csv"
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
    report_path = MODELS_DIR / "svm_v2_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[svm] best_model={best_name}")
    print(f"[svm] selected_features={len(filtered_names)}")
    print(f"[svm] summary={summary_path.name}")


if __name__ == "__main__":
    main()
