#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


TARGET_VISC = "Delta Kin. Viscosity KV100 - relative | - Daimler Oxidation Test (DOT), %"
TARGET_OXID = "Oxidation EOT | DIN 51453 Daimler Oxidation Test (DOT), A/cm"
SCENARIO_COL = "scenario_id"

TARGET_VISC_CANDIDATES = [
    TARGET_VISC,
    "target_viscosity",
    "target_viscosity_delta",
    "target_viscosity_delta_pct",
    "viscosity_delta",
    "delta_viscosity",
    "delta_kin_viscosity",
    "delta_kin_viscosity_kv100",
    "y_viscosity",
    "target_0",
]

TARGET_OXID_CANDIDATES = [
    TARGET_OXID,
    "target_oxidation",
    "target_oxidation_acm",
    "oxidation",
    "oxidation_eot",
    "y_oxidation",
    "target_1",
]


def detect_sep(path: Path) -> str:
    sample = path.read_text(encoding="utf-8", errors="ignore")[:5000]
    for sep in [",", ";", "\t"]:
        if sample.count(sep) > 5:
            return sep
    return ","


def read_csv_auto(path: Path) -> pd.DataFrame:
    sep = detect_sep(path)
    try:
        return pd.read_csv(path, sep=sep)
    except Exception:
        return pd.read_csv(path)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    metrics["rmse_viscosity"] = float(math.sqrt(mean_squared_error(y_true[:, 0], y_pred[:, 0])))
    metrics["rmse_oxidation"] = float(math.sqrt(mean_squared_error(y_true[:, 1], y_pred[:, 1])))
    metrics["mae_viscosity"] = float(mean_absolute_error(y_true[:, 0], y_pred[:, 0]))
    metrics["mae_oxidation"] = float(mean_absolute_error(y_true[:, 1], y_pred[:, 1]))
    metrics["r2_viscosity"] = float(r2_score(y_true[:, 0], y_pred[:, 0]))
    metrics["r2_oxidation"] = float(r2_score(y_true[:, 1], y_pred[:, 1]))
    metrics["mean_rmse"] = float((metrics["rmse_viscosity"] + metrics["rmse_oxidation"]) / 2.0)
    metrics["mean_mae"] = float((metrics["mae_viscosity"] + metrics["mae_oxidation"]) / 2.0)
    metrics["mean_r2"] = float((metrics["r2_viscosity"] + metrics["r2_oxidation"]) / 2.0)
    return metrics


def summarize_cv_results(cv_results: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "rmse_viscosity",
        "rmse_oxidation",
        "mae_viscosity",
        "mae_oxidation",
        "r2_viscosity",
        "r2_oxidation",
        "mean_rmse",
        "mean_mae",
        "mean_r2",
    ]
    return (
        cv_results.groupby(["config_id", "hidden_layers", "alpha", "learning_rate_init"], as_index=False)[metric_cols]
        .mean()
        .sort_values(["mean_rmse", "mean_mae", "mean_r2"], ascending=[True, True, False])
        .reset_index(drop=True)
    )


def resolve_target_columns(df: pd.DataFrame) -> Tuple[str, str]:
    def find_first(candidates: List[str]) -> str | None:
        lower_to_actual = {str(col).strip().lower(): col for col in df.columns}
        for candidate in candidates:
            actual = lower_to_actual.get(candidate.strip().lower())
            if actual is not None:
                return str(actual)

        all_columns_lower = list(lower_to_actual.keys())

        for col_lower in all_columns_lower:
            if "viscosity" in col_lower and ("target" in col_lower or col_lower.startswith("y_")):
                return str(lower_to_actual[col_lower])

        for col_lower in all_columns_lower:
            if "oxid" in col_lower and ("target" in col_lower or col_lower.startswith("y_")):
                return str(lower_to_actual[col_lower])

        return None

    target_visc_col = find_first(TARGET_VISC_CANDIDATES)
    target_oxid_col = find_first(TARGET_OXID_CANDIDATES)

    if target_visc_col is None or target_oxid_col is None:
        raise ValueError(
            "Не удалось найти target-колонки в train-файле.\n"
            f"Искали viscosity среди: {TARGET_VISC_CANDIDATES}\n"
            f"Искали oxidation среди: {TARGET_OXID_CANDIDATES}\n"
            f"Фактические колонки файла: {list(df.columns)}"
        )

    return target_visc_col, target_oxid_col


def select_numeric_feature_columns(df: pd.DataFrame) -> List[str]:
    forbidden = set(TARGET_VISC_CANDIDATES + TARGET_OXID_CANDIDATES)
    if SCENARIO_COL in df.columns:
        forbidden.add(SCENARIO_COL)

    feature_cols = [
        col
        for col in df.columns
        if col not in forbidden and pd.api.types.is_numeric_dtype(df[col])
    ]

    if not feature_cols:
        raise ValueError("Не найдено ни одной числовой feature-колонки.")

    return feature_cols


def drop_all_nan_columns(X: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    keep_cols = [col for col in X.columns if X[col].notna().any()]
    dropped = [col for col in X.columns if col not in keep_cols]
    return X[keep_cols].copy(), dropped


def drop_sparse_columns(X: pd.DataFrame, min_non_na_ratio: float) -> Tuple[pd.DataFrame, List[str]]:
    threshold = int(math.ceil(len(X) * min_non_na_ratio))
    keep_cols = [col for col in X.columns if X[col].notna().sum() >= threshold]
    dropped = [col for col in X.columns if col not in keep_cols]
    return X[keep_cols].copy(), dropped


def drop_low_variance_columns(X: pd.DataFrame, variance_threshold: float) -> Tuple[pd.DataFrame, List[str]]:
    if X.empty:
        return X.copy(), []

    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)

    selector = VarianceThreshold(threshold=variance_threshold)
    selector.fit(X_imputed)

    keep_mask = selector.get_support()
    keep_cols = list(X.columns[keep_mask])
    dropped = [col for col in X.columns if col not in keep_cols]

    return X[keep_cols].copy(), dropped


def rank_features_by_target_correlation(X: pd.DataFrame, y: np.ndarray, max_features: int) -> List[str]:
    if X.shape[1] <= max_features:
        return list(X.columns)

    X_num = X.copy().apply(pd.to_numeric, errors="coerce")
    y_visc = pd.Series(y[:, 0], index=X_num.index)
    y_oxid = pd.Series(y[:, 1], index=X_num.index)

    scores: Dict[str, float] = {}
    for col in X_num.columns:
        s = X_num[col]
        corr_visc = s.corr(y_visc)
        corr_oxid = s.corr(y_oxid)

        score = 0.0
        if pd.notna(corr_visc):
            score += abs(float(corr_visc))
        if pd.notna(corr_oxid):
            score += abs(float(corr_oxid))
        scores[col] = score

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [name for name, _ in ranked[:max_features]]


def compact_feature_selection(
    X: pd.DataFrame,
    y: np.ndarray,
    min_non_na_ratio: float,
    variance_threshold: float,
    max_features: int,
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    report: Dict[str, List[str]] = {}

    X1, dropped_all_nan = drop_all_nan_columns(X)
    report["dropped_all_nan"] = dropped_all_nan

    X2, dropped_sparse = drop_sparse_columns(X1, min_non_na_ratio=min_non_na_ratio)
    report["dropped_sparse"] = dropped_sparse

    X3, dropped_low_variance = drop_low_variance_columns(X2, variance_threshold=variance_threshold)
    report["dropped_low_variance"] = dropped_low_variance

    selected_ranked = rank_features_by_target_correlation(X3, y, max_features=max_features)
    dropped_by_ranking = [col for col in X3.columns if col not in selected_ranked]
    report["dropped_by_ranking"] = dropped_by_ranking

    X_final = X3[selected_ranked].copy()
    return X_final, report


def build_mlp_configs() -> List[Dict[str, object]]:
    return [
        {"config_id": "mlp_1", "hidden_layers": (64, 32), "alpha": 1e-4, "learning_rate_init": 1e-3},
        {"config_id": "mlp_2", "hidden_layers": (128, 64), "alpha": 1e-4, "learning_rate_init": 1e-3},
        {"config_id": "mlp_3", "hidden_layers": (128, 64), "alpha": 1e-3, "learning_rate_init": 1e-3},
        {"config_id": "mlp_4", "hidden_layers": (128, 128), "alpha": 1e-3, "learning_rate_init": 5e-4},
        {"config_id": "mlp_5", "hidden_layers": (256, 128), "alpha": 1e-3, "learning_rate_init": 5e-4},
    ]


def fit_predict_mlp(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_valid: pd.DataFrame,
    hidden_layers: Tuple[int, ...],
    alpha: float,
    learning_rate_init: float,
    random_state: int,
    max_iter: int,
) -> np.ndarray:
    x_imputer = SimpleImputer(strategy="median")
    X_train_imp = x_imputer.fit_transform(X_train)
    X_valid_imp = x_imputer.transform(X_valid)

    x_scaler = StandardScaler()
    X_train_scaled = x_scaler.fit_transform(X_train_imp)
    X_valid_scaled = x_scaler.transform(X_valid_imp)

    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train)

    model = MLPRegressor(
        hidden_layer_sizes=hidden_layers,
        activation="relu",
        solver="adam",
        alpha=alpha,
        batch_size="auto",
        learning_rate_init=learning_rate_init,
        max_iter=max_iter,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=30,
        random_state=random_state,
    )
    model.fit(X_train_scaled, y_train_scaled)

    y_pred_scaled = model.predict(X_valid_scaled)
    if y_pred_scaled.ndim == 1:
        y_pred_scaled = y_pred_scaled.reshape(-1, 1)

    y_pred = y_scaler.inverse_transform(y_pred_scaled)
    return y_pred


def evaluate_mlp_cv(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray | None,
    configs: List[Dict[str, object]],
    n_splits: int,
    random_state: int,
    max_iter: int,
) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []

    if groups is not None:
        unique_groups = np.unique(groups)
        if len(unique_groups) < n_splits:
            raise ValueError(
                f"Недостаточно уникальных групп ({len(unique_groups)}) для GroupKFold с n_splits={n_splits}."
            )

    for config in configs:
        if groups is not None:
            split_iter = GroupKFold(n_splits=n_splits).split(X, y, groups)
        else:
            split_iter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state).split(X, y)

        fold_idx = 0
        for train_idx, valid_idx in split_iter:
            fold_idx += 1

            X_train = X.iloc[train_idx]
            X_valid = X.iloc[valid_idx]
            y_train = y[train_idx]
            y_valid = y[valid_idx]

            y_pred = fit_predict_mlp(
                X_train=X_train,
                y_train=y_train,
                X_valid=X_valid,
                hidden_layers=tuple(config["hidden_layers"]),
                alpha=float(config["alpha"]),
                learning_rate_init=float(config["learning_rate_init"]),
                random_state=random_state,
                max_iter=max_iter,
            )

            fold_metrics = compute_metrics(y_valid, y_pred)
            fold_metrics["config_id"] = str(config["config_id"])
            fold_metrics["hidden_layers"] = str(tuple(config["hidden_layers"]))
            fold_metrics["alpha"] = float(config["alpha"])
            fold_metrics["learning_rate_init"] = float(config["learning_rate_init"])
            fold_metrics["fold"] = fold_idx
            rows.append(fold_metrics)

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Physics-informed MLP on prepared compact flat features."
    )
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, default=Path("mlp_flat_compact_output"))
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--min-non-na-ratio", type=float, default=0.10)
    parser.add_argument("--variance-threshold", type=float, default=1e-12)
    parser.add_argument("--max-features", type=int, default=120)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    print("RUNNING FILE:", Path(__file__).resolve())
    print("INPUT FILE:", args.train.resolve())

    df = read_csv_auto(args.train)

    target_visc_col, target_oxid_col = resolve_target_columns(df)

    print("\n=== Resolved target columns ===")
    print("viscosity target:", target_visc_col)
    print("oxidation target:", target_oxid_col)

    raw_feature_cols = select_numeric_feature_columns(df)
    print("\nRaw numeric feature count:", len(raw_feature_cols))

    X_raw = df[raw_feature_cols].copy()
    y = df[[target_visc_col, target_oxid_col]].to_numpy(dtype=float)

    X, feature_report = compact_feature_selection(
        X=X_raw,
        y=y,
        min_non_na_ratio=args.min_non_na_ratio,
        variance_threshold=args.variance_threshold,
        max_features=args.max_features,
    )
    feature_cols = list(X.columns)

    print("\n=== Compact feature selection summary ===")
    print("dropped_all_nan:", len(feature_report["dropped_all_nan"]))
    print("dropped_sparse:", len(feature_report["dropped_sparse"]))
    print("dropped_low_variance:", len(feature_report["dropped_low_variance"]))
    print("dropped_by_ranking:", len(feature_report["dropped_by_ranking"]))
    print("final_feature_count:", len(feature_cols))

    groups = None
    if SCENARIO_COL in df.columns:
        groups = df[SCENARIO_COL].astype(str).to_numpy()
        print(f"\nUsing GroupKFold by '{SCENARIO_COL}'")
    else:
        print("\n'scenario_id' not found, using plain KFold")

    configs = build_mlp_configs()

    print("\n=== Candidate MLP configs ===")
    for config in configs:
        print(config)

    cv_results = evaluate_mlp_cv(
        X=X,
        y=y,
        groups=groups,
        configs=configs,
        n_splits=args.n_splits,
        random_state=args.random_state,
        max_iter=args.max_iter,
    )
    cv_summary = summarize_cv_results(cv_results)
    best_config_id = str(cv_summary.iloc[0]["config_id"])
    best_config = next(config for config in configs if str(config["config_id"]) == best_config_id)

    cv_results_path = args.outdir / "mlp_flat_compact_cv_fold_metrics.csv"
    cv_summary_path = args.outdir / "mlp_flat_compact_cv_summary.csv"
    feature_cols_path = args.outdir / "mlp_flat_compact_feature_columns.json"
    manifest_path = args.outdir / "mlp_flat_compact_run_manifest.json"
    dropped_report_path = args.outdir / "mlp_flat_compact_dropped_columns.json"

    cv_results.to_csv(cv_results_path, index=False)
    cv_summary.to_csv(cv_summary_path, index=False)
    feature_cols_path.write_text(json.dumps(feature_cols, ensure_ascii=False, indent=2), encoding="utf-8")
    dropped_report_path.write_text(json.dumps(feature_report, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest = {
        "input_train_path": str(args.train),
        "n_splits": args.n_splits,
        "random_state": args.random_state,
        "max_iter": args.max_iter,
        "best_config": best_config,
        "raw_feature_count": len(raw_feature_cols),
        "final_feature_count": len(feature_cols),
        "min_non_na_ratio": args.min_non_na_ratio,
        "variance_threshold": args.variance_threshold,
        "max_features": args.max_features,
        "uses_groupkfold": groups is not None,
        "feature_columns_file": str(feature_cols_path),
        "dropped_columns_file": str(dropped_report_path),
        "outputs": {
            "cv_fold_metrics": str(cv_results_path),
            "cv_summary": str(cv_summary_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== CV fold metrics ===")
    print(cv_results.to_string(index=False))

    print("\n=== CV summary by config ===")
    print(cv_summary.to_string(index=False))

    print("\nBest config:", best_config)
    print(f"Raw feature count: {len(raw_feature_cols)}")
    print(f"Final feature count: {len(feature_cols)}")
    print(f"Artifacts saved to: {args.outdir.resolve()}")


if __name__ == "__main__":
    main()