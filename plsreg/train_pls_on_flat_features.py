#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import Pipeline
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


def build_pipeline(n_components: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", PLSRegression(n_components=n_components, scale=False)),
        ]
    )


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
        cv_results.groupby("n_components", as_index=False)[metric_cols]
        .mean()
        .sort_values(["mean_rmse", "mean_mae", "mean_r2"], ascending=[True, True, False])
        .reset_index(drop=True)
    )


def select_feature_columns(df: pd.DataFrame) -> List[str]:
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


def resolve_target_columns(df: pd.DataFrame) -> Tuple[str, str]:
    def find_first(candidates: List[str]) -> str | None:
        lower_to_actual = {str(col).strip().lower(): col for col in df.columns}
        for candidate in candidates:
            actual = lower_to_actual.get(candidate.strip().lower())
            if actual is not None:
                return actual

        # Fallback: substring heuristics for slightly different flat-feature naming
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

    return str(target_visc_col), str(target_oxid_col)


def evaluate_pls_cv(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray | None,
    n_components_list: Iterable[int],
    n_splits: int,
    random_state: int,
) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []

    if groups is not None:
        unique_groups = np.unique(groups)
        if len(unique_groups) < n_splits:
            raise ValueError(
                f"Недостаточно уникальных групп ({len(unique_groups)}) для GroupKFold с n_splits={n_splits}."
            )
        splitter = GroupKFold(n_splits=n_splits)
        split_iter = splitter.split(X, y, groups)
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        split_iter = splitter.split(X, y)

    max_valid_n_components = min(X.shape[1], X.shape[0] - math.ceil(X.shape[0] / n_splits))
    candidate_components = [n for n in n_components_list if 1 <= n <= max_valid_n_components]
    if not candidate_components:
        raise ValueError("Нет допустимых значений n_components для текущего размера данных.")

    # KFold iterator is exhausted after one pass, so rebuild per n_components
    for n_components in candidate_components:
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

            model = build_pipeline(n_components=n_components)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_valid)

            fold_metrics = compute_metrics(y_valid, y_pred)
            fold_metrics["n_components"] = n_components
            fold_metrics["fold"] = fold_idx
            rows.append(fold_metrics)

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train PLSRegression on already prepared flat features."
    )
    parser.add_argument(
        "--train",
        type=Path,
        required=True,
        help="Path to flat feature train CSV, e.g. train_flat_features_v2.csv",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("pls_flat_output"),
        help="Directory for metrics and artifacts",
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--max-components", type=int, default=12)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    print("RUNNING FILE:", Path(__file__).resolve())
    print("INPUT FILE:", args.train.resolve())

    df = read_csv_auto(args.train)

    target_visc_col, target_oxid_col = resolve_target_columns(df)

    print("\n=== Resolved target columns ===")
    print("viscosity target:", target_visc_col)
    print("oxidation target:", target_oxid_col)

    feature_cols = select_feature_columns(df)

    print("\n=== Selected flat feature columns ===")
    for col in feature_cols:
        print(col)

    X = df[feature_cols].copy()
    y = df[[target_visc_col, target_oxid_col]].to_numpy(dtype=float)

    groups = None
    if SCENARIO_COL in df.columns:
        groups = df[SCENARIO_COL].astype(str).to_numpy()
        print(f"\nUsing GroupKFold by '{SCENARIO_COL}'")
    else:
        print("\n'scenario_id' not found, using plain KFold")

    max_possible = min(args.max_components, len(feature_cols), max(1, X.shape[0] - 1))
    n_components_list = list(range(1, max_possible + 1))

    cv_results = evaluate_pls_cv(
        X=X,
        y=y,
        groups=groups,
        n_components_list=n_components_list,
        n_splits=args.n_splits,
        random_state=args.random_state,
    )
    cv_summary = summarize_cv_results(cv_results)
    best_n_components = int(cv_summary.iloc[0]["n_components"])

    final_model = build_pipeline(best_n_components)
    final_model.fit(X, y)

    cv_results_path = args.outdir / "pls_flat_cv_fold_metrics.csv"
    cv_summary_path = args.outdir / "pls_flat_cv_summary.csv"
    feature_cols_path = args.outdir / "pls_flat_feature_columns.json"
    manifest_path = args.outdir / "pls_flat_run_manifest.json"

    cv_results.to_csv(cv_results_path, index=False)
    cv_summary.to_csv(cv_summary_path, index=False)
    feature_cols_path.write_text(
        json.dumps(feature_cols, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    manifest = {
        "input_train_path": str(args.train),
        "n_splits": args.n_splits,
        "random_state": args.random_state,
        "tested_n_components": n_components_list,
        "best_n_components": best_n_components,
        "n_features": len(feature_cols),
        "feature_columns_file": str(feature_cols_path),
        "uses_groupkfold": groups is not None,
        "outputs": {
            "cv_fold_metrics": str(cv_results_path),
            "cv_summary": str(cv_summary_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== CV fold metrics ===")
    print(cv_results.to_string(index=False))

    print("\n=== CV summary by n_components ===")
    print(cv_summary.to_string(index=False))

    print(f"\nBest n_components: {best_n_components}")
    print(f"Number of features: {len(feature_cols)}")
    print(f"Artifacts saved to: {args.outdir.resolve()}")


if __name__ == "__main__":
    main()