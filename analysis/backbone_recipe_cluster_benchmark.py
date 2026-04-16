#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cross_decomposition import PLSRegression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mlp.train_mlp_targetwise_compact_v3 import (
    SCENARIO_COL,
    add_nonlinear_features,
    build_target_configs,
    fit_predict_single_target_mlp,
    read_csv_auto,
    select_features_for_target,
    select_numeric_feature_columns,
)


OUTDIR = ROOT / "analysis" / "backbone_recipe_cluster"
TRAIN_COMPONENTS = ROOT / "new_datasets" / "train_component_level_transformed.csv"

VISC_TARGET_CANDIDATES = [
    "target_viscosity_delta_pct",
    "Delta Kin. Viscosity KV100 - relative | - Daimler Oxidation Test (DOT), %",
]
OX_TARGET_CANDIDATES = [
    "target_oxidation_acm",
    "Oxidation EOT | DIN 51453 Daimler Oxidation Test (DOT), A/cm",
]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def detect_target_column(df: pd.DataFrame, target_name: str) -> str:
    candidates = VISC_TARGET_CANDIDATES if target_name == "viscosity" else OX_TARGET_CANDIDATES
    lower_to_actual = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        actual = lower_to_actual.get(cand.strip().lower())
        if actual is not None:
            return str(actual)
    raise KeyError(f"Target column not found for {target_name} in {list(df.columns)}")


def build_recipe_cluster_groups(component_df: pd.DataFrame) -> Dict[str, str]:
    comp = component_df.copy()
    comp["Массовая доля, %"] = pd.to_numeric(comp["Массовая доля, %"], errors="coerce").fillna(0.0)
    pivot = comp.pivot_table(index=SCENARIO_COL, columns="Компонент", values="Массовая доля, %", aggfunc="sum", fill_value=0.0)
    km = KMeans(n_clusters=8, random_state=42, n_init=20)
    labels = km.fit_predict(pivot)
    return {str(idx): f"cluster_{label}" for idx, label in zip(pivot.index, labels)}


def prepare_dataset(dataset_name: str, path: Path, target_name: str) -> Tuple[pd.DataFrame, np.ndarray]:
    df = read_csv_auto(path)
    target_col = detect_target_column(df, target_name)

    if dataset_name == "compact_v3_full":
        raw_cols = [c for c in select_numeric_feature_columns(df) if c != target_col]
        X_raw = df[raw_cols].copy()
        X = add_nonlinear_features(X_raw)
    else:
        forbidden = {SCENARIO_COL, target_col}
        X = df[[c for c in df.columns if c not in forbidden and pd.api.types.is_numeric_dtype(df[c])]].copy()

    X.index = df[SCENARIO_COL].astype(str)
    y = pd.to_numeric(df[target_col], errors="coerce").to_numpy(dtype=float)
    return X, y


def evaluate_mlp(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    target_name: str,
) -> Tuple[Dict[str, object], pd.DataFrame]:
    configs = build_target_configs()
    rows: List[Dict[str, object]] = []
    best = None
    best_rmse = float("inf")

    for config in configs:
        oof = np.zeros_like(y, dtype=float)
        split_iter = GroupKFold(n_splits=5).split(X, y, groups)
        for fold, (train_idx, valid_idx) in enumerate(split_iter, start=1):
            X_train_full = X.iloc[train_idx]
            X_valid_full = X.iloc[valid_idx]
            y_train = y[train_idx]
            y_valid = y[valid_idx]
            X_train, X_valid, _report, selected = select_features_for_target(
                X_train_full,
                X_valid_full,
                y_train=y_train,
                target_name=target_name,
                min_non_na_ratio=0.08,
                variance_threshold=1e-8,
                max_features=20,
            )
            y_pred = fit_predict_single_target_mlp(
                X_train=X_train,
                y_train=y_train,
                X_valid=X_valid,
                target_name=target_name,
                config=config,
                random_state=42,
                max_iter=700,
            )
            oof[valid_idx] = y_pred
            rows.append(
                {
                    "model_type": "mlp",
                    "config_id": str(config["config_id"]),
                    "fold": fold,
                    "n_features": len(selected),
                    **compute_metrics(y_valid, y_pred),
                }
            )
        metrics = compute_metrics(y, oof)
        summary = {
            "model_type": "mlp",
            "config_id": str(config["config_id"]),
            "oof_pred": oof,
            **metrics,
        }
        if metrics["rmse"] < best_rmse:
            best_rmse = metrics["rmse"]
            best = summary

    assert best is not None
    return best, pd.DataFrame(rows)


def evaluate_ridge(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
) -> Tuple[Dict[str, object], pd.DataFrame]:
    alphas = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
    rows: List[Dict[str, object]] = []
    best = None
    best_rmse = float("inf")

    for alpha in alphas:
        oof = np.zeros_like(y, dtype=float)
        for fold, (train_idx, valid_idx) in enumerate(GroupKFold(n_splits=5).split(X, y, groups), start=1):
            X_train = X.iloc[train_idx]
            X_valid = X.iloc[valid_idx]
            y_train = y[train_idx]
            y_valid = y[valid_idx]

            imputer = SimpleImputer(strategy="median")
            scaler = StandardScaler()
            X_train_imp = imputer.fit_transform(X_train)
            X_valid_imp = imputer.transform(X_valid)
            X_train_scaled = scaler.fit_transform(X_train_imp)
            X_valid_scaled = scaler.transform(X_valid_imp)

            model = Ridge(alpha=alpha, random_state=42)
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_valid_scaled)
            oof[valid_idx] = y_pred
            rows.append({"model_type": "ridge", "config_id": f"alpha_{alpha}", "fold": fold, **compute_metrics(y_valid, y_pred)})
        metrics = compute_metrics(y, oof)
        summary = {"model_type": "ridge", "config_id": f"alpha_{alpha}", "oof_pred": oof, **metrics}
        if metrics["rmse"] < best_rmse:
            best_rmse = metrics["rmse"]
            best = summary

    assert best is not None
    return best, pd.DataFrame(rows)


def evaluate_pls(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
) -> Tuple[Dict[str, object], pd.DataFrame]:
    rows: List[Dict[str, object]] = []
    best = None
    best_rmse = float("inf")

    max_components = min(12, X.shape[1], max(2, int(len(X) * 0.6)))
    for n_components in range(2, max_components + 1):
        oof = np.zeros_like(y, dtype=float)
        ok = True
        for fold, (train_idx, valid_idx) in enumerate(GroupKFold(n_splits=5).split(X, y, groups), start=1):
            X_train = X.iloc[train_idx]
            X_valid = X.iloc[valid_idx]
            y_train = y[train_idx]
            y_valid = y[valid_idx]

            if n_components >= min(len(train_idx), X_train.shape[1]):
                ok = False
                break

            imputer = SimpleImputer(strategy="median")
            scaler = StandardScaler()
            X_train_imp = imputer.fit_transform(X_train)
            X_valid_imp = imputer.transform(X_valid)
            X_train_scaled = scaler.fit_transform(X_train_imp)
            X_valid_scaled = scaler.transform(X_valid_imp)

            model = PLSRegression(n_components=n_components, scale=False)
            model.fit(X_train_scaled, y_train.reshape(-1, 1))
            y_pred = model.predict(X_valid_scaled).ravel()
            oof[valid_idx] = y_pred
            rows.append({"model_type": "pls", "config_id": f"ncomp_{n_components}", "fold": fold, **compute_metrics(y_valid, y_pred)})

        if not ok:
            continue
        metrics = compute_metrics(y, oof)
        summary = {"model_type": "pls", "config_id": f"ncomp_{n_components}", "oof_pred": oof, **metrics}
        if metrics["rmse"] < best_rmse:
            best_rmse = metrics["rmse"]
            best = summary

    assert best is not None
    return best, pd.DataFrame(rows)


def benchmark_target(target_name: str, candidates: List[Tuple[str, Path]], group_map: Dict[str, str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    fold_frames = []
    for dataset_name, path in candidates:
        X, y = prepare_dataset(dataset_name, path, target_name)
        groups = np.array([group_map[sid] for sid in X.index], dtype=object)

        best_mlp, mlp_fold = evaluate_mlp(X, y, groups, target_name)
        best_ridge, ridge_fold = evaluate_ridge(X, y, groups)
        best_pls, pls_fold = evaluate_pls(X, y, groups)

        for result in [best_mlp, best_ridge, best_pls]:
            summary_rows.append(
                {
                    "target_name": target_name,
                    "dataset_name": dataset_name,
                    "model_type": result["model_type"],
                    "config_id": result["config_id"],
                    "rmse": result["rmse"],
                    "mae": result["mae"],
                    "r2": result["r2"],
                }
            )

        for frame in [mlp_fold, ridge_fold, pls_fold]:
            frame = frame.copy()
            frame["target_name"] = target_name
            frame["dataset_name"] = dataset_name
            fold_frames.append(frame)

    summary_df = pd.DataFrame(summary_rows).sort_values(["target_name", "rmse"]).reset_index(drop=True)
    fold_df = pd.concat(fold_frames, ignore_index=True)
    return summary_df, fold_df


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    component_df = pd.read_csv(TRAIN_COMPONENTS)
    group_map = build_recipe_cluster_groups(component_df)

    viscosity_candidates = [
        ("compact_v3_full", ROOT / "compact_v3" / "train_flat_features_v3_compact.csv"),
        ("viscosity_focus", ROOT / "compact_v3" / "feature_ablation" / "targetwise_feature_sets" / "viscosity_focus.csv"),
    ]
    oxidation_candidates = [
        ("compact_v3_full", ROOT / "compact_v3" / "train_flat_features_v3_compact.csv"),
        ("oxidation_focus_v2", ROOT / "compact_v3" / "oxidation_enhancement" / "oxidation_focus_v2_train.csv"),
    ]

    visc_summary, visc_fold = benchmark_target("viscosity", viscosity_candidates, group_map)
    ox_summary, ox_fold = benchmark_target("oxidation", oxidation_candidates, group_map)

    summary_df = pd.concat([visc_summary, ox_summary], ignore_index=True)
    fold_df = pd.concat([visc_fold, ox_fold], ignore_index=True)

    selected_backbones = (
        summary_df.sort_values(["target_name", "rmse"])
        .groupby("target_name", as_index=False)
        .first()
    )

    summary_df.to_csv(OUTDIR / "recipe_cluster_backbone_summary.csv", index=False)
    fold_df.to_csv(OUTDIR / "recipe_cluster_backbone_fold_metrics.csv", index=False)
    selected_backbones.to_csv(OUTDIR / "selected_backbones.csv", index=False)

    manifest = {
        "grouping": "recipe_cluster",
        "train_components": str(TRAIN_COMPONENTS),
        "candidates": {
            "viscosity": [(name, str(path)) for name, path in viscosity_candidates],
            "oxidation": [(name, str(path)) for name, path in oxidation_candidates],
        },
        "outputs": {
            "summary": str(OUTDIR / "recipe_cluster_backbone_summary.csv"),
            "fold_metrics": str(OUTDIR / "recipe_cluster_backbone_fold_metrics.csv"),
            "selected_backbones": str(OUTDIR / "selected_backbones.csv"),
        },
    }
    (OUTDIR / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(summary_df.to_string(index=False))
    print("\nSelected backbones:")
    print(selected_backbones.to_string(index=False))
    print(f"\nSaved to {OUTDIR}")


if __name__ == "__main__":
    main()
