#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.stage4_oxidation_narrow_experiments import (
    TARGET_OX,
    build_oxidation_chemistry_v4,
    build_recipe_cluster_groups,
)
from analysis.validation_regime_benchmark import build_group_maps
from mlp.train_mlp_targetwise_compact_v3 import (
    SCENARIO_COL,
    add_nonlinear_features,
    read_csv_auto,
)


OUTDIR = ROOT / "analysis" / "stage4b_oxidation_ultra_compact"
TRAIN_COMPONENTS = ROOT / "new_datasets" / "train_component_level_transformed.csv"
TRAIN_SCENARIO = ROOT / "new_datasets" / "train_scenario_level_features.csv"
COMPACT_FULL = ROOT / "compact_v3" / "train_flat_features_v3_compact.csv"


ULTRA_SETS = {
    "chem5_core": [
        "ao_pair_x_biofuel",
        "ao_dpa_phenol_imbalance",
        "synergy_ao_phenol_x_diphenylamine_active_no",
        "ao_homo_max",
        "ao_ionization_min",
    ],
    "chem7_core_mo": [
        "ao_pair_x_biofuel",
        "ao_dpa_phenol_imbalance",
        "synergy_ao_phenol_x_diphenylamine_active_no",
        "ao_homo_max",
        "ao_ionization_min",
        "ao_mo_x_biofuel",
        "synergy_ao_diphenylamine_active_no_x_mo",
    ],
    "chem8_core_mo_zddp": [
        "ao_pair_x_biofuel",
        "ao_dpa_phenol_imbalance",
        "synergy_ao_phenol_x_diphenylamine_active_no",
        "ao_homo_max",
        "ao_ionization_min",
        "ao_mo_x_biofuel",
        "synergy_ao_diphenylamine_active_no_x_mo",
        "zddp_zinc_tbn_interaction",
    ],
    "chem10_balanced": [
        "ao_pair_x_biofuel",
        "ao_dpa_phenol_imbalance",
        "synergy_ao_phenol_x_diphenylamine_active_no",
        "ao_homo_max",
        "ao_ionization_min",
        "ao_mo_x_biofuel",
        "synergy_ao_diphenylamine_active_no_x_mo",
        "zddp_zinc_tbn_interaction",
        "ao_dpa_phenol_min",
        "ao_dpa_phenol_ratio",
    ],
}


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def fit_predict_ridge(X_train: pd.DataFrame, y_train: np.ndarray, X_valid: pd.DataFrame, alpha: float) -> np.ndarray:
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_train_imp = imputer.fit_transform(X_train)
    X_valid_imp = imputer.transform(X_valid)
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_valid_scaled = scaler.transform(X_valid_imp)
    model = Ridge(alpha=alpha, random_state=42)
    model.fit(X_train_scaled, y_train)
    return model.predict(X_valid_scaled)


def fit_predict_pls(X_train: pd.DataFrame, y_train: np.ndarray, X_valid: pd.DataFrame, n_components: int) -> np.ndarray:
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_train_imp = imputer.fit_transform(X_train)
    X_valid_imp = imputer.transform(X_valid)
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_valid_scaled = scaler.transform(X_valid_imp)
    model = PLSRegression(n_components=n_components, scale=False)
    model.fit(X_train_scaled, y_train.reshape(-1, 1))
    return model.predict(X_valid_scaled).ravel()


def fit_predict_mlp(X_train: pd.DataFrame, y_train: np.ndarray, X_valid: pd.DataFrame, hidden: Tuple[int, ...], alpha: float) -> np.ndarray:
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_train_imp = imputer.fit_transform(X_train)
    X_valid_imp = imputer.transform(X_valid)
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_valid_scaled = scaler.transform(X_valid_imp)
    model = MLPRegressor(
        hidden_layer_sizes=hidden,
        activation="relu",
        solver="adam",
        alpha=alpha,
        learning_rate_init=1e-3,
        max_iter=1500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=40,
    )
    model.fit(X_train_scaled, y_train)
    return model.predict(X_valid_scaled)


def evaluate_candidate(
    model_name: str,
    X_model: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    fit_fn,
) -> Tuple[pd.DataFrame, np.ndarray]:
    splitter = GroupKFold(n_splits=5)
    rows = []
    oof = np.zeros_like(y, dtype=float)
    for fold, (train_idx, valid_idx) in enumerate(splitter.split(X_model, y, groups), start=1):
        pred = fit_fn(X_model.iloc[train_idx], y[train_idx], X_model.iloc[valid_idx])
        oof[valid_idx] = pred
        rows.append({"model_name": model_name, "fold": fold, **compute_metrics(y[valid_idx], pred)})
    rows.append({"model_name": model_name, "fold": np.nan, **compute_metrics(y, oof)})
    return pd.DataFrame(rows), oof


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    compact_df = read_csv_auto(COMPACT_FULL)
    component_df = pd.read_csv(TRAIN_COMPONENTS)
    scenario_df = pd.read_csv(TRAIN_SCENARIO)
    chem_df = build_oxidation_chemistry_v4(component_df, scenario_df)
    group_map = build_recipe_cluster_groups(component_df)
    groups = np.array([group_map[sid] for sid in compact_df[SCENARIO_COL].astype(str)], dtype=object)

    merged = compact_df.merge(chem_df, on=SCENARIO_COL, how="left").fillna(0.0)
    merged.index = merged[SCENARIO_COL].astype(str)
    y = pd.to_numeric(merged[TARGET_OX], errors="coerce").to_numpy(dtype=float)

    X_backbone = add_nonlinear_features(
        merged[[c for c in compact_df.columns if c not in {SCENARIO_COL, TARGET_OX, "target_viscosity_delta_pct"}]].copy()
    )

    all_rows = []
    oof_df = pd.DataFrame({SCENARIO_COL: merged[SCENARIO_COL].astype(str), "oxidation_true": y})

    # Backbone reference.
    backbone_metrics, backbone_oof = evaluate_candidate(
        "backbone_ridge",
        X_backbone,
        y,
        groups,
        lambda Xtr, ytr, Xva: fit_predict_ridge(Xtr, ytr, Xva, alpha=30.0),
    )
    all_rows.append(backbone_metrics.assign(feature_set="compact_v3_full", experiment_type="backbone"))
    oof_df["backbone_ridge"] = backbone_oof

    for set_name, cols in ULTRA_SETS.items():
        X_chem = merged[cols].copy()
        X_addon = pd.concat([X_backbone, X_chem], axis=1)

        candidates = [
            (
                f"standalone_ridge__{set_name}",
                X_chem,
                "standalone",
                lambda Xtr, ytr, Xva: fit_predict_ridge(Xtr, ytr, Xva, alpha=10.0),
            ),
            (
                f"standalone_pls__{set_name}",
                X_chem,
                "standalone",
                lambda Xtr, ytr, Xva, n=min(3, max(2, len(cols) - 1)): fit_predict_pls(Xtr, ytr, Xva, n_components=n),
            ),
            (
                f"standalone_mlp__{set_name}",
                X_chem,
                "standalone",
                lambda Xtr, ytr, Xva: fit_predict_mlp(Xtr, ytr, Xva, hidden=(16,), alpha=1e-3),
            ),
            (
                f"addon_ridge__{set_name}",
                X_addon,
                "addon",
                lambda Xtr, ytr, Xva: fit_predict_ridge(Xtr, ytr, Xva, alpha=30.0),
            ),
        ]

        for model_name, X_model, exp_type, fit_fn in candidates:
            metrics, oof = evaluate_candidate(model_name, X_model, y, groups, fit_fn)
            all_rows.append(metrics.assign(feature_set=set_name, experiment_type=exp_type))
            oof_df[model_name] = oof

    metrics_df = pd.concat(all_rows, ignore_index=True)
    fold_df = metrics_df[metrics_df["fold"].notna()].copy()
    summary_df = metrics_df[metrics_df["fold"].isna()].drop(columns=["fold"]).sort_values("rmse").reset_index(drop=True)
    stability_df = (
        fold_df.groupby(["model_name", "feature_set", "experiment_type"], as_index=False)
        .agg(
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            r2_mean=("r2", "mean"),
            r2_std=("r2", "std"),
        )
        .sort_values("rmse_mean")
        .reset_index(drop=True)
    )

    # Dominant base oil sanity for best addon candidate.
    best_addon = summary_df[summary_df["experiment_type"] == "addon"].iloc[0]
    best_set_name = str(best_addon["feature_set"])
    X_best_addon = pd.concat([X_backbone, merged[ULTRA_SETS[best_set_name]].copy()], axis=1)
    group_maps = build_group_maps(component_df)
    dom_groups = np.array([group_maps["dominant_base_oil"][sid] for sid in merged[SCENARIO_COL].astype(str)], dtype=object)
    dom_backbone_metrics, _ = evaluate_candidate(
        "backbone_ridge",
        X_backbone,
        y,
        dom_groups,
        lambda Xtr, ytr, Xva: fit_predict_ridge(Xtr, ytr, Xva, alpha=30.0),
    )
    dom_addon_metrics, _ = evaluate_candidate(
        f"addon_ridge__{best_set_name}",
        X_best_addon,
        y,
        dom_groups,
        lambda Xtr, ytr, Xva: fit_predict_ridge(Xtr, ytr, Xva, alpha=30.0),
    )
    dom_summary = pd.concat([dom_backbone_metrics, dom_addon_metrics], ignore_index=True)
    dom_summary = dom_summary[dom_summary["fold"].isna()].drop(columns=["fold"]).reset_index(drop=True)
    dom_summary["regime"] = "dominant_base_oil"

    metrics_df.to_csv(OUTDIR / "ultra_compact_fold_and_summary_metrics.csv", index=False)
    summary_df.to_csv(OUTDIR / "ultra_compact_summary.csv", index=False)
    stability_df.to_csv(OUTDIR / "ultra_compact_stability.csv", index=False)
    oof_df.to_csv(OUTDIR / "ultra_compact_oof_predictions.csv", index=False)
    dom_summary.to_csv(OUTDIR / "dominant_base_oil_sanity.csv", index=False)

    manifest = {
        "main_validator": "recipe_cluster",
        "backbone_reference": {"dataset": "compact_v3_full", "model": "ridge", "config": "alpha_30.0"},
        "ultra_sets": ULTRA_SETS,
        "outputs": {
            "metrics": str(OUTDIR / "ultra_compact_fold_and_summary_metrics.csv"),
            "summary": str(OUTDIR / "ultra_compact_summary.csv"),
            "stability": str(OUTDIR / "ultra_compact_stability.csv"),
            "oof": str(OUTDIR / "ultra_compact_oof_predictions.csv"),
            "dominant_base_oil_sanity": str(OUTDIR / "dominant_base_oil_sanity.csv"),
        },
    }
    (OUTDIR / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(summary_df.to_string(index=False))
    print(f"\nSaved to {OUTDIR}")


if __name__ == "__main__":
    main()
