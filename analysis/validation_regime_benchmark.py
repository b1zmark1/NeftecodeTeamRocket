#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupKFold, KFold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mlp.train_mlp_targetwise_compact_v3 import (
    SCENARIO_COL,
    add_nonlinear_features,
    build_target_configs,
    compute_metrics,
    fit_predict_single_target_mlp,
    read_csv_auto,
    resolve_target_columns,
    select_features_for_target,
    select_numeric_feature_columns,
)


TRAIN_FEATURES = ROOT / "compact_v3" / "train_flat_features_v3_compact.csv"
TRAIN_COMPONENTS = ROOT / "new_datasets" / "train_component_level_transformed.csv"
OUTDIR = ROOT / "analysis" / "validation_regimes"


def build_group_maps(component_df: pd.DataFrame) -> Dict[str, Dict[str, str]]:
    comp = component_df.copy()
    comp["Массовая доля, %"] = pd.to_numeric(comp["Массовая доля, %"], errors="coerce").fillna(0.0)

    base = comp[comp["Компонент"].astype(str).str.startswith("Базовое_масло")].copy()
    dom_base = (
        base.sort_values([SCENARIO_COL, "Массовая доля, %"], ascending=[True, False], kind="stable")
        .groupby(SCENARIO_COL, sort=False)
        .first()
        .reset_index()[[SCENARIO_COL, "Компонент"]]
    )
    dominant_base_map = dict(zip(dom_base[SCENARIO_COL].astype(str), dom_base["Компонент"].astype(str)))

    pivot = comp.pivot_table(index=SCENARIO_COL, columns="Компонент", values="Массовая доля, %", aggfunc="sum", fill_value=0.0)
    km = KMeans(n_clusters=8, random_state=42, n_init=20)
    labels = km.fit_predict(pivot)
    recipe_cluster_map = {str(idx): f"cluster_{label}" for idx, label in zip(pivot.index, labels)}

    return {
        "dominant_base_oil": dominant_base_map,
        "recipe_cluster": recipe_cluster_map,
    }


def summarize_groups(name: str, groups: np.ndarray) -> Dict[str, object]:
    vc = pd.Series(groups).value_counts()
    return {
        "regime": name,
        "n_groups": int(vc.shape[0]),
        "min_group_size": int(vc.min()),
        "median_group_size": float(vc.median()),
        "max_group_size": int(vc.max()),
        "groups": {str(k): int(v) for k, v in vc.sort_index().items()},
    }


def regime_splits(regime_name: str, scenario_ids: np.ndarray, group_maps: Dict[str, Dict[str, str]], n_splits: int = 5):
    if regime_name == "random_kfold":
        return list(KFold(n_splits=n_splits, shuffle=True, random_state=42).split(scenario_ids)), None
    groups = np.array([group_maps[regime_name][sid] for sid in scenario_ids], dtype=object)
    return list(GroupKFold(n_splits=n_splits).split(scenario_ids, groups=groups, y=None)), groups


def best_config_for_target(target_name: str) -> Dict[str, object]:
    configs = {str(cfg["config_id"]): cfg for cfg in build_target_configs()}
    if target_name == "viscosity":
        return configs["relu_wide"]
    return configs["relu_mid"]


def evaluate_regime(
    X: pd.DataFrame,
    y: np.ndarray,
    target_name: str,
    split_name: str,
    split_iter: Iterable[Tuple[np.ndarray, np.ndarray]],
    config: Dict[str, object],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, object]] = []
    oof = np.zeros_like(y, dtype=float)
    scenario_ids = X.index.to_numpy()

    for fold, (train_idx, valid_idx) in enumerate(split_iter, start=1):
        X_train_full = X.iloc[train_idx]
        X_valid_full = X.iloc[valid_idx]
        y_train = y[train_idx]
        y_valid = y[valid_idx]

        X_train, X_valid, feature_report, selected_features = select_features_for_target(
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
        metrics = compute_metrics(y_valid, y_pred)
        rows.append(
            {
                "target_name": target_name,
                "regime": split_name,
                "fold": fold,
                "config_id": str(config["config_id"]),
                "n_features": len(selected_features),
                **metrics,
            }
        )

    summary = pd.DataFrame(rows)
    overall = pd.DataFrame(
        [
            {
                "target_name": target_name,
                "regime": split_name,
                "config_id": str(config["config_id"]),
                **compute_metrics(y, oof),
            }
        ]
    )
    oof_df = pd.DataFrame(
        {
            SCENARIO_COL: scenario_ids,
            "target_name": target_name,
            "regime": split_name,
            "y_true": y,
            "y_pred": oof,
        }
    )
    return pd.concat([summary, overall], ignore_index=True), oof_df


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    df = read_csv_auto(TRAIN_FEATURES)
    target_visc_col, target_oxid_col = resolve_target_columns(df)
    raw_feature_cols = select_numeric_feature_columns(df)
    X_raw = df[raw_feature_cols].copy()
    X = add_nonlinear_features(X_raw)
    X.index = df[SCENARIO_COL].astype(str)
    scenario_ids = X.index.to_numpy()

    y_visc = pd.to_numeric(df[target_visc_col], errors="coerce").to_numpy(dtype=float)
    y_ox = pd.to_numeric(df[target_oxid_col], errors="coerce").to_numpy(dtype=float)

    comp_df = pd.read_csv(TRAIN_COMPONENTS)
    group_maps = build_group_maps(comp_df)

    regime_names = ["random_kfold", "dominant_base_oil", "recipe_cluster"]

    group_summary_rows = []
    all_cv = []
    all_oof = []
    for regime in regime_names:
        splits, groups = regime_splits(regime, scenario_ids, group_maps)
        if groups is not None:
            group_summary_rows.append(summarize_groups(regime, groups))

        visc_res, visc_oof = evaluate_regime(
            X=X,
            y=y_visc,
            target_name="viscosity",
            split_name=regime,
            split_iter=splits,
            config=best_config_for_target("viscosity"),
        )
        ox_res, ox_oof = evaluate_regime(
            X=X,
            y=y_ox,
            target_name="oxidation",
            split_name=regime,
            split_iter=splits,
            config=best_config_for_target("oxidation"),
        )
        all_cv.extend([visc_res, ox_res])
        all_oof.extend([visc_oof, ox_oof])

    cv_df = pd.concat(all_cv, ignore_index=True)
    oof_df = pd.concat(all_oof, ignore_index=True)

    fold_df = cv_df[cv_df["fold"].notna()].copy()
    overall_df = cv_df[cv_df["fold"].isna()].copy().reset_index(drop=True)
    overall_df = overall_df.sort_values(["target_name", "rmse", "regime"]).reset_index(drop=True)

    fold_df.to_csv(OUTDIR / "validation_regime_fold_metrics.csv", index=False)
    overall_df.to_csv(OUTDIR / "validation_regime_summary.csv", index=False)
    oof_df.to_csv(OUTDIR / "validation_regime_oof_predictions.csv", index=False)
    (OUTDIR / "group_summary.json").write_text(json.dumps(group_summary_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest = {
        "train_features": str(TRAIN_FEATURES),
        "train_components": str(TRAIN_COMPONENTS),
        "regimes": regime_names,
        "configs": {
            "viscosity": best_config_for_target("viscosity"),
            "oxidation": best_config_for_target("oxidation"),
        },
        "outputs": {
            "fold_metrics": str(OUTDIR / "validation_regime_fold_metrics.csv"),
            "summary": str(OUTDIR / "validation_regime_summary.csv"),
            "oof_predictions": str(OUTDIR / "validation_regime_oof_predictions.csv"),
            "group_summary": str(OUTDIR / "group_summary.json"),
        },
    }
    (OUTDIR / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(overall_df.to_string(index=False))
    print(f"\nSaved to {OUTDIR}")


if __name__ == "__main__":
    main()
