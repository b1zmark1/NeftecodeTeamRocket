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
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.stage4_oxidation_narrow_experiments import (
    build_oxidation_chemistry_v4,
    build_recipe_cluster_groups,
)
from mlp.train_mlp_targetwise_compact_v3 import (
    SCENARIO_COL,
    add_nonlinear_features,
    fit_predict_single_target_mlp,
    read_csv_auto,
    select_features_for_target,
)


OUTDIR = ROOT / "analysis" / "stage5_moe_domain_experts"
TRAIN_COMPONENTS = ROOT / "new_datasets" / "train_component_level_transformed.csv"
TRAIN_SCENARIO = ROOT / "new_datasets" / "train_scenario_level_features.csv"
COMPACT_FULL = ROOT / "compact_v3" / "train_flat_features_v3_compact.csv"

TARGET_VISCOUS = "target_viscosity_delta_pct"
TARGET_OX = "target_oxidation_acm"

CHEM5_CORE = [
    "ao_pair_x_biofuel",
    "ao_dpa_phenol_imbalance",
    "synergy_ao_phenol_x_diphenylamine_active_no",
    "ao_homo_max",
    "ao_ionization_min",
]

MIN_EXPERT_SAMPLES = 12
SOFT_TAU = 18.0
TOP_BASE_OILS = 4
RECIPE_EXPERT_CLUSTERS = 4


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


def fit_predict_viscosity_backbone(X_train_full: pd.DataFrame, y_train: np.ndarray, X_valid_full: pd.DataFrame) -> np.ndarray:
    X_train, X_valid, _report, selected = select_features_for_target(
        X_train_full,
        X_valid_full,
        y_train=y_train,
        target_name="viscosity",
        min_non_na_ratio=0.08,
        variance_threshold=1e-8,
        max_features=20,
    )
    return fit_predict_single_target_mlp(
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
        target_name="viscosity",
        config={"config_id": "relu_small", "hidden_layers": (64, 32), "alpha": 1e-4, "learning_rate_init": 1e-3, "activation": "relu"},
        random_state=42,
        max_iter=700,
    )


def crossfit_global_backbone_viscosity(X_train: pd.DataFrame, y_train: np.ndarray, groups_train: np.ndarray) -> np.ndarray:
    inner = GroupKFold(n_splits=4)
    oof = np.zeros_like(y_train, dtype=float)
    for tr_idx, va_idx in inner.split(X_train, y_train, groups_train):
        oof[va_idx] = fit_predict_viscosity_backbone(X_train.iloc[tr_idx], y_train[tr_idx], X_train.iloc[va_idx])
    return oof


def crossfit_global_backbone_oxidation(X_train: pd.DataFrame, y_train: np.ndarray, groups_train: np.ndarray) -> np.ndarray:
    inner = GroupKFold(n_splits=4)
    oof = np.zeros_like(y_train, dtype=float)
    for tr_idx, va_idx in inner.split(X_train, y_train, groups_train):
        oof[va_idx] = fit_predict_ridge(X_train.iloc[tr_idx], y_train[tr_idx], X_train.iloc[va_idx], alpha=30.0)
    return oof


def build_family_meta(component_df: pd.DataFrame) -> pd.DataFrame:
    comp = component_df.copy()
    comp["Массовая доля, %"] = pd.to_numeric(comp["Массовая доля, %"], errors="coerce").fillna(0.0)
    comp["family"] = comp["Компонент"].astype(str).str.replace(r"_\d+$", "", regex=True)

    def dominant_base(group: pd.DataFrame) -> str:
        base = group[group["Компонент"].astype(str).str.startswith("Базовое_масло")][["Компонент", "Массовая доля, %"]]
        if base.empty:
            return "other"
        return str(base.sort_values("Массовая доля, %", ascending=False, kind="stable").iloc[0]["Компонент"])

    rows = []
    for scenario_id, group in comp.groupby(SCENARIO_COL, sort=False):
        family_dose = group.groupby("family")["Массовая доля, %"].sum()
        ao = float(family_dose.get("Антиоксидант", 0.0))
        antiwear = float(family_dose.get("Противоизносная_присадка", 0.0))
        detergent = float(family_dose.get("Детергент", 0.0))
        moly = float(family_dose.get("Соединение_молибдена", 0.0))
        dispersant = float(family_dose.get("Дисперсант", 0.0))
        additive_total = ao + antiwear + detergent + moly + dispersant
        if additive_total <= 0:
            package_type = "other"
        elif moly > 0 and ao > 0:
            package_type = "ao_mo"
        else:
            ratios = {
                "ao_heavy": ao / additive_total,
                "zddp_heavy": antiwear / additive_total,
                "detergent_heavy": detergent / additive_total,
            }
            best_label = max(ratios, key=ratios.get)
            package_type = best_label if ratios[best_label] >= 0.35 else "mixed"

        rows.append(
            {
                SCENARIO_COL: str(scenario_id),
                "dominant_base_oil_raw": dominant_base(group),
                "ao_dose": ao,
                "antiwear_dose": antiwear,
                "detergent_dose": detergent,
                "moly_dose": moly,
                "dispersant_dose": dispersant,
                "package_type": package_type,
            }
        )
    return pd.DataFrame(rows)


def build_composition_pivot(component_df: pd.DataFrame) -> pd.DataFrame:
    comp = component_df.copy()
    comp["Массовая доля, %"] = pd.to_numeric(comp["Массовая доля, %"], errors="coerce").fillna(0.0)
    pivot = comp.pivot_table(index=SCENARIO_COL, columns="Компонент", values="Массовая доля, %", aggfunc="sum", fill_value=0.0)
    pivot.index = pivot.index.astype(str)
    return pivot


def assign_recipe_experts_train_valid(pivot: pd.DataFrame, train_ids: List[str], valid_ids: List[str]) -> Tuple[pd.Series, pd.Series]:
    X_train = pivot.loc[train_ids]
    X_valid = pivot.loc[valid_ids]
    n_clusters = min(RECIPE_EXPERT_CLUSTERS, max(2, min(len(X_train), X_train.shape[0] // 8 or 2)))
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    train_labels = km.fit_predict(X_train)
    valid_labels = km.predict(X_valid)
    train_series = pd.Series([f"recipe_expert_{x}" for x in train_labels], index=train_ids)
    valid_series = pd.Series([f"recipe_expert_{x}" for x in valid_labels], index=valid_ids)
    return train_series, valid_series


def assign_baseoil_experts_train_valid(meta: pd.DataFrame, train_ids: List[str], valid_ids: List[str]) -> Tuple[pd.Series, pd.Series]:
    train_meta = meta.set_index(SCENARIO_COL).loc[train_ids]
    valid_meta = meta.set_index(SCENARIO_COL).loc[valid_ids]
    top = train_meta["dominant_base_oil_raw"].value_counts().head(TOP_BASE_OILS).index.tolist()
    train_labels = train_meta["dominant_base_oil_raw"].where(train_meta["dominant_base_oil_raw"].isin(top), "other")
    valid_labels = valid_meta["dominant_base_oil_raw"].where(valid_meta["dominant_base_oil_raw"].isin(top), "other")
    return train_labels.astype(str), valid_labels.astype(str)


def assign_package_experts_train_valid(meta: pd.DataFrame, train_ids: List[str], valid_ids: List[str]) -> Tuple[pd.Series, pd.Series]:
    train_meta = meta.set_index(SCENARIO_COL).loc[train_ids]
    valid_meta = meta.set_index(SCENARIO_COL).loc[valid_ids]
    return train_meta["package_type"].astype(str), valid_meta["package_type"].astype(str)


def fit_domain_residual_experts(
    X_train: pd.DataFrame,
    residual_train: np.ndarray,
    labels_train: pd.Series,
    X_valid: pd.DataFrame,
    labels_valid: pd.Series,
    alpha: float,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    hard_pred = np.zeros(X_valid.shape[0], dtype=float)
    soft_pred = np.zeros(X_valid.shape[0], dtype=float)
    rows = []
    for label, idx_train in labels_train.groupby(labels_train).groups.items():
        train_idx = np.array(sorted(idx_train))
        mask_train = labels_train == label
        mask_valid = labels_valid == label
        n_train = int(mask_train.sum())
        n_valid = int(mask_valid.sum())
        rows.append({"expert_label": str(label), "n_train": n_train, "n_valid": n_valid})
        if n_train < MIN_EXPERT_SAMPLES or n_valid == 0:
            continue
        pred = fit_predict_ridge(X_train.loc[mask_train], residual_train[mask_train.to_numpy()], X_valid.loc[mask_valid], alpha=alpha)
        valid_positions = np.flatnonzero(mask_valid.to_numpy())
        hard_pred[valid_positions] = pred
        weight = n_train / (n_train + SOFT_TAU)
        soft_pred[valid_positions] = pred * weight
    return hard_pred, soft_pred, pd.DataFrame(rows)


def fit_domain_direct_soft_experts(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    labels_train: pd.Series,
    X_valid: pd.DataFrame,
    labels_valid: pd.Series,
    base_valid_pred: np.ndarray,
    alpha: float,
) -> Tuple[np.ndarray, pd.DataFrame]:
    final_pred = base_valid_pred.copy()
    rows = []
    for label, idx_train in labels_train.groupby(labels_train).groups.items():
        mask_train = labels_train == label
        mask_valid = labels_valid == label
        n_train = int(mask_train.sum())
        n_valid = int(mask_valid.sum())
        rows.append({"expert_label": str(label), "n_train": n_train, "n_valid": n_valid})
        if n_train < MIN_EXPERT_SAMPLES or n_valid == 0:
            continue
        pred = fit_predict_ridge(X_train.loc[mask_train], y_train[mask_train.to_numpy()], X_valid.loc[mask_valid], alpha=alpha)
        valid_positions = np.flatnonzero(mask_valid.to_numpy())
        weight = min(0.25, n_train / (n_train + SOFT_TAU))
        final_pred[valid_positions] = (1.0 - weight) * base_valid_pred[valid_positions] + weight * pred
    return final_pred, pd.DataFrame(rows)


def evaluate_moe() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    compact_df = read_csv_auto(COMPACT_FULL)
    component_df = pd.read_csv(TRAIN_COMPONENTS)
    scenario_df = pd.read_csv(TRAIN_SCENARIO)
    chem_df = build_oxidation_chemistry_v4(component_df, scenario_df)
    family_meta = build_family_meta(component_df)
    comp_pivot = build_composition_pivot(component_df)
    recipe_group_map = build_recipe_cluster_groups(component_df)

    merged = compact_df.merge(chem_df[[SCENARIO_COL] + CHEM5_CORE], on=SCENARIO_COL, how="left").fillna(0.0)
    merged.index = merged[SCENARIO_COL].astype(str)

    X_compact = add_nonlinear_features(
        merged[[c for c in compact_df.columns if c not in {SCENARIO_COL, TARGET_VISCOUS, TARGET_OX}]].copy()
    )
    X_ox_global = pd.concat([X_compact, merged[CHEM5_CORE].copy()], axis=1)
    y_visc = pd.to_numeric(merged[TARGET_VISCOUS], errors="coerce").to_numpy(dtype=float)
    y_ox = pd.to_numeric(merged[TARGET_OX], errors="coerce").to_numpy(dtype=float)

    outer_groups = np.array([recipe_group_map[sid] for sid in merged[SCENARIO_COL].astype(str)], dtype=object)
    splitter = GroupKFold(n_splits=5)

    metrics_rows = []
    expert_rows = []
    oof = pd.DataFrame({SCENARIO_COL: merged[SCENARIO_COL].astype(str), "viscosity_true": y_visc, "oxidation_true": y_ox})
    oof["viscosity_backbone"] = 0.0
    oof["oxidation_backbone"] = 0.0

    for scheme_name in ["recipe_cluster_moe", "dominant_base_oil_moe", "package_type_moe"]:
        oof[f"viscosity__{scheme_name}__hard"] = 0.0
        oof[f"viscosity__{scheme_name}__soft"] = 0.0
        oof[f"viscosity__{scheme_name}__direct_soft"] = 0.0
        oof[f"oxidation__{scheme_name}__hard"] = 0.0
        oof[f"oxidation__{scheme_name}__soft"] = 0.0
        oof[f"oxidation__{scheme_name}__direct_soft"] = 0.0

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(merged, y_visc, outer_groups), start=1):
        train_ids = merged.iloc[train_idx][SCENARIO_COL].astype(str).tolist()
        valid_ids = merged.iloc[valid_idx][SCENARIO_COL].astype(str).tolist()

        X_visc_tr = X_compact.iloc[train_idx]
        X_visc_va = X_compact.iloc[valid_idx]
        X_ox_tr = X_ox_global.iloc[train_idx]
        X_ox_va = X_ox_global.iloc[valid_idx]
        yv_tr, yv_va = y_visc[train_idx], y_visc[valid_idx]
        yo_tr, yo_va = y_ox[train_idx], y_ox[valid_idx]
        groups_tr = outer_groups[train_idx]

        pred_visc_global = fit_predict_viscosity_backbone(X_visc_tr, yv_tr, X_visc_va)
        pred_ox_global = fit_predict_ridge(X_ox_tr, yo_tr, X_ox_va, alpha=30.0)
        oof.iloc[valid_idx, oof.columns.get_loc("viscosity_backbone")] = pred_visc_global
        oof.iloc[valid_idx, oof.columns.get_loc("oxidation_backbone")] = pred_ox_global

        train_visc_global_oof = crossfit_global_backbone_viscosity(X_visc_tr, yv_tr, groups_tr)
        train_ox_global_oof = crossfit_global_backbone_oxidation(X_ox_tr, yo_tr, groups_tr)
        resid_visc = yv_tr - train_visc_global_oof
        resid_ox = yo_tr - train_ox_global_oof

        scheme_assigners = {
            "recipe_cluster_moe": lambda: assign_recipe_experts_train_valid(comp_pivot, train_ids, valid_ids),
            "dominant_base_oil_moe": lambda: assign_baseoil_experts_train_valid(family_meta, train_ids, valid_ids),
            "package_type_moe": lambda: assign_package_experts_train_valid(family_meta, train_ids, valid_ids),
        }

        for scheme_name, assign_fn in scheme_assigners.items():
            labels_train, labels_valid = assign_fn()

            hard_resid_visc, soft_resid_visc, experts_visc = fit_domain_residual_experts(
                X_train=X_visc_tr,
                residual_train=resid_visc,
                labels_train=labels_train,
                X_valid=X_visc_va,
                labels_valid=labels_valid,
                alpha=10.0,
            )
            hard_pred_visc = pred_visc_global + hard_resid_visc
            soft_pred_visc = pred_visc_global + soft_resid_visc

            hard_resid_ox, soft_resid_ox, experts_ox = fit_domain_residual_experts(
                X_train=X_ox_tr,
                residual_train=resid_ox,
                labels_train=labels_train,
                X_valid=X_ox_va,
                labels_valid=labels_valid,
                alpha=10.0,
            )
            hard_pred_ox = pred_ox_global + hard_resid_ox
            soft_pred_ox = pred_ox_global + soft_resid_ox

            direct_soft_visc, experts_visc_direct = fit_domain_direct_soft_experts(
                X_train=X_visc_tr,
                y_train=yv_tr,
                labels_train=labels_train,
                X_valid=X_visc_va,
                labels_valid=labels_valid,
                base_valid_pred=pred_visc_global,
                alpha=10.0,
            )
            direct_soft_ox, experts_ox_direct = fit_domain_direct_soft_experts(
                X_train=X_ox_tr,
                y_train=yo_tr,
                labels_train=labels_train,
                X_valid=X_ox_va,
                labels_valid=labels_valid,
                base_valid_pred=pred_ox_global,
                alpha=10.0,
            )

            oof.iloc[valid_idx, oof.columns.get_loc(f"viscosity__{scheme_name}__hard")] = hard_pred_visc
            oof.iloc[valid_idx, oof.columns.get_loc(f"viscosity__{scheme_name}__soft")] = soft_pred_visc
            oof.iloc[valid_idx, oof.columns.get_loc(f"viscosity__{scheme_name}__direct_soft")] = direct_soft_visc
            oof.iloc[valid_idx, oof.columns.get_loc(f"oxidation__{scheme_name}__hard")] = hard_pred_ox
            oof.iloc[valid_idx, oof.columns.get_loc(f"oxidation__{scheme_name}__soft")] = soft_pred_ox
            oof.iloc[valid_idx, oof.columns.get_loc(f"oxidation__{scheme_name}__direct_soft")] = direct_soft_ox

            for target_name, y_true, hard_pred, soft_pred in [
                ("viscosity", yv_va, hard_pred_visc, soft_pred_visc),
                ("oxidation", yo_va, hard_pred_ox, soft_pred_ox),
            ]:
                metrics_rows.append(
                    {
                        "fold": fold,
                        "target_name": target_name,
                        "model_name": f"{scheme_name}__hard",
                        "scheme_name": scheme_name,
                        "gating": "hard",
                        **compute_metrics(y_true, hard_pred),
                    }
                )
                metrics_rows.append(
                    {
                        "fold": fold,
                        "target_name": target_name,
                        "model_name": f"{scheme_name}__soft",
                        "scheme_name": scheme_name,
                        "gating": "soft",
                        **compute_metrics(y_true, soft_pred),
                    }
                )
            metrics_rows.append(
                {
                    "fold": fold,
                    "target_name": "viscosity",
                    "model_name": f"{scheme_name}__direct_soft",
                    "scheme_name": scheme_name,
                    "gating": "direct_soft",
                    **compute_metrics(yv_va, direct_soft_visc),
                }
            )
            metrics_rows.append(
                {
                    "fold": fold,
                    "target_name": "oxidation",
                    "model_name": f"{scheme_name}__direct_soft",
                    "scheme_name": scheme_name,
                    "gating": "direct_soft",
                    **compute_metrics(yo_va, direct_soft_ox),
                }
            )

            experts_visc["fold"] = fold
            experts_visc["target_name"] = "viscosity"
            experts_visc["scheme_name"] = scheme_name
            experts_visc["expert_style"] = "residual"
            expert_rows.append(experts_visc)
            experts_ox["fold"] = fold
            experts_ox["target_name"] = "oxidation"
            experts_ox["scheme_name"] = scheme_name
            experts_ox["expert_style"] = "residual"
            expert_rows.append(experts_ox)
            experts_visc_direct["fold"] = fold
            experts_visc_direct["target_name"] = "viscosity"
            experts_visc_direct["scheme_name"] = scheme_name
            experts_visc_direct["expert_style"] = "direct"
            expert_rows.append(experts_visc_direct)
            experts_ox_direct["fold"] = fold
            experts_ox_direct["target_name"] = "oxidation"
            experts_ox_direct["scheme_name"] = scheme_name
            experts_ox_direct["expert_style"] = "direct"
            expert_rows.append(experts_ox_direct)

        metrics_rows.append(
            {
                "fold": fold,
                "target_name": "viscosity",
                "model_name": "global_backbone",
                "scheme_name": "global",
                "gating": "none",
                **compute_metrics(yv_va, pred_visc_global),
            }
        )
        metrics_rows.append(
            {
                "fold": fold,
                "target_name": "oxidation",
                "model_name": "global_backbone",
                "scheme_name": "global",
                "gating": "none",
                **compute_metrics(yo_va, pred_ox_global),
            }
        )

    for target_name, true_col, pred_col in [
        ("viscosity", "viscosity_true", "viscosity_backbone"),
        ("oxidation", "oxidation_true", "oxidation_backbone"),
    ]:
        metrics_rows.append(
            {
                "fold": np.nan,
                "target_name": target_name,
                "model_name": "global_backbone",
                "scheme_name": "global",
                "gating": "none",
                **compute_metrics(oof[true_col].to_numpy(dtype=float), oof[pred_col].to_numpy(dtype=float)),
            }
        )

    for scheme_name in ["recipe_cluster_moe", "dominant_base_oil_moe", "package_type_moe"]:
        for target_name, true_col, prefix in [
            ("viscosity", "viscosity_true", "viscosity"),
            ("oxidation", "oxidation_true", "oxidation"),
        ]:
            for gating in ["hard", "soft", "direct_soft"]:
                pred_col = f"{prefix}__{scheme_name}__{gating}"
                metrics_rows.append(
                    {
                        "fold": np.nan,
                        "target_name": target_name,
                        "model_name": f"{scheme_name}__{gating}",
                        "scheme_name": scheme_name,
                        "gating": gating,
                        **compute_metrics(oof[true_col].to_numpy(dtype=float), oof[pred_col].to_numpy(dtype=float)),
                    }
                )

    return pd.DataFrame(metrics_rows), pd.concat(expert_rows, ignore_index=True), oof


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    metrics_df, experts_df, oof_df = evaluate_moe()
    fold_df = metrics_df[metrics_df["fold"].notna()].copy()
    summary_df = (
        metrics_df[metrics_df["fold"].isna()]
        .drop(columns=["fold"])
        .sort_values(["target_name", "rmse", "model_name"])
        .reset_index(drop=True)
    )
    stability_df = (
        fold_df.groupby(["target_name", "model_name", "scheme_name", "gating"], as_index=False)
        .agg(
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            r2_mean=("r2", "mean"),
            r2_std=("r2", "std"),
        )
        .sort_values(["target_name", "rmse_mean"])
        .reset_index(drop=True)
    )

    metrics_df.to_csv(OUTDIR / "moe_fold_and_summary_metrics.csv", index=False)
    summary_df.to_csv(OUTDIR / "moe_summary.csv", index=False)
    stability_df.to_csv(OUTDIR / "moe_stability.csv", index=False)
    experts_df.to_csv(OUTDIR / "moe_expert_group_sizes.csv", index=False)
    oof_df.to_csv(OUTDIR / "moe_oof_predictions.csv", index=False)

    manifest = {
        "main_validator": "recipe_cluster",
        "global_models": {
            "viscosity": {"dataset": "compact_v3_full", "model": "mlp", "config": "relu_small"},
            "oxidation": {"dataset": "compact_v3_full+chem5_core", "model": "ridge", "config": "alpha_30.0"},
        },
        "schemes": ["recipe_cluster_moe", "dominant_base_oil_moe", "package_type_moe"],
        "gating": {"hard_min_samples": MIN_EXPERT_SAMPLES, "soft_tau": SOFT_TAU},
        "outputs": {
            "metrics": str(OUTDIR / "moe_fold_and_summary_metrics.csv"),
            "summary": str(OUTDIR / "moe_summary.csv"),
            "stability": str(OUTDIR / "moe_stability.csv"),
            "expert_group_sizes": str(OUTDIR / "moe_expert_group_sizes.csv"),
            "oof": str(OUTDIR / "moe_oof_predictions.csv"),
        },
    }
    (OUTDIR / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(summary_df.to_string(index=False))
    print(f"\nSaved to {OUTDIR}")


if __name__ == "__main__":
    main()
