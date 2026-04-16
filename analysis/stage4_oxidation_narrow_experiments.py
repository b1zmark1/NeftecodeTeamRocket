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

from mlp.train_mlp_targetwise_compact_v3 import (
    SCENARIO_COL,
    add_nonlinear_features,
    read_csv_auto,
    select_numeric_feature_columns,
)


OUTDIR = ROOT / "analysis" / "stage4_oxidation_narrow"
TRAIN_COMPONENTS = ROOT / "new_datasets" / "train_component_level_transformed.csv"
TRAIN_SCENARIO = ROOT / "new_datasets" / "train_scenario_level_features.csv"
COMPACT_FULL = ROOT / "compact_v3" / "train_flat_features_v3_compact.csv"

TARGET_OX = "target_oxidation_acm"

SCENARIO_COLS = [
    "synergy_ao_phenol_x_diphenylamine_active_no",
    "synergy_ao_diphenylamine_active_no_x_mo",
    "ao_k_avg_weighted_active_no",
    "ao_ionization_min",
    "ao_homo_max",
]

BLOCKS = {
    "ao_core_narrow": [
        "synergy_ao_phenol_x_diphenylamine_active_no",
        "ao_k_avg_weighted_active_no",
        "ao_ionization_min",
        "ao_homo_max",
        "ao_dpa_phenol_ratio",
        "ao_dpa_phenol_min",
        "ao_dpa_phenol_imbalance",
        "ao_ratio_x_baseoil_share",
        "ao_pair_x_biofuel",
        "active_no_coverage_share",
    ],
    "ao_mo_narrow": [
        "synergy_ao_diphenylamine_active_no_x_mo",
        "ao_mo_x_severity",
        "ao_mo_x_biofuel",
        "mo_coverage_share",
    ],
    "zddp_narrow": [
        "zddp_sulfur_calcium_interaction",
        "zddp_sulfur_camg_interaction",
        "zddp_zinc_tbn_interaction",
        "zddp_zinc_boron_log1p",
        "zddp_calcium_coverage_share",
    ],
    "ao_family_narrow": [
        "synergy_ao_phenol_x_diphenylamine_active_no",
        "synergy_ao_diphenylamine_active_no_x_mo",
        "ao_k_avg_weighted_active_no",
        "ao_ionization_min",
        "ao_homo_max",
        "ao_dpa_phenol_ratio",
        "ao_dpa_phenol_min",
        "ao_dpa_phenol_imbalance",
        "ao_ratio_x_baseoil_share",
        "ao_pair_x_biofuel",
        "ao_mo_x_severity",
        "ao_mo_x_biofuel",
        "active_no_coverage_share",
        "mo_coverage_share",
    ],
    "oxidation_v4_narrow": [
        "synergy_ao_phenol_x_diphenylamine_active_no",
        "synergy_ao_diphenylamine_active_no_x_mo",
        "ao_k_avg_weighted_active_no",
        "ao_ionization_min",
        "ao_homo_max",
        "ao_dpa_phenol_ratio",
        "ao_dpa_phenol_min",
        "ao_dpa_phenol_imbalance",
        "ao_ratio_x_baseoil_share",
        "ao_pair_x_biofuel",
        "ao_mo_x_severity",
        "ao_mo_x_biofuel",
        "zddp_sulfur_calcium_interaction",
        "zddp_zinc_tbn_interaction",
        "zddp_zinc_boron_log1p",
    ],
}

RESIDUAL_BLOCKS = ["ao_core_narrow", "ao_mo_narrow", "ao_family_narrow", "zddp_narrow", "oxidation_v4_narrow"]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def build_recipe_cluster_groups(component_df: pd.DataFrame) -> Dict[str, str]:
    comp = component_df.copy()
    comp["Массовая доля, %"] = pd.to_numeric(comp["Массовая доля, %"], errors="coerce").fillna(0.0)
    pivot = comp.pivot_table(index=SCENARIO_COL, columns="Компонент", values="Массовая доля, %", aggfunc="sum", fill_value=0.0)
    km = KMeans(n_clusters=8, random_state=42, n_init=20)
    labels = km.fit_predict(pivot)
    return {str(idx): f"cluster_{label}" for idx, label in zip(pivot.index, labels)}


def weighted_sum(frame: pd.DataFrame, value_col: str, mask: pd.Series) -> float:
    sub = frame.loc[mask, ["Массовая доля, %", value_col]].copy()
    sub["Массовая доля, %"] = pd.to_numeric(sub["Массовая доля, %"], errors="coerce")
    sub[value_col] = pd.to_numeric(sub[value_col], errors="coerce")
    sub = sub.dropna()
    if sub.empty:
        return 0.0
    return float((sub["Массовая доля, %"] * sub[value_col]).sum())


def weighted_mean(frame: pd.DataFrame, value_col: str, mask: pd.Series) -> float:
    sub = frame.loc[mask, ["Массовая доля, %", value_col]].copy()
    sub["Массовая доля, %"] = pd.to_numeric(sub["Массовая доля, %"], errors="coerce")
    sub[value_col] = pd.to_numeric(sub[value_col], errors="coerce")
    sub = sub.dropna()
    if sub.empty:
        return 0.0
    w = sub["Массовая доля, %"].to_numpy(dtype=float)
    v = sub[value_col].to_numpy(dtype=float)
    denom = w.sum()
    if denom <= 0:
        return 0.0
    return float(np.dot(w, v) / denom)


def availability_share(frame: pd.DataFrame, value_col: str, mask: pd.Series) -> float:
    total = pd.to_numeric(frame.loc[mask, "Массовая доля, %"], errors="coerce").fillna(0.0).sum()
    if total <= 0:
        return 0.0
    available = frame.loc[mask, ["Массовая доля, %", value_col]].copy()
    available["Массовая доля, %"] = pd.to_numeric(available["Массовая доля, %"], errors="coerce")
    available[value_col] = pd.to_numeric(available[value_col], errors="coerce")
    available = available[available[value_col].notna()]
    covered = available["Массовая доля, %"].fillna(0.0).sum()
    return float(covered / total)


def build_oxidation_chemistry_v4(component_df: pd.DataFrame, scenario_df: pd.DataFrame) -> pd.DataFrame:
    scen = scenario_df.set_index(SCENARIO_COL)
    rows = []
    for scenario_id, group in component_df.groupby(SCENARIO_COL, sort=False):
        g = group.copy()
        g["Массовая доля, %"] = pd.to_numeric(g["Массовая доля, %"], errors="coerce").fillna(0.0)
        g["active_no"] = pd.to_numeric(g["Активный Азот / Кислород, % масс. (N или O)"], errors="coerce")
        g["mo_pct"] = pd.to_numeric(g["% масс. (Mo)"], errors="coerce")
        g["zinc_pct"] = pd.to_numeric(g["Массовая доля цинка, ASTM D6481"], errors="coerce")
        g["sulfur_pct"] = pd.to_numeric(g["Массовая доля серы, ASTM D6481"], errors="coerce")
        g["phosphorus_pct"] = pd.to_numeric(g["Массовая доля фосфора, ASTM D6481"], errors="coerce")
        g["calcium_pct"] = pd.to_numeric(g["Массовая доля кальция, ASTM D6481"], errors="coerce")
        g["camg_pct"] = pd.to_numeric(g["Содержание металла (Ca/Mg), % масс."], errors="coerce")
        g["boron_pct"] = pd.to_numeric(g["Содержание Бора"], errors="coerce")
        g["tbn"] = pd.to_numeric(g["Щелочное число, ASTM D2896"], errors="coerce")
        g["Тип АО"] = g["Тип АО"].fillna("").astype(str)

        temp = float(g.iloc[0]["Температура испытания | ASTM D445 Daimler Oxidation Test (DOT), °C"])
        time_h = float(g.iloc[0]["Время испытания | - Daimler Oxidation Test (DOT), ч"])
        biofuel = float(g.iloc[0]["Количество биотоплива | - Daimler Oxidation Test (DOT), % масс"])
        severity = time_h * math.exp((temp - 150.0) / 10.0)

        is_dpa = g["Тип АО"].eq("Дифениламин")
        is_phenol = g["Тип АО"].eq("Фенол")
        is_moly = g["Компонент"].astype(str).str.startswith("Соединение_молибдена")
        is_antiwear = g["Компонент"].astype(str).str.startswith("Противоизносная_присадка")
        is_detergent = g["Компонент"].astype(str).str.startswith("Детергент")
        is_baseoil = g["Компонент"].astype(str).str.startswith("Базовое_масло")

        ao_dpa_active_total = weighted_sum(g, "active_no", is_dpa)
        ao_phenol_active_total = weighted_sum(g, "active_no", is_phenol)
        ao_prod = ao_dpa_active_total * ao_phenol_active_total
        ao_ratio = ao_dpa_active_total / (ao_phenol_active_total + 1e-6)
        ao_min = min(ao_dpa_active_total, ao_phenol_active_total)
        ao_imbalance = abs(ao_dpa_active_total - ao_phenol_active_total)

        mo_total = weighted_sum(g, "mo_pct", is_moly)
        antiwear_zinc_total = weighted_sum(g, "zinc_pct", is_antiwear)
        antiwear_sulfur_total = weighted_sum(g, "sulfur_pct", is_antiwear)
        antiwear_phosphorus_total = weighted_sum(g, "phosphorus_pct", is_antiwear)
        calcium_total = weighted_sum(g, "calcium_pct", g.index == g.index)
        camg_total = weighted_sum(g, "camg_pct", g.index == g.index)
        boron_total = weighted_sum(g, "boron_pct", g.index == g.index)
        detergent_tbn_wmean = weighted_mean(g, "tbn", is_detergent)

        base_total = float(g.loc[is_baseoil, "Массовая доля, %"].sum())
        dominant_base_oil_share = 0.0 if base_total <= 0 else float(g.loc[is_baseoil, "Массовая доля, %"].max() / base_total)

        row = {
            SCENARIO_COL: str(scenario_id),
            "ao_dpa_phenol_ratio": ao_ratio,
            "ao_dpa_phenol_min": ao_min,
            "ao_dpa_phenol_imbalance": ao_imbalance,
            "ao_ratio_x_baseoil_share": ao_ratio * dominant_base_oil_share,
            "ao_pair_x_biofuel": ao_prod * biofuel,
            "ao_mo_x_severity": ao_dpa_active_total * mo_total * severity,
            "ao_mo_x_biofuel": ao_dpa_active_total * mo_total * biofuel,
            "zddp_sulfur_calcium_interaction": antiwear_sulfur_total * calcium_total,
            "zddp_sulfur_camg_interaction": antiwear_sulfur_total * camg_total,
            "zddp_zinc_tbn_interaction": antiwear_zinc_total * detergent_tbn_wmean,
            "zddp_zinc_boron_log1p": math.log1p(max(antiwear_zinc_total * boron_total, 0.0)),
            "active_no_coverage_share": availability_share(g, "active_no", is_dpa | is_phenol),
            "mo_coverage_share": availability_share(g, "mo_pct", is_moly),
            "zddp_calcium_coverage_share": availability_share(g, "calcium_pct", g.index == g.index),
            "zddp_phosphorus_total": antiwear_phosphorus_total,
        }
        if str(scenario_id) in scen.index:
            for col in SCENARIO_COLS:
                val = scen.loc[str(scenario_id), col]
                row[col] = 0.0 if pd.isna(val) else float(val)
        else:
            for col in SCENARIO_COLS:
                row[col] = 0.0
        rows.append(row)
    return pd.DataFrame(rows)


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


def fit_ridge_with_coefficients(X_train: pd.DataFrame, y_train: np.ndarray, alpha: float) -> Tuple[Ridge, SimpleImputer, StandardScaler]:
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_train_imp = imputer.fit_transform(X_train)
    X_train_scaled = scaler.fit_transform(X_train_imp)
    model = Ridge(alpha=alpha, random_state=42)
    model.fit(X_train_scaled, y_train)
    return model, imputer, scaler


def crossfit_oxidation_backbone_train_pred(X_train_full: pd.DataFrame, y_train: np.ndarray, groups_train: np.ndarray) -> np.ndarray:
    inner = GroupKFold(n_splits=4)
    oof = np.zeros_like(y_train, dtype=float)
    for tr_idx, va_idx in inner.split(X_train_full, y_train, groups_train):
        pred = fit_predict_ridge(X_train_full.iloc[tr_idx], y_train[tr_idx], X_train_full.iloc[va_idx], alpha=30.0)
        oof[va_idx] = pred
    return oof


def summarize_coefs(X_train: pd.DataFrame, y_train: np.ndarray, alpha: float, feature_set_name: str) -> pd.DataFrame:
    model, imputer, scaler = fit_ridge_with_coefficients(X_train, y_train, alpha=alpha)
    feature_names = list(X_train.columns)
    coef = pd.Series(model.coef_, index=feature_names, dtype=float)
    out = pd.DataFrame(
        {
            "feature_set": feature_set_name,
            "feature_name": feature_names,
            "coef": coef.values,
            "abs_coef": coef.abs().values,
        }
    )
    return out.sort_values("abs_coef", ascending=False).reset_index(drop=True)


def evaluate_stage4(compact_df: pd.DataFrame, chem_df: pd.DataFrame, groups: np.ndarray) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    merged = compact_df.merge(chem_df, on=SCENARIO_COL, how="left")
    merged = merged.fillna(0.0)
    merged.index = merged[SCENARIO_COL].astype(str)

    X_compact = add_nonlinear_features(
        merged[[c for c in compact_df.columns if c not in {SCENARIO_COL, TARGET_OX, "target_viscosity_delta_pct"}]].copy()
    )
    y_ox = pd.to_numeric(merged[TARGET_OX], errors="coerce").to_numpy(dtype=float)

    splitter = GroupKFold(n_splits=5)
    rows = []
    oof_map = {
        "oxidation_backbone_ridge": np.zeros_like(y_ox, dtype=float),
    }
    coef_frames = []

    for block_name in BLOCKS:
        oof_map[f"oxidation_direct_addon_ridge__{block_name}"] = np.zeros_like(y_ox, dtype=float)
    for block_name in RESIDUAL_BLOCKS:
        oof_map[f"oxidation_residual_ridge__{block_name}"] = np.zeros_like(y_ox, dtype=float)

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(merged, y_ox, groups), start=1):
        Xc_tr, Xc_va = X_compact.iloc[train_idx], X_compact.iloc[valid_idx]
        yo_tr, yo_va = y_ox[train_idx], y_ox[valid_idx]
        groups_tr = groups[train_idx]

        pred_backbone = fit_predict_ridge(Xc_tr, yo_tr, Xc_va, alpha=30.0)
        oof_map["oxidation_backbone_ridge"][valid_idx] = pred_backbone
        rows.append(
            {
                "fold": fold,
                "target_name": "oxidation",
                "model_name": "oxidation_backbone_ridge",
                "feature_set": "compact_v3_full",
                **compute_metrics(yo_va, pred_backbone),
            }
        )

        train_backbone_oof = crossfit_oxidation_backbone_train_pred(Xc_tr, yo_tr, groups_tr)
        resid_yo = yo_tr - train_backbone_oof

        for block_name, cols in BLOCKS.items():
            X_block_tr = merged.iloc[train_idx][cols].copy()
            X_block_va = merged.iloc[valid_idx][cols].copy()
            X_direct_tr = pd.concat([Xc_tr, X_block_tr], axis=1)
            X_direct_va = pd.concat([Xc_va, X_block_va], axis=1)
            pred_direct = fit_predict_ridge(X_direct_tr, yo_tr, X_direct_va, alpha=30.0)
            key = f"oxidation_direct_addon_ridge__{block_name}"
            oof_map[key][valid_idx] = pred_direct
            rows.append(
                {
                    "fold": fold,
                    "target_name": "oxidation",
                    "model_name": key,
                    "feature_set": block_name,
                    **compute_metrics(yo_va, pred_direct),
                }
            )

        for block_name in RESIDUAL_BLOCKS:
            cols = BLOCKS[block_name]
            X_block_tr = merged.iloc[train_idx][cols].copy()
            X_block_va = merged.iloc[valid_idx][cols].copy()
            pred_resid = fit_predict_ridge(X_block_tr, resid_yo, X_block_va, alpha=10.0)
            pred_final = pred_backbone + pred_resid
            key = f"oxidation_residual_ridge__{block_name}"
            oof_map[key][valid_idx] = pred_final
            rows.append(
                {
                    "fold": fold,
                    "target_name": "oxidation",
                    "model_name": key,
                    "feature_set": block_name,
                    **compute_metrics(yo_va, pred_final),
                }
            )

    for model_name, pred in oof_map.items():
        feature_set = "compact_v3_full"
        if "__" in model_name:
            feature_set = model_name.split("__", 1)[1]
        rows.append(
            {
                "fold": np.nan,
                "target_name": "oxidation",
                "model_name": model_name,
                "feature_set": feature_set,
                **compute_metrics(y_ox, pred),
            }
        )

    # Full-data coefficient summaries for the most important narrow blocks.
    for block_name in ["ao_core_narrow", "ao_mo_narrow", "ao_family_narrow", "zddp_narrow", "oxidation_v4_narrow"]:
        coef_frames.append(
            summarize_coefs(merged[BLOCKS[block_name]].copy(), y_ox, alpha=10.0, feature_set_name=block_name)
        )

    oof_df = pd.DataFrame({SCENARIO_COL: merged[SCENARIO_COL].astype(str), "oxidation_true": y_ox, **oof_map})
    coef_df = pd.concat(coef_frames, ignore_index=True)
    return pd.DataFrame(rows), oof_df, coef_df


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    compact_df = read_csv_auto(COMPACT_FULL)
    component_df = pd.read_csv(TRAIN_COMPONENTS)
    scenario_df = pd.read_csv(TRAIN_SCENARIO)
    chem_df = build_oxidation_chemistry_v4(component_df, scenario_df)
    group_map = build_recipe_cluster_groups(component_df)
    groups = np.array([group_map[sid] for sid in compact_df[SCENARIO_COL].astype(str)], dtype=object)

    metrics_df, oof_df, coef_df = evaluate_stage4(compact_df, chem_df, groups)
    fold_df = metrics_df[metrics_df["fold"].notna()].copy()
    summary_df = (
        metrics_df[metrics_df["fold"].isna()]
        .drop(columns=["fold"])
        .sort_values("rmse")
        .reset_index(drop=True)
    )
    stability_df = (
        fold_df.groupby(["model_name", "feature_set"], as_index=False)
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

    metrics_df.to_csv(OUTDIR / "stage4_fold_and_summary_metrics.csv", index=False)
    summary_df.to_csv(OUTDIR / "stage4_summary.csv", index=False)
    stability_df.to_csv(OUTDIR / "stage4_stability.csv", index=False)
    oof_df.to_csv(OUTDIR / "stage4_oof_predictions.csv", index=False)
    chem_df.to_csv(OUTDIR / "oxidation_chemistry_v4_narrow.csv", index=False)
    coef_df.to_csv(OUTDIR / "narrow_block_coefficients.csv", index=False)

    manifest = {
        "main_validator": "recipe_cluster",
        "fixed_backbone": {"dataset": "compact_v3_full", "model": "ridge", "config": "alpha_30.0"},
        "blocks": BLOCKS,
        "residual_blocks": RESIDUAL_BLOCKS,
        "outputs": {
            "metrics": str(OUTDIR / "stage4_fold_and_summary_metrics.csv"),
            "summary": str(OUTDIR / "stage4_summary.csv"),
            "stability": str(OUTDIR / "stage4_stability.csv"),
            "oof": str(OUTDIR / "stage4_oof_predictions.csv"),
            "chemistry_v4": str(OUTDIR / "oxidation_chemistry_v4_narrow.csv"),
            "coefficients": str(OUTDIR / "narrow_block_coefficients.csv"),
        },
    }
    (OUTDIR / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(summary_df.to_string(index=False))
    print(f"\nSaved to {OUTDIR}")


if __name__ == "__main__":
    main()
