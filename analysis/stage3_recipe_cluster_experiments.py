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
from sklearn.cross_decomposition import PLSRegression
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
    fit_full_single_target_model,
    fit_predict_single_target_mlp,
    predict_full_model,
    read_csv_auto,
    select_features_for_target,
)


OUTDIR = ROOT / "analysis" / "stage3_recipe_cluster"
TRAIN_COMPONENTS = ROOT / "new_datasets" / "train_component_level_transformed.csv"
TRAIN_SCENARIO = ROOT / "new_datasets" / "train_scenario_level_features.csv"
COMPACT_FULL = ROOT / "compact_v3" / "train_flat_features_v3_compact.csv"

TARGET_VISCOUS = "target_viscosity_delta_pct"
TARGET_OX = "target_oxidation_acm"

SCENARIO_ENGINEERED_COLS = [
    "synergy_ao_phenol_x_diphenylamine_active_no",
    "synergy_ao_diphenylamine_active_no_x_mo",
    "synergy_zn_x_boron_dispersant",
    "synergy_aw_sulfur_x_ca_ca_mg",
    "ao_k_avg_weighted_active_no",
    "ao_k_avg_arithmetic",
    "ao_ionization_min",
    "ao_homo_max",
]

OX_CHEM_V3_COLS = [
    "synergy_ao_phenol_x_diphenylamine_active_no",
    "synergy_ao_diphenylamine_active_no_x_mo",
    "ao_k_avg_weighted_active_no",
    "ao_k_avg_arithmetic",
    "ao_ionization_min",
    "ao_homo_max",
    "ao_dpa_active_total",
    "ao_phenol_active_total",
    "ao_dpa_phenol_product",
    "ao_dpa_phenol_ratio",
    "ao_dpa_phenol_min",
    "ao_dpa_phenol_max",
    "ao_dpa_phenol_imbalance",
    "ao_dpa_phenol_hmean",
    "ao_mo_product",
    "ao_mo_log1p",
    "ao_mo_x_severity",
    "ao_pair_x_severity",
    "zddp_calcium_interaction",
    "zddp_calcium_x_severity",
    "zddp_boron_interaction",
    "dominant_base_oil_share",
    "base_oil_group1_share",
    "ao_ratio_x_baseoil_share",
    "ao_min_x_baseoil_kv100",
]

VISC_RESIDUAL_V3_COLS = [
    "synergy_zn_x_boron_dispersant",
    "synergy_aw_sulfur_x_ca_ca_mg",
    "calcium_total",
    "ca_mg_total",
    "boron_total",
    "antiwear_zinc_total",
    "antiwear_sulfur_total",
    "antiwear_phosphorus_total",
    "zddp_boron_interaction",
    "zddp_boron_log1p",
    "zddp_calcium_interaction",
    "zddp_calcium_x_severity",
    "boron_total_x_severity",
    "antiwear_ca_ratio",
    "dominant_base_oil_share",
    "base_oil_kv100_wmean",
    "base_oil_group1_share",
]


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


def build_chemistry_v3(component_df: pd.DataFrame, scenario_df: pd.DataFrame) -> pd.DataFrame:
    scen = scenario_df.set_index(SCENARIO_COL)
    rows = []
    for scenario_id, group in component_df.groupby(SCENARIO_COL, sort=False):
        g = group.copy()
        g["Массовая доля, %"] = pd.to_numeric(g["Массовая доля, %"], errors="coerce").fillna(0.0)
        g["active_no"] = pd.to_numeric(g["Активный Азот / Кислород, % масс. (N или O)"], errors="coerce").fillna(0.0)
        g["mo_pct"] = pd.to_numeric(g["% масс. (Mo)"], errors="coerce").fillna(0.0)
        g["zinc_pct"] = pd.to_numeric(g["Массовая доля цинка, ASTM D6481"], errors="coerce").fillna(0.0)
        g["sulfur_pct"] = pd.to_numeric(g["Массовая доля серы, ASTM D6481"], errors="coerce").fillna(0.0)
        g["phosphorus_pct"] = pd.to_numeric(g["Массовая доля фосфора, ASTM D6481"], errors="coerce").fillna(0.0)
        g["calcium_pct"] = pd.to_numeric(g["Массовая доля кальция, ASTM D6481"], errors="coerce").fillna(0.0)
        g["camg_pct"] = pd.to_numeric(g["Содержание металла (Ca/Mg), % масс."], errors="coerce").fillna(0.0)
        g["boron_pct"] = pd.to_numeric(g["Содержание Бора"], errors="coerce").fillna(0.0)
        g["kv100"] = pd.to_numeric(g["Кинематическая вязкость, при 100°C, ASTM D445"], errors="coerce")
        g["api_group"] = pd.to_numeric(g["Группа по API"], errors="coerce")
        g["Тип АО"] = g["Тип АО"].fillna("").astype(str)
        temp = float(g.iloc[0]["Температура испытания | ASTM D445 Daimler Oxidation Test (DOT), °C"])
        time_h = float(g.iloc[0]["Время испытания | - Daimler Oxidation Test (DOT), ч"])
        severity = time_h * math.exp((temp - 150.0) / 10.0)

        is_dpa = g["Тип АО"].eq("Дифениламин")
        is_phenol = g["Тип АО"].eq("Фенол")
        is_moly = g["Компонент"].astype(str).str.startswith("Соединение_молибдена")
        is_antiwear = g["Компонент"].astype(str).str.startswith("Противоизносная_присадка")
        is_baseoil = g["Компонент"].astype(str).str.startswith("Базовое_масло")

        def dose_sum(mask: pd.Series, col: str) -> float:
            sub = g.loc[mask, ["Массовая доля, %", col]]
            if sub.empty:
                return 0.0
            return float((sub["Массовая доля, %"] * sub[col]).sum())

        ao_dpa_active_total = dose_sum(is_dpa, "active_no")
        ao_phenol_active_total = dose_sum(is_phenol, "active_no")
        ao_prod = ao_dpa_active_total * ao_phenol_active_total
        ao_ratio = ao_dpa_active_total / (ao_phenol_active_total + 1e-6)
        ao_min = min(ao_dpa_active_total, ao_phenol_active_total)
        ao_max = max(ao_dpa_active_total, ao_phenol_active_total)
        ao_imb = abs(ao_dpa_active_total - ao_phenol_active_total)
        ao_hmean = 0.0 if ao_dpa_active_total <= 0 or ao_phenol_active_total <= 0 else 2.0 * ao_dpa_active_total * ao_phenol_active_total / (ao_dpa_active_total + ao_phenol_active_total)

        moly_total = dose_sum(is_moly, "mo_pct")
        antiwear_zinc_total = dose_sum(is_antiwear, "zinc_pct")
        antiwear_sulfur_total = dose_sum(is_antiwear, "sulfur_pct")
        antiwear_phosphorus_total = dose_sum(is_antiwear, "phosphorus_pct")
        calcium_total = dose_sum(g.index == g.index, "calcium_pct")
        camg_total = dose_sum(g.index == g.index, "camg_pct")
        boron_total = dose_sum(g.index == g.index, "boron_pct")

        base_total = float(g.loc[is_baseoil, "Массовая доля, %"].sum())
        dominant_base_oil_share = 0.0 if base_total <= 0 else float(g.loc[is_baseoil, "Массовая доля, %"].max() / base_total)
        base_kv100 = weighted_mean(g, "kv100", is_baseoil)
        base_group1_share = 0.0 if base_total <= 0 else float(g.loc[is_baseoil & g["api_group"].eq(1.0), "Массовая доля, %"].sum() / base_total)

        row = {
            SCENARIO_COL: str(scenario_id),
            "calcium_total": calcium_total,
            "ca_mg_total": camg_total,
            "boron_total": boron_total,
            "antiwear_zinc_total": antiwear_zinc_total,
            "antiwear_sulfur_total": antiwear_sulfur_total,
            "antiwear_phosphorus_total": antiwear_phosphorus_total,
            "ao_dpa_active_total": ao_dpa_active_total,
            "ao_phenol_active_total": ao_phenol_active_total,
            "ao_dpa_phenol_product": ao_prod,
            "ao_dpa_phenol_ratio": ao_ratio,
            "ao_dpa_phenol_min": ao_min,
            "ao_dpa_phenol_max": ao_max,
            "ao_dpa_phenol_imbalance": ao_imb,
            "ao_dpa_phenol_hmean": ao_hmean,
            "ao_mo_product": ao_dpa_active_total * moly_total,
            "ao_mo_log1p": math.log1p(max(ao_dpa_active_total * moly_total, 0.0)),
            "ao_mo_x_severity": ao_dpa_active_total * moly_total * severity,
            "ao_pair_x_severity": ao_prod * severity,
            "zddp_boron_interaction": antiwear_zinc_total * boron_total,
            "zddp_boron_log1p": math.log1p(max(antiwear_zinc_total * boron_total, 0.0)),
            "zddp_calcium_interaction": antiwear_sulfur_total * (calcium_total + camg_total),
            "zddp_calcium_x_severity": antiwear_sulfur_total * (calcium_total + camg_total) * severity,
            "boron_total_x_severity": boron_total * severity,
            "dominant_base_oil_share": dominant_base_oil_share,
            "base_oil_kv100_wmean": 0.0 if pd.isna(base_kv100) else base_kv100,
            "base_oil_group1_share": base_group1_share,
            "ao_ratio_x_baseoil_share": ao_ratio * dominant_base_oil_share,
            "ao_min_x_baseoil_kv100": ao_min * (0.0 if pd.isna(base_kv100) else base_kv100),
            "antiwear_ca_ratio": (calcium_total + camg_total) / (antiwear_sulfur_total + 1e-6),
        }
        if str(scenario_id) in scen.index:
            for col in SCENARIO_ENGINEERED_COLS:
                val = scen.loc[str(scenario_id), col]
                row[col] = 0.0 if pd.isna(val) else float(val)
        else:
            for col in SCENARIO_ENGINEERED_COLS:
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


def fit_full_ridge_predict(X_train: pd.DataFrame, y_train: np.ndarray, X_valid: pd.DataFrame, alpha: float) -> np.ndarray:
    return fit_predict_ridge(X_train, y_train, X_valid, alpha)


def fit_full_pls_predict(X_train: pd.DataFrame, y_train: np.ndarray, X_valid: pd.DataFrame, n_components: int) -> np.ndarray:
    return fit_predict_pls(X_train, y_train, X_valid, n_components)


def fit_predict_viscosity_backbone(X_train_full: pd.DataFrame, y_train: np.ndarray, X_valid_full: pd.DataFrame) -> np.ndarray:
    X_train, X_valid, _report, _selected = select_features_for_target(
        X_train_full, X_valid_full, y_train=y_train, target_name="viscosity", min_non_na_ratio=0.08, variance_threshold=1e-8, max_features=20
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


def fit_full_viscosity_bundle(X_train_full: pd.DataFrame, y_train: np.ndarray):
    X_train, _X_valid, _report, selected = select_features_for_target(
        X_train_full, X_train_full, y_train=y_train, target_name="viscosity", min_non_na_ratio=0.08, variance_threshold=1e-8, max_features=20
    )
    bundle = fit_full_single_target_model(
        X=X_train,
        y=y_train,
        target_name="viscosity",
        config={"config_id": "relu_small", "hidden_layers": (64, 32), "alpha": 1e-4, "learning_rate_init": 1e-3, "activation": "relu"},
        random_state=42,
        max_iter=700,
    )
    return bundle, selected


def predict_viscosity_bundle(bundle, selected: List[str], X_valid_full: pd.DataFrame) -> np.ndarray:
    return predict_full_model(bundle, X_valid_full[selected].copy())


def crossfit_viscosity_backbone_train_pred(X_train_full: pd.DataFrame, y_train: np.ndarray, groups_train: np.ndarray) -> np.ndarray:
    inner = GroupKFold(n_splits=4)
    oof = np.zeros_like(y_train, dtype=float)
    for tr_idx, va_idx in inner.split(X_train_full, y_train, groups_train):
        pred = fit_predict_viscosity_backbone(X_train_full.iloc[tr_idx], y_train[tr_idx], X_train_full.iloc[va_idx])
        oof[va_idx] = pred
    return oof


def crossfit_oxidation_backbone_train_pred(X_train_full: pd.DataFrame, y_train: np.ndarray, groups_train: np.ndarray) -> np.ndarray:
    inner = GroupKFold(n_splits=4)
    oof = np.zeros_like(y_train, dtype=float)
    for tr_idx, va_idx in inner.split(X_train_full, y_train, groups_train):
        pred = fit_predict_ridge(X_train_full.iloc[tr_idx], y_train[tr_idx], X_train_full.iloc[va_idx], alpha=30.0)
        oof[va_idx] = pred
    return oof


def evaluate_stage3(
    compact_df: pd.DataFrame,
    chem_df: pd.DataFrame,
    groups: np.ndarray,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    merged = compact_df.merge(chem_df, on=SCENARIO_COL, how="left")
    merged = merged.fillna(0.0)
    merged.index = merged[SCENARIO_COL].astype(str)

    X_compact = add_nonlinear_features(merged[[c for c in compact_df.columns if c not in {SCENARIO_COL, TARGET_VISCOUS, TARGET_OX}]].copy())
    y_visc = pd.to_numeric(merged[TARGET_VISCOUS], errors="coerce").to_numpy(dtype=float)
    y_ox = pd.to_numeric(merged[TARGET_OX], errors="coerce").to_numpy(dtype=float)

    X_ox_v3 = merged[OX_CHEM_V3_COLS].copy()
    X_visc_v3 = merged[VISC_RESIDUAL_V3_COLS].copy()
    X_compact_plus_ox = pd.concat([X_compact, X_ox_v3], axis=1)

    rows = []
    oof_rows = []
    splitter = GroupKFold(n_splits=5)
    scenario_ids = merged[SCENARIO_COL].astype(str).to_numpy()

    oof_backbone_visc = np.zeros_like(y_visc, dtype=float)
    oof_resid_visc = np.zeros_like(y_visc, dtype=float)
    oof_backbone_ox = np.zeros_like(y_ox, dtype=float)
    oof_ox_direct_ridge = np.zeros_like(y_ox, dtype=float)
    oof_ox_direct_pls = np.zeros_like(y_ox, dtype=float)
    oof_resid_ox_ridge = np.zeros_like(y_ox, dtype=float)
    oof_resid_ox_pls = np.zeros_like(y_ox, dtype=float)

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(merged, y_visc, groups), start=1):
        Xc_tr, Xc_va = X_compact.iloc[train_idx], X_compact.iloc[valid_idx]
        Xoxv3_tr, Xoxv3_va = X_ox_v3.iloc[train_idx], X_ox_v3.iloc[valid_idx]
        Xviscv3_tr, Xviscv3_va = X_visc_v3.iloc[train_idx], X_visc_v3.iloc[valid_idx]
        Xoxplus_tr, Xoxplus_va = X_compact_plus_ox.iloc[train_idx], X_compact_plus_ox.iloc[valid_idx]
        yv_tr, yv_va = y_visc[train_idx], y_visc[valid_idx]
        yo_tr, yo_va = y_ox[train_idx], y_ox[valid_idx]
        groups_tr = groups[train_idx]

        # Fixed backbones
        pred_backbone_visc = fit_predict_viscosity_backbone(Xc_tr, yv_tr, Xc_va)
        pred_backbone_ox = fit_predict_ridge(Xc_tr, yo_tr, Xc_va, alpha=30.0)

        # Direct oxidation chemistry add-ons
        pred_ox_direct_ridge = fit_predict_ridge(Xoxplus_tr, yo_tr, Xoxplus_va, alpha=30.0)
        pred_ox_direct_pls = fit_predict_pls(Xoxplus_tr, yo_tr, Xoxplus_va, n_components=6)

        # Residual viscosity on v3 interactions
        train_backbone_visc_oof = crossfit_viscosity_backbone_train_pred(Xc_tr, yv_tr, groups_tr)
        resid_yv = yv_tr - train_backbone_visc_oof
        pred_resid_visc = fit_predict_ridge(Xviscv3_tr, resid_yv, Xviscv3_va, alpha=10.0)
        pred_visc_final = pred_backbone_visc + pred_resid_visc

        # Residual oxidation on v3 chemistry
        train_backbone_ox_oof = crossfit_oxidation_backbone_train_pred(Xc_tr, yo_tr, groups_tr)
        resid_yo = yo_tr - train_backbone_ox_oof
        pred_resid_ox_ridge = fit_predict_ridge(Xoxv3_tr, resid_yo, Xoxv3_va, alpha=30.0)
        pred_resid_ox_pls = fit_predict_pls(Xoxv3_tr, resid_yo, Xoxv3_va, n_components=4)
        pred_ox_final_ridge = pred_backbone_ox + pred_resid_ox_ridge
        pred_ox_final_pls = pred_backbone_ox + pred_resid_ox_pls

        for model_name, y_true, y_pred in [
            ("viscosity_backbone_mlp", yv_va, pred_backbone_visc),
            ("viscosity_backbone_plus_residual_ridge_v3", yv_va, pred_visc_final),
            ("oxidation_backbone_ridge", yo_va, pred_backbone_ox),
            ("oxidation_direct_addon_ridge_v3", yo_va, pred_ox_direct_ridge),
            ("oxidation_direct_addon_pls_v3", yo_va, pred_ox_direct_pls),
            ("oxidation_backbone_plus_residual_ridge_v3", yo_va, pred_ox_final_ridge),
            ("oxidation_backbone_plus_residual_pls_v3", yo_va, pred_ox_final_pls),
        ]:
            target_name = "viscosity" if model_name.startswith("viscosity") else "oxidation"
            rows.append({"fold": fold, "target_name": target_name, "model_name": model_name, **compute_metrics(y_true, y_pred)})

        oof_backbone_visc[valid_idx] = pred_backbone_visc
        oof_resid_visc[valid_idx] = pred_visc_final
        oof_backbone_ox[valid_idx] = pred_backbone_ox
        oof_ox_direct_ridge[valid_idx] = pred_ox_direct_ridge
        oof_ox_direct_pls[valid_idx] = pred_ox_direct_pls
        oof_resid_ox_ridge[valid_idx] = pred_ox_final_ridge
        oof_resid_ox_pls[valid_idx] = pred_ox_final_pls

    for model_name, target_name, y_true, y_pred in [
        ("viscosity_backbone_mlp", "viscosity", y_visc, oof_backbone_visc),
        ("viscosity_backbone_plus_residual_ridge_v3", "viscosity", y_visc, oof_resid_visc),
        ("oxidation_backbone_ridge", "oxidation", y_ox, oof_backbone_ox),
        ("oxidation_direct_addon_ridge_v3", "oxidation", y_ox, oof_ox_direct_ridge),
        ("oxidation_direct_addon_pls_v3", "oxidation", y_ox, oof_ox_direct_pls),
        ("oxidation_backbone_plus_residual_ridge_v3", "oxidation", y_ox, oof_resid_ox_ridge),
        ("oxidation_backbone_plus_residual_pls_v3", "oxidation", y_ox, oof_resid_ox_pls),
    ]:
        rows.append({"fold": np.nan, "target_name": target_name, "model_name": model_name, **compute_metrics(y_true, y_pred)})

    oof_df = pd.DataFrame(
        {
            SCENARIO_COL: scenario_ids,
            "viscosity_true": y_visc,
            "viscosity_backbone_mlp": oof_backbone_visc,
            "viscosity_backbone_plus_residual_ridge_v3": oof_resid_visc,
            "oxidation_true": y_ox,
            "oxidation_backbone_ridge": oof_backbone_ox,
            "oxidation_direct_addon_ridge_v3": oof_ox_direct_ridge,
            "oxidation_direct_addon_pls_v3": oof_ox_direct_pls,
            "oxidation_backbone_plus_residual_ridge_v3": oof_resid_ox_ridge,
            "oxidation_backbone_plus_residual_pls_v3": oof_resid_ox_pls,
        }
    )
    return pd.DataFrame(rows), oof_df


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    compact_df = read_csv_auto(COMPACT_FULL)
    component_df = pd.read_csv(TRAIN_COMPONENTS)
    scenario_df = pd.read_csv(TRAIN_SCENARIO)
    chem_df = build_chemistry_v3(component_df, scenario_df)
    group_map = build_recipe_cluster_groups(component_df)
    groups = np.array([group_map[sid] for sid in compact_df[SCENARIO_COL].astype(str)], dtype=object)

    metrics_df, oof_df = evaluate_stage3(compact_df, chem_df, groups)
    fold_df = metrics_df[metrics_df["fold"].notna()].copy()
    summary_df = metrics_df[metrics_df["fold"].isna()].drop(columns=["fold"]).sort_values(["target_name", "rmse"]).reset_index(drop=True)

    metrics_df.to_csv(OUTDIR / "stage3_fold_and_summary_metrics.csv", index=False)
    summary_df.to_csv(OUTDIR / "stage3_summary.csv", index=False)
    oof_df.to_csv(OUTDIR / "stage3_oof_predictions.csv", index=False)
    chem_df.to_csv(OUTDIR / "chemistry_v3_features.csv", index=False)

    manifest = {
        "backbones_fixed": {
            "viscosity": {"dataset": "compact_v3_full", "model": "mlp", "config": "relu_small"},
            "oxidation": {"dataset": "compact_v3_full", "model": "ridge", "config": "alpha_30.0"},
        },
        "recipe_cluster_groups": sorted(set(groups.tolist())),
        "oxidation_v3_cols": OX_CHEM_V3_COLS,
        "viscosity_v3_cols": VISC_RESIDUAL_V3_COLS,
        "outputs": {
            "metrics": str(OUTDIR / "stage3_fold_and_summary_metrics.csv"),
            "summary": str(OUTDIR / "stage3_summary.csv"),
            "oof": str(OUTDIR / "stage3_oof_predictions.csv"),
            "chemistry_v3": str(OUTDIR / "chemistry_v3_features.csv"),
        },
    }
    (OUTDIR / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary_df.to_string(index=False))
    print(f"\nSaved to {OUTDIR}")


if __name__ == "__main__":
    main()
