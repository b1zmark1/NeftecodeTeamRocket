#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# =========================
# Constants
# =========================

TARGET_VISC = "Delta Kin. Viscosity KV100 - relative | - Daimler Oxidation Test (DOT), %"
TARGET_OXID = "Oxidation EOT | DIN 51453 Daimler Oxidation Test (DOT), A/cm"

COL_SCENARIO = "scenario_id"
COL_COMPONENT = "Компонент"
COL_BATCH = "Наименование партии"
COL_DOSE = "Массовая доля, %"
COL_TEMP = "Температура испытания | ASTM D445 Daimler Oxidation Test (DOT), °C"
COL_TIME = "Время испытания | - Daimler Oxidation Test (DOT), ч"
COL_BIOFUEL = "Количество биотоплива | - Daimler Oxidation Test (DOT), % масс"
COL_CATALYST = "Дозировка катализатора, категория"

PROP_COMPONENT = "Компонент"
PROP_BATCH = "Наименование партии"
PROP_NAME = "Наименование показателя"
PROP_VALUE = "Значение показателя"


# =========================
# Utility functions
# =========================


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



def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    text = re.sub(r"\s+", " ", text)
    return text



def normalize_batch(value: object) -> str:
    text = normalize_text(value)
    lowered = text.lower()

    if lowered in {"", "nan", "none", "null"}:
        return ""
    if lowered == "typical":
        return "typical"
    if lowered in {"б/н", "без номера", "без №", "без no", "б н"}:
        return "no_batch"

    return text



def normalize_component(value: object) -> str:
    return normalize_text(value)



def slugify(text: str) -> str:
    text = normalize_text(text).lower()

    replacements = {
        "°c": "c",
        "°": "",
        "% масс": "pct_mass",
        "%": "pct",
        "мм²/с": "mm2_s",
        "мм2/с": "mm2_s",
        "мг koh/г": "mg_koh_g",
        "мпа*с": "mpa_s",
        "мл": "ml",
        "мин": "min",
        "ч": "h",
        "din 51453": "din51453",
        "astm d445": "astm_d445",
        "astm d6481": "astm_d6481",
        "astm d2896": "astm_d2896",
        "astm d5293": "astm_d5293",
        "astm d1401": "astm_d1401",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    text = text.replace("кинематическая вязкость", "kin_visc")
    text = text.replace("динамическая вязкость", "dyn_visc")
    text = text.replace("вязкость", "visc")
    text = text.replace("массовая доля", "mass_frac")
    text = text.replace("щелочное число", "tbn")
    text = text.replace("кальция", "ca")
    text = text.replace("цинка", "zn")
    text = text.replace("фосфора", "p")
    text = text.replace("деэм.вода", "demuls_water")
    text = text.replace("деэм.время", "demuls_time")
    text = text.replace("деэм.масло", "demuls_oil")
    text = text.replace("деэм.эмульсия", "demuls_emulsion")

    text = re.sub(r"[^a-z0-9а-яё]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text



def extract_component_class(component_name: str) -> str:
    text = normalize_component(component_name)
    if "_" in text:
        return text.rsplit("_", 1)[0]
    return text



def extract_component_local_id(component_name: str) -> Optional[int]:
    text = normalize_component(component_name)
    match = re.search(r"_(\d+)$", text)
    if not match:
        return None
    return int(match.group(1))



def safe_weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    mask = values.notna() & weights.notna()
    if not mask.any():
        return np.nan
    v = values[mask].astype(float)
    w = weights[mask].astype(float)
    w_sum = w.sum()
    if w_sum == 0:
        return np.nan
    return float(np.average(v, weights=w))



def infer_property_role(col_name: str) -> Optional[str]:
    lower = col_name.lower()
    if "visc" in lower and "40" in lower:
        return "visc40"
    if "visc" in lower and "100" in lower:
        return "visc100"
    if re.search(r"(^|_)ca($|_)", lower):
        return "ca"
    if re.search(r"(^|_)zn($|_)", lower):
        return "zn"
    if re.search(r"(^|_)p($|_)", lower):
        return "p"
    if "tbn" in lower:
        return "tbn"
    return None


# =========================
# Property table preparation
# =========================


def prepare_properties(properties_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    required = {PROP_COMPONENT, PROP_BATCH, PROP_NAME, PROP_VALUE}
    missing = required - set(properties_df.columns)
    if missing:
        raise ValueError(f"Missing columns in properties file: {sorted(missing)}")

    df = properties_df.copy()
    df[PROP_COMPONENT] = df[PROP_COMPONENT].map(normalize_component)
    df[PROP_BATCH] = df[PROP_BATCH].map(normalize_batch)
    df[PROP_NAME] = df[PROP_NAME].map(normalize_text)
    df[PROP_VALUE] = pd.to_numeric(df[PROP_VALUE], errors="coerce")
    df = df.dropna(subset=[PROP_VALUE])
    df["property_slug"] = df[PROP_NAME].map(slugify)

    agg = (
        df.groupby([PROP_COMPONENT, PROP_BATCH, "property_slug"], as_index=False)[PROP_VALUE]
        .mean()
    )

    exact = agg[agg[PROP_BATCH] != "typical"].copy()
    properties_exact = exact.pivot_table(
        index=[PROP_COMPONENT, PROP_BATCH],
        columns="property_slug",
        values=PROP_VALUE,
        aggfunc="mean",
    ).reset_index()

    typical = agg[agg[PROP_BATCH] == "typical"].copy()
    properties_typical = typical.pivot_table(
        index=[PROP_COMPONENT],
        columns="property_slug",
        values=PROP_VALUE,
        aggfunc="mean",
    ).reset_index()

    properties_exact.columns.name = None
    properties_typical.columns.name = None

    return properties_exact, properties_typical


# =========================
# Mixture enrichment
# =========================


def enrich_mixtures(
    mixtures_df: pd.DataFrame,
    properties_exact: pd.DataFrame,
    properties_typical: pd.DataFrame,
    is_train: bool,
) -> pd.DataFrame:
    required_cols = {
        COL_SCENARIO,
        COL_COMPONENT,
        COL_BATCH,
        COL_DOSE,
        COL_TEMP,
        COL_TIME,
        COL_BIOFUEL,
        COL_CATALYST,
    }
    if is_train:
        required_cols |= {TARGET_VISC, TARGET_OXID}

    missing = required_cols - set(mixtures_df.columns)
    if missing:
        raise ValueError(f"Missing columns in mixtures file: {sorted(missing)}")

    df = mixtures_df.copy()
    df[COL_COMPONENT] = df[COL_COMPONENT].map(normalize_component)
    df[COL_BATCH] = df[COL_BATCH].map(normalize_batch)

    numeric_cols = [COL_DOSE, COL_TEMP, COL_TIME, COL_BIOFUEL, COL_CATALYST]
    if is_train:
        numeric_cols += [TARGET_VISC, TARGET_OXID]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    exact = properties_exact.copy().rename(
        columns={
            COL_COMPONENT: "_prop_component",
            COL_BATCH: "_prop_batch",
        }
    )
    typical = properties_typical.copy().rename(columns={COL_COMPONENT: "_prop_component"})

    property_cols = [
        col for col in properties_exact.columns if col not in {COL_COMPONENT, COL_BATCH}
    ]

    merged = df.merge(
        exact,
        how="left",
        left_on=[COL_COMPONENT, COL_BATCH],
        right_on=["_prop_component", "_prop_batch"],
        suffixes=("", "_exact"),
    )

    typical_value_cols = {
        col: f"{col}_typical" for col in property_cols if col in typical.columns
    }
    typical = typical.rename(columns=typical_value_cols)

    merged = merged.merge(
        typical,
        how="left",
        left_on=COL_COMPONENT,
        right_on="_prop_component",
        suffixes=("", "_typical_right"),
    )

    flag_data: Dict[str, pd.Series] = {}
    for col in property_cols:
        if col not in merged.columns:
            merged[col] = np.nan

        typical_col = f"{col}_typical"
        if typical_col not in merged.columns:
            merged[typical_col] = np.nan

        exact_available = merged[col].notna()
        typical_available = merged[typical_col].notna()

        flag_data[f"is_exact_{col}"] = exact_available.astype(int)
        flag_data[f"is_typical_{col}"] = (~exact_available & typical_available).astype(int)
        flag_data[f"is_missing_{col}"] = (~exact_available & ~typical_available).astype(int)

        merged[col] = merged[col].fillna(merged[typical_col])
        flag_data[f"has_{col}"] = merged[col].notna().astype(int)

    merged = pd.concat([merged, pd.DataFrame(flag_data, index=merged.index)], axis=1)

    drop_cols = [c for c in ["_prop_component", "_prop_batch"] if c in merged.columns]
    drop_cols += [c for c in merged.columns if c.startswith("_prop_component") or c.startswith("_prop_batch")]
    drop_cols += [f"{col}_typical" for col in property_cols if f"{col}_typical" in merged.columns]
    merged = merged.drop(columns=sorted(set(drop_cols)), errors="ignore")

    if COL_COMPONENT not in merged.columns:
        raise KeyError(
            f"After merge, required column {COL_COMPONENT!r} is missing. "
            f"Columns: {list(merged.columns)}"
        )

    merged["component_class"] = merged[COL_COMPONENT].map(extract_component_class)
    merged["component_local_id"] = merged[COL_COMPONENT].map(extract_component_local_id).fillna(-1).astype(int)

    exact_flag_cols = [c for c in merged.columns if c.startswith("is_exact_")]
    typical_flag_cols = [c for c in merged.columns if c.startswith("is_typical_")]
    missing_flag_cols = [c for c in merged.columns if c.startswith("is_missing_")]

    merged["exact_props_count"] = merged[exact_flag_cols].sum(axis=1)
    merged["typical_props_count"] = merged[typical_flag_cols].sum(axis=1)
    merged["missing_props_count"] = merged[missing_flag_cols].sum(axis=1)

    return merged.copy()


# =========================
# Scenario feature engineering
# =========================


def build_scenario_features(enriched_df: pd.DataFrame, is_train: bool) -> pd.DataFrame:
    df = enriched_df.copy()

    property_cols = []
    excluded = {
        COL_SCENARIO,
        COL_COMPONENT,
        COL_BATCH,
        COL_DOSE,
        COL_TEMP,
        COL_TIME,
        COL_BIOFUEL,
        COL_CATALYST,
        "component_class",
        "component_local_id",
        "exact_props_count",
        "typical_props_count",
        "missing_props_count",
    }
    if is_train:
        excluded |= {TARGET_VISC, TARGET_OXID}

    for col in df.columns:
        if col in excluded:
            continue
        if col.startswith(("is_exact_", "is_typical_", "is_missing_", "has_")):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            property_cols.append(col)

    scenario_rows: List[Dict[str, float]] = []

    for scenario_id, group in df.groupby(COL_SCENARIO, sort=False):
        row: Dict[str, float] = {}
        row[COL_SCENARIO] = scenario_id

        row["temperature"] = float(group[COL_TEMP].iloc[0])
        row["time"] = float(group[COL_TIME].iloc[0])
        row["biofuel"] = float(group[COL_BIOFUEL].iloc[0])
        row["catalyst_category"] = float(group[COL_CATALYST].iloc[0])

        row["num_components"] = int(group.shape[0])
        row["num_component_classes"] = int(group["component_class"].nunique())

        class_counts = group["component_class"].value_counts(dropna=False).to_dict()
        class_dose_sums = group.groupby("component_class")[COL_DOSE].sum(min_count=1).to_dict()

        for cls in sorted(group["component_class"].dropna().astype(str).unique()):
            cls_slug = slugify(cls)
            row[f"num_{cls_slug}"] = int(class_counts.get(cls, 0))
            row[f"sum_dose_{cls_slug}"] = float(class_dose_sums.get(cls, 0.0) or 0.0)

        row["sum_dose_total"] = float(group[COL_DOSE].sum())

        for col in property_cols:
            series = pd.to_numeric(group[col], errors="coerce")
            row[f"mean_{col}"] = float(series.mean()) if series.notna().any() else np.nan
            row[f"max_{col}"] = float(series.max()) if series.notna().any() else np.nan
            row[f"std_{col}"] = float(series.std(ddof=0)) if series.notna().any() else np.nan
            row[f"wmean_{col}"] = safe_weighted_mean(series, group[COL_DOSE])

            role = infer_property_role(col)
            if role is not None:
                row[f"wmean_{role}"] = row[f"wmean_{col}"]
                row[f"max_{role}"] = row[f"max_{col}"]
                row[f"std_{role}"] = row[f"std_{col}"]

        total_quality = (
            group["exact_props_count"] +
            group["typical_props_count"] +
            group["missing_props_count"]
        )
        quality_sum = total_quality.sum()

        if quality_sum > 0:
            row["share_exact_properties"] = float(group["exact_props_count"].sum() / quality_sum)
            row["share_typical_properties"] = float(group["typical_props_count"].sum() / quality_sum)
            row["share_missing_properties"] = float(group["missing_props_count"].sum() / quality_sum)
        else:
            row["share_exact_properties"] = np.nan
            row["share_typical_properties"] = np.nan
            row["share_missing_properties"] = np.nan

        row["mean_exact_props_count"] = float(group["exact_props_count"].mean())
        row["mean_typical_props_count"] = float(group["typical_props_count"].mean())
        row["mean_missing_props_count"] = float(group["missing_props_count"].mean())

        temp = row["temperature"]
        time_val = row["time"]
        biofuel = row["biofuel"]
        catalyst = row["catalyst_category"]

        row["temperature_x_time"] = temp * time_val
        row["temperature_x_biofuel"] = temp * biofuel
        row["temperature_x_catalyst"] = temp * catalyst

        def get_feature(*keys: str) -> float:
            for key in keys:
                if key in row and pd.notna(row[key]):
                    return float(row[key])
            return 0.0

        sum_antioxidant = get_feature("sum_dose_антиоксидант", "sum_dose_antioxidant")
        num_antioxidant = get_feature("num_антиоксидант", "num_antioxidant")
        num_detergent = get_feature("num_детергент", "num_detergent")
        sum_detergent = get_feature("sum_dose_детергент", "sum_dose_detergent")
        wmean_zn = get_feature("wmean_zn")
        wmean_p = get_feature("wmean_p")
        wmean_ca = get_feature("wmean_ca")
        wmean_tbn = get_feature("wmean_tbn")

        row["sum_dose_antioxidant_x_temperature"] = sum_antioxidant * temp
        row["sum_dose_antioxidant_x_biofuel"] = sum_antioxidant * biofuel
        row["num_antioxidant_x_num_detergent"] = num_antioxidant * num_detergent
        row["wmean_zn_x_biofuel"] = wmean_zn * biofuel
        row["wmean_p_x_temperature"] = wmean_p * temp
        row["wmean_ca_x_catalyst"] = wmean_ca * catalyst
        row["sum_dose_detergent_x_wmean_tbn"] = sum_detergent * wmean_tbn

        if is_train:
            row[TARGET_VISC] = float(group[TARGET_VISC].iloc[0])
            row[TARGET_OXID] = float(group[TARGET_OXID].iloc[0])

        scenario_rows.append(row)

    return pd.DataFrame(scenario_rows)


# =========================
# Metrics
# =========================


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


# =========================
# Model selection and training
# =========================


def build_pipeline(n_components: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", PLSRegression(n_components=n_components, scale=False)),
        ]
    )



def evaluate_pls_cv(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    n_components_list: Iterable[int],
    n_splits: int,
) -> pd.DataFrame:
    unique_groups = np.unique(groups)
    if len(unique_groups) < n_splits:
        raise ValueError(
            f"Not enough unique groups ({len(unique_groups)}) for GroupKFold with n_splits={n_splits}."
        )

    cv = GroupKFold(n_splits=n_splits)
    rows: List[Dict[str, float]] = []

    max_valid_n_components = min(X.shape[1], X.shape[0] - math.ceil(X.shape[0] / n_splits))
    candidate_components = [n for n in n_components_list if 1 <= n <= max_valid_n_components]
    if not candidate_components:
        raise ValueError("No valid n_components values available for current dataset size.")

    for n_components in candidate_components:
        fold_idx = 0
        for train_idx, valid_idx in cv.split(X, y, groups):
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



def summarize_cv_results(cv_results: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "rmse_viscosity", "rmse_oxidation",
        "mae_viscosity", "mae_oxidation",
        "r2_viscosity", "r2_oxidation",
        "mean_rmse", "mean_mae", "mean_r2",
    ]
    summary = (
        cv_results.groupby("n_components", as_index=False)[metric_cols]
        .mean()
        .sort_values(["mean_rmse", "mean_mae", "mean_r2"], ascending=[True, True, False])
        .reset_index(drop=True)
    )
    return summary


# =========================
# PLS curated feature selection
# =========================


def select_pls_feature_columns(train_scenarios: pd.DataFrame) -> List[str]:
    """
    Compact, manually curated feature set for PLS.
    This is intentionally short because PLS degrades on a wide noisy matrix.
    """
    preferred_features = [
        # Global DOT conditions
        "temperature",
        "time",
        "biofuel",
        "catalyst_category",

        # Mixture structure
        "num_components",
        "num_component_classes",
        "num_антиоксидант",
        "num_детергент",
        "num_дисперсант",
        "num_противоизносная_присадка",
        "num_базовое_масло",

        # Dose aggregates
        "sum_dose_total",
        "sum_dose_антиоксидант",
        "sum_dose_детергент",
        "sum_dose_дисперсант",
        "sum_dose_противоизносная_присадка",
        "sum_dose_базовое_масло",

        # Key weighted property aggregates
        "wmean_visc40",
        "wmean_visc100",
        "wmean_ca",
        "wmean_zn",
        "wmean_p",
        "wmean_tbn",

        # A few maxima for strong local signals
        "max_visc100",
        "max_ca",
        "max_zn",
        "max_p",

        # Data quality
        "share_exact_properties",
        "share_typical_properties",
        "share_missing_properties",

        # Interaction features
        "temperature_x_time",
        "temperature_x_biofuel",
        "temperature_x_catalyst",
        "sum_dose_antioxidant_x_temperature",
        "sum_dose_antioxidant_x_biofuel",
        "num_antioxidant_x_num_detergent",
        "wmean_zn_x_biofuel",
        "wmean_p_x_temperature",
        "wmean_ca_x_catalyst",
        "sum_dose_detergent_x_wmean_tbn",
    ]

    feature_cols = [
        col for col in preferred_features
        if col in train_scenarios.columns and pd.api.types.is_numeric_dtype(train_scenarios[col])
    ]

    if not feature_cols:
        raise ValueError(
            "No curated PLS features were found in train_scenarios. "
            "Check generated scenario feature columns."
        )

    return feature_cols


# =========================
# Main
# =========================


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PLSRegression for Daimler DOT task with fold metrics.")
    parser.add_argument("--train", type=Path, default=Path("daimler_mixtures_train.csv"))
    parser.add_argument("--test", type=Path, default=Path("daimler_mixtures_test.csv"))
    parser.add_argument("--properties", type=Path, default=Path("daimler_component_properties.csv"))
    parser.add_argument("--outdir", type=Path, default=Path("pls_output"))
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--max-components", type=int, default=12)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    print("RUNNING FILE:", Path(__file__).resolve())

    train_raw = read_csv_auto(args.train)
    test_raw = read_csv_auto(args.test)
    properties_raw = read_csv_auto(args.properties)

    properties_exact, properties_typical = prepare_properties(properties_raw)

    train_components = enrich_mixtures(
        mixtures_df=train_raw,
        properties_exact=properties_exact,
        properties_typical=properties_typical,
        is_train=True,
    )
    test_components = enrich_mixtures(
        mixtures_df=test_raw,
        properties_exact=properties_exact,
        properties_typical=properties_typical,
        is_train=False,
    )

    train_scenarios = build_scenario_features(train_components, is_train=True)
    test_scenarios = build_scenario_features(test_components, is_train=False)

    print("\nTRAIN SCENARIO COLUMNS:")
    print(train_scenarios.columns.tolist())

    feature_cols = select_pls_feature_columns(train_scenarios)
    print("\n=== Selected PLS features ===")
    for feature_name in feature_cols:
        print(feature_name)

    X_train = train_scenarios[feature_cols].copy()
    y_train = train_scenarios[[TARGET_VISC, TARGET_OXID]].to_numpy(dtype=float)
    groups = train_scenarios[COL_SCENARIO].astype(str).to_numpy()

    X_test = test_scenarios.reindex(columns=feature_cols).copy()

    max_possible = min(args.max_components, len(feature_cols), max(1, X_train.shape[0] - 1))
    n_components_list = list(range(1, max_possible + 1))

    cv_results = evaluate_pls_cv(
        X=X_train,
        y=y_train,
        groups=groups,
        n_components_list=n_components_list,
        n_splits=args.n_splits,
    )
    cv_summary = summarize_cv_results(cv_results)
    best_n_components = int(cv_summary.iloc[0]["n_components"])

    final_model = build_pipeline(best_n_components)
    final_model.fit(X_train, y_train)
    y_test_pred = final_model.predict(X_test)

    cv_results_path = args.outdir / "pls_cv_fold_metrics.csv"
    cv_summary_path = args.outdir / "pls_cv_summary.csv"
    predictions_path = args.outdir / "pls_test_predictions.csv"
    train_features_path = args.outdir / "train_scenario_features.csv"
    test_features_path = args.outdir / "test_scenario_features.csv"
    manifest_path = args.outdir / "run_manifest.json"

    cv_results.to_csv(cv_results_path, index=False)
    cv_summary.to_csv(cv_summary_path, index=False)
    train_scenarios.to_csv(train_features_path, index=False)
    test_scenarios.to_csv(test_features_path, index=False)

    predictions = pd.DataFrame({
        COL_SCENARIO: test_scenarios[COL_SCENARIO],
        TARGET_VISC: y_test_pred[:, 0],
        TARGET_OXID: y_test_pred[:, 1],
    })
    predictions.to_csv(predictions_path, index=False)

    manifest = {
        "train_path": str(args.train),
        "test_path": str(args.test),
        "properties_path": str(args.properties),
        "n_splits": args.n_splits,
        "tested_n_components": n_components_list,
        "best_n_components": best_n_components,
        "n_features": len(feature_cols),
        "feature_columns": feature_cols,
        "outputs": {
            "cv_fold_metrics": str(cv_results_path),
            "cv_summary": str(cv_summary_path),
            "train_scenario_features": str(train_features_path),
            "test_scenario_features": str(test_features_path),
            "test_predictions": str(predictions_path),
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