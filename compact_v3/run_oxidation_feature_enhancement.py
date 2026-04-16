#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compact_v3.build_compact_v3_from_raw import (  # noqa: E402
    COL_BATCH,
    COL_BIOFUEL,
    COL_CATALYST,
    COL_COMPONENT,
    COL_DOSE,
    COL_SCENARIO,
    COL_TEMP,
    COL_TIME,
    normalize_batch,
    normalize_component,
    parse_numeric_value,
    read_csv_auto,
    slugify,
    to_family,
)


DEFAULT_BASE_TRAIN = ROOT / "compact_v3" / "train_flat_features_v3_compact.csv"
DEFAULT_RAW_TRAIN = ROOT / "docs" / "daimler_mixtures_train.csv"
DEFAULT_PROPERTIES = ROOT / "docs" / "daimler_component_properties.csv"
DEFAULT_OUTDIR = ROOT / "compact_v3" / "oxidation_enhancement"

SCENARIO_COL = "scenario_id"
TARGET_COL = "target_oxidation_acm"
TARGET_CLIP = (-6.0, 6.0)
ANTIOXIDANT_FAMILY = to_family("Антиоксидант_1")
DETERGENT_FAMILY = to_family("Детергент_1")
MOLY_FAMILY = to_family("Соединение_молибдена_1")

BASELINE_FEATURES = [
    "temperature_c",
    "time_h",
    "biofuel_pct",
    "catalyst_category",
    "severity_exp",
    "temperature_x_time",
    "temperature_x_biofuel",
    "severity_x_biofuel",
    "severity_x_catalyst",
    "regime_load",
    "nl__temperature_centered",
    "nl__temperature_centered_sq",
    "nl__time_log1p",
    "nl__biofuel_sq",
    "nl__severity_log1p",
    "nl__severity_sq_scaled",
    "nl__regime_load_log1p",
    "nl__catalyst_x_biofuel_sq",
    "textnum__antioxidant__тип_ао__ao_type_is_phenol__wm",
    "textnum__antioxidant__тип_ао__ao_type_is_diphenylamine__wm",
    "antioxidant__ionization_ev__wm",
    "antioxidant__bde_xh_kcal__wm",
    "antioxidant_balance",
    "nl__phenol_x_severity",
    "nl__dpa_x_severity",
    "nl__antioxidant_energy",
    "nl__antioxidant_balance_abs",
    "detergent__tbn_astm__wm",
    "nl__tbn_x_biofuel",
    "interaction__biofuel_x_zddp",
    "antiwear__sulfur_pct__wm",
    "textnum__antiwear__тип_спиртового_радикала__radical_is_secondary__wm",
    "base_oil__ccs_m35__wm",
    "base_oil__pour_point_c__max",
    "base_oil_visc_ratio",
    "nl__baseoil_visc_ratio_sq",
    "nl__ccs_x_temp",
]

AO_EXTRA_RAW_PROPS = {
    "antioxidant__active_no_pct__wm": "Активный Азот / Кислород, % масс. (N или O)",
    "antioxidant__steric_factor__wm": "Стерический фактор, Å3",
    "antioxidant__chemical_potential__wm": "Химический потенциал, Дж/моль",
    "antioxidant__homo_ev__wm": "Энергия ВЗМО, эВ",
    "antioxidant__lumo_ev__wm": "Энергия НСМО, эВ",
    "antioxidant__dipole_moment__wm": "Дипольный момент, Д",
    "antioxidant__nitrogen_total_pct__wm": "Общее содержание азота | ASTM D3228",
}

MOLY_NUMERIC_RAW_PROPS = {
    "moly__mo_pct__wm": "% масс. (Mo)",
    "moly__s_mo_ratio__wm": "Отношение S:Mo",
    "moly__phosphorus_pct__wm": "Массовая доля фосфора, ASTM D6481",
    "moly__coc_c__wm": "COC (°C)",
}

MOLY_TEXT_RAW_PROPS = {
    "moly__category_modtc__wm": ("Категория", "modtc"),
    "moly__category_modtp__wm": ("Категория", "modtp"),
    "moly__category_amine_complex__wm": ("Категория", "amine_complex"),
    "moly__ligand_dtc__wm": ("Тип лиганда", "ligand_dtc"),
    "moly__ligand_dtp__wm": ("Тип лиганда", "ligand_dtp"),
    "moly__ligand_amide__wm": ("Тип лиганда", "ligand_amide"),
}

CONFIDENCE_PROP_SOURCES = {
    "confidence__ao_ionization_exact_share": (ANTIOXIDANT_FAMILY, "Потенциал ионизации,эВ"),
    "confidence__ao_bde_exact_share": (ANTIOXIDANT_FAMILY, "Энергия диссоциации связи Х-Н, ккал/моль"),
    "confidence__ao_active_no_exact_share": (ANTIOXIDANT_FAMILY, "Активный Азот / Кислород, % масс. (N или O)"),
    "confidence__moly_mo_pct_exact_share": (MOLY_FAMILY, "% масс. (Mo)"),
    "confidence__moly_category_exact_share": (MOLY_FAMILY, "Категория"),
    "confidence__detergent_tbn_exact_share": (DETERGENT_FAMILY, "Щелочное число, ASTM D2896"),
}


def add_nonlinear_features(X: pd.DataFrame) -> pd.DataFrame:
    out = X.copy()

    def s(name: str) -> pd.Series:
        if name in out.columns:
            return pd.to_numeric(out[name], errors="coerce")
        return pd.Series(np.nan, index=out.index, dtype=float)

    temperature = s("temperature_c")
    time_h = s("time_h")
    biofuel = s("biofuel_pct")
    catalyst = s("catalyst_category")
    severity = s("severity_exp")
    zddp_bio = s("interaction__biofuel_x_zddp")
    kv40 = s("base_oil__kv40__wm")
    kv100 = s("base_oil__kv100__wm").replace(0.0, np.nan)
    ccs_m35 = s("base_oil__ccs_m35__wm")
    antiwear_dose = s("family_total_dose__antiwear")
    antiwear_s = s("antiwear__sulfur_pct__wm")
    antiwear_chain = s("antiwear__chain_length__wm")
    phenol = s("textnum__antioxidant__тип_ао__ao_type_is_phenol__wm")
    dpa = s("textnum__antioxidant__тип_ао__ao_type_is_diphenylamine__wm")
    ionization = s("antioxidant__ionization_ev__wm")
    bde = s("antioxidant__bde_xh_kcal__wm")
    tbn = s("detergent__tbn_astm__wm")
    regime_load = s("regime_load")

    temp_centered = temperature - 150.0
    out["nl__temperature_centered"] = temp_centered
    out["nl__temperature_centered_sq"] = (temp_centered / 10.0) ** 2
    out["nl__time_log1p"] = np.log1p(np.clip(time_h, a_min=0.0, a_max=None))
    out["nl__biofuel_sq"] = (biofuel / 10.0) ** 2
    out["nl__severity_log1p"] = np.log1p(np.clip(severity, a_min=0.0, a_max=None))
    out["nl__severity_sq_scaled"] = (severity / 1000.0) ** 2
    out["nl__zddp_bio_log1p"] = np.log1p(np.clip(zddp_bio, a_min=0.0, a_max=None))
    out["nl__antiwear_x_biofuel"] = antiwear_dose * biofuel
    out["nl__antiwear_x_temp"] = antiwear_dose * temp_centered
    out["nl__sulfur_x_chain"] = antiwear_s * antiwear_chain
    out["nl__phenol_x_severity"] = phenol * severity
    out["nl__dpa_x_severity"] = dpa * severity
    out["nl__antioxidant_energy"] = ionization * bde
    out["nl__antioxidant_balance_abs"] = (phenol - dpa).abs()
    out["nl__tbn_x_biofuel"] = tbn * biofuel
    out["nl__baseoil_visc_ratio_sq"] = (kv40 / kv100) ** 2
    out["nl__ccs_x_temp"] = ccs_m35 * temp_centered
    out["nl__regime_load_log1p"] = np.log1p(np.clip(regime_load, a_min=0.0, a_max=None))
    out["nl__catalyst_x_biofuel_sq"] = catalyst * ((biofuel / 10.0) ** 2)
    return out


def transform_target(y: np.ndarray) -> np.ndarray:
    return np.log1p(np.clip(y, a_min=0.0, a_max=None))


def inverse_transform_target(y_t: np.ndarray) -> np.ndarray:
    return np.expm1(np.clip(y_t, *TARGET_CLIP))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def lookup_numeric_with_source(
    comp: str,
    batch: str,
    prop_slug: str,
    exact: Dict[Tuple[str, str, str], float],
    typical: Dict[Tuple[str, str], float],
) -> Tuple[float, str]:
    if (comp, batch, prop_slug) in exact:
        return exact[(comp, batch, prop_slug)], "exact"
    if (comp, prop_slug) in typical:
        return typical[(comp, prop_slug)], "typical"
    return np.nan, "missing"


def lookup_text_with_source(
    comp: str,
    batch: str,
    prop_slug: str,
    exact: Dict[Tuple[str, str, str], str],
    typical: Dict[Tuple[str, str], str],
) -> Tuple[str, str]:
    if (comp, batch, prop_slug) in exact:
        return str(exact[(comp, batch, prop_slug)]), "exact"
    if (comp, prop_slug) in typical:
        return str(typical[(comp, prop_slug)]), "typical"
    return "", "missing"


def map_text_flag(kind: str, value: str) -> float:
    text = str(value).strip().lower()
    if not text:
        return np.nan
    if kind == "modtc":
        return float("modtc" in text)
    if kind == "modtp":
        return float("modtp" in text)
    if kind == "amine_complex":
        return float("амин" in text)
    if kind == "ligand_dtc":
        return float("дитиокарб" in text)
    if kind == "ligand_dtp":
        return float("дитиофосфат" in text)
    if kind == "ligand_amide":
        return float("амид" in text)
    return np.nan


def prepare_custom_property_lookups(
    properties_path: Path,
) -> Tuple[
    Dict[Tuple[str, str, str], float],
    Dict[Tuple[str, str], float],
    Dict[Tuple[str, str, str], str],
    Dict[Tuple[str, str], str],
]:
    df = read_csv_auto(properties_path).copy()
    df["Компонент"] = df["Компонент"].map(normalize_component)
    df["Наименование партии"] = df["Наименование партии"].map(normalize_batch)
    df["Наименование показателя"] = df["Наименование показателя"].map(slugify)

    needed_names = set(AO_EXTRA_RAW_PROPS.values()) | set(MOLY_NUMERIC_RAW_PROPS.values())
    needed_names |= {raw_name for raw_name, _ in MOLY_TEXT_RAW_PROPS.values()}
    needed_names |= {raw_name for _, raw_name in CONFIDENCE_PROP_SOURCES.values()}
    needed_names |= {"Тип АО", "Щелочное число, ASTM D2896"}
    needed_slugs = {slugify(name) for name in needed_names}

    df = df[df["Наименование показателя"].isin(needed_slugs)].copy()

    numeric_exact: Dict[Tuple[str, str, str], float] = {}
    numeric_typical: Dict[Tuple[str, str], float] = {}
    text_exact: Dict[Tuple[str, str, str], str] = {}
    text_typical: Dict[Tuple[str, str], str] = {}

    for row in df.to_dict("records"):
        comp = row["Компонент"]
        batch = row["Наименование партии"]
        prop = row["Наименование показателя"]
        raw_val = row["Значение показателя"]
        num_val = parse_numeric_value(raw_val)
        text_val = str(raw_val).strip() if pd.notna(raw_val) else ""

        if pd.notna(num_val):
            if batch == "typical":
                numeric_typical[(comp, prop)] = float(num_val)
            else:
                numeric_exact[(comp, batch, prop)] = float(num_val)

        if text_val:
            if batch == "typical":
                text_typical[(comp, prop)] = text_val
            else:
                text_exact[(comp, batch, prop)] = text_val

    return numeric_exact, numeric_typical, text_exact, text_typical


def safe_weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    mask = values.notna() & weights.notna()
    if not mask.any():
        return np.nan
    vals = values[mask].astype(float)
    w = weights[mask].astype(float)
    if float(w.sum()) == 0.0:
        return np.nan
    return float(np.average(vals, weights=w))


def build_extra_features(raw_train_path: Path, properties_path: Path) -> pd.DataFrame:
    mixtures = read_csv_auto(raw_train_path).copy()
    mixtures[COL_COMPONENT] = mixtures[COL_COMPONENT].map(normalize_component)
    mixtures[COL_BATCH] = mixtures[COL_BATCH].map(normalize_batch)
    for col in [COL_DOSE, COL_TEMP, COL_TIME, COL_BIOFUEL, COL_CATALYST]:
        mixtures[col] = mixtures[col].map(parse_numeric_value)
    mixtures["family"] = mixtures[COL_COMPONENT].map(to_family)

    num_exact, num_typical, text_exact, text_typical = prepare_custom_property_lookups(properties_path)

    ao_type_slug = slugify("Тип АО")
    detergent_tbn_slug = slugify("Щелочное число, ASTM D2896")
    ao_dpa_vals, ao_phenol_vals, detergent_tbn_vals = [], [], []
    for comp, batch, family in zip(mixtures[COL_COMPONENT], mixtures[COL_BATCH], mixtures["family"]):
        if family == ANTIOXIDANT_FAMILY:
            raw_value, _ = lookup_text_with_source(comp, batch, ao_type_slug, text_exact, text_typical)
            low = str(raw_value).strip().lower()
            ao_dpa_vals.append(float("дифениламин" in low) if low else np.nan)
            ao_phenol_vals.append(float("фенол" in low) if low else np.nan)
        else:
            ao_dpa_vals.append(np.nan)
            ao_phenol_vals.append(np.nan)
        if family == DETERGENT_FAMILY:
            tbn_val, _ = lookup_numeric_with_source(comp, batch, detergent_tbn_slug, num_exact, num_typical)
            detergent_tbn_vals.append(tbn_val)
        else:
            detergent_tbn_vals.append(np.nan)
    mixtures["_tmp_ao_dpa_flag"] = ao_dpa_vals
    mixtures["_tmp_ao_phenol_flag"] = ao_phenol_vals
    mixtures["_tmp_detergent_tbn"] = detergent_tbn_vals

    # Attach extra AO numeric features and their source flags.
    for feat, raw_name in AO_EXTRA_RAW_PROPS.items():
        slug = slugify(raw_name)
        values, sources = [], []
        for comp, batch in zip(mixtures[COL_COMPONENT], mixtures[COL_BATCH]):
            value, source = lookup_numeric_with_source(comp, batch, slug, num_exact, num_typical)
            values.append(value)
            sources.append(source)
        mixtures[feat] = values
        mixtures[f"{feat}__source"] = sources

    # Attach Mo numeric features and source flags.
    for feat, raw_name in MOLY_NUMERIC_RAW_PROPS.items():
        slug = slugify(raw_name)
        values, sources = [], []
        for comp, batch in zip(mixtures[COL_COMPONENT], mixtures[COL_BATCH]):
            value, source = lookup_numeric_with_source(comp, batch, slug, num_exact, num_typical)
            values.append(value)
            sources.append(source)
        mixtures[feat] = values
        mixtures[f"{feat}__source"] = sources

    # Attach Mo text-derived features and source flags.
    for feat, (raw_name, kind) in MOLY_TEXT_RAW_PROPS.items():
        slug = slugify(raw_name)
        values, sources = [], []
        for comp, batch in zip(mixtures[COL_COMPONENT], mixtures[COL_BATCH]):
            raw_value, source = lookup_text_with_source(comp, batch, slug, text_exact, text_typical)
            values.append(map_text_flag(kind, raw_value))
            sources.append(source)
        mixtures[feat] = values
        mixtures[f"{feat}__source"] = sources

    # Confidence flags for current key properties.
    conf_cols = {}
    for feat, (family, raw_name) in CONFIDENCE_PROP_SOURCES.items():
        slug = slugify(raw_name)
        sources = []
        for comp, batch in zip(mixtures[COL_COMPONENT], mixtures[COL_BATCH]):
            fam = to_family(comp)
            if fam != family:
                sources.append("missing")
                continue
            if raw_name in ["Категория"]:
                _, source = lookup_text_with_source(comp, batch, slug, text_exact, text_typical)
            else:
                _, source = lookup_numeric_with_source(comp, batch, slug, num_exact, num_typical)
            sources.append(source)
        conf_cols[feat] = sources
    for feat, sources in conf_cols.items():
        mixtures[f"{feat}__source"] = sources

    rows = []
    for scenario_id, group in mixtures.groupby(COL_SCENARIO, sort=False):
        row: Dict[str, float] = {SCENARIO_COL: str(scenario_id)}

        weights = pd.to_numeric(group[COL_DOSE], errors="coerce")

        def wmean_for_family(col: str, family: str) -> float:
            sub = group[group["family"] == family]
            if sub.empty:
                return np.nan
            return safe_weighted_mean(pd.to_numeric(sub[col], errors="coerce"), pd.to_numeric(sub[COL_DOSE], errors="coerce"))

        def exact_share_for_family(source_col: str, family: str) -> float:
            sub = group[group["family"] == family]
            if sub.empty:
                return np.nan
            w = pd.to_numeric(sub[COL_DOSE], errors="coerce")
            if float(w.sum()) == 0.0:
                return np.nan
            is_exact = (sub[source_col] == "exact").astype(float)
            return float(np.average(is_exact, weights=w))

        row["family_total_dose__moly"] = float(pd.to_numeric(group.loc[group["family"] == MOLY_FAMILY, COL_DOSE], errors="coerce").sum(min_count=1) or 0.0)

        for feat in AO_EXTRA_RAW_PROPS:
            row[feat] = wmean_for_family(feat, ANTIOXIDANT_FAMILY)
        for feat in MOLY_NUMERIC_RAW_PROPS:
            row[feat] = wmean_for_family(feat, MOLY_FAMILY)
        for feat in MOLY_TEXT_RAW_PROPS:
            row[feat] = wmean_for_family(feat, MOLY_FAMILY)

        for feat in CONFIDENCE_PROP_SOURCES:
            family = CONFIDENCE_PROP_SOURCES[feat][0]
            row[feat] = exact_share_for_family(f"{feat}__source", family)

        row["moly__present"] = float(row["family_total_dose__moly"] > 0)
        row["interaction__moly_x_diphenylamine"] = row["family_total_dose__moly"] * float(
            safe_weighted_mean(
                pd.to_numeric(group.loc[group["family"] == ANTIOXIDANT_FAMILY, "_tmp_ao_dpa_flag"], errors="coerce"),
                pd.to_numeric(group.loc[group["family"] == ANTIOXIDANT_FAMILY, COL_DOSE], errors="coerce"),
            )
            if (group["family"] == ANTIOXIDANT_FAMILY).any()
            else 0.0
        )
        row["interaction__moly_x_phenol"] = row["family_total_dose__moly"] * float(
            safe_weighted_mean(
                pd.to_numeric(group.loc[group["family"] == ANTIOXIDANT_FAMILY, "_tmp_ao_phenol_flag"], errors="coerce"),
                pd.to_numeric(group.loc[group["family"] == ANTIOXIDANT_FAMILY, COL_DOSE], errors="coerce"),
            )
            if (group["family"] == ANTIOXIDANT_FAMILY).any()
            else 0.0
        )
        row["interaction__moly_x_tbn"] = row["family_total_dose__moly"] * float(
            safe_weighted_mean(
                pd.to_numeric(group.loc[group["family"] == DETERGENT_FAMILY, "_tmp_detergent_tbn"], errors="coerce"),
                pd.to_numeric(group.loc[group["family"] == DETERGENT_FAMILY, COL_DOSE], errors="coerce"),
            )
            if (group["family"] == DETERGENT_FAMILY).any()
            else 0.0
        )
        row["interaction__active_no_x_severity"] = row.get("antioxidant__active_no_pct__wm", np.nan) * float(
            group[COL_TIME].iloc[0] * np.exp((group[COL_TEMP].iloc[0] - 150.0) / 10.0)
        )
        row["interaction__steric_x_severity"] = row.get("antioxidant__steric_factor__wm", np.nan) * float(
            group[COL_TIME].iloc[0] * np.exp((group[COL_TEMP].iloc[0] - 150.0) / 10.0)
        )
        rows.append(row)

    return pd.DataFrame(rows)


def evaluate_feature_sets(df: pd.DataFrame, feature_sets: Dict[str, Sequence[str]], seed: int, n_splits: int) -> pd.DataFrame:
    y = pd.to_numeric(df[TARGET_COL], errors="coerce").to_numpy(dtype=float)
    rows = []
    for name, feature_list in feature_sets.items():
        feats = [f for f in feature_list if f in df.columns]
        X = df[feats].copy()
        y_pred = np.zeros(len(df), dtype=float)
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for train_idx, valid_idx in kf.split(X):
            pipe = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("model", RidgeCV(alphas=np.logspace(-3, 3, 25))),
                ]
            )
            pipe.fit(X.iloc[train_idx], transform_target(y[train_idx]))
            pred_t = pipe.predict(X.iloc[valid_idx])
            y_pred[valid_idx] = inverse_transform_target(np.asarray(pred_t, dtype=float))
        metrics = compute_metrics(y, y_pred)
        rows.append({"experiment": name, "n_features": len(feats), **metrics})
    return pd.DataFrame(rows).sort_values(["rmse", "mae", "r2"], ascending=[True, True, False]).reset_index(drop=True)


def plot_results(df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(10, 5))
    plt.barh(df["experiment"], df["rmse"], color="#496f5b")
    plt.gca().invert_yaxis()
    plt.xlabel("RMSE")
    plt.title("Oxidation Enhancement Study")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Enhance oxidation-focused representation with Mo/AO/confidence features.")
    parser.add_argument("--base-train", type=Path, default=DEFAULT_BASE_TRAIN)
    parser.add_argument("--raw-train", type=Path, default=DEFAULT_RAW_TRAIN)
    parser.add_argument("--properties", type=Path, default=DEFAULT_PROPERTIES)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    base_df = pd.read_csv(args.base_train)
    base_features = [c for c in base_df.columns if c not in {SCENARIO_COL, "target_viscosity_delta_pct", TARGET_COL}]
    base_full = pd.concat([base_df[[SCENARIO_COL, TARGET_COL]], add_nonlinear_features(base_df[base_features])], axis=1)

    extra_df = build_extra_features(args.raw_train, args.properties)
    merged = base_full.merge(extra_df, on=SCENARIO_COL, how="left")

    ao_extra = [
        "antioxidant__active_no_pct__wm",
        "antioxidant__steric_factor__wm",
        "antioxidant__chemical_potential__wm",
        "antioxidant__homo_ev__wm",
        "antioxidant__lumo_ev__wm",
        "antioxidant__dipole_moment__wm",
        "antioxidant__nitrogen_total_pct__wm",
    ]
    moly_block = [
        "family_total_dose__moly",
        "moly__present",
        "moly__mo_pct__wm",
        "moly__s_mo_ratio__wm",
        "moly__phosphorus_pct__wm",
        "moly__coc_c__wm",
        "moly__category_modtc__wm",
        "moly__category_modtp__wm",
        "moly__category_amine_complex__wm",
        "moly__ligand_dtc__wm",
        "moly__ligand_dtp__wm",
        "moly__ligand_amide__wm",
    ]
    confidence_block = list(CONFIDENCE_PROP_SOURCES.keys())
    interaction_block = [
        "interaction__moly_x_diphenylamine",
        "interaction__moly_x_phenol",
        "interaction__moly_x_tbn",
        "interaction__active_no_x_severity",
        "interaction__steric_x_severity",
    ]

    feature_sets = {
        "baseline_oxidation_focus": BASELINE_FEATURES,
        "baseline_plus_ao_extra": BASELINE_FEATURES + ao_extra,
        "baseline_plus_moly": BASELINE_FEATURES + moly_block,
        "baseline_plus_confidence": BASELINE_FEATURES + confidence_block,
        "baseline_plus_interactions": BASELINE_FEATURES + interaction_block,
        "baseline_plus_ao_plus_moly": BASELINE_FEATURES + ao_extra + moly_block,
        "full_enhanced": BASELINE_FEATURES + ao_extra + moly_block + confidence_block + interaction_block,
    }

    results = evaluate_feature_sets(merged, feature_sets, seed=args.seed, n_splits=args.n_splits)
    baseline_rmse = float(results.loc[results["experiment"] == "baseline_oxidation_focus", "rmse"].iloc[0])
    baseline_mae = float(results.loc[results["experiment"] == "baseline_oxidation_focus", "mae"].iloc[0])
    results["delta_rmse_vs_baseline"] = results["rmse"] - baseline_rmse
    results["delta_mae_vs_baseline"] = results["mae"] - baseline_mae

    dataset_path = args.outdir / "oxidation_enhanced_train.csv"
    results_path = args.outdir / "oxidation_enhancement_results.csv"
    manifest_path = args.outdir / "oxidation_enhancement_manifest.json"
    plot_path = args.outdir / "oxidation_enhancement_rmse.png"

    merged.to_csv(dataset_path, index=False, encoding="utf-8-sig")
    results.to_csv(results_path, index=False, encoding="utf-8-sig")
    plot_results(results.sort_values("rmse"), plot_path)

    manifest = {
        "base_train": str(args.base_train),
        "raw_train": str(args.raw_train),
        "properties": str(args.properties),
        "dataset_csv": str(dataset_path),
        "results_csv": str(results_path),
        "ao_extra": ao_extra,
        "moly_block": moly_block,
        "confidence_block": confidence_block,
        "interaction_block": interaction_block,
        "best_result": results.iloc[0].to_dict(),
        "missing_fraction_top": merged.isna().mean().sort_values(ascending=False).head(30).round(6).to_dict(),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved enhanced dataset to: {dataset_path.resolve()}")
    print(f"Saved results to: {results_path.resolve()}")
    print(f"Saved plot to: {plot_path.resolve()}")
    print(f"Saved manifest to: {manifest_path.resolve()}")


if __name__ == "__main__":
    main()
