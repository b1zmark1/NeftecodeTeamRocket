#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

try:
    from build_compact_v3_dataset import (
        CURATED_SOURCE_FEATURES,
        DERIVED_FEATURES,
        SCENARIO_COL,
        TARGET_OXID_STD,
        TARGET_VISC_STD,
    )
except ModuleNotFoundError:
    from compact_v3.build_compact_v3_dataset import (
        CURATED_SOURCE_FEATURES,
        DERIVED_FEATURES,
        SCENARIO_COL,
        TARGET_OXID_STD,
        TARGET_VISC_STD,
    )


COL_SCENARIO = "scenario_id"
COL_COMPONENT = "Компонент"
COL_BATCH = "Наименование партии"
COL_DOSE = "Массовая доля, %"
COL_TEMP = "Температура испытания | ASTM D445 Daimler Oxidation Test (DOT), °C"
COL_TIME = "Время испытания | - Daimler Oxidation Test (DOT), ч"
COL_BIOFUEL = "Количество биотоплива | - Daimler Oxidation Test (DOT), % масс"
COL_CATALYST = "Дозировка катализатора, категория"

TARGET_VISC_RAW = "Delta Kin. Viscosity KV100 - relative | - Daimler Oxidation Test (DOT), %"
TARGET_OXID_RAW = "Oxidation EOT | DIN 51453 Daimler Oxidation Test (DOT), A/cm"

PROP_COMPONENT = "Компонент"
PROP_BATCH = "Наименование партии"
PROP_NAME = "Наименование показателя"
PROP_VALUE = "Значение показателя"

NUMERIC_PROP_SPECS = {
    "base_oil__kv40__wm": {"slug": "кинематическая_вязкость_при_40c_astm_d445", "family": "base_oil", "agg": "wmean"},
    "base_oil__kv100__wm": {"slug": "кинематическая_вязкость_при_100c_astm_d445", "family": "base_oil", "agg": "wmean"},
    "base_oil__ccs_m35__wm": {"slug": "динамическая_вязкость_ccs_35c_astm_d5293", "family": "base_oil", "agg": "wmean"},
    "base_oil__pour_point_c__max": {"slug": "температура_застывания_гост_20287_метод_б", "family": "base_oil", "agg": "max"},
    "antiwear__sulfur_pct__wm": {"slug": "массовая_доля_серы_astm_d6481", "family": "antiwear", "agg": "wmean"},
    "antiwear__chain_length__wm": {"slug": "длина_углеродной_цепи", "family": "antiwear", "agg": "wmean"},
    "antioxidant__ionization_ev__wm": {"slug": "потенциал_ионизации_эв", "family": "antioxidant", "agg": "wmean"},
    "antioxidant__bde_xh_kcal__wm": {"slug": "энергия_диссоциации_связи_х_н_ккал_моль", "family": "antioxidant", "agg": "wmean"},
    "detergent__tbn_astm__wm": {"slug": "щелочное_число_astm_d2896", "family": "detergent", "agg": "wmean"},
}

TEXT_PROP_SPECS = {
    "catcount__base_oil__группа_по_api__1": {"slug": "группа_по_api", "family": "base_oil", "agg": "count", "mapper": "api_group_1"},
    "textnum__base_oil__происхождение__origin_is_mineral__wm": {"slug": "происхождение", "family": "base_oil", "agg": "wmean", "mapper": "origin_is_mineral"},
    "textnum__antiwear__тип_спиртового_радикала__radical_is_secondary__wm": {
        "slug": "тип_спиртового_радикала", "family": "antiwear", "agg": "wmean", "mapper": "radical_is_secondary"
    },
    "textnum__antiwear__тип_спиртового_радикала__radical_is_mixed__wm": {
        "slug": "тип_спиртового_радикала", "family": "antiwear", "agg": "wmean", "mapper": "radical_is_mixed"
    },
    "textnum__antioxidant__тип_ао__ao_type_is_phenol__wm": {
        "slug": "тип_ао", "family": "antioxidant", "agg": "wmean", "mapper": "ao_type_is_phenol"
    },
    "textnum__antioxidant__тип_ао__ao_type_is_diphenylamine__wm": {
        "slug": "тип_ао", "family": "antioxidant", "agg": "wmean", "mapper": "ao_type_is_diphenylamine"
    },
}

FAMILY_MAP = {
    "базовое_масло": "base_oil",
    "противоизносная_присадка": "antiwear",
    "антиоксидант": "antioxidant",
    "детергент": "detergent",
}


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


def normalize_component(value: object) -> str:
    return normalize_text(value)


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


def slugify(text: str) -> str:
    text = normalize_text(text).lower()
    replacements = {
        "°c": "c",
        "°": "",
        "эв": "эв",
        "х-н": "х_н",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"[^0-9a-zа-яё]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def parse_numeric_value(value: object) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if not text:
        return np.nan
    text = text.replace("−", "-").replace(",", ".")
    text = re.sub(r"\s+", "", text)
    if not text:
        return np.nan

    if text[0] in "<>~≈":
        text = text[1:]

    nums = [float(x) for x in re.findall(r"[-+]?\d+(?:\.\d+)?", text)]
    if not nums:
        return np.nan

    has_range = any(ch in text for ch in ["-", "–", "—"]) and not text.startswith("-") and len(nums) >= 2
    if has_range:
        return float(sum(nums[:2]) / 2.0)

    return float(nums[0])


def extract_component_class(component_name: str) -> str:
    text = normalize_component(component_name)
    if "_" in text:
        return text.rsplit("_", 1)[0]
    return text


def to_family(component_name: str) -> str:
    cls = extract_component_class(component_name).lower()
    return FAMILY_MAP.get(cls, slugify(cls))


def safe_weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    mask = values.notna() & weights.notna()
    if not mask.any():
        return np.nan
    values_f = values[mask].astype(float)
    weights_f = weights[mask].astype(float)
    wsum = weights_f.sum()
    if wsum == 0:
        return np.nan
    return float(np.average(values_f, weights=weights_f))


def map_text_flag(mapper: str, value: str) -> float:
    text = normalize_text(value).lower()
    if not text:
        return np.nan
    if mapper == "api_group_1":
        parsed = parse_numeric_value(text)
        return float(parsed == 1.0) if pd.notna(parsed) else np.nan
    if mapper == "origin_is_mineral":
        return float("минераль" in text)
    if mapper == "radical_is_secondary":
        return float("вторич" in text)
    if mapper == "radical_is_mixed":
        return float("смеш" in text)
    if mapper == "ao_type_is_phenol":
        return float("фенол" in text)
    if mapper == "ao_type_is_diphenylamine":
        return float("дифениламин" in text)
    return np.nan


def prepare_property_lookups(properties_df: pd.DataFrame) -> Tuple[Dict[Tuple[str, str, str], float], Dict[Tuple[str, str], float], Dict[Tuple[str, str, str], str], Dict[Tuple[str, str], str]]:
    required = {PROP_COMPONENT, PROP_BATCH, PROP_NAME, PROP_VALUE}
    missing = required - set(properties_df.columns)
    if missing:
        raise ValueError(f"Missing property columns: {sorted(missing)}")

    df = properties_df.copy()
    df[PROP_COMPONENT] = df[PROP_COMPONENT].map(normalize_component)
    df[PROP_BATCH] = df[PROP_BATCH].map(normalize_batch)
    df[PROP_NAME] = df[PROP_NAME].map(slugify)

    numeric_slugs = {spec["slug"] for spec in NUMERIC_PROP_SPECS.values()}
    text_slugs = {spec["slug"] for spec in TEXT_PROP_SPECS.values()}

    numeric_df = df[df[PROP_NAME].isin(numeric_slugs)].copy()
    numeric_df[PROP_VALUE] = numeric_df[PROP_VALUE].map(parse_numeric_value)
    numeric_df = numeric_df.dropna(subset=[PROP_VALUE])
    numeric_df = numeric_df.groupby([PROP_COMPONENT, PROP_BATCH, PROP_NAME], as_index=False)[PROP_VALUE].mean()

    text_df = df[df[PROP_NAME].isin(text_slugs)].copy()
    text_df[PROP_VALUE] = text_df[PROP_VALUE].map(normalize_text)
    text_df = text_df[text_df[PROP_VALUE] != ""]
    text_df = text_df.groupby([PROP_COMPONENT, PROP_BATCH, PROP_NAME], as_index=False)[PROP_VALUE].first()

    numeric_exact: Dict[Tuple[str, str, str], float] = {}
    numeric_typical: Dict[Tuple[str, str], float] = {}
    for row in numeric_df.to_dict("records"):
        comp = row[PROP_COMPONENT]
        batch = row[PROP_BATCH]
        prop = row[PROP_NAME]
        val = float(row[PROP_VALUE])
        if batch == "typical":
            numeric_typical[(comp, prop)] = val
        else:
            numeric_exact[(comp, batch, prop)] = val

    text_exact: Dict[Tuple[str, str, str], str] = {}
    text_typical: Dict[Tuple[str, str], str] = {}
    for row in text_df.to_dict("records"):
        comp = row[PROP_COMPONENT]
        batch = row[PROP_BATCH]
        prop = row[PROP_NAME]
        val = str(row[PROP_VALUE])
        if batch == "typical":
            text_typical[(comp, prop)] = val
        else:
            text_exact[(comp, batch, prop)] = val

    return numeric_exact, numeric_typical, text_exact, text_typical


def lookup_numeric(comp: str, batch: str, prop_slug: str, exact: Dict[Tuple[str, str, str], float], typical: Dict[Tuple[str, str], float]) -> float:
    if (comp, batch, prop_slug) in exact:
        return exact[(comp, batch, prop_slug)]
    return typical.get((comp, prop_slug), np.nan)


def lookup_text(comp: str, batch: str, prop_slug: str, exact: Dict[Tuple[str, str, str], str], typical: Dict[Tuple[str, str], str]) -> str:
    if (comp, batch, prop_slug) in exact:
        return exact[(comp, batch, prop_slug)]
    return typical.get((comp, prop_slug), "")


def enrich_mixtures(
    mixtures_df: pd.DataFrame,
    numeric_exact: Dict[Tuple[str, str, str], float],
    numeric_typical: Dict[Tuple[str, str], float],
    text_exact: Dict[Tuple[str, str, str], str],
    text_typical: Dict[Tuple[str, str], str],
    is_train: bool,
) -> pd.DataFrame:
    required = {COL_SCENARIO, COL_COMPONENT, COL_BATCH, COL_DOSE, COL_TEMP, COL_TIME, COL_BIOFUEL, COL_CATALYST}
    if is_train:
        required |= {TARGET_VISC_RAW, TARGET_OXID_RAW}
    missing = required - set(mixtures_df.columns)
    if missing:
        raise ValueError(f"Missing mixture columns: {sorted(missing)}")

    df = mixtures_df.copy()
    df[COL_COMPONENT] = df[COL_COMPONENT].map(normalize_component)
    df[COL_BATCH] = df[COL_BATCH].map(normalize_batch)
    df["family"] = df[COL_COMPONENT].map(to_family)

    numeric_cols = [COL_DOSE, COL_TEMP, COL_TIME, COL_BIOFUEL, COL_CATALYST]
    if is_train:
        numeric_cols += [TARGET_VISC_RAW, TARGET_OXID_RAW]
    for col in numeric_cols:
        df[col] = df[col].map(parse_numeric_value)

    for feature_name, spec in NUMERIC_PROP_SPECS.items():
        prop_slug = spec["slug"]
        df[feature_name] = [
            lookup_numeric(comp, batch, prop_slug, numeric_exact, numeric_typical)
            for comp, batch in zip(df[COL_COMPONENT], df[COL_BATCH])
        ]

    for feature_name, spec in TEXT_PROP_SPECS.items():
        prop_slug = spec["slug"]
        mapper = spec["mapper"]
        raw_values = [
            lookup_text(comp, batch, prop_slug, text_exact, text_typical)
            for comp, batch in zip(df[COL_COMPONENT], df[COL_BATCH])
        ]
        df[feature_name] = [map_text_flag(mapper, value) for value in raw_values]

    return df


def agg_numeric(group: pd.DataFrame, feature_name: str, family: str, agg: str) -> float:
    subset = group[group["family"] == family]
    if subset.empty:
        return np.nan
    values = pd.to_numeric(subset[feature_name], errors="coerce")
    if agg == "wmean":
        return safe_weighted_mean(values, pd.to_numeric(subset[COL_DOSE], errors="coerce"))
    if agg == "max":
        return float(values.max()) if values.notna().any() else np.nan
    raise ValueError(f"Unsupported agg {agg}")


def agg_text(group: pd.DataFrame, feature_name: str, family: str, agg: str) -> float:
    subset = group[group["family"] == family]
    if subset.empty:
        return np.nan
    values = pd.to_numeric(subset[feature_name], errors="coerce")
    if agg == "wmean":
        return safe_weighted_mean(values, pd.to_numeric(subset[COL_DOSE], errors="coerce"))
    if agg == "count":
        return float(values.fillna(0.0).sum())
    raise ValueError(f"Unsupported agg {agg}")


def build_compact_v3_from_enriched(enriched_df: pd.DataFrame, is_train: bool) -> pd.DataFrame:
    rows = []
    for scenario_id, group in enriched_df.groupby(COL_SCENARIO, sort=False):
        row: Dict[str, float] = {SCENARIO_COL: str(scenario_id)}
        row["temperature_c"] = float(group[COL_TEMP].iloc[0])
        row["time_h"] = float(group[COL_TIME].iloc[0])
        row["biofuel_pct"] = float(group[COL_BIOFUEL].iloc[0])
        row["catalyst_category"] = float(group[COL_CATALYST].iloc[0])
        row["severity_exp"] = row["time_h"] * float(np.exp((row["temperature_c"] - 150.0) / 10.0))
        row["temperature_x_time"] = row["temperature_c"] * row["time_h"]
        row["temperature_x_biofuel"] = row["temperature_c"] * row["biofuel_pct"]

        antiwear_dose = pd.to_numeric(group.loc[group["family"] == "antiwear", COL_DOSE], errors="coerce").sum(min_count=1)
        row["family_total_dose__antiwear"] = float(antiwear_dose) if pd.notna(antiwear_dose) else 0.0
        row["interaction__biofuel_x_zddp"] = row["biofuel_pct"] * row["family_total_dose__antiwear"]

        for feature_name, spec in NUMERIC_PROP_SPECS.items():
            row[feature_name] = agg_numeric(group, feature_name, spec["family"], spec["agg"])

        for feature_name, spec in TEXT_PROP_SPECS.items():
            row[feature_name] = agg_text(group, feature_name, spec["family"], spec["agg"])

        row["severity_x_biofuel"] = row["severity_exp"] * row["biofuel_pct"]
        row["severity_x_catalyst"] = row["severity_exp"] * row["catalyst_category"]
        row["regime_load"] = row["severity_exp"] * (1.0 + row["biofuel_pct"] / 10.0)

        kv100 = row["base_oil__kv100__wm"]
        row["base_oil_visc_ratio"] = row["base_oil__kv40__wm"] / kv100 if pd.notna(kv100) and kv100 != 0 else np.nan
        row["antiwear_severity"] = row["family_total_dose__antiwear"] * row["severity_exp"]
        row["antioxidant_balance"] = (
            row["textnum__antioxidant__тип_ао__ao_type_is_phenol__wm"]
            - row["textnum__antioxidant__тип_ао__ao_type_is_diphenylamine__wm"]
        )

        if is_train:
            row[TARGET_VISC_STD] = float(group[TARGET_VISC_RAW].iloc[0])
            row[TARGET_OXID_STD] = float(group[TARGET_OXID_RAW].iloc[0])

        rows.append(row)

    compact_df = pd.DataFrame(rows)
    ordered_columns = [SCENARIO_COL]
    if is_train:
        ordered_columns += [TARGET_VISC_STD, TARGET_OXID_STD]
    ordered_columns += CURATED_SOURCE_FEATURES + list(DERIVED_FEATURES.keys())
    return compact_df.reindex(columns=ordered_columns)


def build_compact_v3_from_raw(mixtures_path: Path, properties_path: Path, is_train: bool) -> Tuple[pd.DataFrame, Dict[str, object]]:
    mixtures_df = read_csv_auto(mixtures_path)
    properties_df = read_csv_auto(properties_path)
    numeric_exact, numeric_typical, text_exact, text_typical = prepare_property_lookups(properties_df)
    enriched = enrich_mixtures(mixtures_df, numeric_exact, numeric_typical, text_exact, text_typical, is_train=is_train)
    compact_df = build_compact_v3_from_enriched(enriched, is_train=is_train)

    manifest = {
        "mixtures_path": str(mixtures_path),
        "properties_path": str(properties_path),
        "is_train": is_train,
        "row_count": int(compact_df.shape[0]),
        "column_count": int(compact_df.shape[1]),
        "missing_fraction_by_column": compact_df.isna().mean().round(6).to_dict(),
        "source_feature_count": len(CURATED_SOURCE_FEATURES),
        "derived_feature_count": len(DERIVED_FEATURES),
    }
    return compact_df, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build compact_v3 features directly from raw mixtures/properties CSVs.")
    parser.add_argument("--mixtures", type=Path, required=True, help="Raw mixtures CSV.")
    parser.add_argument("--properties", type=Path, required=True, help="Raw component properties CSV.")
    parser.add_argument("--out", type=Path, required=True, help="Output compact_v3 CSV.")
    parser.add_argument("--manifest", type=Path, default=None, help="Optional manifest JSON output.")
    parser.add_argument("--train", action="store_true", help="Treat mixtures file as train and keep targets.")
    args = parser.parse_args()

    compact_df, manifest = build_compact_v3_from_raw(args.mixtures, args.properties, is_train=args.train)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    compact_df.to_csv(args.out, index=False, encoding="utf-8-sig")
    if args.manifest is not None:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved compact_v3 CSV to: {args.out.resolve()}")
    if args.manifest is not None:
        print(f"Saved manifest to: {args.manifest.resolve()}")
    print(f"Rows: {compact_df.shape[0]}, columns: {compact_df.shape[1]}")


if __name__ == "__main__":
    main()
