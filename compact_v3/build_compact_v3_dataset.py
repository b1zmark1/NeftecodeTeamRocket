#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


SCENARIO_COL = "scenario_id"
TARGET_VISC_STD = "target_viscosity_delta_pct"
TARGET_OXID_STD = "target_oxidation_acm"

TARGET_VISC_CANDIDATES = [
    "Delta Kin. Viscosity KV100 - relative | - Daimler Oxidation Test (DOT), %",
    "target_viscosity",
    "target_viscosity_delta",
    TARGET_VISC_STD,
    "viscosity_delta",
    "delta_viscosity",
    "delta_kin_viscosity",
    "delta_kin_viscosity_kv100",
    "y_viscosity",
    "target_0",
]

TARGET_OXID_CANDIDATES = [
    "Oxidation EOT | DIN 51453 Daimler Oxidation Test (DOT), A/cm",
    "target_oxidation",
    TARGET_OXID_STD,
    "oxidation",
    "oxidation_eot",
    "y_oxidation",
    "target_1",
]


CURATED_SOURCE_FEATURES: List[str] = [
    # Regime block
    "temperature_c",
    "time_h",
    "biofuel_pct",
    "catalyst_category",
    "severity_exp",
    "temperature_x_time",
    "temperature_x_biofuel",
    "interaction__biofuel_x_zddp",
    # Base oil block
    "base_oil__kv40__wm",
    "base_oil__kv100__wm",
    "base_oil__ccs_m35__wm",
    "base_oil__pour_point_c__max",
    "catcount__base_oil__группа_по_api__1",
    "textnum__base_oil__происхождение__origin_is_mineral__wm",
    # Antiwear block
    "family_total_dose__antiwear",
    "antiwear__sulfur_pct__wm",
    "antiwear__chain_length__wm",
    "textnum__antiwear__тип_спиртового_радикала__radical_is_secondary__wm",
    "textnum__antiwear__тип_спиртового_радикала__radical_is_mixed__wm",
    # Antioxidant block
    "textnum__antioxidant__тип_ао__ao_type_is_phenol__wm",
    "textnum__antioxidant__тип_ао__ao_type_is_diphenylamine__wm",
    "antioxidant__ionization_ev__wm",
    "antioxidant__bde_xh_kcal__wm",
    # Detergent block
    "detergent__tbn_astm__wm",
]


DERIVED_FEATURES: Dict[str, str] = {
    "severity_x_biofuel": "severity_exp * biofuel_pct",
    "severity_x_catalyst": "severity_exp * catalyst_category",
    "regime_load": "severity_exp * (1 + biofuel_pct / 10)",
    "base_oil_visc_ratio": "base_oil__kv40__wm / base_oil__kv100__wm",
    "antiwear_severity": "family_total_dose__antiwear * severity_exp",
    "antioxidant_balance": "phenol_share - diphenylamine_share",
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


def resolve_target_columns(df: pd.DataFrame) -> Tuple[str | None, str | None]:
    lower_to_actual = {str(col).strip().lower(): str(col) for col in df.columns}

    def find_first(candidates: List[str]) -> str | None:
        for candidate in candidates:
            actual = lower_to_actual.get(candidate.strip().lower())
            if actual is not None:
                return actual
        return None

    return find_first(TARGET_VISC_CANDIDATES), find_first(TARGET_OXID_CANDIDATES)


def get_numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def build_compact_v3(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    out = pd.DataFrame(index=df.index)
    report: Dict[str, List[str]] = {"selected_source_features": [], "missing_source_features": []}

    if SCENARIO_COL in df.columns:
        out[SCENARIO_COL] = df[SCENARIO_COL].astype(str)

    target_visc_col, target_oxid_col = resolve_target_columns(df)
    if target_visc_col is not None:
        out[TARGET_VISC_STD] = pd.to_numeric(df[target_visc_col], errors="coerce")
    if target_oxid_col is not None:
        out[TARGET_OXID_STD] = pd.to_numeric(df[target_oxid_col], errors="coerce")

    for feature in CURATED_SOURCE_FEATURES:
        if feature in df.columns:
            report["selected_source_features"].append(feature)
        else:
            report["missing_source_features"].append(feature)
        out[feature] = get_numeric_series(df, feature)

    # Derived regime features.
    out["severity_x_biofuel"] = out["severity_exp"] * out["biofuel_pct"]
    out["severity_x_catalyst"] = out["severity_exp"] * out["catalyst_category"]
    out["regime_load"] = out["severity_exp"] * (1.0 + out["biofuel_pct"] / 10.0)

    # Derived chemistry compactors.
    kv100 = out["base_oil__kv100__wm"].replace(0.0, np.nan)
    out["base_oil_visc_ratio"] = out["base_oil__kv40__wm"] / kv100
    out["antiwear_severity"] = out["family_total_dose__antiwear"] * out["severity_exp"]
    out["antioxidant_balance"] = (
        out["textnum__antioxidant__тип_ао__ao_type_is_phenol__wm"]
        - out["textnum__antioxidant__тип_ао__ao_type_is_diphenylamine__wm"]
    )

    return out, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build manually curated compact_v3 dataset from flat feature CSV.")
    parser.add_argument("--train", type=Path, required=True, help="Source train flat features CSV (v2).")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("compact_v3"),
        help="Output directory for compact_v3 dataset.",
    )
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    train_df = read_csv_auto(args.train)
    compact_df, feature_report = build_compact_v3(train_df)

    ordered_columns = (
        ([SCENARIO_COL] if SCENARIO_COL in compact_df.columns else [])
        + ([TARGET_VISC_STD] if TARGET_VISC_STD in compact_df.columns else [])
        + ([TARGET_OXID_STD] if TARGET_OXID_STD in compact_df.columns else [])
        + CURATED_SOURCE_FEATURES
        + list(DERIVED_FEATURES.keys())
    )
    compact_df = compact_df[ordered_columns]

    csv_path = args.outdir / "train_flat_features_v3_compact.csv"
    compact_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    manifest = {
        "input_train": str(args.train),
        "output_train": str(csv_path),
        "row_count": int(compact_df.shape[0]),
        "column_count": int(compact_df.shape[1]),
        "source_feature_count": len(CURATED_SOURCE_FEATURES),
        "derived_feature_count": len(DERIVED_FEATURES),
        "curated_source_features": CURATED_SOURCE_FEATURES,
        "derived_features": DERIVED_FEATURES,
        "feature_report": feature_report,
        "missing_fraction_by_column": compact_df.isna().mean().round(6).to_dict(),
    }

    manifest_path = args.outdir / "train_flat_features_v3_compact_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved compact_v3 train CSV to: {csv_path.resolve()}")
    print(f"Saved manifest to: {manifest_path.resolve()}")
    print(f"Rows: {compact_df.shape[0]}, columns: {compact_df.shape[1]}")
    print(f"Missing source features: {len(feature_report['missing_source_features'])}")
    if feature_report["missing_source_features"]:
        print("Missing source features:")
        for feature in feature_report["missing_source_features"]:
            print(feature)


if __name__ == "__main__":
    main()
