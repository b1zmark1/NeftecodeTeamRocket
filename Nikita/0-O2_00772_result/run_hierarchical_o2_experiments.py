from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BASE_COMPONENT_TRAIN = ROOT / "model_with_structure" / "train_component_level_transformed.csv"
BASE_COMPONENT_TEST = ROOT / "model_with_structure" / "test_component_level_transformed.csv"
BASE_SCENARIO_TRAIN = ROOT / "model_with_structure" / "train_scenario_level_features.csv"
BASE_SCENARIO_TEST = ROOT / "model_with_structure" / "test_scenario_level_features.csv"
BASELINE_METRICS = ROOT / "hierarchical_out" / "validation_metrics_hierarchical_model.json"
TRAIN_SCRIPT = ROOT / "hierarchical_model" / "train_hierarchical_model.py"
OUT_DIR = ROOT / "Nikita" / "0-O2_00772_result"

ID_COL = "scenario_id"
COMPONENT_COL = "Компонент"
WEIGHT_COL = "Массовая доля, %"
AO_TYPE_COL = "Тип АО"
ACTIVE_NO_COL = "Активный Азот / Кислород, % масс. (N или O)"
SUBSTRATE_COL = "Класс субстрата"
TBN_ASTM_COL = "Щелочное число, ASTM D2896"
TBN_GOST_COL = "Щелочное число, ГОСТ 11362"

TARGET_VISC = "Delta Kin. Viscosity KV100 - relative | - Daimler Oxidation Test (DOT), %"
TARGET_OX = "Oxidation EOT | DIN 51453 Daimler Oxidation Test (DOT), A/cm"


def normalize_string(value: Any) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).strip())


def parse_numeric_like(value: Any) -> float:
    if pd.isna(value):
        return float("nan")
    text = normalize_string(value)
    if not text:
        return float("nan")
    text = text.replace("−", "-").replace("–", "-").replace("—", "-")
    text = text.replace("%", "").replace(" ", "")
    if text.count(",") == 1 and text.count(".") == 0:
        text = text.replace(",", ".")
    match = re.search(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?", text.casefold())
    if not match:
        return float("nan")
    try:
        return float(match.group(0))
    except ValueError:
        return float("nan")


def series_numeric(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(dtype=float)
    return df[col].map(parse_numeric_like)


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    mask = values.notna() & weights.notna()
    if not mask.any():
        return 0.0
    v = values[mask].astype(float)
    w = weights[mask].astype(float)
    weight_sum = float(w.sum())
    if weight_sum <= 0.0:
        return float(v.mean())
    return float((v * w).sum() / weight_sum)


def build_o2_features(component_df: pd.DataFrame, scenario_df: pd.DataFrame) -> pd.DataFrame:
    component_groups = {
        scenario_id: group.reset_index(drop=True)
        for scenario_id, group in component_df.groupby(ID_COL, sort=False)
    }
    rows: list[dict[str, Any]] = []

    for _, scenario_row in scenario_df.iterrows():
        scenario_id = scenario_row[ID_COL]
        group = component_groups[scenario_id]
        component_names = group[COMPONENT_COL].map(normalize_string)

        ao_df = group[component_names.map(lambda x: x.startswith("Антиоксидант"))].copy()
        detergent_df = group[component_names.map(lambda x: x.startswith("Детергент"))].copy()

        ao_types = ao_df.get(AO_TYPE_COL, pd.Series(index=ao_df.index, dtype=object)).map(normalize_string)
        ao_active = series_numeric(ao_df, ACTIVE_NO_COL).fillna(0.0)
        dpa_active = float(ao_active[ao_types == "Дифениламин"].sum())
        phenol_active = float(ao_active[ao_types == "Фенол"].sum())

        det_substrate = detergent_df.get(SUBSTRATE_COL, pd.Series(index=detergent_df.index, dtype=object)).map(
            normalize_string
        )
        det_weights = series_numeric(detergent_df, WEIGHT_COL).fillna(0.0)
        det_tbn = series_numeric(detergent_df, TBN_ASTM_COL)
        det_tbn = det_tbn.where(det_tbn.notna(), series_numeric(detergent_df, TBN_GOST_COL))

        salicylate_mask = det_substrate.str.contains("Салицилат", case=False, na=False)
        salicylate_tbn = weighted_mean(det_tbn[salicylate_mask], det_weights[salicylate_mask])
        ca_salicylate_present = float(
            det_substrate.str.contains("Салицилат кальция", case=False, na=False).any()
        )
        mg_detergent_present = float(det_substrate.str.contains("магния", case=False, na=False).any())

        rows.append(
            {
                ID_COL: scenario_id,
                "o2_salicylate_tbn_x_amine_ao": salicylate_tbn * dpa_active,
                "o2_salicylate_tbn_x_phenol_ao": salicylate_tbn * phenol_active,
                "o2_salicylate_tbn_x_amine_x_phenol": salicylate_tbn * dpa_active * phenol_active,
                "o2_ca_salicylate_present": ca_salicylate_present,
                "o2_mg_detergent_present": mg_detergent_present,
            }
        )

    return pd.DataFrame(rows)


def merge_features(base_df: pd.DataFrame, feature_df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    merged = base_df.merge(feature_df[[ID_COL] + columns], on=ID_COL, how="left")
    merged[columns] = merged[columns].fillna(0.0)
    return merged


def train_variant(name: str, train_scenario_path: Path, test_scenario_path: Path) -> dict[str, Any]:
    variant_dir = OUT_DIR / name
    variant_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--component-train",
        str(BASE_COMPONENT_TRAIN),
        "--component-test",
        str(BASE_COMPONENT_TEST),
        "--scenario-train",
        str(train_scenario_path),
        "--scenario-test",
        str(test_scenario_path),
        "--out-dir",
        str(variant_dir),
        "--seed",
        "42",
    ]
    subprocess.run(command, check=True, cwd=ROOT)
    metrics = json.loads((variant_dir / "validation_metrics_hierarchical_model.json").read_text(encoding="utf-8"))
    m = metrics["final_validation_metrics"]
    return {
        "experiment": name,
        "mean_rmse": m["mean_rmse"],
        "mean_mae": m["mean_mae"],
        "mean_r2": m["mean_r2"],
        "viscosity_rmse": m["per_target"][TARGET_VISC]["rmse"],
        "viscosity_mae": m["per_target"][TARGET_VISC]["mae"],
        "oxidation_rmse": m["per_target"][TARGET_OX]["rmse"],
        "oxidation_mae": m["per_target"][TARGET_OX]["mae"],
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    component_train_df = pd.read_csv(BASE_COMPONENT_TRAIN)
    component_test_df = pd.read_csv(BASE_COMPONENT_TEST)
    scenario_train_df = pd.read_csv(BASE_SCENARIO_TRAIN)
    scenario_test_df = pd.read_csv(BASE_SCENARIO_TEST)

    o2_train = build_o2_features(component_train_df, scenario_train_df)
    o2_test = build_o2_features(component_test_df, scenario_test_df)
    o2_train.to_csv(OUT_DIR / "o2_feature_block_train.csv", index=False)
    o2_test.to_csv(OUT_DIR / "o2_feature_block_test.csv", index=False)

    experiments = {
        "oxidation_o2": [
            "o2_salicylate_tbn_x_amine_ao",
            "o2_salicylate_tbn_x_phenol_ao",
            "o2_salicylate_tbn_x_amine_x_phenol",
            "o2_ca_salicylate_present",
            "o2_mg_detergent_present",
        ],
        "oxidation_o2_core3": [
            "o2_salicylate_tbn_x_amine_ao",
            "o2_salicylate_tbn_x_phenol_ao",
            "o2_salicylate_tbn_x_amine_x_phenol",
        ],
        "oxidation_o2_core3_plus_ca": [
            "o2_salicylate_tbn_x_amine_ao",
            "o2_salicylate_tbn_x_phenol_ao",
            "o2_salicylate_tbn_x_amine_x_phenol",
            "o2_ca_salicylate_present",
        ],
    }

    baseline_metrics = json.loads(BASELINE_METRICS.read_text(encoding="utf-8"))["final_validation_metrics"]
    summary_rows = [
        {
            "experiment": "baseline_existing",
            "mean_rmse": baseline_metrics["mean_rmse"],
            "mean_mae": baseline_metrics["mean_mae"],
            "mean_r2": baseline_metrics["mean_r2"],
            "viscosity_rmse": baseline_metrics["per_target"][TARGET_VISC]["rmse"],
            "viscosity_mae": baseline_metrics["per_target"][TARGET_VISC]["mae"],
            "oxidation_rmse": baseline_metrics["per_target"][TARGET_OX]["rmse"],
            "oxidation_mae": baseline_metrics["per_target"][TARGET_OX]["mae"],
        }
    ]

    for experiment_name, feature_cols in experiments.items():
        train_aug = merge_features(scenario_train_df, o2_train, feature_cols)
        test_aug = merge_features(scenario_test_df, o2_test, feature_cols)
        train_path = OUT_DIR / f"{experiment_name}_train_scenario.csv"
        test_path = OUT_DIR / f"{experiment_name}_test_scenario.csv"
        train_aug.to_csv(train_path, index=False)
        test_aug.to_csv(test_path, index=False)
        summary_rows.append(train_variant(experiment_name, train_path, test_path))

    summary = pd.DataFrame(summary_rows)
    baseline = summary.loc[summary["experiment"] == "baseline_existing"].iloc[0]
    summary["delta_mean_rmse_vs_baseline"] = summary["mean_rmse"] - baseline["mean_rmse"]
    summary["delta_viscosity_rmse_vs_baseline"] = summary["viscosity_rmse"] - baseline["viscosity_rmse"]
    summary["delta_oxidation_rmse_vs_baseline"] = summary["oxidation_rmse"] - baseline["oxidation_rmse"]
    summary = summary.sort_values("mean_rmse").reset_index(drop=True)
    summary.to_csv(OUT_DIR / "experiment_summary.csv", index=False)


if __name__ == "__main__":
    main()
