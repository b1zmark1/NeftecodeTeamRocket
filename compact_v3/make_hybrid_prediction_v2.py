#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd

from build_compact_v3_from_raw import build_compact_v3_from_raw
from make_targetwise_prediction_csv import add_nonlinear_features as add_compact_nonlinear_features
from run_oxidation_feature_enhancement import build_extra_features


ROOT = Path(__file__).resolve().parents[1]
RAW_TEST_PATH = ROOT / "docs" / "daimler_mixtures_test.csv"
RAW_PROPERTIES_PATH = ROOT / "docs" / "daimler_component_properties.csv"

VISC_MODEL_PATH = ROOT / "compact_v3" / "mlp_targetwise_output" / "models" / "viscosity_best_model.joblib"
OXID_MODEL_PATH = ROOT / "compact_v3" / "oxidation_focus_v2_output" / "oxidation_focus_v2_best_model.joblib"

OUT_DIR = ROOT / "compact_v3" / "hybrid_predictions_v2"
ROOT_PREDICTION_PATH = ROOT / "prediction.csv"

SCENARIO_COL = "scenario_id"
TARGET_VISC = "Delta Kin. Viscosity KV100 - relative | - Daimler Oxidation Test (DOT), %"
TARGET_OXID = "Oxidation EOT | DIN 51453 Daimler Oxidation Test (DOT), A/cm"


def predict_bundle(bundle: Dict[str, object], X: pd.DataFrame) -> np.ndarray:
    feature_names = list(bundle["feature_names"])
    X_use = X.reindex(columns=feature_names)
    X_imp = bundle["imputer"].transform(X_use)
    X_scaled = bundle["x_scaler"].transform(X_imp)
    pred_scaled = bundle["mlp"].predict(X_scaled)
    pred_t = bundle["y_scaler"].inverse_transform(np.asarray(pred_scaled).reshape(-1, 1)).ravel()
    target_name = str(bundle["target_name"])
    if target_name == "viscosity":
        return 50.0 * np.sinh(np.clip(pred_t, -6.0, 6.0))
    return np.expm1(np.clip(pred_t, -6.0, 6.0))


def build_oxidation_focus_v2_test(base_full: pd.DataFrame, extra_df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    merged = base_full.merge(extra_df, on=SCENARIO_COL, how="left")
    cols = [SCENARIO_COL] + [c for c in feature_names if c in merged.columns]
    return merged[cols].copy()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    compact_base, raw_manifest = build_compact_v3_from_raw(
        mixtures_path=RAW_TEST_PATH,
        properties_path=RAW_PROPERTIES_PATH,
        is_train=False,
    )
    compact_full = add_compact_nonlinear_features(compact_base.drop(columns=[SCENARIO_COL]))
    compact_full.insert(0, SCENARIO_COL, compact_base[SCENARIO_COL])

    extra_df = build_extra_features(RAW_TEST_PATH, RAW_PROPERTIES_PATH)

    visc_bundle = joblib.load(VISC_MODEL_PATH)
    oxid_bundle = joblib.load(OXID_MODEL_PATH)

    viscosity_pred = predict_bundle(visc_bundle, compact_full.drop(columns=[SCENARIO_COL]))

    oxidation_test = build_oxidation_focus_v2_test(compact_full, extra_df, list(oxid_bundle["feature_names"]))
    oxidation_pred = predict_bundle(oxid_bundle, oxidation_test.drop(columns=[SCENARIO_COL]))

    prediction_df = pd.DataFrame(
        {
            SCENARIO_COL: compact_full[SCENARIO_COL],
            TARGET_VISC: viscosity_pred,
            TARGET_OXID: oxidation_pred,
        }
    )

    compact_base_path = OUT_DIR / "test_flat_features_v3_compact.csv"
    compact_full_path = OUT_DIR / "test_flat_features_v3_compact_with_nonlinear.csv"
    oxidation_test_path = OUT_DIR / "oxidation_focus_v2_test.csv"
    prediction_path = OUT_DIR / "prediction.csv"
    manifest_path = OUT_DIR / "prediction_manifest.json"

    compact_base.to_csv(compact_base_path, index=False, encoding="utf-8-sig")
    compact_full.to_csv(compact_full_path, index=False, encoding="utf-8-sig")
    oxidation_test.to_csv(oxidation_test_path, index=False, encoding="utf-8-sig")
    prediction_df.to_csv(prediction_path, index=False, encoding="utf-8-sig")
    prediction_df.to_csv(ROOT_PREDICTION_PATH, index=False, encoding="utf-8-sig")

    manifest = {
        "raw_test_source": str(RAW_TEST_PATH),
        "raw_properties_source": str(RAW_PROPERTIES_PATH),
        "viscosity_model": str(VISC_MODEL_PATH),
        "oxidation_model": str(OXID_MODEL_PATH),
        "prediction_csv": str(prediction_path),
        "root_prediction_csv": str(ROOT_PREDICTION_PATH),
        "compact_base_csv": str(compact_base_path),
        "compact_full_csv": str(compact_full_path),
        "oxidation_focus_v2_test_csv": str(oxidation_test_path),
        "n_test_rows": int(prediction_df.shape[0]),
        "raw_feature_manifest": raw_manifest,
        "missing_fraction_oxidation_test": oxidation_test.drop(columns=[SCENARIO_COL]).isna().mean().round(6).to_dict(),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved hybrid prediction CSV to: {prediction_path.resolve()}")
    print(f"Saved root prediction CSV to: {ROOT_PREDICTION_PATH.resolve()}")
    print(f"Saved oxidation test dataset to: {oxidation_test_path.resolve()}")
    print(f"Saved manifest to: {manifest_path.resolve()}")


if __name__ == "__main__":
    main()
