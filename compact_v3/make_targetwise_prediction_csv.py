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


SCENARIO_COL = "scenario_id"
TARGET_VISC = "Delta Kin. Viscosity KV100 - relative | - Daimler Oxidation Test (DOT), %"
TARGET_OXID = "Oxidation EOT | DIN 51453 Daimler Oxidation Test (DOT), A/cm"

ROOT = Path(__file__).resolve().parents[1]
RAW_TEST_PATH = ROOT / "docs" / "daimler_mixtures_test.csv"
RAW_PROPERTIES_PATH = ROOT / "docs" / "daimler_component_properties.csv"
VISC_MODEL_PATH = ROOT / "compact_v3" / "mlp_targetwise_output" / "models" / "viscosity_best_model.joblib"
OXID_MODEL_PATH = ROOT / "compact_v3" / "mlp_targetwise_output" / "models" / "oxidation_best_model.joblib"
OUT_DIR = ROOT / "compact_v3" / "hybrid_predictions"
ROOT_PREDICTION_PATH = ROOT / "prediction.csv"


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


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    compact_base, raw_manifest = build_compact_v3_from_raw(
        mixtures_path=RAW_TEST_PATH,
        properties_path=RAW_PROPERTIES_PATH,
        is_train=False,
    )
    compact_full = add_nonlinear_features(compact_base.drop(columns=[SCENARIO_COL]))
    compact_full.insert(0, SCENARIO_COL, compact_base[SCENARIO_COL])

    visc_bundle = joblib.load(VISC_MODEL_PATH)
    oxid_bundle = joblib.load(OXID_MODEL_PATH)

    pred_visc = predict_bundle(visc_bundle, compact_full.drop(columns=[SCENARIO_COL]))
    pred_oxid = predict_bundle(oxid_bundle, compact_full.drop(columns=[SCENARIO_COL]))

    prediction_df = pd.DataFrame(
        {
            SCENARIO_COL: compact_full[SCENARIO_COL],
            TARGET_VISC: pred_visc,
            TARGET_OXID: pred_oxid,
        }
    )

    prediction_path = OUT_DIR / "prediction.csv"
    compact_base_path = OUT_DIR / "test_flat_features_v3_compact.csv"
    compact_full_path = OUT_DIR / "test_flat_features_v3_compact_with_nonlinear.csv"
    manifest_path = OUT_DIR / "prediction_manifest.json"

    prediction_df.to_csv(prediction_path, index=False, encoding="utf-8-sig")
    prediction_df.to_csv(ROOT_PREDICTION_PATH, index=False, encoding="utf-8-sig")
    compact_base.to_csv(compact_base_path, index=False, encoding="utf-8-sig")
    compact_full.to_csv(compact_full_path, index=False, encoding="utf-8-sig")

    manifest = {
        "raw_test_source": str(RAW_TEST_PATH),
        "raw_properties_source": str(RAW_PROPERTIES_PATH),
        "viscosity_model": str(VISC_MODEL_PATH),
        "oxidation_model": str(OXID_MODEL_PATH),
        "prediction_csv": str(prediction_path),
        "root_prediction_csv": str(ROOT_PREDICTION_PATH),
        "compact_base_csv": str(compact_base_path),
        "compact_full_csv": str(compact_full_path),
        "n_test_rows": int(prediction_df.shape[0]),
        "raw_feature_manifest": raw_manifest,
        "missing_fraction_with_nonlinear": compact_full.drop(columns=[SCENARIO_COL]).isna().mean().round(6).to_dict(),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved prediction CSV to: {prediction_path.resolve()}")
    print(f"Saved root prediction CSV to: {ROOT_PREDICTION_PATH.resolve()}")
    print(f"Saved compact test features to: {compact_base_path.resolve()}")
    print(f"Saved nonlinear compact test features to: {compact_full_path.resolve()}")
    print(f"Saved manifest to: {manifest_path.resolve()}")


if __name__ == "__main__":
    main()
