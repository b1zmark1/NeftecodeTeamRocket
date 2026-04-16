#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence

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
DEFAULT_TRAIN = ROOT / "compact_v3" / "train_flat_features_v3_compact.csv"
DEFAULT_OUTDIR = ROOT / "compact_v3" / "feature_ablation"

SCENARIO_COL = "scenario_id"
TARGETS = {
    "viscosity": "target_viscosity_delta_pct",
    "oxidation": "target_oxidation_acm",
}
VISC_SCALE = 50.0
TARGET_CLIP = (-6.0, 6.0)


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


def build_feature_blocks() -> Dict[str, List[str]]:
    return {
        "regime": [
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
        ],
        "base_oil": [
            "base_oil__kv40__wm",
            "base_oil__kv100__wm",
            "base_oil__ccs_m35__wm",
            "base_oil__pour_point_c__max",
            "catcount__base_oil__группа_по_api__1",
            "textnum__base_oil__происхождение__origin_is_mineral__wm",
            "base_oil_visc_ratio",
            "nl__baseoil_visc_ratio_sq",
            "nl__ccs_x_temp",
        ],
        "antiwear_zddp": [
            "family_total_dose__antiwear",
            "antiwear__sulfur_pct__wm",
            "antiwear__chain_length__wm",
            "textnum__antiwear__тип_спиртового_радикала__radical_is_secondary__wm",
            "textnum__antiwear__тип_спиртового_радикала__radical_is_mixed__wm",
            "interaction__biofuel_x_zddp",
            "antiwear_severity",
            "nl__zddp_bio_log1p",
            "nl__antiwear_x_biofuel",
            "nl__antiwear_x_temp",
            "nl__sulfur_x_chain",
        ],
        "antioxidant": [
            "textnum__antioxidant__тип_ао__ao_type_is_phenol__wm",
            "textnum__antioxidant__тип_ао__ao_type_is_diphenylamine__wm",
            "antioxidant__ionization_ev__wm",
            "antioxidant__bde_xh_kcal__wm",
            "antioxidant_balance",
            "nl__phenol_x_severity",
            "nl__dpa_x_severity",
            "nl__antioxidant_energy",
            "nl__antioxidant_balance_abs",
        ],
        "detergent": [
            "detergent__tbn_astm__wm",
            "nl__tbn_x_biofuel",
        ],
    }


def build_targetwise_sets(blocks: Dict[str, List[str]]) -> Dict[str, Dict[str, List[str]]]:
    vis_features = list(dict.fromkeys(
        blocks["regime"]
        + blocks["base_oil"]
        + blocks["antiwear_zddp"]
        + ["antioxidant__ionization_ev__wm"]
    ))
    oxid_features = list(dict.fromkeys(
        blocks["regime"]
        + blocks["antioxidant"]
        + blocks["detergent"]
        + [
            "interaction__biofuel_x_zddp",
            "antiwear__sulfur_pct__wm",
            "textnum__antiwear__тип_спиртового_радикала__radical_is_secondary__wm",
            "base_oil__ccs_m35__wm",
            "base_oil__pour_point_c__max",
            "base_oil_visc_ratio",
            "nl__baseoil_visc_ratio_sq",
            "nl__ccs_x_temp",
        ]
    ))
    return {
        "viscosity_focus": {
            "description": "Regime + base oil + antiwear/ZDDP chemistry with minimal antioxidant support.",
            "features": vis_features,
            "blocks": ["regime", "base_oil", "antiwear_zddp", "antioxidant_support"],
        },
        "oxidation_focus": {
            "description": "Regime + antioxidant + detergent chemistry with ZDDP/base-oil stability support.",
            "features": oxid_features,
            "blocks": ["regime", "antioxidant", "detergent", "zddp_support", "base_oil_support"],
        },
    }


def transform_target(y: np.ndarray, target_name: str) -> np.ndarray:
    if target_name == "viscosity":
        return np.arcsinh(y / VISC_SCALE)
    return np.log1p(np.clip(y, a_min=0.0, a_max=None))


def inverse_transform_target(y_t: np.ndarray, target_name: str) -> np.ndarray:
    if target_name == "viscosity":
        return VISC_SCALE * np.sinh(np.clip(y_t, *TARGET_CLIP))
    return np.expm1(np.clip(y_t, *TARGET_CLIP))


def compute_target_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = float(math.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}


def evaluate_target_feature_set(
    X: pd.DataFrame,
    y: np.ndarray,
    features: Sequence[str],
    target_name: str,
    n_splits: int,
    seed: int,
) -> Dict[str, float]:
    feature_list = [f for f in features if f in X.columns]
    X_use = X[feature_list].copy()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    y_pred = np.zeros(len(y), dtype=np.float64)

    for train_idx, valid_idx in kf.split(X_use):
        X_train = X_use.iloc[train_idx]
        X_valid = X_use.iloc[valid_idx]
        y_train = y[train_idx]

        pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", RidgeCV(alphas=np.logspace(-3, 3, 25))),
            ]
        )
        pipe.fit(X_train, transform_target(y_train, target_name))
        pred_t = pipe.predict(X_valid)
        y_pred[valid_idx] = inverse_transform_target(np.asarray(pred_t, dtype=np.float64), target_name)

    metrics = compute_target_metrics(y, y_pred)
    metrics["n_features"] = len(feature_list)
    return metrics


def make_incremental_experiments(blocks: Dict[str, List[str]], target_name: str, targetwise_sets: Dict[str, Dict[str, List[str]]]) -> Dict[str, List[str]]:
    regime = blocks["regime"]
    if target_name == "viscosity":
        return {
            "regime_only": regime,
            "regime_plus_base_oil": regime + blocks["base_oil"],
            "regime_plus_antiwear": regime + blocks["antiwear_zddp"],
            "regime_plus_antioxidant": regime + blocks["antioxidant"],
            "regime_plus_base_oil_plus_antiwear": regime + blocks["base_oil"] + blocks["antiwear_zddp"],
            "all_blocks": regime + blocks["base_oil"] + blocks["antiwear_zddp"] + blocks["antioxidant"] + blocks["detergent"],
            "targetwise_focus": targetwise_sets["viscosity_focus"]["features"],
        }
    return {
        "regime_only": regime,
        "regime_plus_antioxidant": regime + blocks["antioxidant"],
        "regime_plus_detergent": regime + blocks["detergent"],
        "regime_plus_antiwear": regime + blocks["antiwear_zddp"],
        "regime_plus_antioxidant_plus_detergent": regime + blocks["antioxidant"] + blocks["detergent"],
        "all_blocks": regime + blocks["base_oil"] + blocks["antiwear_zddp"] + blocks["antioxidant"] + blocks["detergent"],
        "targetwise_focus": targetwise_sets["oxidation_focus"]["features"],
    }


def make_leave_one_out_experiments(blocks: Dict[str, List[str]]) -> Dict[str, List[str]]:
    base = blocks["regime"] + blocks["base_oil"] + blocks["antiwear_zddp"] + blocks["antioxidant"] + blocks["detergent"]
    experiments = {"all_blocks": base}
    for block_name in ["base_oil", "antiwear_zddp", "antioxidant", "detergent"]:
        reduced = []
        for name, cols in blocks.items():
            if name == block_name:
                continue
            reduced.extend(cols)
        experiments[f"without_{block_name}"] = reduced
    return experiments


def evaluate_experiment_family(
    X: pd.DataFrame,
    y: np.ndarray,
    target_name: str,
    experiments: Dict[str, List[str]],
    n_splits: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    for experiment_name, features in experiments.items():
        metrics = evaluate_target_feature_set(X, y, features, target_name, n_splits, seed)
        rows.append({"experiment": experiment_name, **metrics})
    df = pd.DataFrame(rows).sort_values(["rmse", "mae", "r2"], ascending=[True, True, False]).reset_index(drop=True)
    return df


def save_feature_set_csv(
    df: pd.DataFrame,
    scenario_col: str,
    target_col: str,
    features: Sequence[str],
    out_path: Path,
) -> None:
    cols = [scenario_col, target_col] + [f for f in features if f in df.columns]
    df[cols].to_csv(out_path, index=False, encoding="utf-8-sig")


def plot_bar(df: pd.DataFrame, metric_col: str, title: str, out_path: Path) -> None:
    top = df.copy()
    plt.figure(figsize=(10, 5))
    plt.barh(top["experiment"], top[metric_col], color="#2f5d8a")
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.xlabel(metric_col.upper())
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run target-wise feature block ablation on compact_v3.")
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    feature_set_dir = args.outdir / "targetwise_feature_sets"
    feature_set_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(args.train)
    base_cols = [c for c in train_df.columns if c not in {SCENARIO_COL, TARGETS["viscosity"], TARGETS["oxidation"]}]
    X = add_nonlinear_features(train_df[base_cols])
    full_df = pd.concat([train_df[[SCENARIO_COL, TARGETS["viscosity"], TARGETS["oxidation"]]], X], axis=1)

    blocks = build_feature_blocks()
    targetwise_sets = build_targetwise_sets(blocks)

    all_results: Dict[str, Dict[str, str]] = {}
    recommendations: Dict[str, Dict[str, object]] = {}

    for target_name, target_col in TARGETS.items():
        y = pd.to_numeric(full_df[target_col], errors="coerce").to_numpy(dtype=float)

        incremental = evaluate_experiment_family(
            X=X,
            y=y,
            target_name=target_name,
            experiments=make_incremental_experiments(blocks, target_name, targetwise_sets),
            n_splits=args.n_splits,
            seed=args.seed,
        )
        regime_rmse = float(incremental.loc[incremental["experiment"] == "regime_only", "rmse"].iloc[0])
        regime_mae = float(incremental.loc[incremental["experiment"] == "regime_only", "mae"].iloc[0])
        incremental["delta_rmse_vs_regime"] = incremental["rmse"] - regime_rmse
        incremental["delta_mae_vs_regime"] = incremental["mae"] - regime_mae

        loo = evaluate_experiment_family(
            X=X,
            y=y,
            target_name=target_name,
            experiments=make_leave_one_out_experiments(blocks),
            n_splits=args.n_splits,
            seed=args.seed,
        )
        all_rmse = float(loo.loc[loo["experiment"] == "all_blocks", "rmse"].iloc[0])
        all_mae = float(loo.loc[loo["experiment"] == "all_blocks", "mae"].iloc[0])
        loo["delta_rmse_vs_all"] = loo["rmse"] - all_rmse
        loo["delta_mae_vs_all"] = loo["mae"] - all_mae

        incremental_path = args.outdir / f"{target_name}_incremental_ablation.csv"
        loo_path = args.outdir / f"{target_name}_leave_one_out_ablation.csv"
        incremental.to_csv(incremental_path, index=False, encoding="utf-8-sig")
        loo.to_csv(loo_path, index=False, encoding="utf-8-sig")

        plot_bar(
            incremental.sort_values("rmse"),
            metric_col="rmse",
            title=f"{target_name.title()} Incremental Ablation (RMSE)",
            out_path=args.outdir / f"{target_name}_incremental_ablation_rmse.png",
        )
        plot_bar(
            loo.sort_values("rmse"),
            metric_col="rmse",
            title=f"{target_name.title()} Leave-One-Out Ablation (RMSE)",
            out_path=args.outdir / f"{target_name}_leave_one_out_ablation_rmse.png",
        )

        if target_name == "viscosity":
            rec_name = "viscosity_focus"
        else:
            rec_name = "oxidation_focus"
        rec_features = targetwise_sets[rec_name]["features"]
        feature_set_path = feature_set_dir / f"{rec_name}.csv"
        save_feature_set_csv(full_df, SCENARIO_COL, target_col, rec_features, feature_set_path)

        best_row = incremental.sort_values(["rmse", "mae", "r2"], ascending=[True, True, False]).iloc[0].to_dict()
        recommendations[target_name] = {
            "recommended_feature_set": rec_name,
            "description": targetwise_sets[rec_name]["description"],
            "features": rec_features,
            "feature_csv": str(feature_set_path),
            "best_incremental_result": best_row,
        }
        all_results[target_name] = {
            "incremental_csv": str(incremental_path),
            "leave_one_out_csv": str(loo_path),
        }

    manifest = {
        "train_path": str(args.train),
        "n_rows": int(train_df.shape[0]),
        "blocks": blocks,
        "targetwise_sets": targetwise_sets,
        "results": all_results,
        "recommendations": recommendations,
    }
    manifest_path = args.outdir / "feature_ablation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved feature ablation outputs to: {args.outdir.resolve()}")
    print(f"Saved manifest to: {manifest_path.resolve()}")


if __name__ == "__main__":
    main()
