#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


TARGET_VISC = "Delta Kin. Viscosity KV100 - relative | - Daimler Oxidation Test (DOT), %"
TARGET_OXID = "Oxidation EOT | DIN 51453 Daimler Oxidation Test (DOT), A/cm"
SCENARIO_COL = "scenario_id"

TARGET_VISC_CANDIDATES = [
    TARGET_VISC,
    "target_viscosity",
    "target_viscosity_delta",
    "target_viscosity_delta_pct",
    "viscosity_delta",
    "delta_viscosity",
    "delta_kin_viscosity",
    "delta_kin_viscosity_kv100",
    "y_viscosity",
    "target_0",
]

TARGET_OXID_CANDIDATES = [
    TARGET_OXID,
    "target_oxidation",
    "target_oxidation_acm",
    "oxidation",
    "oxidation_eot",
    "y_oxidation",
    "target_1",
]

VISC_SCALE = 50.0
TRANSFORM_CLIP = (-6.0, 6.0)


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


def resolve_target_columns(df: pd.DataFrame) -> Tuple[str, str]:
    def find_first(candidates: List[str]) -> str | None:
        lower_to_actual = {str(col).strip().lower(): col for col in df.columns}
        for candidate in candidates:
            actual = lower_to_actual.get(candidate.strip().lower())
            if actual is not None:
                return str(actual)

        for col_lower, actual in lower_to_actual.items():
            if "viscosity" in col_lower and ("target" in col_lower or col_lower.startswith("y_")):
                return str(actual)
            if "oxid" in col_lower and ("target" in col_lower or col_lower.startswith("y_")):
                return str(actual)
        return None

    target_visc_col = find_first(TARGET_VISC_CANDIDATES)
    target_oxid_col = find_first(TARGET_OXID_CANDIDATES)
    if target_visc_col is None or target_oxid_col is None:
        raise ValueError(f"Could not resolve target columns from {list(df.columns)}")
    return target_visc_col, target_oxid_col


def select_numeric_feature_columns(df: pd.DataFrame) -> List[str]:
    forbidden = set(TARGET_VISC_CANDIDATES + TARGET_OXID_CANDIDATES)
    if SCENARIO_COL in df.columns:
        forbidden.add(SCENARIO_COL)
    feature_cols = [col for col in df.columns if col not in forbidden and pd.api.types.is_numeric_dtype(df[col])]
    if not feature_cols:
        raise ValueError("No numeric feature columns found.")
    return feature_cols


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


def drop_all_nan_columns(X_train: pd.DataFrame, X_valid: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    keep_cols = [col for col in X_train.columns if X_train[col].notna().any()]
    dropped = [col for col in X_train.columns if col not in keep_cols]
    return X_train[keep_cols].copy(), X_valid[keep_cols].copy(), dropped


def drop_sparse_columns(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    min_non_na_ratio: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    threshold = int(math.ceil(len(X_train) * min_non_na_ratio))
    keep_cols = [col for col in X_train.columns if X_train[col].notna().sum() >= threshold]
    dropped = [col for col in X_train.columns if col not in keep_cols]
    return X_train[keep_cols].copy(), X_valid[keep_cols].copy(), dropped


def drop_low_variance_columns(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    variance_threshold: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    if X_train.empty:
        return X_train.copy(), X_valid.copy(), []
    imputer = SimpleImputer(strategy="median")
    X_train_imputed = imputer.fit_transform(X_train)
    selector = VarianceThreshold(threshold=variance_threshold)
    selector.fit(X_train_imputed)
    keep_mask = selector.get_support()
    keep_cols = list(X_train.columns[keep_mask])
    dropped = [col for col in X_train.columns if col not in keep_cols]
    return X_train[keep_cols].copy(), X_valid[keep_cols].copy(), dropped


def rank_features_by_target_correlation(X_train: pd.DataFrame, y_train: np.ndarray) -> pd.Series:
    target = pd.Series(y_train, index=X_train.index)
    scores: Dict[str, float] = {}
    for col in X_train.columns:
        corr = pd.to_numeric(X_train[col], errors="coerce").corr(target)
        scores[col] = abs(float(corr)) if pd.notna(corr) else 0.0
    return pd.Series(scores).sort_values(ascending=False)


def get_mandatory_features(target_name: str) -> List[str]:
    if target_name == "viscosity":
        return [
            "temperature_c",
            "time_h",
            "severity_exp",
            "temperature_x_time",
            "interaction__biofuel_x_zddp",
            "base_oil__kv40__wm",
            "base_oil__kv100__wm",
            "base_oil__ccs_m35__wm",
            "family_total_dose__antiwear",
            "antiwear__sulfur_pct__wm",
            "antiwear__chain_length__wm",
            "nl__severity_sq_scaled",
            "nl__antiwear_x_temp",
            "nl__baseoil_visc_ratio_sq",
        ]
    return [
        "temperature_c",
        "time_h",
        "biofuel_pct",
        "severity_exp",
        "detergent__tbn_astm__wm",
        "textnum__antioxidant__тип_ао__ao_type_is_phenol__wm",
        "textnum__antioxidant__тип_ао__ao_type_is_diphenylamine__wm",
        "antioxidant__ionization_ev__wm",
        "antioxidant__bde_xh_kcal__wm",
        "interaction__biofuel_x_zddp",
        "nl__phenol_x_severity",
        "nl__dpa_x_severity",
        "nl__antioxidant_energy",
        "nl__tbn_x_biofuel",
    ]


def select_features_for_target(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    y_train: np.ndarray,
    target_name: str,
    min_non_na_ratio: float,
    variance_threshold: float,
    max_features: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, List[str]], List[str]]:
    report: Dict[str, List[str]] = {}
    X1_train, X1_valid, dropped_all_nan = drop_all_nan_columns(X_train, X_valid)
    report["dropped_all_nan"] = dropped_all_nan

    X2_train, X2_valid, dropped_sparse = drop_sparse_columns(X1_train, X1_valid, min_non_na_ratio=min_non_na_ratio)
    report["dropped_sparse"] = dropped_sparse

    X3_train, X3_valid, dropped_low_var = drop_low_variance_columns(X2_train, X2_valid, variance_threshold=variance_threshold)
    report["dropped_low_variance"] = dropped_low_var

    scores = rank_features_by_target_correlation(X3_train, y_train)
    mandatory = [f for f in get_mandatory_features(target_name) if f in X3_train.columns]
    selected = list(scores.head(max_features).index)
    selected = list(dict.fromkeys(mandatory + selected))
    report["selected_features"] = selected
    report["dropped_by_ranking"] = [col for col in X3_train.columns if col not in selected]
    return X3_train[selected].copy(), X3_valid[selected].copy(), report, selected


def build_target_configs() -> List[Dict[str, object]]:
    return [
        {"config_id": "relu_small", "hidden_layers": (64, 32), "alpha": 1e-4, "learning_rate_init": 1e-3, "activation": "relu"},
        {"config_id": "relu_mid", "hidden_layers": (128, 64), "alpha": 1e-4, "learning_rate_init": 1e-3, "activation": "relu"},
        {"config_id": "relu_wide", "hidden_layers": (128, 128), "alpha": 1e-3, "learning_rate_init": 5e-4, "activation": "relu"},
        {"config_id": "tanh_mid", "hidden_layers": (64, 64), "alpha": 1e-3, "learning_rate_init": 1e-3, "activation": "tanh"},
        {"config_id": "tanh_wide", "hidden_layers": (128, 64), "alpha": 1e-3, "learning_rate_init": 5e-4, "activation": "tanh"},
        {"config_id": "relu_deep", "hidden_layers": (256, 128), "alpha": 1e-3, "learning_rate_init": 5e-4, "activation": "relu"},
    ]


def transform_target(y: np.ndarray, target_name: str) -> np.ndarray:
    if target_name == "viscosity":
        return np.arcsinh(y / VISC_SCALE)
    return np.log1p(np.clip(y, a_min=0.0, a_max=None))


def inverse_transform_target(y_t: np.ndarray, target_name: str) -> np.ndarray:
    clipped = np.clip(y_t, TRANSFORM_CLIP[0], TRANSFORM_CLIP[1])
    if target_name == "viscosity":
        return VISC_SCALE * np.sinh(clipped)
    return np.expm1(clipped)


def fit_predict_single_target_mlp(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_valid: pd.DataFrame,
    target_name: str,
    config: Dict[str, object],
    random_state: int,
    max_iter: int,
) -> np.ndarray:
    x_imputer = SimpleImputer(strategy="median")
    X_train_imp = x_imputer.fit_transform(X_train)
    X_valid_imp = x_imputer.transform(X_valid)

    x_scaler = StandardScaler()
    X_train_scaled = x_scaler.fit_transform(X_train_imp)
    X_valid_scaled = x_scaler.transform(X_valid_imp)

    y_train_t = transform_target(y_train, target_name).reshape(-1, 1)
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train_t).ravel()

    model = MLPRegressor(
        hidden_layer_sizes=tuple(config["hidden_layers"]),
        activation=str(config["activation"]),
        solver="adam",
        alpha=float(config["alpha"]),
        learning_rate_init=float(config["learning_rate_init"]),
        max_iter=max_iter,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=30,
        random_state=random_state,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        model.fit(X_train_scaled, y_train_scaled)

    y_pred_scaled = model.predict(X_valid_scaled)
    y_pred_t = y_scaler.inverse_transform(np.asarray(y_pred_scaled).reshape(-1, 1)).ravel()
    return inverse_transform_target(y_pred_t, target_name)


def fit_full_single_target_model(
    X: pd.DataFrame,
    y: np.ndarray,
    target_name: str,
    config: Dict[str, object],
    random_state: int,
    max_iter: int,
) -> Dict[str, object]:
    x_imputer = SimpleImputer(strategy="median")
    X_imp = x_imputer.fit_transform(X)

    x_scaler = StandardScaler()
    X_scaled = x_scaler.fit_transform(X_imp)

    y_t = transform_target(y, target_name).reshape(-1, 1)
    y_scaler = StandardScaler()
    y_scaled = y_scaler.fit_transform(y_t).ravel()

    model = MLPRegressor(
        hidden_layer_sizes=tuple(config["hidden_layers"]),
        activation=str(config["activation"]),
        solver="adam",
        alpha=float(config["alpha"]),
        learning_rate_init=float(config["learning_rate_init"]),
        max_iter=max_iter,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=30,
        random_state=random_state,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        model.fit(X_scaled, y_scaled)

    return {
        "imputer": x_imputer,
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "mlp": model,
        "target_name": target_name,
        "config": config,
        "feature_names": list(X.columns),
    }


def predict_full_model(bundle: Dict[str, object], X: pd.DataFrame) -> np.ndarray:
    X_imp = bundle["imputer"].transform(X)
    X_scaled = bundle["x_scaler"].transform(X_imp)
    pred_scaled = bundle["mlp"].predict(X_scaled)
    pred_t = bundle["y_scaler"].inverse_transform(np.asarray(pred_scaled).reshape(-1, 1)).ravel()
    return inverse_transform_target(pred_t, str(bundle["target_name"]))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def evaluate_target_cv(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray | None,
    target_name: str,
    configs: List[Dict[str, object]],
    n_splits: int,
    random_state: int,
    max_iter: int,
    min_non_na_ratio: float,
    variance_threshold: float,
    max_features: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, object]] = []
    oof_rows: List[Dict[str, object]] = []

    if groups is not None:
        unique_groups = np.unique(groups)
        if len(unique_groups) < n_splits:
            raise ValueError(f"Not enough unique groups ({len(unique_groups)}) for GroupKFold n_splits={n_splits}")

    for config in configs:
        if groups is not None:
            split_iter = GroupKFold(n_splits=n_splits).split(X, y, groups)
        else:
            split_iter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state).split(X, y)

        for fold_idx, (train_idx, valid_idx) in enumerate(split_iter, start=1):
            X_train_full = X.iloc[train_idx]
            X_valid_full = X.iloc[valid_idx]
            y_train = y[train_idx]
            y_valid = y[valid_idx]

            X_train, X_valid, feature_report, selected_features = select_features_for_target(
                X_train_full,
                X_valid_full,
                y_train=y_train,
                target_name=target_name,
                min_non_na_ratio=min_non_na_ratio,
                variance_threshold=variance_threshold,
                max_features=max_features,
            )

            y_pred = fit_predict_single_target_mlp(
                X_train=X_train,
                y_train=y_train,
                X_valid=X_valid,
                target_name=target_name,
                config=config,
                random_state=random_state,
                max_iter=max_iter,
            )

            fold_metrics = compute_metrics(y_valid, y_pred)
            fold_metrics["target_name"] = target_name
            fold_metrics["config_id"] = str(config["config_id"])
            fold_metrics["hidden_layers"] = str(tuple(config["hidden_layers"]))
            fold_metrics["activation"] = str(config["activation"])
            fold_metrics["alpha"] = float(config["alpha"])
            fold_metrics["learning_rate_init"] = float(config["learning_rate_init"])
            fold_metrics["fold"] = fold_idx
            fold_metrics["n_selected_features"] = len(selected_features)
            rows.append(fold_metrics)

            for row_idx, pred in zip(valid_idx, y_pred):
                oof_rows.append(
                    {
                        SCENARIO_COL: X.index[row_idx],
                        "target_name": target_name,
                        "config_id": str(config["config_id"]),
                        "fold": fold_idx,
                        "y_true": float(y[row_idx]),
                        "y_pred": float(pred),
                    }
                )

    return pd.DataFrame(rows), pd.DataFrame(oof_rows)


def summarize_target_cv_results(cv_results: pd.DataFrame) -> pd.DataFrame:
    metric_cols = ["rmse", "mae", "r2", "n_selected_features"]
    return (
        cv_results.groupby(["target_name", "config_id", "hidden_layers", "activation", "alpha", "learning_rate_init"], as_index=False)[metric_cols]
        .mean()
        .sort_values(["target_name", "rmse", "mae", "r2"], ascending=[True, True, True, False])
        .reset_index(drop=True)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Target-wise MLP on compact_v3 with extra nonlinear features.")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, default=Path("mlp_targetwise_compact_v3_output"))
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=700)
    parser.add_argument("--min-non-na-ratio", type=float, default=0.10)
    parser.add_argument("--variance-threshold", type=float, default=1e-12)
    parser.add_argument("--max-features", type=int, default=20)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    print("RUNNING FILE:", Path(__file__).resolve())
    print("INPUT FILE:", args.train.resolve())

    df = read_csv_auto(args.train)
    target_visc_col, target_oxid_col = resolve_target_columns(df)

    raw_feature_cols = select_numeric_feature_columns(df)
    X_raw = df[raw_feature_cols].copy()
    X = add_nonlinear_features(X_raw)
    if SCENARIO_COL in df.columns:
        X.index = df[SCENARIO_COL].astype(str)
        scenario_ids = X.index.to_numpy()
    else:
        X.index = pd.RangeIndex(len(X)).astype(str)
        scenario_ids = X.index.to_numpy()

    y_viscosity = pd.to_numeric(df[target_visc_col], errors="coerce").to_numpy(dtype=float)
    y_oxidation = pd.to_numeric(df[target_oxid_col], errors="coerce").to_numpy(dtype=float)
    groups = scenario_ids if SCENARIO_COL in df.columns else None

    print("\n=== Target columns ===")
    print("viscosity target:", target_visc_col)
    print("oxidation target:", target_oxid_col)
    print("base feature count:", len(raw_feature_cols))
    print("feature count with nonlinear terms:", X.shape[1])

    configs = build_target_configs()
    print("\n=== Candidate target-wise configs ===")
    for config in configs:
        print(config)

    all_cv_results = []
    all_oof = []
    best_feature_reports: Dict[str, Dict[str, object]] = {}

    for target_name, y in [("viscosity", y_viscosity), ("oxidation", y_oxidation)]:
        cv_results, oof_df = evaluate_target_cv(
            X=X,
            y=y,
            groups=groups,
            target_name=target_name,
            configs=configs,
            n_splits=args.n_splits,
            random_state=args.random_state,
            max_iter=args.max_iter,
            min_non_na_ratio=args.min_non_na_ratio,
            variance_threshold=args.variance_threshold,
            max_features=args.max_features,
        )
        all_cv_results.append(cv_results)
        all_oof.append(oof_df)

    cv_results_df = pd.concat(all_cv_results, ignore_index=True)
    cv_summary_df = summarize_target_cv_results(cv_results_df)

    models_dir = args.outdir / "models"
    models_dir.mkdir(exist_ok=True)

    full_models = {}
    for target_name, y in [("viscosity", y_viscosity), ("oxidation", y_oxidation)]:
        best_row = cv_summary_df[cv_summary_df["target_name"] == target_name].iloc[0]
        best_config_id = str(best_row["config_id"])
        best_config = next(config for config in configs if str(config["config_id"]) == best_config_id)

        X_full_train, _, feature_report, selected_features = select_features_for_target(
            X,
            X,
            y_train=y,
            target_name=target_name,
            min_non_na_ratio=args.min_non_na_ratio,
            variance_threshold=args.variance_threshold,
            max_features=args.max_features,
        )
        best_feature_reports[target_name] = {
            "selected_features": selected_features,
            "feature_report": feature_report,
            "best_config": best_config,
        }

        bundle = fit_full_single_target_model(
            X=X_full_train,
            y=y,
            target_name=target_name,
            config=best_config,
            random_state=args.random_state,
            max_iter=args.max_iter,
        )
        full_models[target_name] = bundle
        joblib.dump(bundle, models_dir / f"{target_name}_best_model.joblib")

    cv_results_path = args.outdir / "targetwise_mlp_cv_fold_metrics.csv"
    cv_summary_path = args.outdir / "targetwise_mlp_cv_summary.csv"
    oof_path = args.outdir / "targetwise_mlp_oof_predictions.csv"
    manifest_path = args.outdir / "targetwise_mlp_run_manifest.json"
    selected_features_path = args.outdir / "targetwise_selected_features.json"

    cv_results_df.to_csv(cv_results_path, index=False)
    cv_summary_df.to_csv(cv_summary_path, index=False)
    pd.concat(all_oof, ignore_index=True).to_csv(oof_path, index=False)
    selected_features_path.write_text(json.dumps(best_feature_reports, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest = {
        "input_train_path": str(args.train),
        "n_splits": args.n_splits,
        "random_state": args.random_state,
        "max_iter": args.max_iter,
        "max_features_per_target": args.max_features,
        "base_feature_count": len(raw_feature_cols),
        "feature_count_with_nonlinear_terms": int(X.shape[1]),
        "candidate_configs": configs,
        "selected_features_file": str(selected_features_path),
        "outputs": {
            "cv_fold_metrics": str(cv_results_path),
            "cv_summary": str(cv_summary_path),
            "oof_predictions": str(oof_path),
            "models_dir": str(models_dir),
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== CV summary by target/config ===")
    print(cv_summary_df.to_string(index=False))
    print(f"\nArtifacts saved to: {args.outdir.resolve()}")


if __name__ == "__main__":
    main()
