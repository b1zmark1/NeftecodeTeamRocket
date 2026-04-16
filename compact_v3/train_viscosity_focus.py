#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN = ROOT / "compact_v3" / "feature_ablation" / "targetwise_feature_sets" / "viscosity_focus.csv"
DEFAULT_OUTDIR = ROOT / "compact_v3" / "viscosity_focus_output"
SCENARIO_COL = "scenario_id"
TARGET_COL = "target_viscosity_delta_pct"
VISC_SCALE = 50.0
TARGET_CLIP = (-6.0, 6.0)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def summarize_cv_results(cv_results: pd.DataFrame) -> pd.DataFrame:
    metric_cols = ["rmse", "mae", "r2"]
    return (
        cv_results.groupby(["config_id", "hidden_layers", "activation", "alpha", "learning_rate_init"], as_index=False)[metric_cols]
        .mean()
        .sort_values(["rmse", "mae", "r2"], ascending=[True, True, False])
        .reset_index(drop=True)
    )


def build_configs() -> List[Dict[str, object]]:
    return [
        {"config_id": "relu_small", "hidden_layers": (64, 32), "activation": "relu", "alpha": 1e-4, "learning_rate_init": 1e-3},
        {"config_id": "relu_mid", "hidden_layers": (128, 64), "activation": "relu", "alpha": 1e-4, "learning_rate_init": 1e-3},
        {"config_id": "relu_wide", "hidden_layers": (128, 128), "activation": "relu", "alpha": 1e-3, "learning_rate_init": 5e-4},
        {"config_id": "relu_deep", "hidden_layers": (256, 128), "activation": "relu", "alpha": 1e-3, "learning_rate_init": 5e-4},
        {"config_id": "tanh_mid", "hidden_layers": (64, 64), "activation": "tanh", "alpha": 1e-3, "learning_rate_init": 1e-3},
        {"config_id": "tanh_wide", "hidden_layers": (128, 64), "activation": "tanh", "alpha": 1e-3, "learning_rate_init": 5e-4},
    ]


def transform_target(y: np.ndarray) -> np.ndarray:
    return np.arcsinh(y / VISC_SCALE)


def inverse_transform_target(y_t: np.ndarray) -> np.ndarray:
    return VISC_SCALE * np.sinh(np.clip(y_t, *TARGET_CLIP))


def fit_bundle(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    config: Dict[str, object],
    random_state: int,
) -> Dict[str, object]:
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)

    x_scaler = StandardScaler()
    X_train_scaled = x_scaler.fit_transform(X_train_imp)

    y_scaler = StandardScaler()
    y_train_t = transform_target(y_train).reshape(-1, 1)
    y_train_scaled = y_scaler.fit_transform(y_train_t).ravel()

    mlp = MLPRegressor(
        hidden_layer_sizes=tuple(config["hidden_layers"]),
        activation=str(config["activation"]),
        alpha=float(config["alpha"]),
        learning_rate_init=float(config["learning_rate_init"]),
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=40,
        max_iter=4000,
        random_state=random_state,
    )
    mlp.fit(X_train_scaled, y_train_scaled)
    return {
        "imputer": imputer,
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "mlp": mlp,
    }


def predict_bundle(bundle: Dict[str, object], X: pd.DataFrame) -> np.ndarray:
    X_imp = bundle["imputer"].transform(X)
    X_scaled = bundle["x_scaler"].transform(X_imp)
    pred_scaled = bundle["mlp"].predict(X_scaled)
    pred_t = bundle["y_scaler"].inverse_transform(np.asarray(pred_scaled).reshape(-1, 1)).ravel()
    return inverse_transform_target(pred_t)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train viscosity-only MLP on viscosity_focus dataset.")
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.train)
    feature_cols = [c for c in df.columns if c not in {SCENARIO_COL, TARGET_COL}]
    X = df[feature_cols].copy()
    y = pd.to_numeric(df[TARGET_COL], errors="coerce").to_numpy(dtype=float)
    scenario_ids = df[SCENARIO_COL].astype(str).to_numpy()

    configs = build_configs()
    cv_rows = []
    oof_rows = []
    kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)

    for config in configs:
        oof_pred = np.zeros(len(df), dtype=float)
        for fold, (train_idx, valid_idx) in enumerate(kf.split(X), start=1):
            bundle = fit_bundle(X.iloc[train_idx], y[train_idx], config, random_state=args.seed + fold)
            pred = predict_bundle(bundle, X.iloc[valid_idx])
            oof_pred[valid_idx] = pred
            metrics = compute_metrics(y[valid_idx], pred)
            cv_rows.append(
                {
                    "config_id": config["config_id"],
                    "fold": fold,
                    "hidden_layers": str(config["hidden_layers"]),
                    "activation": config["activation"],
                    "alpha": config["alpha"],
                    "learning_rate_init": config["learning_rate_init"],
                    **metrics,
                }
            )
        for sid, yt, yp in zip(scenario_ids, y, oof_pred):
            oof_rows.append(
                {
                    "config_id": config["config_id"],
                    "scenario_id": sid,
                    TARGET_COL: float(yt),
                    "pred_viscosity": float(yp),
                }
            )
        overall = compute_metrics(y, oof_pred)
        print(
            f"[{config['config_id']}] rmse={overall['rmse']:.4f} "
            f"mae={overall['mae']:.4f} r2={overall['r2']:.4f}"
        )

    cv_df = pd.DataFrame(cv_rows)
    summary_df = summarize_cv_results(cv_df)
    best_row = summary_df.iloc[0].to_dict()
    best_config = next(cfg for cfg in configs if cfg["config_id"] == best_row["config_id"])

    final_bundle = fit_bundle(X, y, best_config, random_state=args.seed)
    final_bundle.update(
        {
            "feature_names": feature_cols,
            "target_name": "viscosity",
            "config": best_config,
        }
    )

    fold_path = args.outdir / "viscosity_focus_cv_fold_metrics.csv"
    summary_path = args.outdir / "viscosity_focus_cv_summary.csv"
    oof_path = args.outdir / "viscosity_focus_oof_predictions.csv"
    model_path = args.outdir / "viscosity_focus_best_model.joblib"
    manifest_path = args.outdir / "viscosity_focus_run_manifest.json"

    cv_df.to_csv(fold_path, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(oof_rows).to_csv(oof_path, index=False, encoding="utf-8-sig")
    joblib.dump(final_bundle, model_path)

    manifest = {
        "train_path": str(args.train),
        "n_rows": int(df.shape[0]),
        "n_features": len(feature_cols),
        "feature_names": feature_cols,
        "best_config": best_config,
        "best_summary_row": best_row,
        "model_path": str(model_path),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved fold metrics to: {fold_path.resolve()}")
    print(f"Saved summary to: {summary_path.resolve()}")
    print(f"Saved OOF predictions to: {oof_path.resolve()}")
    print(f"Saved model to: {model_path.resolve()}")
    print(f"Saved manifest to: {manifest_path.resolve()}")


if __name__ == "__main__":
    main()
