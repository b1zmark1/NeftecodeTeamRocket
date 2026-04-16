#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compact_v3.build_compact_v3_from_raw import (
    COL_BATCH,
    COL_BIOFUEL,
    COL_CATALYST,
    COL_COMPONENT,
    COL_DOSE,
    COL_SCENARIO,
    COL_TEMP,
    COL_TIME,
    TARGET_OXID_RAW,
    TARGET_VISC_RAW,
    build_compact_v3_from_enriched,
    enrich_mixtures,
    prepare_property_lookups,
    read_csv_auto,
    to_family,
)


DEFAULT_TRAIN = ROOT / "docs" / "daimler_mixtures_train.csv"
DEFAULT_PROPERTIES = ROOT / "docs" / "daimler_component_properties.csv"
DEFAULT_OUTDIR = ROOT / "set_transformer" / "set_transformer_output"

SCENARIO_CONTEXT_COLS = [
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
]

COMPONENT_NUMERIC_COLS = [
    COL_DOSE,
    "base_oil__kv40__wm",
    "base_oil__kv100__wm",
    "base_oil__ccs_m35__wm",
    "base_oil__pour_point_c__max",
    "antiwear__sulfur_pct__wm",
    "antiwear__chain_length__wm",
    "antioxidant__ionization_ev__wm",
    "antioxidant__bde_xh_kcal__wm",
    "detergent__tbn_astm__wm",
    "catcount__base_oil__группа_по_api__1",
    "textnum__base_oil__происхождение__origin_is_mineral__wm",
    "textnum__antiwear__тип_спиртового_радикала__radical_is_secondary__wm",
    "textnum__antiwear__тип_спиртового_радикала__radical_is_mixed__wm",
    "textnum__antioxidant__тип_ао__ao_type_is_phenol__wm",
    "textnum__antioxidant__тип_ао__ao_type_is_diphenylamine__wm",
]

VISC_SCALE = 50.0
TARGET_CLIP = (-6.0, 6.0)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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


def transform_targets(y: np.ndarray) -> np.ndarray:
    out = np.zeros_like(y, dtype=np.float32)
    out[:, 0] = np.arcsinh(y[:, 0] / VISC_SCALE)
    out[:, 1] = np.log1p(np.clip(y[:, 1], a_min=0.0, a_max=None))
    return out


def inverse_transform_targets(y_t: np.ndarray) -> np.ndarray:
    out = np.zeros_like(y_t, dtype=np.float32)
    out[:, 0] = VISC_SCALE * np.sinh(np.clip(y_t[:, 0], *TARGET_CLIP))
    out[:, 1] = np.expm1(np.clip(y_t[:, 1], *TARGET_CLIP))
    return out


def fit_standardizer(values: np.ndarray, mask: np.ndarray | None = None) -> Dict[str, np.ndarray]:
    arr = values.astype(np.float32).copy()
    if mask is None:
        valid = ~np.isnan(arr)
    else:
        valid = mask.astype(bool) & ~np.isnan(arr)
    medians = np.zeros(arr.shape[-1], dtype=np.float32)
    means = np.zeros(arr.shape[-1], dtype=np.float32)
    stds = np.ones(arr.shape[-1], dtype=np.float32)
    for j in range(arr.shape[-1]):
        col_valid = valid[..., j]
        col_vals = arr[..., j][col_valid]
        if col_vals.size == 0:
            medians[j] = 0.0
            means[j] = 0.0
            stds[j] = 1.0
        else:
            medians[j] = np.median(col_vals)
            filled = np.where(np.isnan(arr[..., j]), medians[j], arr[..., j])
            if mask is None:
                fit_vals = filled
            else:
                fit_vals = filled[mask.astype(bool)[..., j]]
            means[j] = float(np.mean(fit_vals))
            std = float(np.std(fit_vals))
            stds[j] = std if std > 1e-6 else 1.0
    return {"median": medians, "mean": means, "std": stds}


def apply_standardizer(values: np.ndarray, stats: Dict[str, np.ndarray]) -> np.ndarray:
    arr = values.astype(np.float32).copy()
    for j in range(arr.shape[-1]):
        arr[..., j] = np.where(np.isnan(arr[..., j]), stats["median"][j], arr[..., j])
        arr[..., j] = (arr[..., j] - stats["mean"][j]) / stats["std"][j]
    return arr


@dataclass
class ScenarioSample:
    scenario_id: str
    component_numeric: np.ndarray
    component_mask: np.ndarray
    family_ids: np.ndarray
    component_ids: np.ndarray
    context: np.ndarray
    target: np.ndarray


class ScenarioDataset(Dataset):
    def __init__(self, samples: Sequence[ScenarioSample]) -> None:
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        return {
            "component_numeric": torch.tensor(sample.component_numeric, dtype=torch.float32),
            "component_mask": torch.tensor(sample.component_mask, dtype=torch.bool),
            "family_ids": torch.tensor(sample.family_ids, dtype=torch.long),
            "component_ids": torch.tensor(sample.component_ids, dtype=torch.long),
            "context": torch.tensor(sample.context, dtype=torch.float32),
            "target": torch.tensor(sample.target, dtype=torch.float32),
        }


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.net(x))


class SetAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.ff = FeedForwardBlock(d_model, dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
        x = self.norm(x + attn_out)
        return self.ff(x)


class PoolingByMultiheadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float, n_seeds: int = 1) -> None:
        super().__init__()
        self.seed = nn.Parameter(torch.randn(1, n_seeds, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.ff = FeedForwardBlock(d_model, dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        q = self.seed.expand(batch_size, -1, -1)
        out, _ = self.attn(q, x, x, key_padding_mask=key_padding_mask)
        out = self.norm(q + out)
        out = self.ff(out)
        return out[:, 0, :]


class SetTransformerRegressor(nn.Module):
    def __init__(
        self,
        num_component_features: int,
        num_context_features: int,
        n_families: int,
        n_components: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
        family_emb_dim: int = 8,
        component_emb_dim: int = 12,
    ) -> None:
        super().__init__()
        self.family_emb = nn.Embedding(n_families, family_emb_dim)
        self.component_emb = nn.Embedding(n_components, component_emb_dim)
        input_dim = num_component_features + family_emb_dim + component_emb_dim
        self.component_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model),
        )
        self.blocks = nn.ModuleList([SetAttentionBlock(d_model, n_heads, dropout) for _ in range(n_layers)])
        self.pool = PoolingByMultiheadAttention(d_model, n_heads, dropout, n_seeds=1)
        self.context_mlp = nn.Sequential(
            nn.Linear(num_context_features, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model),
        )
        self.head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 2),
        )

    def forward(
        self,
        component_numeric: torch.Tensor,
        component_mask: torch.Tensor,
        family_ids: torch.Tensor,
        component_ids: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        fam_emb = self.family_emb(family_ids)
        comp_emb = self.component_emb(component_ids)
        x = torch.cat([component_numeric, fam_emb, comp_emb], dim=-1)
        x = self.component_proj(x)
        for block in self.blocks:
            x = block(x, key_padding_mask=~component_mask)
        pooled = self.pool(x, key_padding_mask=~component_mask)
        ctx = self.context_mlp(context)
        return self.head(torch.cat([pooled, ctx], dim=-1))


def build_samples(
    train_path: Path,
    properties_path: Path,
) -> Tuple[List[ScenarioSample], Dict[str, int], Dict[str, int]]:
    train_df = read_csv_auto(train_path)
    properties_df = read_csv_auto(properties_path)
    numeric_exact, numeric_typical, text_exact, text_typical = prepare_property_lookups(properties_df)
    enriched = enrich_mixtures(
        mixtures_df=train_df,
        numeric_exact=numeric_exact,
        numeric_typical=numeric_typical,
        text_exact=text_exact,
        text_typical=text_typical,
        is_train=True,
    )
    scenario_df = build_compact_v3_from_enriched(enriched, is_train=True)
    scenario_map = scenario_df.set_index(COL_SCENARIO)

    family_vocab = {"<pad>": 0}
    component_vocab = {"<pad>": 0}
    for family in sorted(enriched["family"].astype(str).unique()):
        family_vocab[family] = len(family_vocab)
    for component in sorted(enriched[COL_COMPONENT].astype(str).unique()):
        component_vocab[component] = len(component_vocab)

    max_len = int(enriched.groupby(COL_SCENARIO).size().max())
    samples: List[ScenarioSample] = []

    for scenario_id, group in enriched.groupby(COL_SCENARIO, sort=False):
        group = group.sort_values([COL_DOSE, COL_COMPONENT], ascending=[False, True]).reset_index(drop=True)
        n = len(group)
        comp_numeric = np.full((max_len, len(COMPONENT_NUMERIC_COLS)), np.nan, dtype=np.float32)
        comp_mask = np.zeros(max_len, dtype=bool)
        family_ids = np.zeros(max_len, dtype=np.int64)
        component_ids = np.zeros(max_len, dtype=np.int64)

        for i, row in enumerate(group.to_dict("records")):
            comp_mask[i] = True
            family_ids[i] = family_vocab[row["family"]]
            component_ids[i] = component_vocab[row[COL_COMPONENT]]
            for j, col in enumerate(COMPONENT_NUMERIC_COLS):
                value = row.get(col, np.nan)
                comp_numeric[i, j] = float(value) if pd.notna(value) else np.nan

        scenario_row = scenario_map.loc[scenario_id]
        context = scenario_row[SCENARIO_CONTEXT_COLS].to_numpy(dtype=np.float32)
        target = np.array(
            [float(scenario_row["target_viscosity_delta_pct"]), float(scenario_row["target_oxidation_acm"])],
            dtype=np.float32,
        )
        samples.append(
            ScenarioSample(
                scenario_id=str(scenario_id),
                component_numeric=comp_numeric,
                component_mask=comp_mask,
                family_ids=family_ids,
                component_ids=component_ids,
                context=context,
                target=target,
            )
        )
    return samples, family_vocab, component_vocab


def split_train_inner(indices: np.ndarray, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    shuffled = indices.copy()
    rng.shuffle(shuffled)
    n_val = max(1, int(round(0.15 * len(shuffled))))
    return shuffled[n_val:], shuffled[:n_val]


def summarize_cv_results(cv_results: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "rmse_viscosity",
        "rmse_oxidation",
        "mae_viscosity",
        "mae_oxidation",
        "r2_viscosity",
        "r2_oxidation",
        "mean_rmse",
        "mean_mae",
        "mean_r2",
    ]
    return (
        cv_results.groupby(["config_id", "d_model", "n_heads", "n_layers", "dropout", "lr", "weight_decay"], as_index=False)[metric_cols]
        .mean()
        .sort_values(["mean_rmse", "mean_mae", "mean_r2"], ascending=[True, True, False])
        .reset_index(drop=True)
    )


def build_configs() -> List[Dict[str, float | int | str]]:
    return [
        {"config_id": "st_small", "d_model": 64, "n_heads": 4, "n_layers": 2, "dropout": 0.10, "lr": 3e-4, "weight_decay": 1e-4},
        {"config_id": "st_medium", "d_model": 96, "n_heads": 4, "n_layers": 2, "dropout": 0.10, "lr": 3e-4, "weight_decay": 1e-4},
        {"config_id": "st_deep", "d_model": 96, "n_heads": 4, "n_layers": 3, "dropout": 0.15, "lr": 2e-4, "weight_decay": 1e-4},
    ]


def make_batch_arrays(samples: Sequence[ScenarioSample], idx: Sequence[int]) -> Dict[str, np.ndarray]:
    return {
        "component_numeric": np.stack([samples[i].component_numeric for i in idx], axis=0),
        "component_mask": np.stack([samples[i].component_mask for i in idx], axis=0),
        "family_ids": np.stack([samples[i].family_ids for i in idx], axis=0),
        "component_ids": np.stack([samples[i].component_ids for i in idx], axis=0),
        "context": np.stack([samples[i].context for i in idx], axis=0),
        "target": np.stack([samples[i].target for i in idx], axis=0),
        "scenario_ids": [samples[i].scenario_id for i in idx],
    }


def transform_split(
    samples: Sequence[ScenarioSample],
    train_idx: Sequence[int],
    valid_idx: Sequence[int],
) -> Tuple[List[ScenarioSample], List[ScenarioSample], Dict[str, Dict[str, np.ndarray]]]:
    train_arrays = make_batch_arrays(samples, train_idx)
    valid_arrays = make_batch_arrays(samples, valid_idx)

    train_mask_3d = np.repeat(train_arrays["component_mask"][..., None], len(COMPONENT_NUMERIC_COLS), axis=2)
    component_stats = fit_standardizer(train_arrays["component_numeric"], mask=train_mask_3d)
    context_stats = fit_standardizer(train_arrays["context"])

    y_train_t = transform_targets(train_arrays["target"])
    target_stats = fit_standardizer(y_train_t)

    def rebuild(arrays: Dict[str, np.ndarray]) -> List[ScenarioSample]:
        comp_scaled = apply_standardizer(arrays["component_numeric"], component_stats)
        ctx_scaled = apply_standardizer(arrays["context"], context_stats)
        y_scaled = apply_standardizer(transform_targets(arrays["target"]), target_stats)
        out_samples: List[ScenarioSample] = []
        for i, scenario_id in enumerate(arrays["scenario_ids"]):
            out_samples.append(
                ScenarioSample(
                    scenario_id=scenario_id,
                    component_numeric=comp_scaled[i],
                    component_mask=arrays["component_mask"][i],
                    family_ids=arrays["family_ids"][i],
                    component_ids=arrays["component_ids"][i],
                    context=ctx_scaled[i],
                    target=y_scaled[i],
                )
            )
        return out_samples

    stats = {"component": component_stats, "context": context_stats, "target": target_stats}
    return rebuild(train_arrays), rebuild(valid_arrays), stats


def inverse_scaled_targets(y_scaled: np.ndarray, target_stats: Dict[str, np.ndarray]) -> np.ndarray:
    y_t = y_scaled.astype(np.float32).copy()
    for j in range(y_t.shape[1]):
        y_t[:, j] = y_t[:, j] * target_stats["std"][j] + target_stats["mean"][j]
    return inverse_transform_targets(y_t)


def train_one_model(
    train_samples: Sequence[ScenarioSample],
    config: Dict[str, float | int | str],
    n_families: int,
    n_components: int,
    seed: int,
    max_epochs: int,
    patience: int,
    device: torch.device,
) -> Dict[str, object]:
    all_indices = np.arange(len(train_samples))
    train_inner_idx, val_inner_idx = split_train_inner(all_indices, seed)
    train_inner = [train_samples[i] for i in train_inner_idx]
    val_inner = [train_samples[i] for i in val_inner_idx]

    model = SetTransformerRegressor(
        num_component_features=len(COMPONENT_NUMERIC_COLS),
        num_context_features=len(SCENARIO_CONTEXT_COLS),
        n_families=n_families,
        n_components=n_components,
        d_model=int(config["d_model"]),
        n_heads=int(config["n_heads"]),
        n_layers=int(config["n_layers"]),
        dropout=float(config["dropout"]),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["lr"]), weight_decay=float(config["weight_decay"]))
    loss_fn = nn.SmoothL1Loss(beta=0.5)

    train_loader = DataLoader(ScenarioDataset(train_inner), batch_size=min(16, len(train_inner)), shuffle=True)
    val_loader = DataLoader(ScenarioDataset(val_inner), batch_size=min(32, len(val_inner)), shuffle=False)

    best_state = None
    best_val = float("inf")
    no_improve = 0
    history: List[Dict[str, float]] = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_losses: List[float] = []
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            pred = model(
                component_numeric=batch["component_numeric"].to(device),
                component_mask=batch["component_mask"].to(device),
                family_ids=batch["family_ids"].to(device),
                component_ids=batch["component_ids"].to(device),
                context=batch["context"].to(device),
            )
            loss = loss_fn(pred, batch["target"].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(float(loss.item()))

        model.eval()
        val_losses: List[float] = []
        with torch.no_grad():
            for batch in val_loader:
                pred = model(
                    component_numeric=batch["component_numeric"].to(device),
                    component_mask=batch["component_mask"].to(device),
                    family_ids=batch["family_ids"].to(device),
                    component_ids=batch["component_ids"].to(device),
                    context=batch["context"].to(device),
                )
                loss = loss_fn(pred, batch["target"].to(device))
                val_losses.append(float(loss.item()))

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {"model": model, "best_val_loss": best_val, "history": history}


def predict_model(model: nn.Module, samples: Sequence[ScenarioSample], device: torch.device) -> np.ndarray:
    loader = DataLoader(ScenarioDataset(samples), batch_size=min(32, len(samples)), shuffle=False)
    preds = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            pred = model(
                component_numeric=batch["component_numeric"].to(device),
                component_mask=batch["component_mask"].to(device),
                family_ids=batch["family_ids"].to(device),
                component_ids=batch["component_ids"].to(device),
                context=batch["context"].to(device),
            )
            preds.append(pred.detach().cpu().numpy())
    return np.concatenate(preds, axis=0)


def refit_full_model(
    samples: Sequence[ScenarioSample],
    config: Dict[str, float | int | str],
    n_families: int,
    n_components: int,
    seed: int,
    max_epochs: int,
    patience: int,
    device: torch.device,
) -> Dict[str, object]:
    trained = train_one_model(samples, config, n_families, n_components, seed, max_epochs, patience, device)
    return trained


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Set Transformer on raw Daimler mixture data.")
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--properties", type=Path, default=DEFAULT_PROPERTIES)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, default=220)
    parser.add_argument("--patience", type=int, default=35)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    set_global_seed(args.seed)

    samples, family_vocab, component_vocab = build_samples(args.train, args.properties)
    scenario_ids = np.array([s.scenario_id for s in samples])
    y_true_raw = np.stack([s.target for s in samples], axis=0)
    kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    configs = build_configs()

    cv_rows: List[Dict[str, object]] = []
    oof_rows: List[Dict[str, object]] = []

    for config in configs:
        for fold, (train_idx, valid_idx) in enumerate(kf.split(samples), start=1):
            train_fold, valid_fold, stats = transform_split(samples, train_idx, valid_idx)
            trained = train_one_model(
                train_samples=train_fold,
                config=config,
                n_families=len(family_vocab),
                n_components=len(component_vocab),
                seed=args.seed + fold,
                max_epochs=args.max_epochs,
                patience=args.patience,
                device=device,
            )
            pred_scaled = predict_model(trained["model"], valid_fold, device)
            pred_raw = inverse_scaled_targets(pred_scaled, stats["target"])
            true_raw = y_true_raw[valid_idx]
            metrics = compute_metrics(true_raw, pred_raw)
            cv_rows.append(
                {
                    "config_id": config["config_id"],
                    "fold": fold,
                    "d_model": config["d_model"],
                    "n_heads": config["n_heads"],
                    "n_layers": config["n_layers"],
                    "dropout": config["dropout"],
                    "lr": config["lr"],
                    "weight_decay": config["weight_decay"],
                    "best_val_loss": trained["best_val_loss"],
                    **metrics,
                }
            )
            for local_i, global_i in enumerate(valid_idx):
                oof_rows.append(
                    {
                        "config_id": config["config_id"],
                        "fold": fold,
                        "scenario_id": scenario_ids[global_i],
                        TARGET_VISC_RAW: float(true_raw[local_i, 0]),
                        TARGET_OXID_RAW: float(true_raw[local_i, 1]),
                        "pred_viscosity": float(pred_raw[local_i, 0]),
                        "pred_oxidation": float(pred_raw[local_i, 1]),
                    }
                )

            print(
                f"[{config['config_id']}] fold={fold} "
                f"rmse=({metrics['rmse_viscosity']:.2f}, {metrics['rmse_oxidation']:.2f}) "
                f"mae=({metrics['mae_viscosity']:.2f}, {metrics['mae_oxidation']:.2f}) "
                f"r2=({metrics['r2_viscosity']:.3f}, {metrics['r2_oxidation']:.3f})"
            )

    cv_df = pd.DataFrame(cv_rows)
    summary_df = summarize_cv_results(cv_df)
    best_row = summary_df.iloc[0].to_dict()
    best_config = next(cfg for cfg in configs if cfg["config_id"] == best_row["config_id"])

    full_samples, _, full_stats = transform_split(samples, np.arange(len(samples)), np.arange(len(samples)))
    final_model_bundle = refit_full_model(
        samples=full_samples,
        config=best_config,
        n_families=len(family_vocab),
        n_components=len(component_vocab),
        seed=args.seed,
        max_epochs=args.max_epochs,
        patience=args.patience,
        device=device,
    )

    cv_path = args.outdir / "set_transformer_cv_fold_metrics.csv"
    summary_path = args.outdir / "set_transformer_cv_summary.csv"
    oof_path = args.outdir / "set_transformer_oof_predictions.csv"
    manifest_path = args.outdir / "set_transformer_run_manifest.json"
    model_path = args.outdir / "set_transformer_best_model.pt"

    cv_df.to_csv(cv_path, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(oof_rows).to_csv(oof_path, index=False, encoding="utf-8-sig")
    torch.save(
        {
            "model_state_dict": final_model_bundle["model"].state_dict(),
            "best_config": best_config,
            "family_vocab": family_vocab,
            "component_vocab": component_vocab,
            "component_numeric_cols": COMPONENT_NUMERIC_COLS,
            "scenario_context_cols": SCENARIO_CONTEXT_COLS,
            "full_data_stats": full_stats,
        },
        model_path,
    )

    manifest = {
        "train_path": str(args.train),
        "properties_path": str(args.properties),
        "n_samples": len(samples),
        "max_set_size": int(max(s.component_mask.sum() for s in samples)),
        "n_families": len(family_vocab),
        "n_components": len(component_vocab),
        "configs": configs,
        "best_config": best_config,
        "best_summary_row": best_row,
        "component_numeric_cols": COMPONENT_NUMERIC_COLS,
        "scenario_context_cols": SCENARIO_CONTEXT_COLS,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved fold metrics to: {cv_path.resolve()}")
    print(f"Saved summary to: {summary_path.resolve()}")
    print(f"Saved OOF predictions to: {oof_path.resolve()}")
    print(f"Saved model bundle to: {model_path.resolve()}")
    print(f"Saved manifest to: {manifest_path.resolve()}")


if __name__ == "__main__":
    main()
