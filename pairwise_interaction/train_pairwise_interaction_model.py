#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from torch import nn
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN = ROOT / "new_datasets" / "train_component_level_transformed.csv"
DEFAULT_TEST = ROOT / "new_datasets" / "test_component_level_transformed.csv"
DEFAULT_OUTDIR = ROOT / "pairwise_interaction" / "pairwise_output"

SCENARIO_COL = "scenario_id"
COMPONENT_COL = "Компонент"
BATCH_COL = "Наименование партии"
DOSE_COL = "Массовая доля, %"
TEMP_COL = "Температура испытания | ASTM D445 Daimler Oxidation Test (DOT), °C"
TIME_COL = "Время испытания | - Daimler Oxidation Test (DOT), ч"
BIOFUEL_COL = "Количество биотоплива | - Daimler Oxidation Test (DOT), % масс"
CATALYST_COL = "Дозировка катализатора, категория"
TARGET_VISC_COL = "Delta Kin. Viscosity KV100 - relative | - Daimler Oxidation Test (DOT), %"
TARGET_OX_COL = "Oxidation EOT | DIN 51453 Daimler Oxidation Test (DOT), A/cm"

COMPONENT_NUMERIC_COLS = [
    DOSE_COL,
    "% масс. (Mo)",
    "Активный Азот / Кислород, % масс. (N или O)",
    "Потенциал ионизации,эВ",
    "Энергия ВЗМО, эВ",
    "Дипольный момент, Д",
    "Массовая доля цинка, ASTM D6481",
    "Массовая доля серы, ASTM D6481",
    "Массовая доля кальция, ASTM D6481",
    "Содержание металла (Ca/Mg), % масс.",
    "Содержание Бора",
    "Щелочное число, ASTM D2896",
    "Кинематическая вязкость, при 40°C, ASTM D445",
    "Кинематическая вязкость, при 100°C, ASTM D445",
    "Динамическая вязкость CCS -35°C, ASTM D5293",
    "Температура застывания, ГОСТ 20287, метод Б",
    "Индекс вязкости, ГОСТ 25371",
    "Испаряемость по NOACK, ASTM D5800",
    "Массовая доля фосфора, ASTM D6481",
    "Длина углеродной цепи",
]

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

VISC_SCALE = 50.0
TARGET_CLIP = (-6.0, 6.0)
SEED = 42


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "rmse_viscosity": float(math.sqrt(mean_squared_error(y_true[:, 0], y_pred[:, 0]))),
        "rmse_oxidation": float(math.sqrt(mean_squared_error(y_true[:, 1], y_pred[:, 1]))),
        "mae_viscosity": float(mean_absolute_error(y_true[:, 0], y_pred[:, 0])),
        "mae_oxidation": float(mean_absolute_error(y_true[:, 1], y_pred[:, 1])),
        "r2_viscosity": float(r2_score(y_true[:, 0], y_pred[:, 0])),
        "r2_oxidation": float(r2_score(y_true[:, 1], y_pred[:, 1])),
    }


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


def family_of(component_name: str) -> str:
    if not isinstance(component_name, str) or not component_name:
        return "unknown"
    return component_name.split("_", 1)[0]


def build_scenario_context(row: pd.Series) -> Dict[str, float]:
    temperature = float(row[TEMP_COL])
    time_h = float(row[TIME_COL])
    biofuel = float(row[BIOFUEL_COL])
    catalyst = float(row[CATALYST_COL])
    severity = time_h * math.exp((temperature - 150.0) / 10.0)
    return {
        "temperature_c": temperature,
        "time_h": time_h,
        "biofuel_pct": biofuel,
        "catalyst_category": catalyst,
        "severity_exp": severity,
        "temperature_x_time": temperature * time_h,
        "temperature_x_biofuel": temperature * biofuel,
        "severity_x_biofuel": severity * biofuel,
        "severity_x_catalyst": severity * catalyst,
        "regime_load": severity * (1.0 + biofuel / 10.0),
    }


def fit_standardizer(values: np.ndarray, valid_mask: np.ndarray | None = None) -> Dict[str, np.ndarray]:
    arr = values.astype(np.float32).copy()
    base_valid = ~np.isnan(arr)
    if valid_mask is not None:
        base_valid &= valid_mask.astype(bool)
    medians = np.zeros(arr.shape[-1], dtype=np.float32)
    means = np.zeros(arr.shape[-1], dtype=np.float32)
    stds = np.ones(arr.shape[-1], dtype=np.float32)
    for j in range(arr.shape[-1]):
        col_vals = arr[..., j][base_valid[..., j]]
        if col_vals.size == 0:
            continue
        med = float(np.median(col_vals))
        medians[j] = med
        filled = np.where(np.isnan(arr[..., j]), med, arr[..., j])
        fit_vals = filled[base_valid[..., j]]
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
    component_valid: np.ndarray
    family_ids: np.ndarray
    component_ids: np.ndarray
    context: np.ndarray
    target: np.ndarray | None = None


class PairDataset(Dataset):
    def __init__(self, samples: Sequence[ScenarioSample]) -> None:
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        item = {
            "component_numeric": torch.tensor(sample.component_numeric, dtype=torch.float32),
            "component_valid": torch.tensor(sample.component_valid, dtype=torch.bool),
            "family_ids": torch.tensor(sample.family_ids, dtype=torch.long),
            "component_ids": torch.tensor(sample.component_ids, dtype=torch.long),
            "context": torch.tensor(sample.context, dtype=torch.float32),
        }
        if sample.target is not None:
            item["target"] = torch.tensor(sample.target, dtype=torch.float32)
        return item


class PairwiseInteractionRegressor(nn.Module):
    def __init__(
        self,
        num_component_features: int,
        num_context_features: int,
        n_families: int,
        n_components: int,
        max_components: int,
        d_model: int,
        dropout: float,
        family_emb_dim: int = 8,
        component_emb_dim: int = 12,
        ctx_dim: int = 32,
        pair_hidden: int = 96,
    ) -> None:
        super().__init__()
        self.family_emb = nn.Embedding(n_families, family_emb_dim)
        self.component_emb = nn.Embedding(n_components, component_emb_dim)
        comp_in = num_component_features + family_emb_dim + component_emb_dim
        self.component_encoder = nn.Sequential(
            nn.Linear(comp_in, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.context_mlp = nn.Sequential(
            nn.Linear(num_context_features, ctx_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ctx_dim, ctx_dim),
            nn.GELU(),
        )
        pair_in = d_model * 4 + ctx_dim
        self.pair_mlp = nn.Sequential(
            nn.Linear(pair_in, pair_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(pair_hidden, d_model),
            nn.GELU(),
        )
        head_in = d_model * 4 + ctx_dim
        self.head = nn.Sequential(
            nn.Linear(head_in, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 2),
        )
        idx_i, idx_j = torch.triu_indices(max_components, max_components, offset=1)
        self.register_buffer("pair_i", idx_i, persistent=False)
        self.register_buffer("pair_j", idx_j, persistent=False)

    def masked_mean(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.float().unsqueeze(-1)
        denom = weights.sum(dim=1).clamp_min(1.0)
        return (x * weights).sum(dim=1) / denom

    def masked_max(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        neg = torch.finfo(x.dtype).min
        masked = x.masked_fill(~mask.unsqueeze(-1), neg)
        out = masked.max(dim=1).values
        out = torch.where(torch.isfinite(out), out, torch.zeros_like(out))
        return out

    def forward(
        self,
        component_numeric: torch.Tensor,
        component_valid: torch.Tensor,
        family_ids: torch.Tensor,
        component_ids: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        fam_emb = self.family_emb(family_ids)
        comp_emb = self.component_emb(component_ids)
        comp_input = torch.cat([component_numeric, fam_emb, comp_emb], dim=-1)
        comp_repr = self.component_encoder(comp_input)
        self_mean = self.masked_mean(comp_repr, component_valid)
        self_max = self.masked_max(comp_repr, component_valid)

        ctx = self.context_mlp(context)
        xi = comp_repr[:, self.pair_i, :]
        xj = comp_repr[:, self.pair_j, :]
        pair_valid = component_valid[:, self.pair_i] & component_valid[:, self.pair_j]
        ctx_expanded = ctx.unsqueeze(1).expand(-1, xi.size(1), -1)
        pair_input = torch.cat([xi, xj, torch.abs(xi - xj), xi * xj, ctx_expanded], dim=-1)
        pair_repr = self.pair_mlp(pair_input)
        pair_mean = self.masked_mean(pair_repr, pair_valid)
        pair_max = self.masked_max(pair_repr, pair_valid)

        head_input = torch.cat([self_mean, self_max, pair_mean, pair_max, ctx], dim=-1)
        return self.head(head_input)


def build_samples(
    df: pd.DataFrame,
    family_to_id: Dict[str, int],
    component_to_id: Dict[str, int],
    max_components: int,
    include_target: bool,
) -> List[ScenarioSample]:
    samples: List[ScenarioSample] = []
    numeric_cols = COMPONENT_NUMERIC_COLS
    for scenario_id, group in df.groupby(SCENARIO_COL, sort=False):
        g = group.sort_values([COMPONENT_COL, BATCH_COL], kind="stable").reset_index(drop=True)
        n = min(len(g), max_components)
        g = g.iloc[:n].copy()

        raw_numeric = g[numeric_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        missing_flags = np.isnan(raw_numeric).astype(np.float32)
        batch_str = g[BATCH_COL].fillna("").astype(str).str.strip().str.lower()
        is_typical = batch_str.eq("typical").astype(np.float32).to_numpy()[:, None]
        is_measured = (1.0 - is_typical).astype(np.float32)
        component_numeric = np.concatenate([raw_numeric, missing_flags, is_typical, is_measured], axis=1)

        comp_valid = np.zeros(max_components, dtype=bool)
        comp_valid[:n] = True
        padded_numeric = np.zeros((max_components, component_numeric.shape[1]), dtype=np.float32)
        padded_numeric[:n] = component_numeric

        family_ids = np.zeros(max_components, dtype=np.int64)
        component_ids = np.zeros(max_components, dtype=np.int64)
        for idx, row in g.iterrows():
            family_ids[idx] = family_to_id.get(family_of(str(row[COMPONENT_COL])), 0)
            component_ids[idx] = component_to_id.get(str(row[COMPONENT_COL]), 0)

        context = build_scenario_context(g.iloc[0])
        context_arr = np.array([context[c] for c in SCENARIO_CONTEXT_COLS], dtype=np.float32)

        target = None
        if include_target:
            target = np.array(
                [float(g.iloc[0][TARGET_VISC_COL]), float(g.iloc[0][TARGET_OX_COL])],
                dtype=np.float32,
            )

        samples.append(
            ScenarioSample(
                scenario_id=str(scenario_id),
                component_numeric=padded_numeric,
                component_valid=comp_valid,
                family_ids=family_ids,
                component_ids=component_ids,
                context=context_arr,
                target=target,
            )
        )
    return samples


def split_and_standardize(
    train_samples: Sequence[ScenarioSample],
    val_samples: Sequence[ScenarioSample],
) -> tuple[list[ScenarioSample], list[ScenarioSample], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    train_numeric = np.stack([s.component_numeric[:, : len(COMPONENT_NUMERIC_COLS)] for s in train_samples], axis=0)
    train_valid = np.stack([np.repeat(s.component_valid[:, None], len(COMPONENT_NUMERIC_COLS), axis=1) for s in train_samples], axis=0)
    comp_stats = fit_standardizer(train_numeric, valid_mask=train_valid)

    train_context = np.stack([s.context for s in train_samples], axis=0)
    ctx_stats = fit_standardizer(train_context)

    def transform(samples: Sequence[ScenarioSample]) -> list[ScenarioSample]:
        out: list[ScenarioSample] = []
        for sample in samples:
            numeric = sample.component_numeric.copy()
            raw = numeric[:, : len(COMPONENT_NUMERIC_COLS)]
            flags = numeric[:, len(COMPONENT_NUMERIC_COLS) :]
            raw_scaled = apply_standardizer(raw[None, ...], comp_stats)[0]
            numeric_scaled = np.concatenate([raw_scaled, flags], axis=1)
            ctx_scaled = apply_standardizer(sample.context[None, :], ctx_stats)[0]
            out.append(
                ScenarioSample(
                    scenario_id=sample.scenario_id,
                    component_numeric=numeric_scaled,
                    component_valid=sample.component_valid,
                    family_ids=sample.family_ids,
                    component_ids=sample.component_ids,
                    context=ctx_scaled,
                    target=transform_targets(sample.target[None, :])[0] if sample.target is not None else None,
                )
            )
        return out

    return transform(train_samples), transform(val_samples), comp_stats, ctx_stats


def train_one_fold(
    train_samples: Sequence[ScenarioSample],
    val_samples: Sequence[ScenarioSample],
    config: Dict[str, float | int | str],
    n_families: int,
    n_components: int,
    max_components: int,
    device: torch.device,
) -> Dict[str, object]:
    train_scaled, val_scaled, comp_stats, ctx_stats = split_and_standardize(train_samples, val_samples)
    model = PairwiseInteractionRegressor(
        num_component_features=train_scaled[0].component_numeric.shape[1],
        num_context_features=train_scaled[0].context.shape[0],
        n_families=n_families,
        n_components=n_components,
        max_components=max_components,
        d_model=int(config["d_model"]),
        dropout=float(config["dropout"]),
        pair_hidden=int(config["pair_hidden"]),
        ctx_dim=int(config["ctx_dim"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["lr"]),
        weight_decay=float(config["weight_decay"]),
    )
    criterion = nn.MSELoss()

    train_loader = DataLoader(PairDataset(train_scaled), batch_size=min(int(config["batch_size"]), len(train_scaled)), shuffle=True)
    val_loader = DataLoader(PairDataset(val_scaled), batch_size=min(32, len(val_scaled)), shuffle=False)

    best_state = None
    best_val = float("inf")
    wait = 0
    for epoch in range(int(config["epochs"])):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            preds = model(
                batch["component_numeric"].to(device),
                batch["component_valid"].to(device),
                batch["family_ids"].to(device),
                batch["component_ids"].to(device),
                batch["context"].to(device),
            )
            loss = criterion(preds, batch["target"].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        val_losses: list[float] = []
        with torch.no_grad():
            for batch in val_loader:
                preds = model(
                    batch["component_numeric"].to(device),
                    batch["component_valid"].to(device),
                    batch["family_ids"].to(device),
                    batch["component_ids"].to(device),
                    batch["context"].to(device),
                )
                val_losses.append(float(criterion(preds, batch["target"].to(device)).item()))
        val_loss = float(np.mean(val_losses))
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            wait = 0
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= int(config["patience"]):
                break

    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    preds_t: list[np.ndarray] = []
    with torch.no_grad():
        for batch in val_loader:
            pred = model(
                batch["component_numeric"].to(device),
                batch["component_valid"].to(device),
                batch["family_ids"].to(device),
                batch["component_ids"].to(device),
                batch["context"].to(device),
            )
            preds_t.append(pred.cpu().numpy())
    pred_val = inverse_transform_targets(np.vstack(preds_t))
    true_val = np.stack([s.target for s in val_samples], axis=0)
    metrics = compute_metrics(true_val, pred_val)
    return {
        "model_state": best_state,
        "comp_stats": comp_stats,
        "ctx_stats": ctx_stats,
        "val_pred": pred_val,
        "metrics": metrics,
    }


def predict_with_state(
    samples: Sequence[ScenarioSample],
    model_state: Dict[str, torch.Tensor],
    comp_stats: Dict[str, np.ndarray],
    ctx_stats: Dict[str, np.ndarray],
    config: Dict[str, float | int | str],
    n_families: int,
    n_components: int,
    max_components: int,
    device: torch.device,
) -> np.ndarray:
    transformed: list[ScenarioSample] = []
    for sample in samples:
        numeric = sample.component_numeric.copy()
        raw = numeric[:, : len(COMPONENT_NUMERIC_COLS)]
        flags = numeric[:, len(COMPONENT_NUMERIC_COLS) :]
        raw_scaled = apply_standardizer(raw[None, ...], comp_stats)[0]
        numeric_scaled = np.concatenate([raw_scaled, flags], axis=1)
        ctx_scaled = apply_standardizer(sample.context[None, :], ctx_stats)[0]
        transformed.append(
            ScenarioSample(
                scenario_id=sample.scenario_id,
                component_numeric=numeric_scaled,
                component_valid=sample.component_valid,
                family_ids=sample.family_ids,
                component_ids=sample.component_ids,
                context=ctx_scaled,
                target=sample.target,
            )
        )
    loader = DataLoader(PairDataset(transformed), batch_size=min(32, len(transformed)), shuffle=False)
    model = PairwiseInteractionRegressor(
        num_component_features=transformed[0].component_numeric.shape[1],
        num_context_features=transformed[0].context.shape[0],
        n_families=n_families,
        n_components=n_components,
        max_components=max_components,
        d_model=int(config["d_model"]),
        dropout=float(config["dropout"]),
        pair_hidden=int(config["pair_hidden"]),
        ctx_dim=int(config["ctx_dim"]),
    ).to(device)
    model.load_state_dict(model_state)
    model.eval()
    preds_t: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            pred = model(
                batch["component_numeric"].to(device),
                batch["component_valid"].to(device),
                batch["family_ids"].to(device),
                batch["component_ids"].to(device),
                batch["context"].to(device),
            )
            preds_t.append(pred.cpu().numpy())
    return inverse_transform_targets(np.vstack(preds_t))


def fit_full_standardizers(samples: Sequence[ScenarioSample]) -> tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    train_numeric = np.stack([s.component_numeric[:, : len(COMPONENT_NUMERIC_COLS)] for s in samples], axis=0)
    train_valid = np.stack([np.repeat(s.component_valid[:, None], len(COMPONENT_NUMERIC_COLS), axis=1) for s in samples], axis=0)
    comp_stats = fit_standardizer(train_numeric, valid_mask=train_valid)
    ctx_stats = fit_standardizer(np.stack([s.context for s in samples], axis=0))
    return comp_stats, ctx_stats


def train_full_model(
    samples: Sequence[ScenarioSample],
    config: Dict[str, float | int | str],
    n_families: int,
    n_components: int,
    max_components: int,
    device: torch.device,
) -> tuple[Dict[str, torch.Tensor], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    comp_stats, ctx_stats = fit_full_standardizers(samples)
    indices = np.arange(len(samples))
    rng = np.random.default_rng(SEED)
    rng.shuffle(indices)
    split = max(1, int(len(indices) * 0.15))
    val_idx = indices[:split]
    train_idx = indices[split:]
    train_samples = [samples[i] for i in train_idx]
    val_samples = [samples[i] for i in val_idx]
    result = train_one_fold(train_samples, val_samples, config, n_families, n_components, max_components, device)
    return result["model_state"], comp_stats, ctx_stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Train explicit pairwise interaction model on component-level mixtures.")
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args()

    set_global_seed(SEED)
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(args.train)
    test_df = pd.read_csv(args.test)
    print(f"train rows={len(train_df)} scenarios={train_df[SCENARIO_COL].nunique()}")
    print(f"test rows={len(test_df)} scenarios={test_df[SCENARIO_COL].nunique()}")

    family_vocab = sorted({family_of(v) for v in pd.concat([train_df[COMPONENT_COL], test_df[COMPONENT_COL]], axis=0).astype(str)})
    component_vocab = sorted(set(pd.concat([train_df[COMPONENT_COL], test_df[COMPONENT_COL]], axis=0).astype(str)))
    family_to_id = {name: idx + 1 for idx, name in enumerate(family_vocab)}
    component_to_id = {name: idx + 1 for idx, name in enumerate(component_vocab)}
    max_components = int(
        max(
            train_df.groupby(SCENARIO_COL).size().max(),
            test_df.groupby(SCENARIO_COL).size().max(),
        )
    )
    print(f"max_components={max_components}, families={len(family_vocab)}, components={len(component_vocab)}")

    train_samples = build_samples(train_df, family_to_id, component_to_id, max_components=max_components, include_target=True)
    test_samples = build_samples(test_df, family_to_id, component_to_id, max_components=max_components, include_target=False)
    y_true = np.stack([s.target for s in train_samples], axis=0)
    scenario_ids = np.array([s.scenario_id for s in train_samples])

    configs = [
        {
            "name": "pair_small",
            "d_model": 48,
            "ctx_dim": 24,
            "pair_hidden": 80,
            "dropout": 0.15,
            "lr": 1e-3,
            "weight_decay": 2e-4,
            "batch_size": 16,
            "epochs": 140,
            "patience": 20,
        },
        {
            "name": "pair_medium",
            "d_model": 64,
            "ctx_dim": 32,
            "pair_hidden": 96,
            "dropout": 0.20,
            "lr": 8e-4,
            "weight_decay": 3e-4,
            "batch_size": 16,
            "epochs": 160,
            "patience": 24,
        },
    ]
    device = torch.device("cpu")
    splitter = KFold(n_splits=5, shuffle=True, random_state=SEED)

    summary_rows: list[dict] = []
    best_config = None
    best_oof = None
    best_mean_rmse = float("inf")

    for config in configs:
        print(f"\n=== config={config['name']} ===")
        oof_pred = np.zeros_like(y_true, dtype=np.float32)
        fold_rows = []
        for fold, (train_idx, val_idx) in enumerate(splitter.split(scenario_ids), start=1):
            fold_train = [train_samples[i] for i in train_idx]
            fold_val = [train_samples[i] for i in val_idx]
            result = train_one_fold(
                fold_train,
                fold_val,
                config=config,
                n_families=len(family_to_id) + 1,
                n_components=len(component_to_id) + 1,
                max_components=max_components,
                device=device,
            )
            oof_pred[val_idx] = result["val_pred"]
            row = {"config": config["name"], "fold": fold, **result["metrics"]}
            fold_rows.append(row)
            print(
                f"fold={fold} rmse_visc={row['rmse_viscosity']:.2f} "
                f"rmse_ox={row['rmse_oxidation']:.2f} "
                f"mae_visc={row['mae_viscosity']:.2f} mae_ox={row['mae_oxidation']:.2f}"
            )

        metrics = compute_metrics(y_true, oof_pred)
        mean_rmse = (metrics["rmse_viscosity"] + metrics["rmse_oxidation"]) / 2.0
        summary_rows.append({"config": config["name"], **metrics, "mean_rmse": mean_rmse})
        pd.DataFrame(fold_rows).to_csv(outdir / f"{config['name']}_fold_metrics.csv", index=False)
        pd.DataFrame(
            {
                SCENARIO_COL: scenario_ids,
                "target_viscosity_true": y_true[:, 0],
                "target_oxidation_true": y_true[:, 1],
                "target_viscosity_pred": oof_pred[:, 0],
                "target_oxidation_pred": oof_pred[:, 1],
            }
        ).to_csv(outdir / f"{config['name']}_oof_predictions.csv", index=False)
        print(
            f"OOF config={config['name']} rmse_visc={metrics['rmse_viscosity']:.2f} "
            f"rmse_ox={metrics['rmse_oxidation']:.2f} mean_rmse={mean_rmse:.2f}"
        )
        if mean_rmse < best_mean_rmse:
            best_mean_rmse = mean_rmse
            best_config = config
            best_oof = oof_pred.copy()

    assert best_config is not None
    summary_df = pd.DataFrame(summary_rows).sort_values("mean_rmse").reset_index(drop=True)
    summary_df.to_csv(outdir / "pairwise_cv_summary.csv", index=False)
    print("\nBest config:", best_config["name"])
    print(summary_df.to_string(index=False))

    final_state, final_comp_stats, final_ctx_stats = train_full_model(
        train_samples,
        config=best_config,
        n_families=len(family_to_id) + 1,
        n_components=len(component_to_id) + 1,
        max_components=max_components,
        device=device,
    )
    test_pred = predict_with_state(
        test_samples,
        model_state=final_state,
        comp_stats=final_comp_stats,
        ctx_stats=final_ctx_stats,
        config=best_config,
        n_families=len(family_to_id) + 1,
        n_components=len(component_to_id) + 1,
        max_components=max_components,
        device=device,
    )
    prediction_df = pd.DataFrame(
        {
            SCENARIO_COL: [s.scenario_id for s in test_samples],
            TARGET_VISC_COL: test_pred[:, 0],
            TARGET_OX_COL: test_pred[:, 1],
        }
    )
    prediction_df.to_csv(outdir / "prediction.csv", index=False)
    if best_oof is not None:
        pd.DataFrame(
            {
                SCENARIO_COL: scenario_ids,
                "target_viscosity_true": y_true[:, 0],
                "target_oxidation_true": y_true[:, 1],
                "target_viscosity_pred": best_oof[:, 0],
                "target_oxidation_pred": best_oof[:, 1],
            }
        ).to_csv(outdir / "best_oof_predictions.csv", index=False)

    torch.save(
        {
            "config": best_config,
            "model_state": final_state,
            "component_stats": final_comp_stats,
            "context_stats": final_ctx_stats,
            "component_numeric_cols": COMPONENT_NUMERIC_COLS,
            "context_cols": SCENARIO_CONTEXT_COLS,
            "family_to_id": family_to_id,
            "component_to_id": component_to_id,
            "max_components": max_components,
        },
        outdir / "pairwise_best_model.pt",
    )
    manifest = {
        "train_path": str(args.train),
        "test_path": str(args.test),
        "best_config": best_config,
        "component_numeric_cols": COMPONENT_NUMERIC_COLS,
        "component_feature_layout": {
            "raw_numeric": len(COMPONENT_NUMERIC_COLS),
            "missing_flags": len(COMPONENT_NUMERIC_COLS),
            "source_flags": ["is_typical_batch", "is_measured_batch"],
        },
        "context_cols": SCENARIO_CONTEXT_COLS,
        "max_components": max_components,
        "n_families": len(family_to_id),
        "n_components": len(component_to_id),
        "cv_summary": summary_df.to_dict(orient="records"),
    }
    (outdir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved outputs to {outdir}")


if __name__ == "__main__":
    main()
