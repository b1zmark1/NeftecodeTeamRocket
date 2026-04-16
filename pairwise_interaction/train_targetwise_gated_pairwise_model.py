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
DEFAULT_TRAIN_COMPONENTS = ROOT / "new_datasets" / "train_component_level_transformed.csv"
DEFAULT_TEST_COMPONENTS = ROOT / "new_datasets" / "test_component_level_transformed.csv"
DEFAULT_TRAIN_SCENARIO = ROOT / "new_datasets" / "train_scenario_level_features.csv"
DEFAULT_TEST_SCENARIO = ROOT / "new_datasets" / "test_scenario_level_features.csv"
DEFAULT_OUTDIR = ROOT / "pairwise_interaction" / "targetwise_gated_output"

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

PAIR_TYPE_NONE = 0
PAIR_TYPE_AO_AO = 1
PAIR_TYPE_AO_MO = 2
PAIR_TYPE_ZDDP_DETERGENT = 3
PAIR_TYPE_ZN_DISPERSANT = 4
PAIR_TYPE_ANTIWEAR_CA_MG = 5
PAIR_TYPE_NAMES = {
    0: "none",
    1: "ao_ao",
    2: "ao_mo",
    3: "zddp_detergent",
    4: "zn_dispersant",
    5: "antiwear_ca_mg",
}

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

BASE_CONTEXT_COLS = [
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

SCENARIO_ENGINEERED_COLS = [
    "synergy_ao_phenol_x_diphenylamine_active_no",
    "synergy_ao_diphenylamine_active_no_x_mo",
    "synergy_zn_x_boron_dispersant",
    "synergy_aw_sulfur_x_ca_ca_mg",
    "ao_k_avg_weighted_active_no",
    "ao_k_avg_arithmetic",
    "ao_ionization_min",
    "ao_homo_max",
]

CONTEXT_COLS = BASE_CONTEXT_COLS + SCENARIO_ENGINEERED_COLS

VISC_SCALE = 50.0
TARGET_CLIP = (-6.0, 6.0)
SEED = 42


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def family_of(component_name: str) -> str:
    if not isinstance(component_name, str) or not component_name:
        return "unknown"
    return component_name.split("_", 1)[0]


def build_base_context(row: pd.Series) -> Dict[str, float]:
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


def target_transform(y: np.ndarray, target_name: str) -> np.ndarray:
    y = y.astype(np.float32)
    if target_name == "viscosity":
        return np.arcsinh(y / VISC_SCALE)
    if target_name == "oxidation":
        return np.log1p(np.clip(y, a_min=0.0, a_max=None))
    raise ValueError(target_name)


def target_inverse(y_t: np.ndarray, target_name: str) -> np.ndarray:
    y_t = y_t.astype(np.float32)
    if target_name == "viscosity":
        return VISC_SCALE * np.sinh(np.clip(y_t, *TARGET_CLIP))
    if target_name == "oxidation":
        return np.expm1(np.clip(y_t, *TARGET_CLIP))
    raise ValueError(target_name)


def compute_target_metrics(y_true: np.ndarray, y_pred: np.ndarray, target_name: str) -> Dict[str, float]:
    return {
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "target": target_name,
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
        vals = arr[..., j][base_valid[..., j]]
        if vals.size == 0:
            continue
        med = float(np.median(vals))
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
    pair_allowed: np.ndarray
    pair_type_ids: np.ndarray
    target: float | None = None


class ScenarioDataset(Dataset):
    def __init__(self, samples: Sequence[ScenarioSample]) -> None:
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s = self.samples[idx]
        item = {
            "component_numeric": torch.tensor(s.component_numeric, dtype=torch.float32),
            "component_valid": torch.tensor(s.component_valid, dtype=torch.bool),
            "family_ids": torch.tensor(s.family_ids, dtype=torch.long),
            "component_ids": torch.tensor(s.component_ids, dtype=torch.long),
            "context": torch.tensor(s.context, dtype=torch.float32),
            "pair_allowed": torch.tensor(s.pair_allowed, dtype=torch.bool),
            "pair_type_ids": torch.tensor(s.pair_type_ids, dtype=torch.long),
        }
        if s.target is not None:
            item["target"] = torch.tensor(float(s.target), dtype=torch.float32)
        return item


class TargetwiseGatedPairwiseRegressor(nn.Module):
    def __init__(
        self,
        num_component_features: int,
        num_context_features: int,
        n_families: int,
        n_components: int,
        max_components: int,
        d_model: int,
        dropout: float,
        ctx_dim: int,
        pair_hidden: int,
        pair_type_emb_dim: int = 6,
        family_emb_dim: int = 8,
        component_emb_dim: int = 12,
    ) -> None:
        super().__init__()
        self.family_emb = nn.Embedding(n_families, family_emb_dim)
        self.component_emb = nn.Embedding(n_components, component_emb_dim)
        self.pair_type_emb = nn.Embedding(len(PAIR_TYPE_NAMES), pair_type_emb_dim)

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
        pair_in = d_model * 4 + pair_type_emb_dim + ctx_dim
        self.pair_mlp = nn.Sequential(
            nn.Linear(pair_in, pair_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(pair_hidden, d_model),
            nn.GELU(),
        )
        self.gate_mlp = nn.Sequential(
            nn.Linear(pair_in, pair_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(pair_hidden // 2, 1),
        )
        self.attn_mlp = nn.Sequential(
            nn.Linear(pair_in, pair_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(pair_hidden // 2, 1),
        )
        head_in = d_model * 4 + ctx_dim
        self.head = nn.Sequential(
            nn.Linear(head_in, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        idx_i, idx_j = torch.triu_indices(max_components, max_components, offset=1)
        self.register_buffer("pair_i", idx_i, persistent=False)
        self.register_buffer("pair_j", idx_j, persistent=False)

    def masked_mean(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        w = mask.float().unsqueeze(-1)
        denom = w.sum(dim=1).clamp_min(1.0)
        return (x * w).sum(dim=1) / denom

    def masked_max(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        neg = torch.finfo(x.dtype).min
        masked = x.masked_fill(~mask.unsqueeze(-1), neg)
        out = masked.max(dim=1).values
        return torch.where(torch.isfinite(out), out, torch.zeros_like(out))

    def masked_softmax(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        neg = torch.finfo(logits.dtype).min
        masked = logits.masked_fill(~mask, neg)
        weights = torch.softmax(masked, dim=1)
        weights = torch.where(mask, weights, torch.zeros_like(weights))
        denom = weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return weights / denom

    def forward(
        self,
        component_numeric: torch.Tensor,
        component_valid: torch.Tensor,
        family_ids: torch.Tensor,
        component_ids: torch.Tensor,
        context: torch.Tensor,
        pair_allowed: torch.Tensor,
        pair_type_ids: torch.Tensor,
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
        type_emb = self.pair_type_emb(pair_type_ids)
        valid_pairs = component_valid[:, self.pair_i] & component_valid[:, self.pair_j] & pair_allowed
        ctx_exp = ctx.unsqueeze(1).expand(-1, xi.size(1), -1)
        pair_input = torch.cat([xi, xj, torch.abs(xi - xj), xi * xj, type_emb, ctx_exp], dim=-1)
        pair_base = self.pair_mlp(pair_input)
        gates = torch.sigmoid(self.gate_mlp(pair_input)).squeeze(-1)
        logits = self.attn_mlp(pair_input).squeeze(-1)
        attn = self.masked_softmax(logits, valid_pairs)
        gated = pair_base * gates.unsqueeze(-1)
        pair_attn = (gated * attn.unsqueeze(-1)).sum(dim=1)
        pair_max = self.masked_max(gated, valid_pairs)

        head_input = torch.cat([self_mean, self_max, pair_attn, pair_max, ctx], dim=-1)
        return self.head(head_input).squeeze(-1)


def build_pair_type(
    is_ao_i: bool,
    is_ao_j: bool,
    is_moly_i: bool,
    is_moly_j: bool,
    is_antiwear_i: bool,
    is_antiwear_j: bool,
    is_detergent_i: bool,
    is_detergent_j: bool,
    is_dispersant_i: bool,
    is_dispersant_j: bool,
    zinc_i: float,
    zinc_j: float,
    boron_i: float,
    boron_j: float,
    ca_i: float,
    ca_j: float,
    ca_mg_i: float,
    ca_mg_j: float,
) -> int:
    if is_ao_i and is_ao_j:
        return PAIR_TYPE_AO_AO
    if (is_ao_i and is_moly_j) or (is_ao_j and is_moly_i):
        return PAIR_TYPE_AO_MO
    if (is_antiwear_i and is_detergent_j) or (is_antiwear_j and is_detergent_i):
        return PAIR_TYPE_ZDDP_DETERGENT
    if ((zinc_i > 0 and is_dispersant_j and boron_j > 0) or (zinc_j > 0 and is_dispersant_i and boron_i > 0)):
        return PAIR_TYPE_ZN_DISPERSANT
    if (is_antiwear_i and (ca_j > 0 or ca_mg_j > 0)) or (is_antiwear_j and (ca_i > 0 or ca_mg_i > 0)):
        return PAIR_TYPE_ANTIWEAR_CA_MG
    return PAIR_TYPE_NONE


def build_samples(
    comp_df: pd.DataFrame,
    scenario_df: pd.DataFrame,
    family_to_id: Dict[str, int],
    component_to_id: Dict[str, int],
    max_components: int,
    target_name: str,
    include_target: bool,
) -> List[ScenarioSample]:
    scenario_extra = scenario_df.set_index(SCENARIO_COL)[SCENARIO_ENGINEERED_COLS].apply(pd.to_numeric, errors="coerce")
    pair_i, pair_j = torch.triu_indices(max_components, max_components, offset=1)
    pair_i = pair_i.numpy()
    pair_j = pair_j.numpy()

    samples: List[ScenarioSample] = []
    for scenario_id, group in comp_df.groupby(SCENARIO_COL, sort=False):
        g = group.sort_values([COMPONENT_COL, BATCH_COL], kind="stable").reset_index(drop=True)
        n = min(len(g), max_components)
        g = g.iloc[:n].copy()

        raw_numeric = g[COMPONENT_NUMERIC_COLS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
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
        fam_names = ["unknown"] * max_components
        comp_names = [""] * max_components
        for idx, row in g.iterrows():
            fam = family_of(str(row[COMPONENT_COL]))
            comp_name = str(row[COMPONENT_COL])
            fam_names[idx] = fam
            comp_names[idx] = comp_name
            family_ids[idx] = family_to_id.get(fam, 0)
            component_ids[idx] = component_to_id.get(comp_name, 0)

        zinc = pd.to_numeric(g["Массовая доля цинка, ASTM D6481"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        boron = pd.to_numeric(g["Содержание Бора"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        calcium = pd.to_numeric(g["Массовая доля кальция, ASTM D6481"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        ca_mg = pd.to_numeric(g["Содержание металла (Ca/Mg), % масс."], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)

        allowed = np.zeros(len(pair_i), dtype=bool)
        pair_types = np.zeros(len(pair_i), dtype=np.int64)
        for k, (i, j) in enumerate(zip(pair_i, pair_j)):
            if i >= n or j >= n:
                continue
            fam_i = fam_names[i]
            fam_j = fam_names[j]
            pair_type = build_pair_type(
                fam_i == "Антиоксидант",
                fam_j == "Антиоксидант",
                comp_names[i].startswith("Соединение_молибдена"),
                comp_names[j].startswith("Соединение_молибдена"),
                comp_names[i].startswith("Противоизносная_присадка"),
                comp_names[j].startswith("Противоизносная_присадка"),
                fam_i == "Детергент",
                fam_j == "Детергент",
                fam_i == "Дисперсант",
                fam_j == "Дисперсант",
                float(zinc[i]),
                float(zinc[j]),
                float(boron[i]),
                float(boron[j]),
                float(calcium[i]),
                float(calcium[j]),
                float(ca_mg[i]),
                float(ca_mg[j]),
            )
            pair_types[k] = pair_type
            allowed[k] = pair_type != PAIR_TYPE_NONE

        base_context = build_base_context(g.iloc[0])
        scenario_values = scenario_extra.loc[scenario_id] if scenario_id in scenario_extra.index else pd.Series(index=SCENARIO_ENGINEERED_COLS, dtype=float)
        ctx = {**base_context}
        for col in SCENARIO_ENGINEERED_COLS:
            val = scenario_values.get(col, 0.0)
            ctx[col] = 0.0 if pd.isna(val) else float(val)
        context_arr = np.array([ctx[c] for c in CONTEXT_COLS], dtype=np.float32)

        target = None
        if include_target:
            target_col = TARGET_VISC_COL if target_name == "viscosity" else TARGET_OX_COL
            target = float(g.iloc[0][target_col])

        samples.append(
            ScenarioSample(
                scenario_id=str(scenario_id),
                component_numeric=padded_numeric,
                component_valid=comp_valid,
                family_ids=family_ids,
                component_ids=component_ids,
                context=context_arr,
                pair_allowed=allowed,
                pair_type_ids=pair_types,
                target=target,
            )
        )
    return samples


def standardize_samples(
    train_samples: Sequence[ScenarioSample],
    val_samples: Sequence[ScenarioSample],
    target_name: str,
) -> tuple[list[ScenarioSample], list[ScenarioSample], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    raw_dim = len(COMPONENT_NUMERIC_COLS)
    train_numeric = np.stack([s.component_numeric[:, :raw_dim] for s in train_samples], axis=0)
    train_valid = np.stack([np.repeat(s.component_valid[:, None], raw_dim, axis=1) for s in train_samples], axis=0)
    comp_stats = fit_standardizer(train_numeric, valid_mask=train_valid)
    ctx_stats = fit_standardizer(np.stack([s.context for s in train_samples], axis=0))

    def transform(samples: Sequence[ScenarioSample]) -> list[ScenarioSample]:
        out = []
        for s in samples:
            raw = s.component_numeric[:, :raw_dim]
            flags = s.component_numeric[:, raw_dim:]
            raw_scaled = apply_standardizer(raw[None, ...], comp_stats)[0]
            ctx_scaled = apply_standardizer(s.context[None, :], ctx_stats)[0]
            out.append(
                ScenarioSample(
                    scenario_id=s.scenario_id,
                    component_numeric=np.concatenate([raw_scaled, flags], axis=1),
                    component_valid=s.component_valid,
                    family_ids=s.family_ids,
                    component_ids=s.component_ids,
                    context=ctx_scaled,
                    pair_allowed=s.pair_allowed,
                    pair_type_ids=s.pair_type_ids,
                    target=float(target_transform(np.array([s.target], dtype=np.float32), target_name)[0]) if s.target is not None else None,
                )
            )
        return out

    return transform(train_samples), transform(val_samples), comp_stats, ctx_stats


def train_one_fold(
    train_samples: Sequence[ScenarioSample],
    val_samples: Sequence[ScenarioSample],
    target_name: str,
    config: Dict[str, float | int | str],
    n_families: int,
    n_components: int,
    max_components: int,
    device: torch.device,
) -> Dict[str, object]:
    train_scaled, val_scaled, comp_stats, ctx_stats = standardize_samples(train_samples, val_samples, target_name)
    model = TargetwiseGatedPairwiseRegressor(
        num_component_features=train_scaled[0].component_numeric.shape[1],
        num_context_features=train_scaled[0].context.shape[0],
        n_families=n_families,
        n_components=n_components,
        max_components=max_components,
        d_model=int(config["d_model"]),
        dropout=float(config["dropout"]),
        ctx_dim=int(config["ctx_dim"]),
        pair_hidden=int(config["pair_hidden"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["lr"]),
        weight_decay=float(config["weight_decay"]),
    )
    criterion = nn.MSELoss()

    train_loader = DataLoader(ScenarioDataset(train_scaled), batch_size=min(int(config["batch_size"]), len(train_scaled)), shuffle=True)
    val_loader = DataLoader(ScenarioDataset(val_scaled), batch_size=min(32, len(val_scaled)), shuffle=False)

    best_state = None
    best_loss = float("inf")
    wait = 0
    for _epoch in range(int(config["epochs"])):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            pred = model(
                batch["component_numeric"].to(device),
                batch["component_valid"].to(device),
                batch["family_ids"].to(device),
                batch["component_ids"].to(device),
                batch["context"].to(device),
                batch["pair_allowed"].to(device),
                batch["pair_type_ids"].to(device),
            )
            loss = criterion(pred, batch["target"].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        losses = []
        with torch.no_grad():
            for batch in val_loader:
                pred = model(
                    batch["component_numeric"].to(device),
                    batch["component_valid"].to(device),
                    batch["family_ids"].to(device),
                    batch["component_ids"].to(device),
                    batch["context"].to(device),
                    batch["pair_allowed"].to(device),
                    batch["pair_type_ids"].to(device),
                )
                losses.append(float(criterion(pred, batch["target"].to(device)).item()))
        val_loss = float(np.mean(losses))
        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            wait = 0
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= int(config["patience"]):
                break

    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in val_loader:
            pred = model(
                batch["component_numeric"].to(device),
                batch["component_valid"].to(device),
                batch["family_ids"].to(device),
                batch["component_ids"].to(device),
                batch["context"].to(device),
                batch["pair_allowed"].to(device),
                batch["pair_type_ids"].to(device),
            )
            preds.append(pred.cpu().numpy())
    pred_val = target_inverse(np.concatenate(preds, axis=0), target_name)
    true_val = np.array([s.target for s in val_samples], dtype=np.float32)
    metrics = compute_target_metrics(true_val, pred_val, target_name)
    return {
        "model_state": best_state,
        "comp_stats": comp_stats,
        "ctx_stats": ctx_stats,
        "val_pred": pred_val,
        "metrics": metrics,
    }


def predict_samples(
    samples: Sequence[ScenarioSample],
    target_name: str,
    model_state: Dict[str, torch.Tensor],
    comp_stats: Dict[str, np.ndarray],
    ctx_stats: Dict[str, np.ndarray],
    config: Dict[str, float | int | str],
    n_families: int,
    n_components: int,
    max_components: int,
    device: torch.device,
) -> np.ndarray:
    raw_dim = len(COMPONENT_NUMERIC_COLS)
    transformed = []
    for s in samples:
        raw = s.component_numeric[:, :raw_dim]
        flags = s.component_numeric[:, raw_dim:]
        raw_scaled = apply_standardizer(raw[None, ...], comp_stats)[0]
        ctx_scaled = apply_standardizer(s.context[None, :], ctx_stats)[0]
        transformed.append(
            ScenarioSample(
                scenario_id=s.scenario_id,
                component_numeric=np.concatenate([raw_scaled, flags], axis=1),
                component_valid=s.component_valid,
                family_ids=s.family_ids,
                component_ids=s.component_ids,
                context=ctx_scaled,
                pair_allowed=s.pair_allowed,
                pair_type_ids=s.pair_type_ids,
                target=s.target,
            )
        )
    model = TargetwiseGatedPairwiseRegressor(
        num_component_features=transformed[0].component_numeric.shape[1],
        num_context_features=transformed[0].context.shape[0],
        n_families=n_families,
        n_components=n_components,
        max_components=max_components,
        d_model=int(config["d_model"]),
        dropout=float(config["dropout"]),
        ctx_dim=int(config["ctx_dim"]),
        pair_hidden=int(config["pair_hidden"]),
    ).to(device)
    model.load_state_dict(model_state)
    model.eval()
    loader = DataLoader(ScenarioDataset(transformed), batch_size=min(32, len(transformed)), shuffle=False)
    preds = []
    with torch.no_grad():
        for batch in loader:
            pred = model(
                batch["component_numeric"].to(device),
                batch["component_valid"].to(device),
                batch["family_ids"].to(device),
                batch["component_ids"].to(device),
                batch["context"].to(device),
                batch["pair_allowed"].to(device),
                batch["pair_type_ids"].to(device),
            )
            preds.append(pred.cpu().numpy())
    return target_inverse(np.concatenate(preds, axis=0), target_name)


def train_full_model(
    samples: Sequence[ScenarioSample],
    target_name: str,
    config: Dict[str, float | int | str],
    n_families: int,
    n_components: int,
    max_components: int,
    device: torch.device,
) -> tuple[Dict[str, torch.Tensor], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    indices = np.arange(len(samples))
    rng = np.random.default_rng(SEED)
    rng.shuffle(indices)
    split = max(1, int(len(indices) * 0.15))
    val_idx = indices[:split]
    train_idx = indices[split:]
    train_samples = [samples[i] for i in train_idx]
    val_samples = [samples[i] for i in val_idx]
    result = train_one_fold(train_samples, val_samples, target_name, config, n_families, n_components, max_components, device)
    return result["model_state"], result["comp_stats"], result["ctx_stats"]


def train_target(
    target_name: str,
    train_samples: Sequence[ScenarioSample],
    test_samples: Sequence[ScenarioSample],
    outdir: Path,
    n_families: int,
    n_components: int,
    max_components: int,
    device: torch.device,
) -> dict:
    y_true = np.array([s.target for s in train_samples], dtype=np.float32)
    scenario_ids = np.array([s.scenario_id for s in train_samples])
    configs = [
        {
            "name": f"{target_name}_gated_small",
            "d_model": 48,
            "ctx_dim": 24,
            "pair_hidden": 72,
            "dropout": 0.15,
            "lr": 1e-3,
            "weight_decay": 2e-4,
            "batch_size": 16,
            "epochs": 140,
            "patience": 20,
        },
        {
            "name": f"{target_name}_gated_medium",
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
    splitter = KFold(n_splits=5, shuffle=True, random_state=SEED)

    summary_rows = []
    best_config = None
    best_rmse = float("inf")
    best_oof = None
    for config in configs:
        print(f"\\n=== {target_name} config={config['name']} ===")
        oof = np.zeros_like(y_true, dtype=np.float32)
        fold_rows = []
        for fold, (tr_idx, va_idx) in enumerate(splitter.split(scenario_ids), start=1):
            tr = [train_samples[i] for i in tr_idx]
            va = [train_samples[i] for i in va_idx]
            result = train_one_fold(tr, va, target_name, config, n_families, n_components, max_components, device)
            oof[va_idx] = result["val_pred"]
            row = {"config": config["name"], "fold": fold, **result["metrics"]}
            fold_rows.append(row)
            print(f"fold={fold} rmse={row['rmse']:.2f} mae={row['mae']:.2f} r2={row['r2']:.3f}")
        metrics = compute_target_metrics(y_true, oof, target_name)
        summary_rows.append({"config": config["name"], **metrics})
        pd.DataFrame(fold_rows).to_csv(outdir / f"{config['name']}_fold_metrics.csv", index=False)
        pd.DataFrame({SCENARIO_COL: scenario_ids, f"{target_name}_true": y_true, f"{target_name}_pred": oof}).to_csv(
            outdir / f"{config['name']}_oof_predictions.csv", index=False
        )
        if metrics["rmse"] < best_rmse:
            best_rmse = metrics["rmse"]
            best_config = config
            best_oof = oof.copy()
        print(f"OOF {target_name} config={config['name']} rmse={metrics['rmse']:.2f} mae={metrics['mae']:.2f}")

    assert best_config is not None and best_oof is not None
    summary_df = pd.DataFrame(summary_rows).sort_values("rmse").reset_index(drop=True)
    summary_df.to_csv(outdir / f"{target_name}_cv_summary.csv", index=False)

    model_state, comp_stats, ctx_stats = train_full_model(
        train_samples, target_name, best_config, n_families, n_components, max_components, device
    )
    test_pred = predict_samples(
        test_samples, target_name, model_state, comp_stats, ctx_stats, best_config, n_families, n_components, max_components, device
    )
    torch.save(
        {
            "target_name": target_name,
            "config": best_config,
            "model_state": model_state,
            "component_stats": comp_stats,
            "context_stats": ctx_stats,
            "component_numeric_cols": COMPONENT_NUMERIC_COLS,
            "context_cols": CONTEXT_COLS,
            "pair_type_names": PAIR_TYPE_NAMES,
            "max_components": max_components,
        },
        outdir / f"{target_name}_best_model.pt",
    )
    return {
        "target_name": target_name,
        "best_config": best_config,
        "summary": summary_df.to_dict(orient="records"),
        "oof": best_oof,
        "test_pred": test_pred,
    }


def pair_coverage_summary(samples: Sequence[ScenarioSample]) -> dict:
    counts = {name: 0 for name in PAIR_TYPE_NAMES.values() if name != "none"}
    per_scenario = []
    for s in samples:
        active = s.pair_type_ids[s.pair_allowed]
        per_scenario.append(int(active.size))
        for pair_type in active:
            if int(pair_type) != PAIR_TYPE_NONE:
                counts[PAIR_TYPE_NAMES[int(pair_type)]] += 1
    return {
        "mean_allowed_pairs_per_scenario": float(np.mean(per_scenario)) if per_scenario else 0.0,
        "median_allowed_pairs_per_scenario": float(np.median(per_scenario)) if per_scenario else 0.0,
        "pair_type_counts": counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train target-wise gated pairwise models on chemically selected pairs.")
    parser.add_argument("--train-components", type=Path, default=DEFAULT_TRAIN_COMPONENTS)
    parser.add_argument("--test-components", type=Path, default=DEFAULT_TEST_COMPONENTS)
    parser.add_argument("--train-scenario", type=Path, default=DEFAULT_TRAIN_SCENARIO)
    parser.add_argument("--test-scenario", type=Path, default=DEFAULT_TEST_SCENARIO)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args()

    set_global_seed(SEED)
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    train_comp = pd.read_csv(args.train_components)
    test_comp = pd.read_csv(args.test_components)
    train_scen = pd.read_csv(args.train_scenario)
    test_scen = pd.read_csv(args.test_scenario)

    family_vocab = sorted({family_of(v) for v in pd.concat([train_comp[COMPONENT_COL], test_comp[COMPONENT_COL]], axis=0).astype(str)})
    component_vocab = sorted(set(pd.concat([train_comp[COMPONENT_COL], test_comp[COMPONENT_COL]], axis=0).astype(str)))
    family_to_id = {name: idx + 1 for idx, name in enumerate(family_vocab)}
    component_to_id = {name: idx + 1 for idx, name in enumerate(component_vocab)}
    max_components = int(max(train_comp.groupby(SCENARIO_COL).size().max(), test_comp.groupby(SCENARIO_COL).size().max()))

    print(f"train scenarios={train_comp[SCENARIO_COL].nunique()} test scenarios={test_comp[SCENARIO_COL].nunique()}")
    print(f"max_components={max_components} families={len(family_vocab)} components={len(component_vocab)}")

    train_samples_visc = build_samples(train_comp, train_scen, family_to_id, component_to_id, max_components, "viscosity", True)
    train_samples_ox = build_samples(train_comp, train_scen, family_to_id, component_to_id, max_components, "oxidation", True)
    test_samples_visc = build_samples(test_comp, test_scen, family_to_id, component_to_id, max_components, "viscosity", False)
    test_samples_ox = build_samples(test_comp, test_scen, family_to_id, component_to_id, max_components, "oxidation", False)

    coverage = pair_coverage_summary(train_samples_visc)
    print("pair coverage:", json.dumps(coverage, ensure_ascii=False))

    device = torch.device("cpu")
    visc_result = train_target(
        "viscosity", train_samples_visc, test_samples_visc, outdir, len(family_to_id) + 1, len(component_to_id) + 1, max_components, device
    )
    ox_result = train_target(
        "oxidation", train_samples_ox, test_samples_ox, outdir, len(family_to_id) + 1, len(component_to_id) + 1, max_components, device
    )

    prediction = pd.DataFrame(
        {
            SCENARIO_COL: [s.scenario_id for s in test_samples_visc],
            TARGET_VISC_COL: visc_result["test_pred"],
            TARGET_OX_COL: ox_result["test_pred"],
        }
    )
    prediction.to_csv(outdir / "prediction.csv", index=False)
    pd.DataFrame(
        {
            SCENARIO_COL: [s.scenario_id for s in train_samples_visc],
            "viscosity_true": [s.target for s in train_samples_visc],
            "viscosity_pred": visc_result["oof"],
            "oxidation_true": [s.target for s in train_samples_ox],
            "oxidation_pred": ox_result["oof"],
        }
    ).to_csv(outdir / "targetwise_best_oof_predictions.csv", index=False)

    manifest = {
        "train_components": str(args.train_components),
        "test_components": str(args.test_components),
        "train_scenario": str(args.train_scenario),
        "test_scenario": str(args.test_scenario),
        "component_numeric_cols": COMPONENT_NUMERIC_COLS,
        "context_cols": CONTEXT_COLS,
        "pair_type_names": PAIR_TYPE_NAMES,
        "pair_coverage": coverage,
        "viscosity": {"best_config": visc_result["best_config"], "summary": visc_result["summary"]},
        "oxidation": {"best_config": ox_result["best_config"], "summary": ox_result["summary"]},
    }
    (outdir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\\nSaved outputs to {outdir}")


if __name__ == "__main__":
    main()
