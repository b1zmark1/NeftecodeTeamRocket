#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import math
import random
import sys
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pairwise_interaction.train_targetwise_gated_pairwise_model import (
    BASE_CONTEXT_COLS,
    BATCH_COL,
    BIOFUEL_COL,
    CATALYST_COL,
    COMPONENT_COL,
    COMPONENT_NUMERIC_COLS,
    DEFAULT_TEST_COMPONENTS,
    DEFAULT_TEST_SCENARIO,
    DEFAULT_TRAIN_COMPONENTS,
    DEFAULT_TRAIN_SCENARIO,
    DOSE_COL,
    PAIR_TYPE_AO_AO,
    PAIR_TYPE_AO_MO,
    PAIR_TYPE_ANTIWEAR_CA_MG,
    PAIR_TYPE_NAMES,
    PAIR_TYPE_ZDDP_DETERGENT,
    PAIR_TYPE_ZN_DISPERSANT,
    SCENARIO_COL,
    SCENARIO_ENGINEERED_COLS,
    TARGET_OX_COL,
    TARGET_VISC_COL,
    TEMP_COL,
    TIME_COL,
    TargetwiseGatedPairwiseRegressor,
    apply_standardizer,
    build_pair_type,
    family_of,
    fit_standardizer,
)

BASELINE_OOF = ROOT / "compact_v3" / "mlp_targetwise_output" / "targetwise_mlp_oof_predictions.csv"
BASELINE_TEST_PRED = ROOT / "compact_v3" / "hybrid_predictions" / "prediction.csv"
OUTDIR = ROOT / "pairwise_interaction" / "residual_v2_output"

SEED = 42
RESIDUAL_SCALE = {"viscosity": 50.0, "oxidation": 20.0}

V2_EXTRA_CONTEXT_COLS = [
    "calcium_total",
    "ca_mg_total",
    "boron_total",
    "antiwear_zinc_total",
    "antiwear_sulfur_total",
    "antiwear_phosphorus_total",
    "ao_dpa_active_total",
    "ao_phenol_active_total",
    "ao_active_total",
    "ao_dpa_phenol_product",
    "ao_dpa_phenol_ratio",
    "ao_dpa_phenol_min",
    "ao_dpa_phenol_max",
    "ao_dpa_phenol_imbalance",
    "ao_dpa_phenol_hmean",
    "ao_mo_product",
    "ao_mo_log1p",
    "ao_mo_x_severity",
    "ao_pair_x_severity",
    "zddp_boron_interaction",
    "zddp_boron_log1p",
    "zddp_calcium_interaction",
    "zddp_calcium_x_severity",
    "boron_total_x_severity",
    "dominant_base_oil_share",
    "base_oil_kv100_wmean",
    "base_oil_group1_share",
    "ao_ratio_x_baseoil_share",
    "ao_min_x_baseoil_kv100",
    "antiwear_ca_ratio",
]

CONTEXT_COLS_V2 = BASE_CONTEXT_COLS + SCENARIO_ENGINEERED_COLS + V2_EXTRA_CONTEXT_COLS


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(math.sqrt(mean_squared_error(y_true, y_pred)))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, target_name: str) -> Dict[str, float]:
    return {
        "target_name": target_name,
        "rmse": rmse(y_true, y_pred),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def residual_transform(y: np.ndarray, target_name: str) -> np.ndarray:
    scale = RESIDUAL_SCALE[target_name]
    return np.arcsinh(y.astype(np.float32) / scale)


def residual_inverse(y_t: np.ndarray, target_name: str) -> np.ndarray:
    scale = RESIDUAL_SCALE[target_name]
    return scale * np.sinh(np.clip(y_t.astype(np.float32), -6.0, 6.0))


@dataclass
class ResidualSample:
    scenario_id: str
    component_numeric: np.ndarray
    component_valid: np.ndarray
    family_ids: np.ndarray
    component_ids: np.ndarray
    context: np.ndarray
    pair_allowed: np.ndarray
    pair_type_ids: np.ndarray
    target: float | None = None


class ResidualDataset(Dataset):
    def __init__(self, samples: Sequence[ResidualSample]) -> None:
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


def load_baseline_predictions() -> tuple[dict, dict, dict, dict]:
    oof = pd.read_csv(BASELINE_OOF)
    test_pred = pd.read_csv(BASELINE_TEST_PRED)

    best_cfg = {"viscosity": "relu_wide", "oxidation": "relu_mid"}
    oof_maps = {}
    true_maps = {}
    for target_name, cfg in best_cfg.items():
        subset = oof[(oof["target_name"] == target_name) & (oof["config_id"] == cfg)].copy()
        subset = subset.sort_values(SCENARIO_COL).drop_duplicates(SCENARIO_COL, keep="last")
        oof_maps[target_name] = dict(zip(subset[SCENARIO_COL], subset["y_pred"]))
        true_maps[target_name] = dict(zip(subset[SCENARIO_COL], subset["y_true"]))

    test_maps = {
        "viscosity": dict(zip(test_pred[SCENARIO_COL], test_pred[TARGET_VISC_COL])),
        "oxidation": dict(zip(test_pred[SCENARIO_COL], test_pred[TARGET_OX_COL])),
    }
    return oof_maps["viscosity"], oof_maps["oxidation"], test_maps["viscosity"], test_maps["oxidation"]


def weighted_mean(frame: pd.DataFrame, value_col: str, mask: pd.Series) -> float:
    sub = frame.loc[mask, [DOSE_COL, value_col]].copy()
    sub[DOSE_COL] = pd.to_numeric(sub[DOSE_COL], errors="coerce")
    sub[value_col] = pd.to_numeric(sub[value_col], errors="coerce")
    sub = sub.dropna()
    if sub.empty:
        return 0.0
    weights = sub[DOSE_COL].to_numpy(dtype=float)
    vals = sub[value_col].to_numpy(dtype=float)
    w_sum = weights.sum()
    if w_sum <= 0:
        return 0.0
    return float(np.dot(weights, vals) / w_sum)


def build_context_v2(group: pd.DataFrame, scenario_row: pd.Series) -> Dict[str, float]:
    temperature = float(group.iloc[0][TEMP_COL])
    time_h = float(group.iloc[0][TIME_COL])
    biofuel = float(group.iloc[0][BIOFUEL_COL])
    catalyst = float(group.iloc[0][CATALYST_COL])
    severity = time_h * math.exp((temperature - 150.0) / 10.0)
    ctx = {
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
    for col in SCENARIO_ENGINEERED_COLS:
        val = scenario_row.get(col, 0.0)
        ctx[col] = 0.0 if pd.isna(val) else float(val)

    comp = group.copy()
    comp[DOSE_COL] = pd.to_numeric(comp[DOSE_COL], errors="coerce").fillna(0.0)
    comp["active_no"] = pd.to_numeric(comp["Активный Азот / Кислород, % масс. (N или O)"], errors="coerce").fillna(0.0)
    comp["mo_pct"] = pd.to_numeric(comp["% масс. (Mo)"], errors="coerce").fillna(0.0)
    comp["zinc_pct"] = pd.to_numeric(comp["Массовая доля цинка, ASTM D6481"], errors="coerce").fillna(0.0)
    comp["sulfur_pct"] = pd.to_numeric(comp["Массовая доля серы, ASTM D6481"], errors="coerce").fillna(0.0)
    comp["phosphorus_pct"] = pd.to_numeric(comp["Массовая доля фосфора, ASTM D6481"], errors="coerce").fillna(0.0)
    comp["calcium_pct"] = pd.to_numeric(comp["Массовая доля кальция, ASTM D6481"], errors="coerce").fillna(0.0)
    comp["camg_pct"] = pd.to_numeric(comp["Содержание металла (Ca/Mg), % масс."], errors="coerce").fillna(0.0)
    comp["boron_pct"] = pd.to_numeric(comp["Содержание Бора"], errors="coerce").fillna(0.0)
    comp["kv100"] = pd.to_numeric(comp["Кинематическая вязкость, при 100°C, ASTM D445"], errors="coerce")
    comp["api_group"] = pd.to_numeric(comp["Группа по API"], errors="coerce")
    comp["Тип АО"] = comp["Тип АО"].fillna("").astype(str)

    is_dpa = comp["Тип АО"].eq("Дифениламин")
    is_phenol = comp["Тип АО"].eq("Фенол")
    is_moly = comp[COMPONENT_COL].astype(str).str.startswith("Соединение_молибдена")
    is_antiwear = comp[COMPONENT_COL].astype(str).str.startswith("Противоизносная_присадка")
    is_baseoil = comp[COMPONENT_COL].astype(str).str.startswith("Базовое_масло")

    def dose_sum(mask: pd.Series, col: str) -> float:
        sub = comp.loc[mask, [DOSE_COL, col]].copy()
        if sub.empty:
            return 0.0
        return float((sub[DOSE_COL] * sub[col]).sum())

    ao_dpa_active_total = dose_sum(is_dpa, "active_no")
    ao_phenol_active_total = dose_sum(is_phenol, "active_no")
    ao_active_total = ao_dpa_active_total + ao_phenol_active_total
    ao_prod = ao_dpa_active_total * ao_phenol_active_total
    ao_ratio = ao_dpa_active_total / (ao_phenol_active_total + 1e-6)
    ao_min = min(ao_dpa_active_total, ao_phenol_active_total)
    ao_max = max(ao_dpa_active_total, ao_phenol_active_total)
    ao_imb = abs(ao_dpa_active_total - ao_phenol_active_total)
    ao_hmean = 0.0 if ao_dpa_active_total <= 0 or ao_phenol_active_total <= 0 else 2.0 * ao_dpa_active_total * ao_phenol_active_total / (ao_dpa_active_total + ao_phenol_active_total)

    moly_total = dose_sum(is_moly, "mo_pct")
    antiwear_zinc_total = dose_sum(is_antiwear, "zinc_pct")
    antiwear_sulfur_total = dose_sum(is_antiwear, "sulfur_pct")
    antiwear_phosphorus_total = dose_sum(is_antiwear, "phosphorus_pct")
    calcium_total = dose_sum(comp.index == comp.index, "calcium_pct")
    ca_mg_total = dose_sum(comp.index == comp.index, "camg_pct")
    boron_total = dose_sum(comp.index == comp.index, "boron_pct")

    baseoil_share_total = float(comp.loc[is_baseoil, DOSE_COL].sum())
    dominant_base_oil_share = 0.0
    if baseoil_share_total > 0:
        dominant_base_oil_share = float(comp.loc[is_baseoil, DOSE_COL].max() / baseoil_share_total)
    base_oil_kv100_wmean = weighted_mean(comp, "kv100", is_baseoil)
    api_group1_share = 0.0
    if baseoil_share_total > 0:
        api_group1_share = float(comp.loc[is_baseoil & comp["api_group"].eq(1.0), DOSE_COL].sum() / baseoil_share_total)

    ao_mo_product = ao_dpa_active_total * moly_total
    zddp_boron = antiwear_zinc_total * boron_total
    zddp_calcium = antiwear_sulfur_total * (calcium_total + ca_mg_total)

    ctx.update(
        {
            "calcium_total": calcium_total,
            "ca_mg_total": ca_mg_total,
            "boron_total": boron_total,
            "antiwear_zinc_total": antiwear_zinc_total,
            "antiwear_sulfur_total": antiwear_sulfur_total,
            "antiwear_phosphorus_total": antiwear_phosphorus_total,
            "ao_dpa_active_total": ao_dpa_active_total,
            "ao_phenol_active_total": ao_phenol_active_total,
            "ao_active_total": ao_active_total,
            "ao_dpa_phenol_product": ao_prod,
            "ao_dpa_phenol_ratio": ao_ratio,
            "ao_dpa_phenol_min": ao_min,
            "ao_dpa_phenol_max": ao_max,
            "ao_dpa_phenol_imbalance": ao_imb,
            "ao_dpa_phenol_hmean": ao_hmean,
            "ao_mo_product": ao_mo_product,
            "ao_mo_log1p": math.log1p(max(ao_mo_product, 0.0)),
            "ao_mo_x_severity": ao_mo_product * severity,
            "ao_pair_x_severity": ao_prod * severity,
            "zddp_boron_interaction": zddp_boron,
            "zddp_boron_log1p": math.log1p(max(zddp_boron, 0.0)),
            "zddp_calcium_interaction": zddp_calcium,
            "zddp_calcium_x_severity": zddp_calcium * severity,
            "boron_total_x_severity": boron_total * severity,
            "dominant_base_oil_share": dominant_base_oil_share,
            "base_oil_kv100_wmean": base_oil_kv100_wmean if pd.notna(base_oil_kv100_wmean) else 0.0,
            "base_oil_group1_share": api_group1_share,
            "ao_ratio_x_baseoil_share": ao_ratio * dominant_base_oil_share,
            "ao_min_x_baseoil_kv100": ao_min * (0.0 if pd.isna(base_oil_kv100_wmean) else base_oil_kv100_wmean),
            "antiwear_ca_ratio": (calcium_total + ca_mg_total) / (antiwear_sulfur_total + 1e-6),
        }
    )
    return ctx


def build_samples_v2(
    comp_df: pd.DataFrame,
    scenario_df: pd.DataFrame,
    family_to_id: Dict[str, int],
    component_to_id: Dict[str, int],
    max_components: int,
    target_name: str,
    baseline_map: Dict[str, float],
    include_target: bool,
) -> List[ResidualSample]:
    scen_lookup = scenario_df.set_index(SCENARIO_COL)
    pair_i, pair_j = torch.triu_indices(max_components, max_components, offset=1)
    pair_i = pair_i.numpy()
    pair_j = pair_j.numpy()
    out: List[ResidualSample] = []
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

        padded_numeric = np.zeros((max_components, component_numeric.shape[1]), dtype=np.float32)
        padded_numeric[:n] = component_numeric
        comp_valid = np.zeros(max_components, dtype=bool)
        comp_valid[:n] = True

        family_ids = np.zeros(max_components, dtype=np.int64)
        component_ids = np.zeros(max_components, dtype=np.int64)
        fam_names = ["unknown"] * max_components
        comp_names = [""] * max_components
        for idx, row in g.iterrows():
            fam = family_of(str(row[COMPONENT_COL]))
            name = str(row[COMPONENT_COL])
            fam_names[idx] = fam
            comp_names[idx] = name
            family_ids[idx] = family_to_id.get(fam, 0)
            component_ids[idx] = component_to_id.get(name, 0)

        zinc = pd.to_numeric(g["Массовая доля цинка, ASTM D6481"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        boron = pd.to_numeric(g["Содержание Бора"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        calcium = pd.to_numeric(g["Массовая доля кальция, ASTM D6481"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        ca_mg = pd.to_numeric(g["Содержание металла (Ca/Mg), % масс."], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)

        allowed = np.zeros(len(pair_i), dtype=bool)
        pair_types = np.zeros(len(pair_i), dtype=np.int64)
        for k, (i, j) in enumerate(zip(pair_i, pair_j)):
            if i >= n or j >= n:
                continue
            pair_type = build_pair_type(
                fam_names[i] == "Антиоксидант",
                fam_names[j] == "Антиоксидант",
                comp_names[i].startswith("Соединение_молибдена"),
                comp_names[j].startswith("Соединение_молибдена"),
                comp_names[i].startswith("Противоизносная_присадка"),
                comp_names[j].startswith("Противоизносная_присадка"),
                fam_names[i] == "Детергент",
                fam_names[j] == "Детергент",
                fam_names[i] == "Дисперсант",
                fam_names[j] == "Дисперсант",
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
            allowed[k] = pair_type != 0

        scenario_row = scen_lookup.loc[scenario_id]
        ctx = build_context_v2(g, scenario_row)
        context_arr = np.array([ctx[c] for c in CONTEXT_COLS_V2], dtype=np.float32)

        target = None
        if include_target:
            true_col = TARGET_VISC_COL if target_name == "viscosity" else TARGET_OX_COL
            target = float(g.iloc[0][true_col]) - float(baseline_map[str(scenario_id)])

        out.append(
            ResidualSample(
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
    return out


def standardize_samples(train_samples: Sequence[ResidualSample], val_samples: Sequence[ResidualSample], target_name: str):
    raw_dim = len(COMPONENT_NUMERIC_COLS)
    train_num = np.stack([s.component_numeric[:, :raw_dim] for s in train_samples], axis=0)
    train_valid = np.stack([np.repeat(s.component_valid[:, None], raw_dim, axis=1) for s in train_samples], axis=0)
    comp_stats = fit_standardizer(train_num, valid_mask=train_valid)
    ctx_stats = fit_standardizer(np.stack([s.context for s in train_samples], axis=0))

    def transform(samples: Sequence[ResidualSample]) -> List[ResidualSample]:
        res: List[ResidualSample] = []
        for s in samples:
            raw = s.component_numeric[:, :raw_dim]
            flags = s.component_numeric[:, raw_dim:]
            raw_scaled = apply_standardizer(raw[None, ...], comp_stats)[0]
            ctx_scaled = apply_standardizer(s.context[None, :], ctx_stats)[0]
            res.append(
                ResidualSample(
                    scenario_id=s.scenario_id,
                    component_numeric=np.concatenate([raw_scaled, flags], axis=1),
                    component_valid=s.component_valid,
                    family_ids=s.family_ids,
                    component_ids=s.component_ids,
                    context=ctx_scaled,
                    pair_allowed=s.pair_allowed,
                    pair_type_ids=s.pair_type_ids,
                    target=float(residual_transform(np.array([s.target]), target_name)[0]) if s.target is not None else None,
                )
            )
        return res

    return transform(train_samples), transform(val_samples), comp_stats, ctx_stats


def train_fold(train_samples: Sequence[ResidualSample], val_samples: Sequence[ResidualSample], target_name: str, config: dict, n_families: int, n_components: int, max_components: int):
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
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["lr"]), weight_decay=float(config["weight_decay"]))
    criterion = nn.MSELoss()
    train_loader = DataLoader(ResidualDataset(train_scaled), batch_size=min(int(config["batch_size"]), len(train_scaled)), shuffle=True)
    val_loader = DataLoader(ResidualDataset(val_scaled), batch_size=min(32, len(val_scaled)), shuffle=False)
    best_state = None
    best_loss = float("inf")
    wait = 0
    for _ in range(int(config["epochs"])):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            pred = model(batch["component_numeric"], batch["component_valid"], batch["family_ids"], batch["component_ids"], batch["context"], batch["pair_allowed"], batch["pair_type_ids"])
            loss = criterion(pred, batch["target"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                pred = model(batch["component_numeric"], batch["component_valid"], batch["family_ids"], batch["component_ids"], batch["context"], batch["pair_allowed"], batch["pair_type_ids"])
                val_losses.append(float(criterion(pred, batch["target"]).item()))
        v = float(np.mean(val_losses))
        if v < best_loss - 1e-5:
            best_loss = v
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= int(config["patience"]):
                break
    model.load_state_dict(best_state)
    preds = []
    with torch.no_grad():
        for batch in val_loader:
            pred = model(batch["component_numeric"], batch["component_valid"], batch["family_ids"], batch["component_ids"], batch["context"], batch["pair_allowed"], batch["pair_type_ids"])
            preds.append(pred.numpy())
    residual_pred = residual_inverse(np.concatenate(preds, axis=0), target_name)
    return {"model_state": best_state, "comp_stats": comp_stats, "ctx_stats": ctx_stats, "residual_pred": residual_pred}


def predict_residual(samples: Sequence[ResidualSample], target_name: str, state: dict, comp_stats: dict, ctx_stats: dict, config: dict, n_families: int, n_components: int, max_components: int):
    raw_dim = len(COMPONENT_NUMERIC_COLS)
    transformed = []
    for s in samples:
        raw = s.component_numeric[:, :raw_dim]
        flags = s.component_numeric[:, raw_dim:]
        raw_scaled = apply_standardizer(raw[None, ...], comp_stats)[0]
        ctx_scaled = apply_standardizer(s.context[None, :], ctx_stats)[0]
        transformed.append(ResidualSample(s.scenario_id, np.concatenate([raw_scaled, flags], axis=1), s.component_valid, s.family_ids, s.component_ids, ctx_scaled, s.pair_allowed, s.pair_type_ids, s.target))
    model = TargetwiseGatedPairwiseRegressor(
        num_component_features=transformed[0].component_numeric.shape[1],
        num_context_features=transformed[0].context.shape[0],
        n_families=n_families, n_components=n_components, max_components=max_components,
        d_model=int(config["d_model"]), dropout=float(config["dropout"]), ctx_dim=int(config["ctx_dim"]), pair_hidden=int(config["pair_hidden"]),
    )
    model.load_state_dict(state)
    model.eval()
    loader = DataLoader(ResidualDataset(transformed), batch_size=min(32, len(transformed)), shuffle=False)
    preds = []
    with torch.no_grad():
        for batch in loader:
            pred = model(batch["component_numeric"], batch["component_valid"], batch["family_ids"], batch["component_ids"], batch["context"], batch["pair_allowed"], batch["pair_type_ids"])
            preds.append(pred.numpy())
    return residual_inverse(np.concatenate(preds, axis=0), target_name), transformed, model


def signed_pair_effect(model: TargetwiseGatedPairwiseRegressor, transformed_samples: Sequence[ResidualSample], target_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for s in transformed_samples:
        full_loader = DataLoader(ResidualDataset([s]), batch_size=1, shuffle=False)
        with torch.no_grad():
            for batch in full_loader:
                full = model(batch["component_numeric"], batch["component_valid"], batch["family_ids"], batch["component_ids"], batch["context"], batch["pair_allowed"], batch["pair_type_ids"]).numpy()
        full_pred = float(residual_inverse(full, target_name)[0])
        for pair_type_id, pair_name in PAIR_TYPE_NAMES.items():
            if pair_type_id == 0:
                continue
            modified = ResidualSample(
                s.scenario_id,
                s.component_numeric,
                s.component_valid,
                s.family_ids,
                s.component_ids,
                s.context,
                s.pair_allowed & (s.pair_type_ids != pair_type_id),
                s.pair_type_ids,
                s.target,
            )
            with torch.no_grad():
                for batch in DataLoader(ResidualDataset([modified]), batch_size=1, shuffle=False):
                    ablated = model(batch["component_numeric"], batch["component_valid"], batch["family_ids"], batch["component_ids"], batch["context"], batch["pair_allowed"], batch["pair_type_ids"]).numpy()
            ablated_pred = float(residual_inverse(ablated, target_name)[0])
            rows.append(
                {
                    SCENARIO_COL: s.scenario_id,
                    "target_name": target_name,
                    "pair_type": pair_name,
                    "full_residual_pred": full_pred,
                    "ablated_residual_pred": ablated_pred,
                    "signed_effect": full_pred - ablated_pred,
                    "pair_count": int((s.pair_allowed & (s.pair_type_ids == pair_type_id)).sum()),
                }
            )
    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby("pair_type")
        .agg(
            mean_signed_effect=("signed_effect", "mean"),
            median_signed_effect=("signed_effect", "median"),
            mean_abs_signed_effect=("signed_effect", lambda s: float(np.mean(np.abs(s)))),
            positive_share=("signed_effect", lambda s: float(np.mean(np.array(s) > 0))),
            nonzero_scenarios=("pair_count", lambda s: int(np.sum(np.array(s) > 0))),
        )
        .reset_index()
        .sort_values("mean_abs_signed_effect", ascending=False)
    )
    return detail, summary


def main() -> None:
    set_global_seed(SEED)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    train_comp = pd.read_csv(DEFAULT_TRAIN_COMPONENTS)
    test_comp = pd.read_csv(DEFAULT_TEST_COMPONENTS)
    train_scen = pd.read_csv(DEFAULT_TRAIN_SCENARIO)
    test_scen = pd.read_csv(DEFAULT_TEST_SCENARIO)
    base_oof_visc, base_oof_ox, base_test_visc, base_test_ox = load_baseline_predictions()

    family_vocab = sorted({family_of(v) for v in pd.concat([train_comp[COMPONENT_COL], test_comp[COMPONENT_COL]], axis=0).astype(str)})
    component_vocab = sorted(set(pd.concat([train_comp[COMPONENT_COL], test_comp[COMPONENT_COL]], axis=0).astype(str)))
    family_to_id = {name: idx + 1 for idx, name in enumerate(family_vocab)}
    component_to_id = {name: idx + 1 for idx, name in enumerate(component_vocab)}
    max_components = int(max(train_comp.groupby(SCENARIO_COL).size().max(), test_comp.groupby(SCENARIO_COL).size().max()))

    configs = [
        {"name": "resid_small", "d_model": 48, "ctx_dim": 24, "pair_hidden": 72, "dropout": 0.15, "lr": 1e-3, "weight_decay": 2e-4, "batch_size": 16, "epochs": 140, "patience": 20},
        {"name": "resid_medium", "d_model": 64, "ctx_dim": 32, "pair_hidden": 96, "dropout": 0.20, "lr": 8e-4, "weight_decay": 3e-4, "batch_size": 16, "epochs": 160, "patience": 24},
    ]

    all_results = []
    final_test = {SCENARIO_COL: None}
    comparison_rows = []

    for target_name, baseline_oof, baseline_test in [("viscosity", base_oof_visc, base_test_visc), ("oxidation", base_oof_ox, base_test_ox)]:
        train_samples = build_samples_v2(train_comp, train_scen, family_to_id, component_to_id, max_components, target_name, baseline_oof, True)
        test_samples = build_samples_v2(test_comp, test_scen, family_to_id, component_to_id, max_components, target_name, baseline_test, False)
        true_target = np.array([float(train_comp[train_comp[SCENARIO_COL] == s.scenario_id].iloc[0][TARGET_VISC_COL if target_name == "viscosity" else TARGET_OX_COL]) for s in train_samples], dtype=np.float32)
        baseline_train = np.array([baseline_oof[s.scenario_id] for s in train_samples], dtype=np.float32)
        scenario_ids = np.array([s.scenario_id for s in train_samples])
        baseline_metrics = compute_metrics(true_target, baseline_train, target_name)

        best_cfg = None
        best_final_rmse = float("inf")
        best_oof_resid = None
        best_states = None
        splitter = KFold(n_splits=5, shuffle=True, random_state=SEED)
        for cfg in configs:
            oof_resid = np.zeros_like(true_target)
            for tr_idx, va_idx in splitter.split(scenario_ids):
                tr = [train_samples[i] for i in tr_idx]
                va = [train_samples[i] for i in va_idx]
                res = train_fold(tr, va, target_name, cfg, len(family_to_id) + 1, len(component_to_id) + 1, max_components)
                oof_resid[va_idx] = res["residual_pred"]
            final_oof = baseline_train + oof_resid
            m = compute_metrics(true_target, final_oof, target_name)
            all_results.append({"target_name": target_name, "config": cfg["name"], **m})
            if m["rmse"] < best_final_rmse:
                best_final_rmse = m["rmse"]
                best_cfg = cfg
                best_oof_resid = oof_resid.copy()

        assert best_cfg is not None and best_oof_resid is not None
        final_oof = baseline_train + best_oof_resid
        final_metrics = compute_metrics(true_target, final_oof, target_name)
        comparison_rows.append({"target_name": target_name, "model": "baseline", **baseline_metrics})
        comparison_rows.append({"target_name": target_name, "model": "baseline_plus_residual_v2", **final_metrics})

        full_state = train_fold(train_samples[: int(len(train_samples)*0.85)], train_samples[int(len(train_samples)*0.85):], target_name, best_cfg, len(family_to_id)+1, len(component_to_id)+1, max_components)
        test_resid, transformed_test, model = predict_residual(test_samples, target_name, full_state["model_state"], full_state["comp_stats"], full_state["ctx_stats"], best_cfg, len(family_to_id)+1, len(component_to_id)+1, max_components)
        test_ids = [s.scenario_id for s in test_samples]
        test_base_arr = np.array([baseline_test[sid] for sid in test_ids], dtype=np.float32)
        if final_test[SCENARIO_COL] is None:
            final_test[SCENARIO_COL] = test_ids
        final_test[target_name] = test_base_arr + test_resid

        pd.DataFrame({SCENARIO_COL: scenario_ids, f"{target_name}_true": true_target, f"{target_name}_baseline_oof": baseline_train, f"{target_name}_residual_oof": best_oof_resid, f"{target_name}_final_oof": final_oof}).to_csv(OUTDIR / f"{target_name}_residual_oof.csv", index=False)

        _, transformed_train_for_sign, sign_model = predict_residual(train_samples, target_name, full_state["model_state"], full_state["comp_stats"], full_state["ctx_stats"], best_cfg, len(family_to_id)+1, len(component_to_id)+1, max_components)
        detail, summary = signed_pair_effect(sign_model, transformed_train_for_sign, target_name)
        detail.to_csv(OUTDIR / f"{target_name}_signed_pair_effect_detail.csv", index=False)
        summary.to_csv(OUTDIR / f"{target_name}_signed_pair_effect_summary.csv", index=False)

    pred_df = pd.DataFrame({SCENARIO_COL: final_test[SCENARIO_COL], TARGET_VISC_COL: final_test["viscosity"], TARGET_OX_COL: final_test["oxidation"]})
    pred_df.to_csv(OUTDIR / "prediction.csv", index=False)
    pd.DataFrame(all_results).sort_values(["target_name", "rmse"]).to_csv(OUTDIR / "residual_cv_results.csv", index=False)
    pd.DataFrame(comparison_rows).to_csv(OUTDIR / "baseline_vs_residual_comparison.csv", index=False)
    manifest = {
        "context_cols_v2": CONTEXT_COLS_V2,
        "extra_context_cols": V2_EXTRA_CONTEXT_COLS,
        "residual_scale": RESIDUAL_SCALE,
        "baseline_oof": str(BASELINE_OOF),
        "baseline_test_prediction": str(BASELINE_TEST_PRED),
    }
    (OUTDIR / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved to {OUTDIR}")


if __name__ == "__main__":
    main()
