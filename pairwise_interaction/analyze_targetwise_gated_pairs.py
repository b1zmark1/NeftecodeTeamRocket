#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from pairwise_interaction.train_targetwise_gated_pairwise_model import (
    BATCH_COL,
    COMPONENT_COL,
    COMPONENT_NUMERIC_COLS,
    CONTEXT_COLS,
    DEFAULT_OUTDIR,
    DEFAULT_TRAIN_COMPONENTS,
    DEFAULT_TRAIN_SCENARIO,
    DEFAULT_TEST_COMPONENTS,
    PAIR_TYPE_NAMES,
    SCENARIO_COL,
    ScenarioSample,
    TargetwiseGatedPairwiseRegressor,
    apply_standardizer,
    build_samples,
    family_of,
)

ANALYSIS_DIR = DEFAULT_OUTDIR / "analysis"


def load_checkpoint(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def transform_samples(samples: List[ScenarioSample], comp_stats: Dict[str, np.ndarray], ctx_stats: Dict[str, np.ndarray]) -> List[ScenarioSample]:
    raw_dim = len(COMPONENT_NUMERIC_COLS)
    transformed: List[ScenarioSample] = []
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
    return transformed


def build_component_lookup(comp_df: pd.DataFrame, max_components: int) -> Dict[str, dict]:
    lookup: Dict[str, dict] = {}
    for scenario_id, group in comp_df.groupby(SCENARIO_COL, sort=False):
        g = group.sort_values([COMPONENT_COL, BATCH_COL], kind="stable").reset_index(drop=True)
        g = g.iloc[:max_components].copy()
        names = g[COMPONENT_COL].astype(str).tolist()
        families = [family_of(x) for x in names]
        batches = g[BATCH_COL].astype(str).tolist()
        lookup[str(scenario_id)] = {"names": names, "families": families, "batches": batches}
    return lookup


def analyze_target(
    target_name: str,
    checkpoint_path: Path,
    train_comp: pd.DataFrame,
    train_scen: pd.DataFrame,
    family_to_id: Dict[str, int],
    component_to_id: Dict[str, int],
    max_components: int,
) -> None:
    checkpoint = load_checkpoint(checkpoint_path)
    config = checkpoint["config"]
    comp_stats = checkpoint["component_stats"]
    ctx_stats = checkpoint["context_stats"]

    samples = build_samples(
        train_comp,
        train_scen,
        family_to_id,
        component_to_id,
        max_components=max_components,
        target_name=target_name,
        include_target=True,
    )
    samples = transform_samples(samples, comp_stats, ctx_stats)
    component_lookup = build_component_lookup(train_comp, max_components)

    model = TargetwiseGatedPairwiseRegressor(
        num_component_features=samples[0].component_numeric.shape[1],
        num_context_features=len(CONTEXT_COLS),
        n_families=len(family_to_id) + 1,
        n_components=len(component_to_id) + 1,
        max_components=max_components,
        d_model=int(config["d_model"]),
        dropout=float(config["dropout"]),
        ctx_dim=int(config["ctx_dim"]),
        pair_hidden=int(config["pair_hidden"]),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    pair_i = model.pair_i.numpy()
    pair_j = model.pair_j.numpy()

    pair_rows: List[dict] = []
    scenario_rows: List[dict] = []

    with torch.no_grad():
        for sample in samples:
            x_num = torch.tensor(sample.component_numeric[None, ...], dtype=torch.float32)
            x_valid = torch.tensor(sample.component_valid[None, ...], dtype=torch.bool)
            fam_ids = torch.tensor(sample.family_ids[None, ...], dtype=torch.long)
            comp_ids = torch.tensor(sample.component_ids[None, ...], dtype=torch.long)
            ctx = torch.tensor(sample.context[None, ...], dtype=torch.float32)
            pair_allowed = torch.tensor(sample.pair_allowed[None, ...], dtype=torch.bool)
            pair_type_ids = torch.tensor(sample.pair_type_ids[None, ...], dtype=torch.long)

            fam_emb = model.family_emb(fam_ids)
            comp_emb = model.component_emb(comp_ids)
            comp_input = torch.cat([x_num, fam_emb, comp_emb], dim=-1)
            comp_repr = model.component_encoder(comp_input)
            ctx_repr = model.context_mlp(ctx)
            xi = comp_repr[:, model.pair_i, :]
            xj = comp_repr[:, model.pair_j, :]
            type_emb = model.pair_type_emb(pair_type_ids)
            valid_pairs = x_valid[:, model.pair_i] & x_valid[:, model.pair_j] & pair_allowed
            ctx_exp = ctx_repr.unsqueeze(1).expand(-1, xi.size(1), -1)
            pair_input = torch.cat([xi, xj, torch.abs(xi - xj), xi * xj, type_emb, ctx_exp], dim=-1)
            pair_base = model.pair_mlp(pair_input)
            gates = torch.sigmoid(model.gate_mlp(pair_input)).squeeze(-1)
            logits = model.attn_mlp(pair_input).squeeze(-1)
            attn = model.masked_softmax(logits, valid_pairs)
            pair_strength = torch.norm(pair_base, dim=-1)
            proxy_score = attn * gates * pair_strength

            scenario_total = 0.0
            per_type = {name: 0.0 for name in PAIR_TYPE_NAMES.values() if name != "none"}
            info = component_lookup[sample.scenario_id]
            for idx, allowed in enumerate(sample.pair_allowed):
                if not allowed:
                    continue
                pair_type = int(sample.pair_type_ids[idx])
                pair_name = PAIR_TYPE_NAMES[pair_type]
                score = float(proxy_score[0, idx].item())
                gate = float(gates[0, idx].item())
                attn_w = float(attn[0, idx].item())
                strength = float(pair_strength[0, idx].item())
                i = int(pair_i[idx])
                j = int(pair_j[idx])
                scenario_total += score
                per_type[pair_name] += score
                pair_rows.append(
                    {
                        SCENARIO_COL: sample.scenario_id,
                        "target_name": target_name,
                        "pair_type": pair_name,
                        "component_i": info["names"][i],
                        "component_j": info["names"][j],
                        "family_i": info["families"][i],
                        "family_j": info["families"][j],
                        "batch_i": info["batches"][i],
                        "batch_j": info["batches"][j],
                        "proxy_score": score,
                        "gate": gate,
                        "attention": attn_w,
                        "pair_strength": strength,
                        "target_true": sample.target,
                    }
                )
            scenario_row = {
                SCENARIO_COL: sample.scenario_id,
                "target_name": target_name,
                "total_pair_proxy_score": scenario_total,
                "target_true": sample.target,
            }
            scenario_row.update({f"score__{k}": v for k, v in per_type.items()})
            scenario_rows.append(scenario_row)

    pair_df = pd.DataFrame(pair_rows)
    scen_df = pd.DataFrame(scenario_rows)

    type_importance = (
        pair_df.groupby("pair_type", dropna=False)
        .agg(
            pair_count=("pair_type", "size"),
            mean_proxy_score=("proxy_score", "mean"),
            median_proxy_score=("proxy_score", "median"),
            sum_proxy_score=("proxy_score", "sum"),
            mean_gate=("gate", "mean"),
            mean_attention=("attention", "mean"),
            mean_pair_strength=("pair_strength", "mean"),
        )
        .reset_index()
        .sort_values("sum_proxy_score", ascending=False)
    )

    top_pairs = pair_df.sort_values("proxy_score", ascending=False).head(200).reset_index(drop=True)
    top_scenarios = scen_df.sort_values("total_pair_proxy_score", ascending=False).reset_index(drop=True)

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    pair_df.to_csv(ANALYSIS_DIR / f"{target_name}_all_pair_scores.csv", index=False)
    scen_df.to_csv(ANALYSIS_DIR / f"{target_name}_scenario_pair_summary.csv", index=False)
    type_importance.to_csv(ANALYSIS_DIR / f"{target_name}_pair_type_importance.csv", index=False)
    top_pairs.to_csv(ANALYSIS_DIR / f"{target_name}_top_active_pairs.csv", index=False)
    top_scenarios.head(100).to_csv(ANALYSIS_DIR / f"{target_name}_top_scenarios.csv", index=False)


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    train_comp = pd.read_csv(DEFAULT_TRAIN_COMPONENTS)
    test_comp = pd.read_csv(DEFAULT_TEST_COMPONENTS)
    train_scen = pd.read_csv(DEFAULT_TRAIN_SCENARIO)

    family_vocab = sorted({family_of(v) for v in pd.concat([train_comp[COMPONENT_COL], test_comp[COMPONENT_COL]], axis=0).astype(str)})
    component_vocab = sorted(set(pd.concat([train_comp[COMPONENT_COL], test_comp[COMPONENT_COL]], axis=0).astype(str)))
    family_to_id = {name: idx + 1 for idx, name in enumerate(family_vocab)}
    component_to_id = {name: idx + 1 for idx, name in enumerate(component_vocab)}
    max_components = int(train_comp.groupby(SCENARIO_COL).size().max())

    analyze_target(
        "viscosity",
        DEFAULT_OUTDIR / "viscosity_best_model.pt",
        train_comp,
        train_scen,
        family_to_id,
        component_to_id,
        max_components,
    )
    analyze_target(
        "oxidation",
        DEFAULT_OUTDIR / "oxidation_best_model.pt",
        train_comp,
        train_scen,
        family_to_id,
        component_to_id,
        max_components,
    )

    manifest = {
        "analysis_dir": str(ANALYSIS_DIR),
        "targets": ["viscosity", "oxidation"],
        "score_definition": "proxy_score = attention * gate * ||pair_embedding||_2",
        "note": "Это proxy-важность pair block, а не строгая SHAP/causal attribution.",
    }
    (ANALYSIS_DIR / "analysis_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved to {ANALYSIS_DIR}")


if __name__ == "__main__":
    main()
