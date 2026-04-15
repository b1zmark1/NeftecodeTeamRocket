from __future__ import annotations

import json
import math
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from build_training_data import (
    ARTIFACTS_DIR,
    FAMILY_ID_BY_RU,
    FAMILY_SPECS,
    PROPERTY_PATH,
    TEST_PATH,
    TRAIN_PATH,
    aggregate_numeric_property,
    get_property_meta,
    get_property_value,
    rows_to_matrix,
    safe_div,
    write_flat_csv,
    write_jsonl,
)
from chem_text_features import extract_text_feature_result
from dot_data import Scenario, load_scenarios


TEXT_PROPERTY_NAMES = {
    "Группа по API",
    "Тип АО",
    "Номер CAS / SMILES",
    "Класс субстрата",
    "Структура УВ-радикала",
    "Класс полиамина",
    "Модификация",
    "Тип сукцинимида",
    "Тип спиртового радикала",
    "Разветвленность радикала / радикалов",
    "Происхождение",
    "Категория",
    "Тип лиганда",
    "Тип полимера",
    "Соотношение мономеров (EO:PO)",
    "SMILES для наиболее вероятной (средней) молекулы сульфокислоты",
    "Номер CAS",
    "CAS",
}


def build_token_payload_v2(scenario: Scenario) -> List[dict]:
    total_dose = sum(token.mass_share for token in scenario.components)
    ranked = sorted(scenario.components, key=lambda t: t.mass_share, reverse=True)
    rank_map = {id(token): rank + 1 for rank, token in enumerate(ranked)}

    payload = []
    for token in scenario.components:
        text_properties_raw: Dict[str, Optional[str]] = {}
        text_properties_normalized: Dict[str, Optional[str]] = {}
        text_numeric_features: Dict[str, float] = {}

        for property_name, prop in token.properties.items():
            if property_name not in TEXT_PROPERTY_NAMES:
                continue
            text_properties_raw[property_name] = prop.raw
            result = extract_text_feature_result(property_name, prop.raw)
            text_properties_normalized[property_name] = result.normalized_value
            for feat_name, feat_value in result.numeric_features.items():
                text_numeric_features[f"{property_name}__{feat_name}"] = feat_value

        payload.append(
            {
                "component_id": token.component_id,
                "component_family": token.component_family,
                "component_family_id": FAMILY_ID_BY_RU.get(token.component_family, "other"),
                "batch_id": token.batch_id,
                "dose_transformed": token.mass_share,
                "dose_rank_in_scenario": rank_map[id(token)],
                "dose_share_of_total_transformed": safe_div(token.mass_share, total_dose),
                "row_count_after_merge": token.row_count,
                "text_properties_raw": text_properties_raw,
                "text_properties_normalized": text_properties_normalized,
                "text_numeric_features": text_numeric_features,
            }
        )
    return payload


def collect_category_vocab(scenarios: Sequence[Scenario]) -> Dict[str, List[str]]:
    counter: Dict[str, Counter] = defaultdict(Counter)
    for scenario in scenarios:
        for token in scenario.components:
            family_id = FAMILY_ID_BY_RU.get(token.component_family, "other")
            for property_name, prop in token.properties.items():
                if property_name not in TEXT_PROPERTY_NAMES:
                    continue
                result = extract_text_feature_result(property_name, prop.raw)
                if result.normalized_value:
                    key = f"{family_id}::{property_name}"
                    counter[key][result.normalized_value] += 1

    vocab: Dict[str, List[str]] = {}
    for key, counts in counter.items():
        values = [value for value, cnt in counts.most_common() if cnt >= 2][:8]
        vocab[key] = values
    return vocab


def aggregate_text_numeric(tokens, property_name: str, feature_name: str) -> Dict[str, float]:
    pairs = []
    values = []
    for token in tokens:
        prop = token.properties.get(property_name)
        if not prop:
            continue
        result = extract_text_feature_result(property_name, prop.raw)
        value = result.numeric_features.get(feature_name)
        if value is None:
            continue
        pairs.append((value, token.mass_share))
        values.append(value)
    if not values:
        return {"wm": math.nan, "max": math.nan, "count": 0.0}
    weighted_mean = sum(value * weight for value, weight in pairs) / sum(weight for _, weight in pairs)
    return {"wm": weighted_mean, "max": max(values), "count": float(len(values))}


def collect_text_numeric_feature_names(scenarios: Sequence[Scenario]) -> Dict[str, Dict[str, List[str]]]:
    result: Dict[str, Dict[str, set]] = defaultdict(lambda: defaultdict(set))
    for scenario in scenarios:
        for token in scenario.components:
            family_id = FAMILY_ID_BY_RU.get(token.component_family, "other")
            for property_name, prop in token.properties.items():
                if property_name not in TEXT_PROPERTY_NAMES:
                    continue
                features = extract_text_feature_result(property_name, prop.raw).numeric_features
                result[family_id][property_name].update(features.keys())
    return {family_id: {prop: sorted(names) for prop, names in props.items()} for family_id, props in result.items()}


def build_flat_row_v2(
    scenario: Scenario,
    category_vocab: Dict[str, List[str]],
    text_numeric_schema: Dict[str, Dict[str, List[str]]],
) -> OrderedDict[str, object]:
    row: OrderedDict[str, object] = OrderedDict()
    row["scenario_id"] = scenario.scenario_id
    row["temperature_c"] = scenario.temperature_c
    row["time_h"] = scenario.time_h
    row["biofuel_pct"] = scenario.biofuel_pct
    row["catalyst_category"] = float(scenario.catalyst_category)
    row["temperature_x_time"] = scenario.temperature_c * scenario.time_h
    row["temperature_x_biofuel"] = scenario.temperature_c * scenario.biofuel_pct
    row["time_x_biofuel"] = scenario.time_h * scenario.biofuel_pct
    row["biofuel_x_catalyst"] = scenario.biofuel_pct * scenario.catalyst_category
    row["severity_exp"] = scenario.time_h * math.exp((scenario.temperature_c - 150.0) / 10.0)
    row["n_components"] = float(len(scenario.components))

    family_tokens: Dict[str, list] = {spec.family_ru: [] for spec in FAMILY_SPECS}
    for token in scenario.components:
        family_tokens.setdefault(token.component_family, []).append(token)

    family_dose_by_id: Dict[str, float] = {}
    for spec in FAMILY_SPECS:
        tokens = family_tokens.get(spec.family_ru, [])
        family_id = spec.family_id
        family_total_dose = sum(token.mass_share for token in tokens)
        family_count = float(len(tokens))
        family_dose_by_id[family_id] = family_total_dose
        row[f"family_total_dose__{family_id}"] = family_total_dose
        row[f"family_component_count__{family_id}"] = family_count
        row[f"family_present__{family_id}"] = 1.0 if family_count else 0.0

        for prop_spec in spec.numeric_properties:
            stats = aggregate_numeric_property(tokens, prop_spec.aliases)
            row[f"{family_id}__{prop_spec.feature_id}__wm"] = stats["wm"]
            row[f"{family_id}__{prop_spec.feature_id}__max"] = stats["max"]
            row[f"{family_id}__{prop_spec.feature_id}__count"] = stats["count"]

        # Aggregated categorical buckets.
        for property_name in spec.categorical_properties:
            key = f"{family_id}::{property_name}"
            vocab_values = category_vocab.get(key, [])
            category_dose = {value: 0.0 for value in vocab_values}
            category_count = {value: 0.0 for value in vocab_values}
            unknown_dose = 0.0
            for token in tokens:
                prop = token.properties.get(property_name)
                if not prop:
                    continue
                normalized = extract_text_feature_result(property_name, prop.raw).normalized_value
                if normalized in category_dose:
                    category_dose[normalized] += token.mass_share
                    category_count[normalized] += 1.0
                elif normalized:
                    unknown_dose += token.mass_share
            prop_slug = property_name.lower().replace(" ", "_").replace("/", "_")
            for value in vocab_values:
                row[f"catdose__{family_id}__{prop_slug}__{value}"] = category_dose[value]
                row[f"catcount__{family_id}__{prop_slug}__{value}"] = category_count[value]
            row[f"catdose__{family_id}__{prop_slug}__other"] = unknown_dose

        # Text-derived numeric descriptors, including SMILES/formula/ratio parsing.
        for property_name, feature_names in text_numeric_schema.get(family_id, {}).items():
            prop_slug = property_name.lower().replace(" ", "_").replace("/", "_")
            for feature_name in feature_names:
                stats = aggregate_text_numeric(tokens, property_name, feature_name)
                row[f"textnum__{family_id}__{prop_slug}__{feature_name}__wm"] = stats["wm"]
                row[f"textnum__{family_id}__{prop_slug}__{feature_name}__max"] = stats["max"]
                row[f"textnum__{family_id}__{prop_slug}__{feature_name}__count"] = stats["count"]

    antioxidant_dose = family_dose_by_id.get("antioxidant", 0.0)
    antiwear_dose = family_dose_by_id.get("antiwear", 0.0)
    moly_dose = family_dose_by_id.get("moly", 0.0)
    detergent_dose = family_dose_by_id.get("detergent", 0.0)
    dispersant_dose = family_dose_by_id.get("dispersant", 0.0)
    base_oil_dose = family_dose_by_id.get("base_oil", 0.0)
    polymer_dose = family_dose_by_id.get("polymer", 0.0)

    row["interaction__ao_x_zddp"] = antioxidant_dose * antiwear_dose
    row["interaction__ao_x_mo"] = antioxidant_dose * moly_dose
    row["interaction__detergent_x_dispersant"] = detergent_dose * dispersant_dose
    row["interaction__base_oil_x_ao"] = base_oil_dose * antioxidant_dose
    row["interaction__base_oil_x_dispersant"] = base_oil_dose * dispersant_dose
    row["interaction__biofuel_x_ao"] = scenario.biofuel_pct * antioxidant_dose
    row["interaction__biofuel_x_zddp"] = scenario.biofuel_pct * antiwear_dose
    row["interaction__temperature_x_ao"] = scenario.temperature_c * antioxidant_dose
    row["interaction__temperature_x_polymer"] = scenario.temperature_c * polymer_dose

    phosphorus = row.get("antiwear__phosphorus_pct__wm", math.nan)
    sulfur = row.get("antiwear__sulfur_pct__wm", math.nan)
    mo_pct = row.get("moly__mo_pct__wm", math.nan)
    row["interaction__catalyst_x_phosphorus"] = scenario.catalyst_category * (0.0 if math.isnan(phosphorus) else phosphorus)
    row["interaction__catalyst_x_sulfur"] = scenario.catalyst_category * (0.0 if math.isnan(sulfur) else sulfur)
    row["interaction__catalyst_x_mo"] = scenario.catalyst_category * (0.0 if math.isnan(mo_pct) else mo_pct)

    if scenario.target_viscosity_delta_pct is not None:
        row["target_viscosity_delta_pct"] = scenario.target_viscosity_delta_pct
        row["target_oxidation_acm"] = scenario.target_oxidation_acm
    return row


def main() -> None:
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    train_scenarios = load_scenarios(TRAIN_PATH, PROPERTY_PATH, include_targets=True)
    test_scenarios = load_scenarios(TEST_PATH, PROPERTY_PATH, include_targets=False)
    all_scenarios = list(train_scenarios) + list(test_scenarios)

    category_vocab = collect_category_vocab(all_scenarios)
    text_numeric_schema = collect_text_numeric_feature_names(all_scenarios)

    train_flat = [build_flat_row_v2(scenario, category_vocab, text_numeric_schema) for scenario in train_scenarios]
    test_flat = [build_flat_row_v2(scenario, category_vocab, text_numeric_schema) for scenario in test_scenarios]
    train_tokens = []
    for scenario in train_scenarios:
        train_tokens.append(
            {
                "scenario_id": scenario.scenario_id,
                "conditions": {
                    "temperature_c": scenario.temperature_c,
                    "time_h": scenario.time_h,
                    "biofuel_pct": scenario.biofuel_pct,
                    "catalyst_category": scenario.catalyst_category,
                },
                "targets": {
                    "target_viscosity_delta_pct": scenario.target_viscosity_delta_pct,
                    "target_oxidation_acm": scenario.target_oxidation_acm,
                },
                "tokens": build_token_payload_v2(scenario),
            }
        )
    test_tokens = []
    for scenario in test_scenarios:
        test_tokens.append(
            {
                "scenario_id": scenario.scenario_id,
                "conditions": {
                    "temperature_c": scenario.temperature_c,
                    "time_h": scenario.time_h,
                    "biofuel_pct": scenario.biofuel_pct,
                    "catalyst_category": scenario.catalyst_category,
                },
                "tokens": build_token_payload_v2(scenario),
            }
        )

    write_flat_csv(ARTIFACTS_DIR / "train_flat_features_v2.csv", train_flat)
    write_flat_csv(ARTIFACTS_DIR / "test_flat_features_v2.csv", test_flat)
    write_jsonl(ARTIFACTS_DIR / "train_tokens_v2.jsonl", train_tokens)
    write_jsonl(ARTIFACTS_DIR / "test_tokens_v2.jsonl", test_tokens)

    X_train, feature_names, y_train = rows_to_matrix(train_flat, target_mode=True)
    X_test, feature_names_test, _ = rows_to_matrix(test_flat, target_mode=False)
    if feature_names != feature_names_test:
        raise ValueError("Train/test feature columns mismatch for V2.")

    np.savez_compressed(
        ARTIFACTS_DIR / "train_flat_features_v2.npz",
        X=X_train,
        y=y_train,
        feature_names=np.array(feature_names, dtype=object),
        scenario_ids=np.array([row["scenario_id"] for row in train_flat], dtype=object),
    )
    np.savez_compressed(
        ARTIFACTS_DIR / "test_flat_features_v2.npz",
        X=X_test,
        feature_names=np.array(feature_names, dtype=object),
        scenario_ids=np.array([row["scenario_id"] for row in test_flat], dtype=object),
    )

    schema = {
        "flat_feature_count": len(feature_names),
        "category_vocab": category_vocab,
        "text_numeric_schema": text_numeric_schema,
        "files": {
            "train_flat_csv": "train_flat_features_v2.csv",
            "test_flat_csv": "test_flat_features_v2.csv",
            "train_flat_npz": "train_flat_features_v2.npz",
            "test_flat_npz": "test_flat_features_v2.npz",
            "train_tokens_jsonl": "train_tokens_v2.jsonl",
            "test_tokens_jsonl": "test_tokens_v2.jsonl",
        },
    }
    (ARTIFACTS_DIR / "feature_schema_v2.json").write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"train_flat_rows={len(train_flat)}")
    print(f"test_flat_rows={len(test_flat)}")
    print(f"flat_feature_count={len(feature_names)}")
    print(ARTIFACTS_DIR)


if __name__ == "__main__":
    main()
