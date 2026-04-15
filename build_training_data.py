from __future__ import annotations

import csv
import json
import math
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from dot_data import Scenario, load_scenarios


ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = ROOT / "artifacts"

TRAIN_PATH = ROOT / "daimler_mixtures_train.csv"
TEST_PATH = ROOT / "daimler_mixtures_test.csv"
PROPERTY_PATH = ROOT / "daimler_component_properties.csv"


@dataclass(frozen=True)
class NumericPropertySpec:
    feature_id: str
    aliases: Sequence[str]


@dataclass(frozen=True)
class FamilySpec:
    family_ru: str
    family_id: str
    numeric_properties: Sequence[NumericPropertySpec]
    categorical_properties: Sequence[str]


FAMILY_SPECS: Sequence[FamilySpec] = [
    FamilySpec(
        family_ru="Базовое_масло",
        family_id="base_oil",
        numeric_properties=[
            NumericPropertySpec("kv40", ["Кинематическая вязкость, при 40°C, ASTM D445"]),
            NumericPropertySpec("kv100", ["Кинематическая вязкость, при 100°C, ASTM D445"]),
            NumericPropertySpec("ccs_m15", ["Динамическая вязкость CCS -15°C, ASTM D5293"]),
            NumericPropertySpec("ccs_m20", ["Динамическая вязкость CCS -20°C, ASTM D5293"]),
            NumericPropertySpec("ccs_m25", ["Динамическая вязкость CCS -25°C, ASTM D5293"]),
            NumericPropertySpec("ccs_m30", ["Динамическая вязкость CCS -30°C, ASTM D5293"]),
            NumericPropertySpec("ccs_m35", ["Динамическая вязкость CCS -35°C, ASTM D5293"]),
            NumericPropertySpec("vi", ["Индекс вязкости, ГОСТ 25371"]),
            NumericPropertySpec("pour_point_c", ["Температура застывания, ГОСТ 20287, метод Б"]),
            NumericPropertySpec("noack_pct", ["Испаряемость по NOACK, ASTM D5800"]),
            NumericPropertySpec("density_15", ["Плотность при 15°С, ASTM D4052"]),
            NumericPropertySpec("density_20", ["Плотность при 20°С, ASTM D4052"]),
            NumericPropertySpec("aniline_point", ["Анилиновая точка"]),
            NumericPropertySpec("aromatics_pct", ["Содержание ароматики"]),
            NumericPropertySpec("saturates_pct", ["Содержание насыщ. у/в"]),
            NumericPropertySpec("sulfur_mgkg", ["Содержание серы, мг/кг"]),
            NumericPropertySpec("sulfur_pct", ["Содержание серы, % масс."]),
            NumericPropertySpec("deaeration_min", ["Деаэрация | ASTM D3427"]),
            NumericPropertySpec("deem_water_ml", ["Деэм.вода | ASTM D1401"]),
            NumericPropertySpec("deem_time_min", ["Деэм.время | ASTM D1401"]),
            NumericPropertySpec("deem_oil_ml", ["Деэм.масло | ASTM D1401"]),
            NumericPropertySpec("deem_emulsion_ml", ["Деэм.эмульсия | ASTM D1401"]),
            NumericPropertySpec("foam_seq1_cm3", ["Последовательность 1 | ASTM D892"]),
            NumericPropertySpec("foam_seq2_cm3", ["Последовательность 2 | ASTM D892"]),
            NumericPropertySpec("foam_seq3_cm3", ["Последовательность 3 | ASTM D892"]),
        ],
        categorical_properties=["Группа по API"],
    ),
    FamilySpec(
        family_ru="Антиоксидант",
        family_id="antioxidant",
        numeric_properties=[
            NumericPropertySpec("active_no_pct", ["Активный Азот / Кислород, % масс. (N или O)"]),
            NumericPropertySpec("melting_point_c", ["Температура плавления, °C"]),
            NumericPropertySpec("bde_xh_kcal", ["Энергия диссоциации связи Х-Н, ккал/моль"]),
            NumericPropertySpec("ionization_ev", ["Потенциал ионизации,эВ"]),
            NumericPropertySpec("chemical_potential_jmol", ["Химический потенциал, Дж/моль"]),
            NumericPropertySpec("homo_ev", ["Энергия ВЗМО, эВ"]),
            NumericPropertySpec("lumo_ev", ["Энергия НСМО, эВ"]),
            NumericPropertySpec("dipole", ["Дипольный момент, Д"]),
            NumericPropertySpec("steric_factor", ["Стерический фактор, Å3"]),
        ],
        categorical_properties=["Тип АО", "Номер CAS / SMILES"],
    ),
    FamilySpec(
        family_ru="Детергент",
        family_id="detergent",
        numeric_properties=[
            NumericPropertySpec("tbn_astm", ["Щелочное число, ASTM D2896"]),
            NumericPropertySpec("tbn_gost", ["Щелочное число, ГОСТ 11362"]),
            NumericPropertySpec("ca_pct", ["Массовая доля кальция, ASTM D6481", "Массовая доля кальция | ASTM D6481"]),
            NumericPropertySpec("soap_pct", ["Содержание мыла, % масс."]),
            NumericPropertySpec("oil_pct", ["Содержание масла, % масс."]),
            NumericPropertySpec("carbonate_pct", ["Содержание MgCO3, CaCO3, % масс."]),
            NumericPropertySpec("soap_base_ratio", ["Отношение Мыло/Основание"]),
            NumericPropertySpec("metal_content_pct", ["Содержание металла (Ca/Mg), % масс."]),
            NumericPropertySpec("micelle_size_nm", ["Размер мицелл, нм"]),
            NumericPropertySpec("water_pct", ["Содержание воды, % масс."]),
        ],
        categorical_properties=["Класс субстрата", "Структура УВ-радикала"],
    ),
    FamilySpec(
        family_ru="Дисперсант",
        family_id="dispersant",
        numeric_properties=[
            NumericPropertySpec("nitrogen_content", ["Содержание Азота", "Общее содержание азота | ASTM D3228"]),
            NumericPropertySpec("boron_content", ["Содержание Бора"]),
            NumericPropertySpec("tail_mass_gmol", ["Масса гидрофобного хвоста, г/моль"]),
            NumericPropertySpec("pdi", ["Индекс полидисперсности"]),
            NumericPropertySpec("oil_content", ["Содержание масла"]),
        ],
        categorical_properties=["Класс полиамина", "Модификация", "Тип сукцинимида"],
    ),
    FamilySpec(
        family_ru="Противоизносная_присадка",
        family_id="antiwear",
        numeric_properties=[
            NumericPropertySpec("phosphorus_pct", ["Массовая доля фосфора, ASTM D6481", "Массовая доля фосфора | ASTM D6481"]),
            NumericPropertySpec("zinc_pct", ["Массовая доля цинка, ASTM D6481", "Массовая доля цинка | ASTM D6481"]),
            NumericPropertySpec("sulfur_pct", ["Массовая доля серы, ASTM D6481", "Массовая доля серы | ASTM D6481"]),
            NumericPropertySpec("p_zn_ratio", ["Атомное отношение P:Zn"]),
            NumericPropertySpec("chain_length", ["Длина углеродной цепи"]),
            NumericPropertySpec("polysulfidity", ["Степень полисульфидности"]),
            NumericPropertySpec("sulfated_ash_pct", ["Массовая доля сульфатной золы, ГОСТ 12417"]),
        ],
        categorical_properties=["Тип спиртового радикала", "Разветвленность радикала / радикалов"],
    ),
    FamilySpec(
        family_ru="Соединение_молибдена",
        family_id="moly",
        numeric_properties=[
            NumericPropertySpec("mo_pct", ["% масс. (Mo)"]),
            NumericPropertySpec("s_mo_ratio", ["Отношение S:Mo"]),
            NumericPropertySpec("coc_c", ["COC (°C)"]),
        ],
        categorical_properties=["Категория", "Тип лиганда"],
    ),
    FamilySpec(
        family_ru="Загуститель",
        family_id="polymer",
        numeric_properties=[
            NumericPropertySpec("polymer_content", ["Содержание полимера"]),
            NumericPropertySpec("weight_avg_mw", ["Средневесовая масса"]),
            NumericPropertySpec("stability_index_pct", ["Индекс стабильности, %"]),
            NumericPropertySpec("kv100", ["Кинематическая вязкость, при 100°C, ASTM D445"]),
        ],
        categorical_properties=["Тип полимера", "Соотношение мономеров (EO:PO)"],
    ),
    FamilySpec(
        family_ru="Антипенная_присадка",
        family_id="antifoam",
        numeric_properties=[],
        categorical_properties=[],
    ),
    FamilySpec(
        family_ru="Депрессорная_присадка",
        family_id="depressant",
        numeric_properties=[],
        categorical_properties=[],
    ),
]


FAMILY_ID_BY_RU = {spec.family_ru: spec.family_id for spec in FAMILY_SPECS}
FAMILY_SPEC_BY_RU = {spec.family_ru: spec for spec in FAMILY_SPECS}


def get_property_value(token, aliases: Sequence[str]) -> Optional[float]:
    for alias in aliases:
        prop = token.properties.get(alias)
        if prop and prop.value is not None:
            return prop.value
    return None


def get_property_meta(token, aliases: Sequence[str]) -> Dict[str, Optional[str]]:
    for alias in aliases:
        prop = token.properties.get(alias)
        if prop:
            return {
                "property_name": alias,
                "raw": prop.raw,
                "parse_kind": prop.parse_kind,
                "source": prop.source,
            }
    return {"property_name": aliases[0], "raw": None, "parse_kind": None, "source": None}


def weighted_mean(pairs: Iterable[tuple[float, float]]) -> Optional[float]:
    total_weight = 0.0
    total_value = 0.0
    for value, weight in pairs:
        total_value += value * weight
        total_weight += weight
    if total_weight == 0.0:
        return None
    return total_value / total_weight


def aggregate_numeric_property(tokens, aliases: Sequence[str]) -> Dict[str, float]:
    pairs = []
    values = []
    for token in tokens:
        value = get_property_value(token, aliases)
        if value is None:
            continue
        pairs.append((value, token.mass_share))
        values.append(value)
    if not values:
        return {"wm": math.nan, "max": math.nan, "count": 0.0}
    return {
        "wm": weighted_mean(pairs),
        "max": max(values),
        "count": float(len(values)),
    }


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def build_token_payload(scenario: Scenario) -> List[dict]:
    total_dose = sum(token.mass_share for token in scenario.components)
    ranked = sorted(scenario.components, key=lambda t: t.mass_share, reverse=True)
    rank_map = {id(token): rank + 1 for rank, token in enumerate(ranked)}

    payload = []
    for token in scenario.components:
        spec = FAMILY_SPEC_BY_RU.get(token.component_family)
        numeric_properties = {}
        numeric_present = {}
        parse_kinds = {}
        property_sources = {}
        categorical_properties = {}

        if spec:
            for prop_spec in spec.numeric_properties:
                numeric_properties[prop_spec.feature_id] = get_property_value(token, prop_spec.aliases)
                meta = get_property_meta(token, prop_spec.aliases)
                numeric_present[prop_spec.feature_id] = 1 if numeric_properties[prop_spec.feature_id] is not None else 0
                parse_kinds[prop_spec.feature_id] = meta["parse_kind"]
                property_sources[prop_spec.feature_id] = meta["source"]
            for prop_name in spec.categorical_properties:
                prop = token.properties.get(prop_name)
                categorical_properties[prop_name] = prop.raw if prop else None

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
                "numeric_properties": numeric_properties,
                "categorical_properties": categorical_properties,
                "property_present": numeric_present,
                "property_parse_kind": parse_kinds,
                "property_source": property_sources,
            }
        )
    return payload


def build_flat_row(scenario: Scenario) -> OrderedDict[str, object]:
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

    # Key cross-family and condition-aware interactions from the chemistry hypotheses.
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


def write_jsonl(path: Path, items: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def write_flat_csv(path: Path, rows: List[OrderedDict[str, object]]) -> List[str]:
    columns = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return columns


def rows_to_matrix(rows: List[OrderedDict[str, object]], target_mode: bool) -> tuple[np.ndarray, list[str], Optional[np.ndarray]]:
    drop = {"scenario_id"}
    if target_mode:
        drop.update({"target_viscosity_delta_pct", "target_oxidation_acm"})
    feature_names = [name for name in rows[0].keys() if name not in drop]

    X = np.empty((len(rows), len(feature_names)), dtype=np.float32)
    for i, row in enumerate(rows):
        for j, name in enumerate(feature_names):
            value = row[name]
            if value is None or (isinstance(value, float) and math.isnan(value)):
                X[i, j] = np.nan
            else:
                X[i, j] = float(value)

    y = None
    if target_mode:
        y = np.array(
            [[float(row["target_viscosity_delta_pct"]), float(row["target_oxidation_acm"])] for row in rows],
            dtype=np.float32,
        )
    return X, feature_names, y


def build_outputs(scenarios: Sequence[Scenario], include_targets: bool) -> tuple[List[OrderedDict[str, object]], List[dict]]:
    flat_rows = [build_flat_row(scenario) for scenario in scenarios]
    token_rows = []
    for scenario in scenarios:
        payload = {
            "scenario_id": scenario.scenario_id,
            "conditions": {
                "temperature_c": scenario.temperature_c,
                "time_h": scenario.time_h,
                "biofuel_pct": scenario.biofuel_pct,
                "catalyst_category": scenario.catalyst_category,
            },
            "tokens": build_token_payload(scenario),
        }
        if include_targets:
            payload["targets"] = {
                "target_viscosity_delta_pct": scenario.target_viscosity_delta_pct,
                "target_oxidation_acm": scenario.target_oxidation_acm,
            }
        token_rows.append(payload)
    return flat_rows, token_rows


def main() -> None:
    ARTIFACTS_DIR.mkdir(exist_ok=True)

    train_scenarios = load_scenarios(TRAIN_PATH, PROPERTY_PATH, include_targets=True)
    test_scenarios = load_scenarios(TEST_PATH, PROPERTY_PATH, include_targets=False)

    train_flat, train_tokens = build_outputs(train_scenarios, include_targets=True)
    test_flat, test_tokens = build_outputs(test_scenarios, include_targets=False)

    train_columns = write_flat_csv(ARTIFACTS_DIR / "train_flat_features_v1.csv", train_flat)
    test_columns = write_flat_csv(ARTIFACTS_DIR / "test_flat_features_v1.csv", test_flat)
    write_jsonl(ARTIFACTS_DIR / "train_tokens_v1.jsonl", train_tokens)
    write_jsonl(ARTIFACTS_DIR / "test_tokens_v1.jsonl", test_tokens)

    X_train, feature_names, y_train = rows_to_matrix(train_flat, target_mode=True)
    X_test, feature_names_test, _ = rows_to_matrix(test_flat, target_mode=False)
    if feature_names != feature_names_test:
        raise ValueError("Train/test feature columns mismatch.")

    np.savez_compressed(
        ARTIFACTS_DIR / "train_flat_features_v1.npz",
        X=X_train,
        y=y_train,
        feature_names=np.array(feature_names, dtype=object),
        scenario_ids=np.array([row["scenario_id"] for row in train_flat], dtype=object),
    )
    np.savez_compressed(
        ARTIFACTS_DIR / "test_flat_features_v1.npz",
        X=X_test,
        feature_names=np.array(feature_names, dtype=object),
        scenario_ids=np.array([row["scenario_id"] for row in test_flat], dtype=object),
    )

    schema = {
        "artifact_dir": str(ARTIFACTS_DIR),
        "train_scenarios": len(train_scenarios),
        "test_scenarios": len(test_scenarios),
        "flat_feature_count": len(feature_names),
        "family_specs": [
            {
                "family_ru": spec.family_ru,
                "family_id": spec.family_id,
                "numeric_properties": [prop.feature_id for prop in spec.numeric_properties],
                "categorical_properties": list(spec.categorical_properties),
            }
            for spec in FAMILY_SPECS
        ],
        "files": {
            "train_flat_csv": "train_flat_features_v1.csv",
            "test_flat_csv": "test_flat_features_v1.csv",
            "train_flat_npz": "train_flat_features_v1.npz",
            "test_flat_npz": "test_flat_features_v1.npz",
            "train_tokens_jsonl": "train_tokens_v1.jsonl",
            "test_tokens_jsonl": "test_tokens_v1.jsonl",
        },
        "feature_names": feature_names,
    }
    (ARTIFACTS_DIR / "feature_schema_v1.json").write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"train_flat_rows={len(train_flat)}")
    print(f"test_flat_rows={len(test_flat)}")
    print(f"flat_feature_count={len(feature_names)}")
    print(ARTIFACTS_DIR)


if __name__ == "__main__":
    main()
