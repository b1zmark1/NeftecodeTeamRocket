from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


MIXTURE_TRAIN_COLUMNS = {
    "scenario_id": 0,
    "component_id": 1,
    "batch_id": 2,
    "mass_share": 3,
    "temperature_c": 4,
    "time_h": 5,
    "target_viscosity_delta_pct": 6,
    "target_oxidation_acm": 7,
    "biofuel_pct": 8,
    "catalyst_category": 9,
}

MIXTURE_TEST_COLUMNS = {
    "scenario_id": 0,
    "component_id": 1,
    "batch_id": 2,
    "mass_share": 3,
    "temperature_c": 4,
    "time_h": 5,
    "biofuel_pct": 6,
    "catalyst_category": 7,
}

PROPERTY_COLUMNS = {
    "component_id": 0,
    "batch_id": 1,
    "property_name": 2,
    "property_unit": 3,
    "property_value": 4,
}

RANGE_RE = re.compile(r"^\s*(-?\d+(?:[.,]\d+)?)\s*[-–]\s*(-?\d+(?:[.,]\d+)?)\s*$")
FIRST_NUMBER_RE = re.compile(r"-?\d+(?:[.,]\d+)?")


@dataclass
class ParsedValue:
    value: Optional[float]
    kind: str
    raw: str


@dataclass
class PropertyValue:
    name: str
    unit: str
    value: Optional[float]
    parse_kind: str
    raw: str
    source: str


@dataclass
class ComponentToken:
    component_id: str
    component_family: str
    batch_id: str
    mass_share: float
    row_count: int = 1
    properties: Dict[str, PropertyValue] = field(default_factory=dict)


@dataclass
class Scenario:
    scenario_id: str
    temperature_c: float
    time_h: float
    biofuel_pct: float
    catalyst_category: int
    components: List[ComponentToken]
    target_viscosity_delta_pct: Optional[float] = None
    target_oxidation_acm: Optional[float] = None


def extract_family(component_id: str) -> str:
    match = re.match(r"(.+)_\d+$", component_id)
    return match.group(1) if match else component_id


def parse_numeric_value(raw: str) -> ParsedValue:
    text = (raw or "").strip()
    if not text:
        return ParsedValue(None, "missing", raw)

    text = text.replace("\xa0", " ")

    try:
        return ParsedValue(float(text.replace(",", ".")), "exact", raw)
    except ValueError:
        pass

    if any(sep in text for sep in ("T", ":")) and re.search(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}", text):
        return ParsedValue(None, "datetime_like", raw)

    range_match = RANGE_RE.match(text)
    if range_match:
        left = float(range_match.group(1).replace(",", "."))
        right = float(range_match.group(2).replace(",", "."))
        return ParsedValue((left + right) / 2.0, "range_mid", raw)

    for prefix, kind in (("<=", "upper_bound"), ("≤", "upper_bound"), ("<", "upper_bound"),
                         (">=", "lower_bound"), ("≥", "lower_bound"), (">", "lower_bound")):
        if text.startswith(prefix):
            number_match = FIRST_NUMBER_RE.search(text)
            if number_match:
                return ParsedValue(float(number_match.group(0).replace(",", ".")), kind, raw)
            return ParsedValue(None, f"{kind}_missing_number", raw)

    number_match = FIRST_NUMBER_RE.search(text)
    if number_match:
        return ParsedValue(float(number_match.group(0).replace(",", ".")), "parsed_first_number", raw)

    return ParsedValue(None, "non_numeric", raw)


def load_property_table(path: Path) -> Dict[Tuple[str, str], Dict[str, PropertyValue]]:
    table: Dict[Tuple[str, str], Dict[str, PropertyValue]] = {}

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)

        for row in reader:
            if len(row) < 5:
                continue

            component_id = row[PROPERTY_COLUMNS["component_id"]].strip()
            batch_id = row[PROPERTY_COLUMNS["batch_id"]].strip()
            property_name = row[PROPERTY_COLUMNS["property_name"]].strip()
            unit = row[PROPERTY_COLUMNS["property_unit"]].strip()
            raw_value = row[PROPERTY_COLUMNS["property_value"]].strip()

            parsed = parse_numeric_value(raw_value)
            source = "typical" if batch_id.lower() == "typical" else "measured"

            table.setdefault((component_id, batch_id), {})[property_name] = PropertyValue(
                name=property_name,
                unit=unit,
                value=parsed.value,
                parse_kind=parsed.kind,
                raw=raw_value,
                source=source,
            )

    return table


def resolve_component_properties(
    property_table: Dict[Tuple[str, str], Dict[str, PropertyValue]],
    component_id: str,
    batch_id: str,
) -> Dict[str, PropertyValue]:
    resolved: Dict[str, PropertyValue] = {}

    typical = property_table.get((component_id, "typical"), {})
    measured = property_table.get((component_id, batch_id), {})

    for name, value in typical.items():
        resolved[name] = value
    for name, value in measured.items():
        resolved[name] = value

    return resolved


def _new_scenario(row: List[str], columns: Dict[str, int], include_targets: bool) -> Scenario:
    return Scenario(
        scenario_id=row[columns["scenario_id"]].strip(),
        temperature_c=float(row[columns["temperature_c"]]),
        time_h=float(row[columns["time_h"]]),
        biofuel_pct=float(row[columns["biofuel_pct"]]),
        catalyst_category=int(float(row[columns["catalyst_category"]])),
        components=[],
        target_viscosity_delta_pct=(
            float(row[columns["target_viscosity_delta_pct"]]) if include_targets else None
        ),
        target_oxidation_acm=(
            float(row[columns["target_oxidation_acm"]]) if include_targets else None
        ),
    )


def load_scenarios(
    mixture_path: Path,
    property_path: Path,
    include_targets: bool,
    aggregate_duplicate_component_batches: bool = True,
) -> List[Scenario]:
    columns = MIXTURE_TRAIN_COLUMNS if include_targets else MIXTURE_TEST_COLUMNS
    property_table = load_property_table(property_path)
    scenarios: Dict[str, Scenario] = {}
    component_index: Dict[Tuple[str, str, str], ComponentToken] = {}

    with mixture_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)

        for row in reader:
            if len(row) < max(columns.values()) + 1:
                continue

            scenario_id = row[columns["scenario_id"]].strip()
            if scenario_id not in scenarios:
                scenarios[scenario_id] = _new_scenario(row, columns=columns, include_targets=include_targets)

            component_id = row[columns["component_id"]].strip()
            batch_id = row[columns["batch_id"]].strip()
            mass_share = float(row[columns["mass_share"]])
            key = (scenario_id, component_id, batch_id)

            if aggregate_duplicate_component_batches and key in component_index:
                token = component_index[key]
                token.mass_share += mass_share
                token.row_count += 1
                continue

            token = ComponentToken(
                component_id=component_id,
                component_family=extract_family(component_id),
                batch_id=batch_id,
                mass_share=mass_share,
                properties=resolve_component_properties(property_table, component_id, batch_id),
            )

            component_index[key] = token
            scenarios[scenario_id].components.append(token)

    return [scenarios[key] for key in sorted(scenarios)]


def safe_mean(values: Iterable[float]) -> Optional[float]:
    values = list(values)
    if not values:
        return None
    return sum(values) / len(values)


def weighted_mean(pairs: Iterable[Tuple[float, float]]) -> Optional[float]:
    total_weight = 0.0
    weighted_sum = 0.0
    for value, weight in pairs:
        if value is None or weight is None:
            continue
        weighted_sum += value * weight
        total_weight += weight
    if total_weight == 0.0:
        return None
    return weighted_sum / total_weight


def weighted_sum(pairs: Iterable[Tuple[float, float]]) -> float:
    total = 0.0
    for value, weight in pairs:
        if value is None or weight is None:
            continue
        total += value * weight
    return total


def scenario_summary(scenario: Scenario) -> Dict[str, object]:
    family_mass_share: Dict[str, float] = {}
    property_presence: Dict[str, int] = {}
    property_uncertainty: Dict[str, int] = {}

    for token in scenario.components:
        family_mass_share[token.component_family] = family_mass_share.get(token.component_family, 0.0) + token.mass_share
        for name, prop in token.properties.items():
            property_presence[name] = property_presence.get(name, 0) + 1
            if prop.parse_kind not in {"exact", "range_mid"}:
                property_uncertainty[name] = property_uncertainty.get(name, 0) + 1

    exposure_index = scenario.time_h * math.exp((scenario.temperature_c - 150.0) / 10.0)

    return {
        "scenario_id": scenario.scenario_id,
        "n_components": len(scenario.components),
        "temperature_c": scenario.temperature_c,
        "time_h": scenario.time_h,
        "biofuel_pct": scenario.biofuel_pct,
        "catalyst_category": scenario.catalyst_category,
        "exposure_index": exposure_index,
        "family_mass_share": family_mass_share,
        "property_presence": property_presence,
        "property_uncertainty": property_uncertainty,
        "target_viscosity_delta_pct": scenario.target_viscosity_delta_pct,
        "target_oxidation_acm": scenario.target_oxidation_acm,
    }


def infer_property_group(property_name: str, component_family: str) -> str:
    name = property_name.lower()
    family = component_family.lower()

    if family == "антиоксидант" or any(
        token in name
        for token in (
            "энергия диссоциации",
            "потенциал ионизации",
            "энергия взмо",
            "энергия нсмо",
            "дипольный момент",
            "тип ао",
        )
    ):
        return "antioxidant_activity"

    if family == "базовое_масло" or any(
        token in name
        for token in (
            "кинематическая вязкость",
            "ccs",
            "индекс вязкости",
            "температура застывания",
            "noack",
            "плотность",
            "содержание насыщ",
        )
    ):
        return "base_oil_structure"

    if family == "детергент" or any(
        token in name
        for token in (
            "щелочное число",
            "кальция",
            "магния",
            "содержание металла",
            "мицелл",
            "сульфонат",
            "салицилат",
            "мыло",
        )
    ):
        return "detergency_reserve"

    if family == "дисперсант" or any(
        token in name
        for token in (
            "азота",
            "бора",
            "полиамина",
            "сукцинимида",
            "гидрофобного хвоста",
            "полидисперсности",
        )
    ):
        return "dispersancy_polarity"

    if family in {"противоизносная_присадка", "соединение_молибдена"} or any(
        token in name
        for token in (
            "цинка",
            "фосфора",
            "s:mo",
            "% масс. (mo)",
            "лиганда",
            "p:zn",
        )
    ):
        return "antiwear_redox"

    if family == "загуститель" or any(
        token in name
        for token in (
            "тип полимера",
            "содержание полимера",
            "средневесовая масса",
            "соотношение мономеров",
            "индекс стабильности",
        )
    ):
        return "polymer_rheology"

    if family == "депрессорная_присадка":
        return "low_temp_flow"

    if family == "антипенная_присадка":
        return "foam_control"

    return "other"


def mechanism_aggregates(scenario: Scenario) -> Dict[str, Dict[str, object]]:
    buckets: Dict[str, Dict[str, List[Tuple[Optional[float], float]]]] = {}
    group_mass_share: Dict[str, float] = {}
    group_token_count: Dict[str, int] = {}

    for token in scenario.components:
        seen_groups = set()
        for name, prop in token.properties.items():
            if prop.value is None:
                continue
            group = infer_property_group(name, token.component_family)
            buckets.setdefault(group, {}).setdefault(name, []).append((prop.value, token.mass_share))
            seen_groups.add(group)
        for group in seen_groups:
            group_mass_share[group] = group_mass_share.get(group, 0.0) + token.mass_share
            group_token_count[group] = group_token_count.get(group, 0) + 1

    aggregates: Dict[str, Dict[str, object]] = {}
    for group, properties in buckets.items():
        group_payload: Dict[str, object] = {
            "token_count": group_token_count.get(group, 0),
            "mass_share_total": group_mass_share.get(group, 0.0),
            "properties": {},
        }
        for property_name, pairs in properties.items():
            values = [value for value, _ in pairs if value is not None]
            weights = [weight for _, weight in pairs]
            group_payload["properties"][property_name] = {
                "weighted_mean": weighted_mean(pairs),
                "weighted_sum": weighted_sum(pairs),
                "max_value": max(values) if values else None,
                "count": len(values),
                "mass_share_total": sum(weights),
            }
        aggregates[group] = group_payload

    return aggregates
