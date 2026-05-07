from __future__ import annotations

import math

import torch  # noqa: F401
import pandas as pd

from src.feature_engineering import (
    GLOBAL_COLS,
    PROP_ACTIVE_NO,
    PROP_TYPE_AO,
    build_o2_features,
    build_property_tables,
    enrich_mixture_with_properties,
    normalize_property_value,
    parse_numeric_like_o2,
)
from src.train_hierarchical_model import parse_numeric_like


def test_parse_numeric_like_handles_common_lab_formats() -> None:
    assert parse_numeric_like(" 12,5 % ") == 12.5
    assert parse_numeric_like("≤ 3.2") == 3.2
    assert parse_numeric_like("1e-3") == 0.001
    assert math.isnan(parse_numeric_like("2026-05-07"))
    assert math.isnan(parse_numeric_like("нет"))


def test_parse_numeric_like_o2_extracts_numbers_from_text() -> None:
    assert parse_numeric_like_o2("100 ppm") == 100.0
    assert parse_numeric_like_o2("12,5 %") == 12.5
    assert parse_numeric_like_o2("about 7.4") == 7.4
    assert math.isnan(parse_numeric_like_o2("not measured"))


def test_exact_property_overrides_typical_and_missing_exact_falls_back() -> None:
    props_df = pd.DataFrame(
        [
            {
                "Компонент": "Антиоксидант_1",
                "Наименование партии": "typical",
                "Наименование показателя": PROP_ACTIVE_NO,
                "Значение показателя": normalize_property_value("1.5"),
            },
            {
                "Компонент": "Антиоксидант_1",
                "Наименование партии": "batch_a",
                "Наименование показателя": PROP_ACTIVE_NO,
                "Значение показателя": normalize_property_value("2.5"),
            },
        ]
    )
    mixture_df = pd.DataFrame(
        [
            {
                "scenario_id": "s1",
                "Компонент": "Антиоксидант_1",
                "Наименование партии": "batch_a",
            },
            {
                "scenario_id": "s2",
                "Компонент": "Антиоксидант_1",
                "Наименование партии": "unknown_batch",
            },
        ]
    )

    wide_exact, wide_typical, property_columns = build_property_tables(props_df)
    enriched = enrich_mixture_with_properties(
        mixture_df=mixture_df,
        wide_exact=wide_exact,
        wide_typical=wide_typical,
        property_columns=property_columns,
    )

    by_scenario = enriched.set_index("scenario_id")
    assert by_scenario.loc["s1", PROP_ACTIVE_NO] == 2.5
    assert by_scenario.loc["s2", PROP_ACTIVE_NO] == 1.5


def test_build_o2_features_for_salicylate_detergent_and_antioxidants() -> None:
    component_df = pd.DataFrame(
        [
            {
                "scenario_id": "s1",
                "Компонент": "Детергент_1",
                "Массовая доля, %": 10.0,
                "Класс субстрата": "Салицилат кальция",
                "Щелочное число, ASTM D2896": 100.0,
            },
            {
                "scenario_id": "s1",
                "Компонент": "Антиоксидант_1",
                PROP_TYPE_AO: "Дифениламин",
                PROP_ACTIVE_NO: 2.0,
            },
            {
                "scenario_id": "s1",
                "Компонент": "Антиоксидант_2",
                PROP_TYPE_AO: "Фенол",
                PROP_ACTIVE_NO: 3.0,
            },
        ]
    )
    scenario_df = pd.DataFrame(
        [
            {
                "scenario_id": "s1",
                GLOBAL_COLS[0]: 160,
                GLOBAL_COLS[1]: 168,
                GLOBAL_COLS[2]: 5,
                GLOBAL_COLS[3]: 1,
            }
        ]
    )

    features = build_o2_features(component_df, scenario_df).set_index("scenario_id")

    assert features.loc["s1", "o2_salicylate_tbn_x_amine_ao"] == 200.0
    assert features.loc["s1", "o2_salicylate_tbn_x_phenol_ao"] == 300.0
    assert features.loc["s1", "o2_salicylate_tbn_x_amine_x_phenol"] == 600.0
    assert features.loc["s1", "o2_ca_salicylate_present"] == 1.0
