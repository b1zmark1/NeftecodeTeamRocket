from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


R_GAS_CONSTANT = 8.314

TRAIN_TARGET_COLS = [
    "Delta Kin. Viscosity KV100 - relative | - Daimler Oxidation Test (DOT), %",
    "Oxidation EOT | DIN 51453 Daimler Oxidation Test (DOT), A/cm",
]

GLOBAL_COLS = [
    "Температура испытания | ASTM D445 Daimler Oxidation Test (DOT), °C",
    "Время испытания | - Daimler Oxidation Test (DOT), ч",
    "Количество биотоплива | - Daimler Oxidation Test (DOT), % масс",
    "Дозировка катализатора, категория",
]

MIXTURE_KEY_COLS = [
    "scenario_id",
    "Компонент",
    "Наименование партии",
]

REQUIRED_MIXTURE_COLS = MIXTURE_KEY_COLS + [
    "Массовая доля, %",
    *GLOBAL_COLS,
]

PROPS_REQUIRED_COLS = [
    "Компонент",
    "Наименование партии",
    "Наименование показателя",
    "Значение показателя",
]

PROP_TYPE_AO = "Тип АО"
PROP_ACTIVE_NO = "Активный Азот / Кислород, % масс. (N или O)"
PROP_MO = "% масс. (Mo)"
PROP_ZN = "Массовая доля цинка, ASTM D6481"
PROP_BORON = "Содержание Бора"
PROP_S = "Массовая доля серы, ASTM D6481"
PROP_CA = "Массовая доля кальция, ASTM D6481"
PROP_CA_MG = "Содержание металла (Ca/Mg), % масс."
PROP_E_BOND = "Энергия диссоциации связи Х-Н, ккал/моль"
PROP_IONIZATION = "Потенциал ионизации,эВ"
PROP_STERIC = "Стерический фактор, Å3"
PROP_CHEM_POT = "Химический потенциал, Дж/моль"
PROP_HOMO = "Энергия ВЗМО, эВ"
PROP_LUMO = "Энергия НСМО, эВ"

AO_TYPE_PHENOL = "фенол"
AO_TYPE_DPA = "дифениламин"

PROPERTY_NAME_ALIASES = {
    "массовая доля цинка, astm d6481": PROP_ZN,
    "массовая доля цинка | astm d6481": PROP_ZN,
    "массовая доля серы, astm d6481": PROP_S,
    "массовая доля серы | astm d6481": PROP_S,
    "массовая доля кальция, astm d6481": PROP_CA,
    "массовая доля кальция | astm d6481": PROP_CA,
    "% масс. mo": PROP_MO,
    "% масс. (mo)": PROP_MO,
    "энергия взмо эв": PROP_HOMO,
    "энергия взмо, эв": PROP_HOMO,
}


def normalize_string(value: Any) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).strip())


def normalize_casefold(value: Any) -> str:
    return normalize_string(value).casefold()


def normalize_property_name(value: Any) -> str:
    normalized = normalize_string(value)
    return PROPERTY_NAME_ALIASES.get(normalized.casefold(), normalized)


def normalize_property_value(value: Any) -> Any:
    if pd.isna(value):
        return np.nan

    normalized = normalize_string(value)
    if not normalized:
        return np.nan

    numeric_value = pd.to_numeric(normalized, errors="coerce")
    if pd.notna(numeric_value):
        return float(numeric_value)

    return normalized


def assert_required_columns(df: pd.DataFrame, required: list[str], df_name: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"В {df_name} отсутствуют обязательные колонки: {missing}")


def read_csv_strict(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")
    return pd.read_csv(path)


def component_is(component_name: str, prefix: str) -> bool:
    return normalize_string(component_name).startswith(prefix)


def build_property_tables(props_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    exact_df = props_df[props_df["Наименование партии"].map(normalize_casefold) != "typical"].copy()
    typical_df = props_df[props_df["Наименование партии"].map(normalize_casefold) == "typical"].copy()

    wide_exact = exact_df.pivot_table(
        index=["Компонент", "Наименование партии"],
        columns="Наименование показателя",
        values="Значение показателя",
        aggfunc="first",
    ).reset_index()

    wide_typical = typical_df.pivot_table(
        index=["Компонент"],
        columns="Наименование показателя",
        values="Значение показателя",
        aggfunc="first",
    ).reset_index()

    property_columns = sorted(
        (set(wide_exact.columns) - {"Компонент", "Наименование партии"})
        | (set(wide_typical.columns) - {"Компонент"})
    )

    for col in property_columns:
        if col not in wide_exact.columns:
            wide_exact[col] = np.nan
        if col not in wide_typical.columns:
            wide_typical[col] = np.nan

    wide_exact = wide_exact[["Компонент", "Наименование партии"] + property_columns]
    wide_typical = wide_typical[["Компонент"] + property_columns]

    return wide_exact, wide_typical, property_columns


def enrich_mixture_with_properties(
    mixture_df: pd.DataFrame,
    wide_exact: pd.DataFrame,
    wide_typical: pd.DataFrame,
    property_columns: list[str],
) -> pd.DataFrame:
    enriched = mixture_df.merge(
        wide_exact,
        on=["Компонент", "Наименование партии"],
        how="left",
    )
    enriched = enriched.merge(
        wide_typical,
        on=["Компонент"],
        how="left",
        suffixes=("", "__typ"),
    )

    for col in property_columns:
        typ_col = f"{col}__typ"
        enriched[col] = enriched[col].where(enriched[col].notna(), enriched[typ_col])

    drop_cols = [f"{col}__typ" for col in property_columns]
    enriched = enriched.drop(columns=drop_cols)

    return enriched


def safe_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def sum_product_cross(a: pd.Series, b: pd.Series) -> float:
    a_clean = pd.to_numeric(a, errors="coerce").dropna()
    b_clean = pd.to_numeric(b, errors="coerce").dropna()
    if a_clean.empty or b_clean.empty:
        return 0.0
    return float(a_clean.sum() * b_clean.sum())


def calculate_k(e_value: float, a_value: float, temperature_c: float) -> float:
    temperature_k = float(temperature_c) + 273.0
    return float(
        float(a_value) * math.exp(-(4184.0 * float(e_value)) / (R_GAS_CONSTANT * temperature_k))
    )


def scenario_features_strict(scenario_df: pd.DataFrame, is_train: bool) -> dict[str, Any]:
    row: dict[str, Any] = {
        "scenario_id": scenario_df["scenario_id"].iloc[0],
        GLOBAL_COLS[0]: scenario_df[GLOBAL_COLS[0]].iloc[0],
        GLOBAL_COLS[1]: scenario_df[GLOBAL_COLS[1]].iloc[0],
        GLOBAL_COLS[2]: scenario_df[GLOBAL_COLS[2]].iloc[0],
        GLOBAL_COLS[3]: scenario_df[GLOBAL_COLS[3]].iloc[0],
    }

    if is_train:
        row[TRAIN_TARGET_COLS[0]] = scenario_df[TRAIN_TARGET_COLS[0]].iloc[0]
        row[TRAIN_TARGET_COLS[1]] = scenario_df[TRAIN_TARGET_COLS[1]].iloc[0]

    temperature_c = float(scenario_df[GLOBAL_COLS[0]].iloc[0])

    ao_df = scenario_df[scenario_df["Компонент"].map(lambda x: component_is(x, "Антиоксидант"))].copy()
    dispersant_df = scenario_df[scenario_df["Компонент"].map(lambda x: component_is(x, "Дисперсант"))].copy()
    molybdenum_df = scenario_df[
        scenario_df["Компонент"].map(lambda x: component_is(x, "Соединение_молибдена"))
    ].copy()
    antiwear_df = scenario_df[
        scenario_df["Компонент"].map(lambda x: component_is(x, "Противоизносная_присадка"))
    ].copy()

    ao_synergy_value = 0.0
    if PROP_TYPE_AO in ao_df.columns and PROP_ACTIVE_NO in ao_df.columns:
        ao_work = ao_df[[PROP_TYPE_AO, PROP_ACTIVE_NO]].copy()
        ao_work[PROP_TYPE_AO] = ao_work[PROP_TYPE_AO].map(normalize_casefold)
        ao_work[PROP_ACTIVE_NO] = pd.to_numeric(ao_work[PROP_ACTIVE_NO], errors="coerce")

        phenol_values = ao_work.loc[ao_work[PROP_TYPE_AO] == AO_TYPE_PHENOL, PROP_ACTIVE_NO].dropna()
        dpa_values = ao_work.loc[ao_work[PROP_TYPE_AO] == AO_TYPE_DPA, PROP_ACTIVE_NO].dropna()

        ao_synergy_value = sum_product_cross(dpa_values, phenol_values)

    row["synergy_ao_phenol_x_diphenylamine_active_no"] = ao_synergy_value

    dpa_active_values = pd.Series(dtype=float)
    if PROP_TYPE_AO in ao_df.columns and PROP_ACTIVE_NO in ao_df.columns:
        ao_work = ao_df[[PROP_TYPE_AO, PROP_ACTIVE_NO]].copy()
        ao_work[PROP_TYPE_AO] = ao_work[PROP_TYPE_AO].map(normalize_casefold)
        ao_work[PROP_ACTIVE_NO] = pd.to_numeric(ao_work[PROP_ACTIVE_NO], errors="coerce")
        dpa_active_values = ao_work.loc[ao_work[PROP_TYPE_AO] == AO_TYPE_DPA, PROP_ACTIVE_NO].dropna()

    mo_values = safe_series(molybdenum_df, PROP_MO).dropna()
    row["synergy_ao_diphenylamine_active_no_x_mo"] = sum_product_cross(dpa_active_values, mo_values)

    zinc_values = safe_series(scenario_df, PROP_ZN).dropna()
    boron_values = safe_series(dispersant_df, PROP_BORON)
    boron_values = boron_values[boron_values != 0].dropna()
    row["synergy_zn_x_boron_dispersant"] = sum_product_cross(zinc_values, boron_values)

    sulfur_sum = float(safe_series(antiwear_df, PROP_S).dropna().sum())
    metal_sum = float(
        safe_series(scenario_df, PROP_CA).dropna().sum()
        + safe_series(scenario_df, PROP_CA_MG).dropna().sum()
    )
    row["synergy_aw_sulfur_x_ca_ca_mg"] = float(sulfur_sum * metal_sum)

    ao_k_values: list[float] = []
    ao_k_weighted_terms: list[float] = []

    if PROP_E_BOND in ao_df.columns and PROP_STERIC in ao_df.columns:
        e_values = pd.to_numeric(ao_df[PROP_E_BOND], errors="coerce")
        steric_values = pd.to_numeric(ao_df[PROP_STERIC], errors="coerce")

        if PROP_ACTIVE_NO in ao_df.columns:
            active_no_values = pd.to_numeric(ao_df[PROP_ACTIVE_NO], errors="coerce").fillna(0.0)
        else:
            active_no_values = pd.Series(0.0, index=ao_df.index)

        valid_mask = e_values.notna() & steric_values.notna()

        for idx in ao_df.index[valid_mask]:
            k_value = calculate_k(
                e_value=float(e_values.loc[idx]),
                a_value=float(steric_values.loc[idx]),
                temperature_c=temperature_c,
            )
            ao_k_values.append(k_value)
            ao_k_weighted_terms.append(k_value * float(active_no_values.loc[idx]))

    if len(ao_k_values) == 1:
        row["ao_k_avg_weighted_active_no"] = float(ao_k_values[0])
        row["ao_k_avg_arithmetic"] = float(ao_k_values[0])
    elif ao_k_values:
        row["ao_k_avg_weighted_active_no"] = float(np.sum(ao_k_weighted_terms))
        row["ao_k_avg_arithmetic"] = float(np.mean(ao_k_values))
    else:
        row["ao_k_avg_weighted_active_no"] = 0.0
        row["ao_k_avg_arithmetic"] = 0.0

    ionization_values = safe_series(ao_df, PROP_IONIZATION).dropna()
    row["ao_ionization_min"] = float(ionization_values.min()) if not ionization_values.empty else np.nan

    homo_values = safe_series(ao_df, PROP_HOMO).dropna()
    row["ao_homo_max"] = float(homo_values.max()) if not homo_values.empty else np.nan

    return row


def build_component_level_output(enriched_df: pd.DataFrame) -> pd.DataFrame:
    component_df = enriched_df.copy()

    drop_cols = [
        PROP_CHEM_POT,
        PROP_LUMO,
        PROP_E_BOND,
        PROP_STERIC,
    ]
    existing_drop_cols = [col for col in drop_cols if col in component_df.columns]
    component_df = component_df.drop(columns=existing_drop_cols)

    return component_df


def build_scenario_level_output(enriched_df: pd.DataFrame, is_train: bool) -> pd.DataFrame:
    rows = [
        scenario_features_strict(group, is_train=is_train)
        for _, group in enriched_df.groupby("scenario_id", sort=True)
    ]
    return pd.DataFrame(rows).sort_values("scenario_id").reset_index(drop=True)


def transform_dataset(
    mixture_df: pd.DataFrame,
    wide_exact: pd.DataFrame,
    wide_typical: pd.DataFrame,
    property_columns: list[str],
    is_train: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    enriched = enrich_mixture_with_properties(
        mixture_df=mixture_df,
        wide_exact=wide_exact,
        wide_typical=wide_typical,
        property_columns=property_columns,
    )

    component_level = build_component_level_output(enriched)
    scenario_level = build_scenario_level_output(enriched, is_train=is_train)

    return component_level, scenario_level


O2_ID_COL = "scenario_id"
O2_COMPONENT_COL = "Компонент"
O2_WEIGHT_COL = "Массовая доля, %"
O2_AO_TYPE_COL = "Тип АО"
O2_ACTIVE_NO_COL = "Активный Азот / Кислород, % масс. (N или O)"
O2_SUBSTRATE_COL = "Класс субстрата"
O2_TBN_ASTM_COL = "Щелочное число, ASTM D2896"
O2_TBN_GOST_COL = "Щелочное число, ГОСТ 11362"

O2_COLUMNS = [
    "o2_salicylate_tbn_x_amine_ao",
    "o2_salicylate_tbn_x_phenol_ao",
    "o2_salicylate_tbn_x_amine_x_phenol",
    "o2_ca_salicylate_present",
    "o2_mg_detergent_present",
]


def normalize_string_o2(value: Any) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).strip())


def parse_numeric_like_o2(value: Any) -> float:
    if pd.isna(value):
        return float("nan")
    text = normalize_string_o2(value)
    if not text:
        return float("nan")
    text = text.replace("−", "-").replace("–", "-").replace("—", "-")
    text = text.replace("%", "").replace(" ", "")
    if text.count(",") == 1 and text.count(".") == 0:
        text = text.replace(",", ".")
    match = re.search(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?", text.casefold())
    if not match:
        return float("nan")
    try:
        return float(match.group(0))
    except ValueError:
        return float("nan")


def series_numeric_o2(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(dtype=float)
    return df[col].map(parse_numeric_like_o2)


def weighted_mean_o2(values: pd.Series, weights: pd.Series) -> float:
    mask = values.notna() & weights.notna()
    if not mask.any():
        return 0.0
    v = values[mask].astype(float)
    w = weights[mask].astype(float)
    weight_sum = float(w.sum())
    if weight_sum <= 0.0:
        return float(v.mean())
    return float((v * w).sum() / weight_sum)


def build_o2_features(component_df: pd.DataFrame, scenario_df: pd.DataFrame) -> pd.DataFrame:
    component_groups = {
        scenario_id: group.reset_index(drop=True)
        for scenario_id, group in component_df.groupby(O2_ID_COL, sort=False)
    }
    rows: list[dict[str, Any]] = []

    for _, scenario_row in scenario_df.iterrows():
        scenario_id = scenario_row[O2_ID_COL]
        group = component_groups[scenario_id]
        component_names = group[O2_COMPONENT_COL].map(normalize_string_o2)

        ao_df = group[component_names.map(lambda x: x.startswith("Антиоксидант"))].copy()
        detergent_df = group[component_names.map(lambda x: x.startswith("Детергент"))].copy()

        ao_types = ao_df.get(O2_AO_TYPE_COL, pd.Series(index=ao_df.index, dtype=object)).map(normalize_string_o2)
        ao_active = series_numeric_o2(ao_df, O2_ACTIVE_NO_COL).fillna(0.0)
        det_substrate = detergent_df.get(
            O2_SUBSTRATE_COL,
            pd.Series(index=detergent_df.index, dtype=object),
        ).map(normalize_string_o2)
        det_weights = series_numeric_o2(detergent_df, O2_WEIGHT_COL).fillna(0.0)
        det_tbn = series_numeric_o2(detergent_df, O2_TBN_ASTM_COL)
        det_tbn = det_tbn.where(det_tbn.notna(), series_numeric_o2(detergent_df, O2_TBN_GOST_COL))

        salicylate_mask = det_substrate.str.contains("Салицилат", case=False, na=False)
        salicylate_tbn = weighted_mean_o2(det_tbn[salicylate_mask], det_weights[salicylate_mask])
        ca_salicylate_present = float(
            det_substrate.str.contains("Салицилат кальция", case=False, na=False).any()
        )
        mg_detergent_present = float(det_substrate.str.contains("магния", case=False, na=False).any())
        dpa_active = float(ao_active[ao_types == "Дифениламин"].sum())
        phenol_active = float(ao_active[ao_types == "Фенол"].sum())

        rows.append(
            {
                O2_ID_COL: scenario_id,
                "o2_salicylate_tbn_x_amine_ao": salicylate_tbn * dpa_active,
                "o2_salicylate_tbn_x_phenol_ao": salicylate_tbn * phenol_active,
                "o2_salicylate_tbn_x_amine_x_phenol": salicylate_tbn * dpa_active * phenol_active,
                "o2_ca_salicylate_present": ca_salicylate_present,
                "o2_mg_detergent_present": mg_detergent_present,
            }
        )

    return pd.DataFrame(rows)


def merge_features(base_df: pd.DataFrame, feature_df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    merged = base_df.merge(feature_df[[O2_ID_COL] + columns], on=O2_ID_COL, how="left")
    merged[columns] = merged[columns].fillna(0.0)
    return merged


def build_o2_feature_tables(
    train_path: Path,
    test_path: Path,
    props_path: Path,
    runtime_dir: Path,
) -> tuple[Path, Path, Path, Path]:
    for path in (train_path, test_path, props_path):
        if not path.exists():
            raise FileNotFoundError(path)

    train_df = read_csv_strict(train_path)
    test_df = read_csv_strict(test_path)
    props_df = read_csv_strict(props_path)

    assert_required_columns(train_df, REQUIRED_MIXTURE_COLS + TRAIN_TARGET_COLS, "train")
    assert_required_columns(test_df, REQUIRED_MIXTURE_COLS, "test")
    assert_required_columns(props_df, PROPS_REQUIRED_COLS, "properties")

    for df in (train_df, test_df):
        df["scenario_id"] = df["scenario_id"].map(normalize_string)
        df["Компонент"] = df["Компонент"].map(normalize_string)
        df["Наименование партии"] = df["Наименование партии"].map(normalize_string)

    props_df["Компонент"] = props_df["Компонент"].map(normalize_string)
    props_df["Наименование партии"] = props_df["Наименование партии"].map(normalize_string)
    props_df["Наименование показателя"] = props_df["Наименование показателя"].map(normalize_property_name)
    props_df["Значение показателя"] = props_df["Значение показателя"].map(normalize_property_value)

    wide_exact, wide_typical, property_columns = build_property_tables(props_df)
    component_train_df, scenario_train_df = transform_dataset(
        train_df,
        wide_exact,
        wide_typical,
        property_columns,
        True,
    )
    component_test_df, scenario_test_df = transform_dataset(
        test_df,
        wide_exact,
        wide_typical,
        property_columns,
        False,
    )

    o2_train = build_o2_features(component_train_df, scenario_train_df)
    o2_test = build_o2_features(component_test_df, scenario_test_df)
    scenario_train_aug = merge_features(scenario_train_df, o2_train, O2_COLUMNS)
    scenario_test_aug = merge_features(scenario_test_df, o2_test, O2_COLUMNS)

    runtime_dir.mkdir(parents=True, exist_ok=True)
    component_train_path = runtime_dir / "train_component_level_transformed.csv"
    component_test_path = runtime_dir / "test_component_level_transformed.csv"
    scenario_train_path = runtime_dir / "train_scenario_level_features_o2.csv"
    scenario_test_path = runtime_dir / "test_scenario_level_features_o2.csv"

    component_train_df.to_csv(component_train_path, index=False)
    component_test_df.to_csv(component_test_path, index=False)
    scenario_train_aug.to_csv(scenario_train_path, index=False)
    scenario_test_aug.to_csv(scenario_test_path, index=False)

    return component_train_path, component_test_path, scenario_train_path, scenario_test_path
