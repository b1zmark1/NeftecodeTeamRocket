from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from catboost import CatBoostRegressor


TARGET_COLUMNS = [
    "Delta Kin. Viscosity KV100 - relative | - Daimler Oxidation Test (DOT), %",
    "Oxidation EOT | DIN 51453 Daimler Oxidation Test (DOT), A/cm",
]

CONDITION_COLUMNS = [
    "Температура испытания | ASTM D445 Daimler Oxidation Test (DOT), °C",
    "Время испытания | - Daimler Oxidation Test (DOT), ч",
    "Количество биотоплива | - Daimler Oxidation Test (DOT), % масс",
    "Дозировка катализатора, категория",
]

PROPERTY_COLUMNS = [
    "% масс. (Mo)",
    "Тип АО",
    "Активный Азот / Кислород, % масс. (N или O)",
    "Содержание Азота",
    "Общее содержание азота | ASTM D3228",
]


def log(message: str) -> None:
    print(f"[moly-amine] {message}")


def sanitize_name(value: str) -> str:
    return "".join(char.lower() if char.isalnum() else "_" for char in value).strip("_")


def load_data(project_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(project_root / "docs" / "daimler_mixtures_train.csv")
    props = pd.read_csv(project_root / "docs" / "daimler_component_properties.csv")
    props["value_num"] = pd.to_numeric(
        props["Значение показателя"].astype(str).str.replace(",", ".", regex=False),
        errors="coerce",
    )
    return train, props


def build_property_tables(props: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    relevant = props[props["Наименование показателя"].isin(PROPERTY_COLUMNS)].copy()

    numeric_columns = [
        "% масс. (Mo)",
        "Активный Азот / Кислород, % масс. (N или O)",
        "Содержание Азота",
        "Общее содержание азота | ASTM D3228",
    ]
    text_columns = ["Тип АО"]

    measured_numeric = (
        relevant[
            (relevant["Наименование партии"].astype(str).str.lower() != "typical")
            & (relevant["Наименование показателя"].isin(numeric_columns))
        ]
        .pivot_table(
            index=["Компонент", "Наименование партии"],
            columns="Наименование показателя",
            values="value_num",
            aggfunc="mean",
        )
        .reset_index()
    )
    measured_text = (
        relevant[
            (relevant["Наименование партии"].astype(str).str.lower() != "typical")
            & (relevant["Наименование показателя"].isin(text_columns))
        ][["Компонент", "Наименование партии", "Наименование показателя", "Значение показателя"]]
        .drop_duplicates()
        .pivot(index=["Компонент", "Наименование партии"], columns="Наименование показателя", values="Значение показателя")
        .reset_index()
    )
    measured = measured_numeric.merge(measured_text, on=["Компонент", "Наименование партии"], how="outer")

    typical_numeric = (
        relevant[
            (relevant["Наименование партии"].astype(str).str.lower() == "typical")
            & (relevant["Наименование показателя"].isin(numeric_columns))
        ]
        .pivot_table(
            index="Компонент",
            columns="Наименование показателя",
            values="value_num",
            aggfunc="mean",
        )
        .add_suffix("__typical")
        .reset_index()
    )
    typical_text = (
        relevant[
            (relevant["Наименование партии"].astype(str).str.lower() == "typical")
            & (relevant["Наименование показателя"].isin(text_columns))
        ][["Компонент", "Наименование показателя", "Значение показателя"]]
        .drop_duplicates()
        .pivot(index="Компонент", columns="Наименование показателя", values="Значение показателя")
        .add_suffix("__typical")
        .reset_index()
    )
    typical = typical_numeric.merge(typical_text, on="Компонент", how="outer")
    return measured, typical


def build_scenario_table(train: pd.DataFrame, measured_table: pd.DataFrame, typical_table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    mix = train.merge(measured_table, on=["Компонент", "Наименование партии"], how="left")
    mix = mix.merge(typical_table, on="Компонент", how="left")
    for column in PROPERTY_COLUMNS:
        typical_column = f"{column}__typical"
        if column not in mix.columns:
            mix[column] = np.nan
        if typical_column in mix.columns:
            mix[column] = mix[column].fillna(mix[typical_column])
    scenario_total = mix.groupby("scenario_id")["Массовая доля, %"].transform("sum")
    mix["norm_share"] = mix["Массовая доля, %"] / scenario_total

    mix["is_moly"] = mix["Компонент"].str.startswith("Соединение_молибдена")
    mix["is_amine_antioxidant"] = (
        mix["Компонент"].str.startswith("Антиоксидант")
        & mix["Тип АО"].astype(str).str.contains("Дифениламин", case=False, na=False)
    )
    mix["nitrogen_proxy"] = mix[
        ["Активный Азот / Кислород, % масс. (N или O)", "Содержание Азота", "Общее содержание азота | ASTM D3228"]
    ].max(axis=1)

    scenario = (
        mix.groupby("scenario_id")
        .agg(
            {
                **{column: "first" for column in CONDITION_COLUMNS},
                **{column: "first" for column in TARGET_COLUMNS},
            }
        )
        .reset_index()
    )
    scenario["moly_share"] = mix.groupby("scenario_id").apply(
        lambda group: group.loc[group["is_moly"], "norm_share"].sum(),
        include_groups=False,
    ).values
    scenario["amine_ao_share"] = mix.groupby("scenario_id").apply(
        lambda group: group.loc[group["is_amine_antioxidant"], "norm_share"].sum(),
        include_groups=False,
    ).values
    scenario["weighted_mo"] = mix.groupby("scenario_id").apply(
        lambda group: np.nansum(group.loc[group["is_moly"], "norm_share"] * group.loc[group["is_moly"], "% масс. (Mo)"].fillna(0)),
        include_groups=False,
    ).values
    scenario["weighted_amine_n"] = mix.groupby("scenario_id").apply(
        lambda group: np.nansum(
            group.loc[group["is_amine_antioxidant"], "norm_share"]
            * group.loc[group["is_amine_antioxidant"], "nitrogen_proxy"].fillna(0)
        ),
        include_groups=False,
    ).values
    scenario["moly_amine_interaction"] = scenario["moly_share"] * scenario["amine_ao_share"]
    scenario["moly_n_interaction"] = scenario["moly_share"] * scenario["weighted_amine_n"]
    scenario["moly_present"] = scenario["moly_share"] > 0
    scenario["amine_present"] = scenario["amine_ao_share"] > 0

    relevant_components = mix.loc[
        mix["is_moly"] | mix["is_amine_antioxidant"],
        ["Компонент", "Наименование партии", "% масс. (Mo)", "Тип АО", "nitrogen_proxy"],
    ].drop_duplicates().sort_values(["Компонент", "Наименование партии"])
    return scenario, relevant_components


def save_heatmaps(scenario: pd.DataFrame, output_dir: Path) -> None:
    frame = scenario.copy()
    frame["moly_bin"] = pd.qcut(frame["moly_share"].rank(method="first"), q=4, labels=["Q1", "Q2", "Q3", "Q4"])
    frame["amine_bin"] = pd.qcut(frame["amine_ao_share"].rank(method="first"), q=4, labels=["Q1", "Q2", "Q3", "Q4"])

    count_pivot = frame.pivot_table(
        index="amine_bin", columns="moly_bin", values="scenario_id", aggfunc="count", observed=False
    )
    count_pivot.to_csv(output_dir / "moly_amine_bin_counts.csv")
    plt.figure(figsize=(6, 5))
    sns.heatmap(count_pivot, annot=True, fmt=".0f", cmap="Blues")
    plt.title("Counts by amine AO share x moly share bins")
    plt.tight_layout()
    plt.savefig(output_dir / "moly_amine_bin_counts.png", dpi=180, bbox_inches="tight")
    plt.close()

    for target in TARGET_COLUMNS:
        pivot = frame.pivot_table(
            index="amine_bin", columns="moly_bin", values=target, aggfunc="mean", observed=False
        )
        pivot.to_csv(output_dir / f"{sanitize_name(target)}_moly_amine_heatmap.csv")
        plt.figure(figsize=(6, 5))
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="coolwarm")
        plt.title(f"Mean {target}\nby amine AO share x moly share bins")
        plt.tight_layout()
        plt.savefig(output_dir / f"{sanitize_name(target)}_moly_amine_heatmap.png", dpi=180, bbox_inches="tight")
        plt.close()


def save_within_condition_comparison(scenario: pd.DataFrame, output_dir: Path) -> None:
    subset = scenario[scenario["amine_ao_share"] > 0].copy()
    rows = []
    for condition_values, group in subset.groupby(CONDITION_COLUMNS):
        if group["moly_present"].nunique() < 2:
            continue
        row = dict(zip(CONDITION_COLUMNS, condition_values))
        row["n_amine_only"] = int((~group["moly_present"]).sum())
        row["n_amine_plus_moly"] = int(group["moly_present"].sum())
        for target in TARGET_COLUMNS:
            amine_only = group.loc[~group["moly_present"], target].mean()
            amine_plus_moly = group.loc[group["moly_present"], target].mean()
            row[f"{sanitize_name(target)}_amine_only"] = amine_only
            row[f"{sanitize_name(target)}_amine_plus_moly"] = amine_plus_moly
            row[f"{sanitize_name(target)}_diff"] = amine_plus_moly - amine_only
        rows.append(row)
    pd.DataFrame(rows).to_csv(output_dir / "within_condition_amine_vs_moly.csv", index=False)


def save_focused_shap(scenario: pd.DataFrame, output_dir: Path) -> None:
    feature_columns = CONDITION_COLUMNS + [
        "moly_share",
        "amine_ao_share",
        "weighted_mo",
        "weighted_amine_n",
        "moly_amine_interaction",
        "moly_n_interaction",
    ]
    summary = {}
    for target in TARGET_COLUMNS:
        model = CatBoostRegressor(
            iterations=500,
            learning_rate=0.04,
            depth=5,
            loss_function="RMSE",
            random_seed=42,
            allow_writing_files=False,
            verbose=False,
        )
        model.fit(scenario[feature_columns], scenario[target])
        explainer = shap.TreeExplainer(model)
        shap_values = np.asarray(explainer.shap_values(scenario[feature_columns]))

        importance = pd.DataFrame(
            {"feature": feature_columns, "mean_abs_shap": np.abs(shap_values).mean(axis=0)}
        ).sort_values("mean_abs_shap", ascending=False)
        importance.to_csv(output_dir / f"{sanitize_name(target)}_focused_shap_importance.csv", index=False)

        plt.figure(figsize=(9, 6))
        shap.summary_plot(shap_values, scenario[feature_columns], plot_type="bar", max_display=10, show=False)
        plt.tight_layout()
        plt.savefig(output_dir / f"{sanitize_name(target)}_focused_shap_bar.png", dpi=180, bbox_inches="tight")
        plt.close()

        for feature in ["moly_share", "amine_ao_share", "moly_amine_interaction", "moly_n_interaction"]:
            plt.figure(figsize=(8, 6))
            shap.dependence_plot(
                feature,
                shap_values,
                scenario[feature_columns],
                interaction_index="auto",
                show=False,
            )
            plt.tight_layout()
            plt.savefig(
                output_dir / f"{sanitize_name(target)}_dependence_{sanitize_name(feature)}.png",
                dpi=180,
                bbox_inches="tight",
            )
            plt.close()

        summary[target] = importance.head(10).to_dict(orient="records")

    with open(output_dir / "focused_shap_summary.json", "w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    output_dir = project_root / "baseline" / "interactions" / "moly_amine_antioxidants"
    output_dir.mkdir(parents=True, exist_ok=True)

    train, props = load_data(project_root)
    measured_table, typical_table = build_property_tables(props)
    scenario, relevant_components = build_scenario_table(train, measured_table, typical_table)
    relevant_components.to_csv(output_dir / "moly_amine_components.csv", index=False)
    scenario.to_csv(output_dir / "scenario_level_moly_amine_features.csv", index=False)
    save_heatmaps(scenario, output_dir)
    save_within_condition_comparison(scenario, output_dir)
    save_focused_shap(scenario, output_dir)
    log(f"scenario table shape={scenario.shape}, relevant_components={len(relevant_components)}")
    log(f"saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
