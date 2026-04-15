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

TOP_BASE_OILS = 6


def log(message: str) -> None:
    print(f"[amine-phenol] {message}")


def sanitize_name(value: str) -> str:
    return "".join(char.lower() if char.isalnum() else "_" for char in value).strip("_")


def load_data(project_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(project_root / "docs" / "daimler_mixtures_train.csv")
    props = pd.read_csv(project_root / "docs" / "daimler_component_properties.csv")
    return train, props


def build_antioxidant_type_table(props: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ao = props[
        props["Наименование показателя"].eq("Тип АО")
        & props["Компонент"].astype(str).str.startswith("Антиоксидант")
    ][["Компонент", "Наименование партии", "Значение показателя"]].copy()

    measured = ao[ao["Наименование партии"].astype(str).str.lower() != "typical"].rename(
        columns={"Значение показателя": "Тип АО"}
    )
    typical = (
        ao[ao["Наименование партии"].astype(str).str.lower() == "typical"][["Компонент", "Значение показателя"]]
        .drop_duplicates(subset=["Компонент"])
        .rename(columns={"Значение показателя": "Тип АО"})
    )
    return measured, typical


def build_scenario_table(
    train: pd.DataFrame,
    ao_measured: pd.DataFrame,
    ao_typical: pd.DataFrame,
) -> pd.DataFrame:
    mix = train.merge(ao_measured, on=["Компонент", "Наименование партии"], how="left")
    mix = mix.merge(ao_typical.rename(columns={"Тип АО": "Тип АО typical"}), on="Компонент", how="left")
    mix["Тип АО"] = mix["Тип АО"].fillna(mix["Тип АО typical"])

    scenario_total = mix.groupby("scenario_id")["Массовая доля, %"].transform("sum")
    mix["norm_share"] = mix["Массовая доля, %"] / scenario_total
    mix["is_base_oil"] = mix["Компонент"].str.startswith("Базовое_масло")
    mix["is_amine"] = mix["Тип АО"].astype(str).str.contains("Дифениламин", case=False, na=False)
    mix["is_phenol"] = mix["Тип АО"].astype(str).str.contains("Фенол", case=False, na=False)

    top_base_oils = (
        mix.loc[mix["is_base_oil"], "Компонент"].value_counts().head(TOP_BASE_OILS).index.tolist()
    )

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
    scenario["amine_share"] = mix.groupby("scenario_id").apply(
        lambda group: group.loc[group["is_amine"], "norm_share"].sum(),
        include_groups=False,
    ).values
    scenario["phenol_share"] = mix.groupby("scenario_id").apply(
        lambda group: group.loc[group["is_phenol"], "norm_share"].sum(),
        include_groups=False,
    ).values
    scenario["ao_total_share"] = scenario["amine_share"] + scenario["phenol_share"]
    scenario["amine_phenol_interaction"] = scenario["amine_share"] * scenario["phenol_share"]
    scenario["amine_to_phenol_ratio"] = scenario["amine_share"] / (scenario["phenol_share"] + 1e-6)
    scenario["both_ao_present"] = (scenario["amine_share"] > 0) & (scenario["phenol_share"] > 0)
    scenario["ao_group"] = np.select(
        [
            (scenario["amine_share"] > 0) & (scenario["phenol_share"] > 0),
            (scenario["amine_share"] > 0) & (scenario["phenol_share"] == 0),
            (scenario["amine_share"] == 0) & (scenario["phenol_share"] > 0),
        ],
        ["both", "amine_only", "phenol_only"],
        default="none",
    )

    for base_oil in top_base_oils:
        scenario[f"share__{base_oil}"] = mix.groupby("scenario_id").apply(
            lambda group, base_oil=base_oil: group.loc[group["Компонент"].eq(base_oil), "norm_share"].sum(),
            include_groups=False,
        ).values

    base_oil_shares = (
        mix.loc[mix["is_base_oil"], ["scenario_id", "Компонент", "norm_share"]]
        .sort_values(["scenario_id", "norm_share"], ascending=[True, False])
        .drop_duplicates(subset=["scenario_id"])
        .rename(columns={"Компонент": "dominant_base_oil", "norm_share": "dominant_base_oil_share"})
    )
    scenario = scenario.merge(base_oil_shares, on="scenario_id", how="left")
    return scenario


def save_heatmaps(scenario: pd.DataFrame, output_dir: Path) -> None:
    frame = scenario.copy()
    frame["amine_bin"] = pd.qcut(frame["amine_share"].rank(method="first"), q=4, labels=["Q1", "Q2", "Q3", "Q4"])
    frame["phenol_bin"] = pd.qcut(frame["phenol_share"].rank(method="first"), q=4, labels=["Q1", "Q2", "Q3", "Q4"])

    count_pivot = frame.pivot_table(
        index="phenol_bin", columns="amine_bin", values="scenario_id", aggfunc="count", observed=False
    )
    count_pivot.to_csv(output_dir / "amine_phenol_bin_counts.csv")
    plt.figure(figsize=(6, 5))
    sns.heatmap(count_pivot, annot=True, fmt=".0f", cmap="Blues")
    plt.title("Counts by phenol share x amine share bins")
    plt.tight_layout()
    plt.savefig(output_dir / "amine_phenol_bin_counts.png", dpi=180, bbox_inches="tight")
    plt.close()

    for target in TARGET_COLUMNS:
        pivot = frame.pivot_table(
            index="phenol_bin", columns="amine_bin", values=target, aggfunc="mean", observed=False
        )
        pivot.to_csv(output_dir / f"{sanitize_name(target)}_amine_phenol_heatmap.csv")
        plt.figure(figsize=(6, 5))
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="coolwarm")
        plt.title(f"Mean {target}\nby phenol share x amine share bins")
        plt.tight_layout()
        plt.savefig(output_dir / f"{sanitize_name(target)}_amine_phenol_heatmap.png", dpi=180, bbox_inches="tight")
        plt.close()


def save_within_condition_comparison(scenario: pd.DataFrame, output_dir: Path) -> None:
    subset = scenario[scenario["ao_group"] != "none"].copy()
    rows = []
    for condition_values, group in subset.groupby(CONDITION_COLUMNS):
        if group["ao_group"].nunique() < 2:
            continue
        row = dict(zip(CONDITION_COLUMNS, condition_values))
        for group_name in ["both", "amine_only", "phenol_only"]:
            row[f"n_{group_name}"] = int((group["ao_group"] == group_name).sum())
        for target in TARGET_COLUMNS:
            for group_name in ["both", "amine_only", "phenol_only"]:
                group_values = group.loc[group["ao_group"] == group_name, target]
                row[f"{sanitize_name(target)}_{group_name}"] = group_values.mean() if not group_values.empty else np.nan
            row[f"{sanitize_name(target)}_both_minus_amine_only"] = row[f"{sanitize_name(target)}_both"] - row[f"{sanitize_name(target)}_amine_only"]
            row[f"{sanitize_name(target)}_both_minus_phenol_only"] = row[f"{sanitize_name(target)}_both"] - row[f"{sanitize_name(target)}_phenol_only"]
        rows.append(row)
    pd.DataFrame(rows).to_csv(output_dir / "within_condition_ao_group_comparison.csv", index=False)


def save_base_oil_summary(scenario: pd.DataFrame, output_dir: Path) -> None:
    subset = scenario[scenario["both_ao_present"]].copy()
    summary = (
        subset.groupby("dominant_base_oil")
        .agg(
            n_scenarios=("scenario_id", "nunique"),
            mean_amine_share=("amine_share", "mean"),
            mean_phenol_share=("phenol_share", "mean"),
            mean_delta_visc=(TARGET_COLUMNS[0], "mean"),
            mean_oxidation=(TARGET_COLUMNS[1], "mean"),
        )
        .sort_values("n_scenarios", ascending=False)
        .reset_index()
    )
    summary.to_csv(output_dir / "both_ao_by_dominant_base_oil.csv", index=False)


def save_focused_shap(scenario: pd.DataFrame, output_dir: Path) -> None:
    base_features = [column for column in scenario.columns if column.startswith("share__базовое_масло_")]
    feature_columns = CONDITION_COLUMNS + [
        "amine_share",
        "phenol_share",
        "ao_total_share",
        "amine_phenol_interaction",
        "amine_to_phenol_ratio",
    ] + base_features

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
        shap.summary_plot(shap_values, scenario[feature_columns], plot_type="bar", max_display=12, show=False)
        plt.tight_layout()
        plt.savefig(output_dir / f"{sanitize_name(target)}_focused_shap_bar.png", dpi=180, bbox_inches="tight")
        plt.close()

        for feature in ["amine_share", "phenol_share", "amine_phenol_interaction", "amine_to_phenol_ratio"]:
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

        summary[target] = importance.head(12).to_dict(orient="records")

    with open(output_dir / "focused_shap_summary.json", "w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    output_dir = project_root / "baseline" / "interactions" / "amine_phenol_baseoil"
    output_dir.mkdir(parents=True, exist_ok=True)

    train, props = load_data(project_root)
    ao_measured, ao_typical = build_antioxidant_type_table(props)
    scenario = build_scenario_table(train, ao_measured, ao_typical)
    scenario.to_csv(output_dir / "scenario_level_amine_phenol_baseoil.csv", index=False)
    save_heatmaps(scenario, output_dir)
    save_within_condition_comparison(scenario, output_dir)
    save_base_oil_summary(scenario, output_dir)
    save_focused_shap(scenario, output_dir)
    log(f"scenario table shape={scenario.shape}")
    log(f"saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
