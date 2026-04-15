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
    "Массовая доля цинка, ASTM D6481",
    "Массовая доля кальция, ASTM D6481",
    "Щелочное число, ASTM D2896",
    "Щелочное число, ГОСТ 11362",
]


def log(message: str) -> None:
    print(f"[zinc-detergent] {message}")


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


def build_property_table(props: pd.DataFrame) -> pd.DataFrame:
    props = props[props["Наименование показателя"].isin(PROPERTY_COLUMNS)].copy()
    measured = (
        props[props["Наименование партии"].astype(str).str.lower() != "typical"]
        .pivot_table(
            index=["Компонент", "Наименование партии"],
            columns="Наименование показателя",
            values="value_num",
            aggfunc="mean",
        )
        .reset_index()
    )
    typical = (
        props[props["Наименование партии"].astype(str).str.lower() == "typical"]
        .pivot_table(
            index="Компонент",
            columns="Наименование показателя",
            values="value_num",
            aggfunc="mean",
        )
        .add_suffix("__typical")
        .reset_index()
    )
    property_table = measured.merge(typical, on="Компонент", how="left")
    for column in PROPERTY_COLUMNS:
        if column not in property_table.columns:
            property_table[column] = np.nan
        typical_column = f"{column}__typical"
        if typical_column in property_table.columns:
            property_table[column] = property_table[column].fillna(property_table[typical_column])
            property_table = property_table.drop(columns=[typical_column])
    return property_table


def build_scenario_table(train: pd.DataFrame, property_table: pd.DataFrame) -> pd.DataFrame:
    mix = train.merge(property_table, on=["Компонент", "Наименование партии"], how="left")
    scenario_total = mix.groupby("scenario_id")["Массовая доля, %"].transform("sum")
    mix["norm_share"] = mix["Массовая доля, %"] / scenario_total
    mix["is_detergent"] = mix["Компонент"].str.startswith("Детергент")
    mix["is_zinc_rich"] = mix["Массовая доля цинка, ASTM D6481"].fillna(0).gt(0.5)
    mix["det_tbn"] = mix[["Щелочное число, ASTM D2896", "Щелочное число, ГОСТ 11362"]].max(axis=1)

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
    scenario["detergent_share"] = mix.groupby("scenario_id").apply(
        lambda group: group.loc[group["is_detergent"], "norm_share"].sum(),
        include_groups=False,
    ).values
    scenario["weighted_zinc"] = mix.groupby("scenario_id").apply(
        lambda group: np.nansum(group["norm_share"] * group["Массовая доля цинка, ASTM D6481"].fillna(0)),
        include_groups=False,
    ).values
    scenario["zinc_rich_share"] = mix.groupby("scenario_id").apply(
        lambda group: group.loc[group["is_zinc_rich"], "norm_share"].sum(),
        include_groups=False,
    ).values
    scenario["detergent_weighted_ca"] = mix.groupby("scenario_id").apply(
        lambda group: np.nansum(
            group.loc[group["is_detergent"], "norm_share"]
            * group.loc[group["is_detergent"], "Массовая доля кальция, ASTM D6481"].fillna(0)
        ),
        include_groups=False,
    ).values
    scenario["detergent_weighted_tbn"] = mix.groupby("scenario_id").apply(
        lambda group: np.nansum(
            group.loc[group["is_detergent"], "norm_share"]
            * group.loc[group["is_detergent"], "det_tbn"].fillna(0)
        ),
        include_groups=False,
    ).values
    scenario["detergent_zinc_interaction"] = scenario["detergent_share"] * scenario["weighted_zinc"]
    scenario["detergent_zinc_presence_interaction"] = scenario["detergent_share"] * scenario["zinc_rich_share"]
    return scenario


def save_heatmaps(scenario: pd.DataFrame, output_dir: Path) -> None:
    heatmap_frame = scenario.copy()
    heatmap_frame["detergent_bin"] = pd.qcut(
        heatmap_frame["detergent_share"].rank(method="first"),
        q=4,
        labels=["Q1", "Q2", "Q3", "Q4"],
    )
    heatmap_frame["zinc_bin"] = pd.qcut(
        heatmap_frame["weighted_zinc"].rank(method="first"),
        q=4,
        labels=["Q1", "Q2", "Q3", "Q4"],
    )

    count_pivot = heatmap_frame.pivot_table(
        index="detergent_bin", columns="zinc_bin", values="scenario_id", aggfunc="count"
    )
    count_pivot.to_csv(output_dir / "zinc_detergent_bin_counts.csv")
    plt.figure(figsize=(6, 5))
    sns.heatmap(count_pivot, annot=True, fmt=".0f", cmap="Blues")
    plt.title("Counts by detergent share x weighted zinc bins")
    plt.tight_layout()
    plt.savefig(output_dir / "zinc_detergent_bin_counts.png", dpi=180, bbox_inches="tight")
    plt.close()

    for target in TARGET_COLUMNS:
        pivot = heatmap_frame.pivot_table(
            index="detergent_bin", columns="zinc_bin", values=target, aggfunc="mean"
        )
        pivot.to_csv(output_dir / f"{sanitize_name(target)}_zinc_detergent_heatmap.csv")
        plt.figure(figsize=(6, 5))
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="coolwarm")
        plt.title(f"Mean {target}\nby detergent share x weighted zinc bins")
        plt.tight_layout()
        plt.savefig(
            output_dir / f"{sanitize_name(target)}_zinc_detergent_heatmap.png",
            dpi=180,
            bbox_inches="tight",
        )
        plt.close()


def save_within_condition_comparison(scenario: pd.DataFrame, output_dir: Path) -> None:
    subset = scenario[scenario["detergent_share"] > 0].copy()
    subset["zinc_flag"] = subset["zinc_rich_share"] > 0
    rows = []
    for condition_values, group in subset.groupby(CONDITION_COLUMNS):
        if group["zinc_flag"].nunique() < 2:
            continue
        row = dict(zip(CONDITION_COLUMNS, condition_values))
        row["n_det_only"] = int((~group["zinc_flag"]).sum())
        row["n_det_plus_zinc"] = int(group["zinc_flag"].sum())
        for target in TARGET_COLUMNS:
            det_only = group.loc[~group["zinc_flag"], target].mean()
            det_plus_zinc = group.loc[group["zinc_flag"], target].mean()
            row[f"{sanitize_name(target)}_det_only"] = det_only
            row[f"{sanitize_name(target)}_det_plus_zinc"] = det_plus_zinc
            row[f"{sanitize_name(target)}_diff"] = det_plus_zinc - det_only
        rows.append(row)
    comparison = pd.DataFrame(rows)
    comparison.to_csv(output_dir / "within_condition_detergent_vs_zinc.csv", index=False)


def save_focused_shap(scenario: pd.DataFrame, output_dir: Path) -> None:
    feature_columns = CONDITION_COLUMNS + [
        "detergent_share",
        "weighted_zinc",
        "zinc_rich_share",
        "detergent_weighted_ca",
        "detergent_weighted_tbn",
        "detergent_zinc_interaction",
        "detergent_zinc_presence_interaction",
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
            {
                "feature": feature_columns,
                "mean_abs_shap": np.abs(shap_values).mean(axis=0),
            }
        ).sort_values("mean_abs_shap", ascending=False)
        importance.to_csv(output_dir / f"{sanitize_name(target)}_focused_shap_importance.csv", index=False)

        plt.figure(figsize=(9, 6))
        shap.summary_plot(shap_values, scenario[feature_columns], plot_type="bar", max_display=10, show=False)
        plt.tight_layout()
        plt.savefig(output_dir / f"{sanitize_name(target)}_focused_shap_bar.png", dpi=180, bbox_inches="tight")
        plt.close()

        for feature in ["detergent_share", "weighted_zinc", "detergent_zinc_interaction"]:
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
    output_dir = project_root / "baseline" / "interactions" / "zinc_detergent"
    output_dir.mkdir(parents=True, exist_ok=True)

    train, props = load_data(project_root)
    property_table = build_property_table(props)
    scenario = build_scenario_table(train, property_table)
    log(f"scenario table shape={scenario.shape}")

    save_heatmaps(scenario, output_dir)
    save_within_condition_comparison(scenario, output_dir)
    save_focused_shap(scenario, output_dir)
    scenario.to_csv(output_dir / "scenario_level_zinc_detergent_features.csv", index=False)
    log(f"saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
