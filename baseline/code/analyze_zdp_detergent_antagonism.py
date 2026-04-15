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
    "Массовая доля фосфора, ASTM D6481",
    "Массовая доля серы, ASTM D6481",
    "Массовая доля кальция, ASTM D6481",
    "Щелочное число, ASTM D2896",
    "Щелочное число, ГОСТ 11362",
]


def log(message: str) -> None:
    print(f"[zdp-detergent] {message}")


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


def is_zdp_like(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["Компонент"].str.startswith("Противоизносная_присадка")
        & frame["Массовая доля цинка, ASTM D6481"].fillna(0).gt(0.5)
        & frame["Массовая доля фосфора, ASTM D6481"].fillna(0).gt(1.0)
        & (
            frame["Массовая доля серы, ASTM D6481"].fillna(2.0).gt(1.0)
        )
    )


def build_scenario_table(train: pd.DataFrame, property_table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    mix = train.merge(property_table, on=["Компонент", "Наименование партии"], how="left")
    scenario_total = mix.groupby("scenario_id")["Массовая доля, %"].transform("sum")
    mix["norm_share"] = mix["Массовая доля, %"] / scenario_total
    mix["is_detergent"] = mix["Компонент"].str.startswith("Детергент")
    mix["is_zdp_like"] = is_zdp_like(mix)
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
    scenario["zdp_share"] = mix.groupby("scenario_id").apply(
        lambda group: group.loc[group["is_zdp_like"], "norm_share"].sum(),
        include_groups=False,
    ).values
    scenario["weighted_zdp_zinc"] = mix.groupby("scenario_id").apply(
        lambda group: np.nansum(
            group.loc[group["is_zdp_like"], "norm_share"]
            * group.loc[group["is_zdp_like"], "Массовая доля цинка, ASTM D6481"].fillna(0)
        ),
        include_groups=False,
    ).values
    scenario["weighted_zdp_phosphorus"] = mix.groupby("scenario_id").apply(
        lambda group: np.nansum(
            group.loc[group["is_zdp_like"], "norm_share"]
            * group.loc[group["is_zdp_like"], "Массовая доля фосфора, ASTM D6481"].fillna(0)
        ),
        include_groups=False,
    ).values
    scenario["weighted_zdp_sulfur"] = mix.groupby("scenario_id").apply(
        lambda group: np.nansum(
            group.loc[group["is_zdp_like"], "norm_share"]
            * group.loc[group["is_zdp_like"], "Массовая доля серы, ASTM D6481"].fillna(0)
        ),
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
    scenario["zdp_detergent_interaction"] = scenario["zdp_share"] * scenario["detergent_share"]
    scenario["zdp_tbn_interaction"] = scenario["zdp_share"] * scenario["detergent_weighted_tbn"]
    scenario["zdp_ca_interaction"] = scenario["zdp_share"] * scenario["detergent_weighted_ca"]
    scenario["zdp_present"] = scenario["zdp_share"] > 0

    zdp_components = (
        mix.loc[mix["is_zdp_like"], ["Компонент", "Наименование партии", "Массовая доля цинка, ASTM D6481", "Массовая доля фосфора, ASTM D6481", "Массовая доля серы, ASTM D6481"]]
        .drop_duplicates()
        .sort_values(["Компонент", "Наименование партии"])
        .reset_index(drop=True)
    )
    return scenario, zdp_components


def save_heatmaps(scenario: pd.DataFrame, output_dir: Path) -> None:
    heatmap_frame = scenario.copy()
    heatmap_frame["detergent_bin"] = pd.qcut(
        heatmap_frame["detergent_share"].rank(method="first"),
        q=4,
        labels=["Q1", "Q2", "Q3", "Q4"],
    )
    heatmap_frame["zdp_bin"] = pd.qcut(
        heatmap_frame["zdp_share"].rank(method="first"),
        q=4,
        labels=["Q1", "Q2", "Q3", "Q4"],
    )

    count_pivot = heatmap_frame.pivot_table(
        index="detergent_bin",
        columns="zdp_bin",
        values="scenario_id",
        aggfunc="count",
        observed=False,
    )
    count_pivot.to_csv(output_dir / "zdp_detergent_bin_counts.csv")
    plt.figure(figsize=(6, 5))
    sns.heatmap(count_pivot, annot=True, fmt=".0f", cmap="Blues")
    plt.title("Counts by detergent share x ZDP-like share bins")
    plt.tight_layout()
    plt.savefig(output_dir / "zdp_detergent_bin_counts.png", dpi=180, bbox_inches="tight")
    plt.close()

    for target in TARGET_COLUMNS:
        pivot = heatmap_frame.pivot_table(
            index="detergent_bin",
            columns="zdp_bin",
            values=target,
            aggfunc="mean",
            observed=False,
        )
        pivot.to_csv(output_dir / f"{sanitize_name(target)}_zdp_detergent_heatmap.csv")
        plt.figure(figsize=(6, 5))
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="coolwarm")
        plt.title(f"Mean {target}\nby detergent share x ZDP-like share bins")
        plt.tight_layout()
        plt.savefig(
            output_dir / f"{sanitize_name(target)}_zdp_detergent_heatmap.png",
            dpi=180,
            bbox_inches="tight",
        )
        plt.close()


def save_within_condition_comparison(scenario: pd.DataFrame, output_dir: Path) -> None:
    subset = scenario[scenario["detergent_share"] > 0].copy()
    rows = []
    for condition_values, group in subset.groupby(CONDITION_COLUMNS):
        if group["zdp_present"].nunique() < 2:
            continue
        row = dict(zip(CONDITION_COLUMNS, condition_values))
        row["n_det_only"] = int((~group["zdp_present"]).sum())
        row["n_det_plus_zdp"] = int(group["zdp_present"].sum())
        for target in TARGET_COLUMNS:
            det_only = group.loc[~group["zdp_present"], target].mean()
            det_plus_zdp = group.loc[group["zdp_present"], target].mean()
            row[f"{sanitize_name(target)}_det_only"] = det_only
            row[f"{sanitize_name(target)}_det_plus_zdp"] = det_plus_zdp
            row[f"{sanitize_name(target)}_diff"] = det_plus_zdp - det_only
        rows.append(row)
    pd.DataFrame(rows).to_csv(output_dir / "within_condition_detergent_vs_zdp.csv", index=False)


def save_focused_shap(scenario: pd.DataFrame, output_dir: Path) -> None:
    feature_columns = CONDITION_COLUMNS + [
        "detergent_share",
        "zdp_share",
        "weighted_zdp_zinc",
        "weighted_zdp_phosphorus",
        "weighted_zdp_sulfur",
        "detergent_weighted_ca",
        "detergent_weighted_tbn",
        "zdp_detergent_interaction",
        "zdp_tbn_interaction",
        "zdp_ca_interaction",
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

        for feature in ["zdp_share", "detergent_share", "zdp_detergent_interaction", "zdp_tbn_interaction"]:
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
    output_dir = project_root / "baseline" / "interactions" / "zdp_detergent"
    output_dir.mkdir(parents=True, exist_ok=True)

    train, props = load_data(project_root)
    property_table = build_property_table(props)
    scenario, zdp_components = build_scenario_table(train, property_table)

    zdp_components.to_csv(output_dir / "zdp_like_components.csv", index=False)
    scenario.to_csv(output_dir / "scenario_level_zdp_detergent_features.csv", index=False)
    save_heatmaps(scenario, output_dir)
    save_within_condition_comparison(scenario, output_dir)
    save_focused_shap(scenario, output_dir)
    log(f"scenario table shape={scenario.shape}, zdp_like_components={len(zdp_components)}")
    log(f"saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
