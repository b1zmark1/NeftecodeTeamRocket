from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


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
]


def log(message: str) -> None:
    print(f"[zdp-pairs] {message}")


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
        & frame["Массовая доля серы, ASTM D6481"].fillna(2.0).gt(1.0)
    )


def build_pair_table(train: pd.DataFrame, property_table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    mix = train.merge(property_table, on=["Компонент", "Наименование партии"], how="left")
    mix["is_detergent"] = mix["Компонент"].str.startswith("Детергент")
    mix["is_zdp_like"] = is_zdp_like(mix)

    scenario_targets = (
        mix.groupby("scenario_id")
        .agg({**{column: "first" for column in CONDITION_COLUMNS}, **{column: "first" for column in TARGET_COLUMNS}})
        .reset_index()
    )

    rows = []
    for scenario_id, group in mix.groupby("scenario_id"):
        detergents = sorted(group.loc[group["is_detergent"], "Компонент"].unique())
        zdps = sorted(group.loc[group["is_zdp_like"], "Компонент"].unique())
        if not detergents or not zdps:
            continue
        for detergent in detergents:
            for zdp in zdps:
                rows.append({"scenario_id": scenario_id, "detergent": detergent, "zdp_component": zdp})

    pair_presence = pd.DataFrame(rows).merge(scenario_targets, on="scenario_id", how="left")
    return pair_presence, scenario_targets


def build_pair_summary(pair_presence: pd.DataFrame, scenario_targets: pd.DataFrame) -> pd.DataFrame:
    pair_stats = (
        pair_presence.groupby(["detergent", "zdp_component"])
        .agg(
            n_scenarios=("scenario_id", "nunique"),
            mean_delta_visc=(TARGET_COLUMNS[0], "mean"),
            mean_oxidation=(TARGET_COLUMNS[1], "mean"),
        )
        .reset_index()
    )

    all_targets = scenario_targets[TARGET_COLUMNS]
    pair_stats["delta_visc_shift_vs_global"] = pair_stats["mean_delta_visc"] - all_targets[TARGET_COLUMNS[0]].mean()
    pair_stats["oxidation_shift_vs_global"] = pair_stats["mean_oxidation"] - all_targets[TARGET_COLUMNS[1]].mean()

    within_rows = []
    for (detergent, zdp_component), pair_rows in pair_presence.groupby(["detergent", "zdp_component"]):
        pair_scenarios = set(pair_rows["scenario_id"])
        for condition_values, cond_group in scenario_targets.groupby(CONDITION_COLUMNS):
            present = cond_group[cond_group["scenario_id"].isin(pair_scenarios)]
            absent = cond_group[~cond_group["scenario_id"].isin(pair_scenarios)]
            if present.empty or absent.empty:
                continue
            within_rows.append(
                {
                    "detergent": detergent,
                    "zdp_component": zdp_component,
                    "condition_key": "|".join(map(str, condition_values)),
                    "n_present": len(present),
                    "n_absent": len(absent),
                    "delta_visc_diff": present[TARGET_COLUMNS[0]].mean() - absent[TARGET_COLUMNS[0]].mean(),
                    "oxidation_diff": present[TARGET_COLUMNS[1]].mean() - absent[TARGET_COLUMNS[1]].mean(),
                }
            )
    within = pd.DataFrame(within_rows)
    if not within.empty:
        within_summary = (
            within.groupby(["detergent", "zdp_component"])
            .agg(
                comparable_condition_groups=("condition_key", "nunique"),
                avg_delta_visc_diff=("delta_visc_diff", "mean"),
                avg_oxidation_diff=("oxidation_diff", "mean"),
            )
            .reset_index()
        )
        pair_stats = pair_stats.merge(within_summary, on=["detergent", "zdp_component"], how="left")
    else:
        pair_stats["comparable_condition_groups"] = 0
        pair_stats["avg_delta_visc_diff"] = np.nan
        pair_stats["avg_oxidation_diff"] = np.nan

    pair_stats["antagonism_score"] = (
        pair_stats["avg_oxidation_diff"].fillna(0) + pair_stats["avg_delta_visc_diff"].fillna(0) / 10.0
    )
    return pair_stats.sort_values(["n_scenarios", "antagonism_score"], ascending=[False, False]), within


def save_outputs(pair_stats: pd.DataFrame, within: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pair_stats.to_csv(output_dir / "pair_summary.csv", index=False)
    within.to_csv(output_dir / "pair_within_condition_diffs.csv", index=False)

    top_pairs = pair_stats.head(15).copy()
    top_pairs["pair_label"] = top_pairs["detergent"] + " x " + top_pairs["zdp_component"]

    plt.figure(figsize=(12, 8))
    sns.barplot(data=top_pairs, x="n_scenarios", y="pair_label", color="steelblue")
    plt.title("Most frequent detergent x ZDP-like pairs")
    plt.tight_layout()
    plt.savefig(output_dir / "top_pairs_by_frequency.png", dpi=180, bbox_inches="tight")
    plt.close()

    scored = pair_stats.dropna(subset=["avg_delta_visc_diff", "avg_oxidation_diff"]).copy()
    if not scored.empty:
        scored = scored.sort_values("antagonism_score", ascending=False).head(20)
        scored["pair_label"] = scored["detergent"] + " x " + scored["zdp_component"]
        plt.figure(figsize=(10, 8))
        sns.scatterplot(
            data=scored,
            x="avg_delta_visc_diff",
            y="avg_oxidation_diff",
            size="n_scenarios",
            hue="detergent",
            sizes=(80, 400),
        )
        plt.axhline(0, color="grey", linewidth=1)
        plt.axvline(0, color="grey", linewidth=1)
        plt.title("Pair effect inside comparable conditions")
        plt.tight_layout()
        plt.savefig(output_dir / "pair_effect_scatter.png", dpi=180, bbox_inches="tight")
        plt.close()


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    output_dir = project_root / "baseline" / "interactions" / "zdp_detergent_pairs"

    train, props = load_data(project_root)
    property_table = build_property_table(props)
    pair_presence, scenario_targets = build_pair_table(train, property_table)
    pair_stats, within = build_pair_summary(pair_presence, scenario_targets)
    save_outputs(pair_stats, within, output_dir)
    log(f"pair_presence_rows={len(pair_presence)}, unique_pairs={pair_stats[['detergent','zdp_component']].drop_duplicates().shape[0]}")
    log(f"saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
