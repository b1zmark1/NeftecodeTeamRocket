from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import seaborn as sns
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold


RANDOM_STATE = 42
N_SPLITS = 5
TOP_COMPONENTS = 25
TOP_FAMILIES = 12
SHAP_MAX_DISPLAY = 50
TOP_DEPENDENCE_PLOTS = 10
TOP_SCATTER_PLOTS = 10
TOP_INTERACTIONS = 8

TARGET_COLUMNS = [
    "Delta Kin. Viscosity KV100 - relative | - Daimler Oxidation Test (DOT), %",
    "Oxidation EOT | DIN 51453 Daimler Oxidation Test (DOT), A/cm",
]

SCENARIO_COLUMNS = [
    "Температура испытания | ASTM D445 Daimler Oxidation Test (DOT), °C",
    "Время испытания | - Daimler Oxidation Test (DOT), ч",
    "Количество биотоплива | - Daimler Oxidation Test (DOT), % масс",
    "Дозировка катализатора, категория",
]

MIXTURE_COLUMNS = ["scenario_id", "Компонент", "Наименование партии", "Массовая доля, %"]


def log(message: str) -> None:
    print(f"[baseline] {message}")


def sanitize_name(value: str) -> str:
    value = re.sub(r"[^0-9A-Za-zА-Яа-я_]+", "_", value)
    return re.sub(r"_+", "_", value).strip("_").lower()


def sign_log1p(values: pd.Series) -> pd.Series:
    return np.sign(values) * np.log1p(np.abs(values))


def inverse_sign_log1p(values: np.ndarray) -> np.ndarray:
    return np.sign(values) * np.expm1(np.abs(values))


def load_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(data_dir / "daimler_mixtures_train.csv")
    test = pd.read_csv(data_dir / "daimler_mixtures_test.csv")
    props = pd.read_csv(data_dir / "daimler_component_properties.csv")
    log(
        "loaded data: "
        f"train_rows={len(train)}, train_scenarios={train['scenario_id'].nunique()}, "
        f"test_rows={len(test)}, test_scenarios={test['scenario_id'].nunique()}, "
        f"property_rows={len(props)}"
    )
    return train, test, props


def build_property_label_map(property_alias: dict[str, str]) -> dict[str, str]:
    return {alias: source_name for source_name, alias in property_alias.items()}


def make_feature_display_name(feature_name: str, property_label_map: dict[str, str]) -> str:
    match = re.match(r"^(prop_\d+)(__(wmean|max|coverage))$", feature_name)
    if match:
        property_alias = match.group(1)
        suffix = match.group(2)
        suffix_map = {
            "__wmean": "weighted_mean",
            "__max": "max",
            "__coverage": "coverage",
        }
        property_name = property_label_map.get(property_alias, property_alias)
        return f"{property_name} [{suffix_map[suffix]}]"

    if feature_name.startswith("component_share__"):
        return f"Доля компонента: {feature_name.removeprefix('component_share__')}"
    if feature_name.startswith("component_count__"):
        return f"Число вхождений компонента: {feature_name.removeprefix('component_count__')}"
    if feature_name.startswith("family_share__"):
        return f"Доля семейства: {feature_name.removeprefix('family_share__')}"
    if feature_name.startswith("family_count__"):
        return f"Число компонентов семейства: {feature_name.removeprefix('family_count__')}"
    if feature_name.startswith("scenario__"):
        return f"Условие сценария: {feature_name.removeprefix('scenario__')}"
    return feature_name


def safe_filename(value: str) -> str:
    return sanitize_name(value)[:120]


def build_property_table(props: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    numeric_props = props.copy()
    numeric_props["value_num"] = pd.to_numeric(
        numeric_props["Значение показателя"].astype(str).str.replace(",", ".", regex=False),
        errors="coerce",
    )
    numeric_props = numeric_props.dropna(subset=["value_num"]).copy()
    log(
        "property values parsed: "
        f"numeric_rows={len(numeric_props)}, "
        f"numeric_properties={numeric_props['Наименование показателя'].nunique()}"
    )

    property_alias = {
        name: f"prop_{idx:03d}"
        for idx, name in enumerate(sorted(numeric_props["Наименование показателя"].unique()), start=1)
    }
    numeric_props["property_alias"] = numeric_props["Наименование показателя"].map(property_alias)

    measured = (
        numeric_props[numeric_props["Наименование партии"].astype(str).str.lower() != "typical"]
        .pivot_table(
            index=["Компонент", "Наименование партии"],
            columns="property_alias",
            values="value_num",
            aggfunc="mean",
        )
        .reset_index()
    )
    typical = (
        numeric_props[numeric_props["Наименование партии"].astype(str).str.lower() == "typical"]
        .pivot_table(
            index="Компонент",
            columns="property_alias",
            values="value_num",
            aggfunc="mean",
        )
        .add_suffix("__typical")
        .reset_index()
    )
    log(
        "property tables built: "
        f"measured_component_parties={len(measured)}, "
        f"typical_components={len(typical)}, "
        f"property_features={len(property_alias)}"
    )

    property_columns = sorted(property_alias.values())
    property_table = measured.merge(typical, on="Компонент", how="left")
    fallback_fills = 0
    for column in property_columns:
        typical_column = f"{column}__typical"
        if column not in property_table.columns:
            property_table[column] = np.nan
        if typical_column in property_table.columns:
            fallback_fills += int(property_table[column].isna().sum() - property_table[column].fillna(property_table[typical_column]).isna().sum())
            property_table[column] = property_table[column].fillna(property_table[typical_column])
            property_table = property_table.drop(columns=[typical_column])

    coverage = float(property_table[property_columns].notna().mean().mean()) if property_columns else 0.0
    log(
        "merged property table ready: "
        f"rows={len(property_table)}, cols={len(property_table.columns)}, "
        f"fallback_fills={fallback_fills}, avg_numeric_coverage={coverage:.3f}"
    )

    return property_table, property_alias


def component_family(component_name: str) -> str:
    return re.sub(r"_\d+$", "", component_name)


def weighted_mean(series: pd.Series, weights: pd.Series) -> float:
    mask = series.notna()
    if not mask.any():
        return np.nan
    return float(np.average(series[mask], weights=weights[mask]))


def build_feature_config(train: pd.DataFrame) -> dict[str, list[str]]:
    families = train["Компонент"].map(component_family)
    config = {
        "top_components": train["Компонент"].value_counts().head(TOP_COMPONENTS).index.tolist(),
        "top_families": families.value_counts().head(TOP_FAMILIES).index.tolist(),
    }
    log(
        "feature config selected: "
        f"top_components={len(config['top_components'])}, "
        f"top_families={len(config['top_families'])}"
    )
    log(f"top components: {', '.join(config['top_components'])}")
    log(f"top families: {', '.join(config['top_families'])}")
    return config


def build_scenario_features(
    mixtures: pd.DataFrame,
    property_table: pd.DataFrame,
    feature_config: dict[str, list[str]],
) -> pd.DataFrame:
    log(
        "building scenario features: "
        f"rows={len(mixtures)}, scenarios={mixtures['scenario_id'].nunique()}"
    )
    merged = mixtures.merge(property_table, on=["Компонент", "Наименование партии"], how="left")
    merged["component_family"] = merged["Компонент"].map(component_family)
    merged["scenario_total_share"] = merged.groupby("scenario_id")["Массовая доля, %"].transform("sum")
    merged["normalized_share"] = merged["Массовая доля, %"] / merged["scenario_total_share"]

    property_columns = [column for column in property_table.columns if column.startswith("prop_")]
    rows: list[dict[str, float | str]] = []

    for scenario_id, group in merged.groupby("scenario_id", sort=True):
        base = {
            "scenario_id": scenario_id,
            "n_components": float(len(group)),
            "n_unique_components": float(group["Компонент"].nunique()),
            "n_unique_families": float(group["component_family"].nunique()),
            "share_sum": float(group["Массовая доля, %"].sum()),
            "share_mean": float(group["Массовая доля, %"].mean()),
            "share_std": float(group["Массовая доля, %"].std(ddof=0)),
            "share_min": float(group["Массовая доля, %"].min()),
            "share_max": float(group["Массовая доля, %"].max()),
            "share_entropy": float(
                -(group["normalized_share"] * np.log(group["normalized_share"].clip(lower=1e-12))).sum()
            ),
        }

        for column in SCENARIO_COLUMNS:
            base[f"scenario__{sanitize_name(column)}"] = float(group[column].iloc[0])

        for family in feature_config["top_families"]:
            family_mask = group["component_family"] == family
            base[f"family_share__{sanitize_name(family)}"] = float(group.loc[family_mask, "normalized_share"].sum())
            base[f"family_count__{sanitize_name(family)}"] = float(family_mask.sum())

        for component in feature_config["top_components"]:
            component_mask = group["Компонент"] == component
            base[f"component_share__{sanitize_name(component)}"] = float(
                group.loc[component_mask, "normalized_share"].sum()
            )
            base[f"component_count__{sanitize_name(component)}"] = float(component_mask.sum())

        for column in property_columns:
            base[f"{column}__wmean"] = weighted_mean(group[column], group["normalized_share"])
            base[f"{column}__max"] = float(group[column].max()) if group[column].notna().any() else np.nan
            base[f"{column}__coverage"] = float(group[column].notna().mean())

        rows.append(base)

    features = pd.DataFrame(rows).set_index("scenario_id").sort_index()
    log(
        "scenario features ready: "
        f"shape={features.shape}, "
        f"missing_ratio={features.isna().mean().mean():.3f}"
    )
    sample_columns = ", ".join(features.columns[:12])
    log(f"sample feature columns: {sample_columns}")
    return features


def fit_target_model(
    X: pd.DataFrame,
    y: pd.Series,
    target_name: str,
    output_dir: Path,
) -> tuple[np.ndarray, np.ndarray, CatBoostRegressor]:
    log(f"training target: {target_name}")
    transformed = sign_log1p(y) if target_name.startswith("Delta Kin.") else y.copy()
    splitter = KFold(n_splits=min(N_SPLITS, len(X)), shuffle=True, random_state=RANDOM_STATE)
    oof_pred = np.zeros(len(X), dtype=float)

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(X), start=1):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train = transformed.iloc[train_idx]
        model = CatBoostRegressor(
            iterations=700,
            learning_rate=0.03,
            depth=6,
            l2_leaf_reg=6.0,
            loss_function="RMSE",
            eval_metric="RMSE",
            random_seed=RANDOM_STATE + fold,
            allow_writing_files=False,
            verbose=False,
        )
        model.fit(X_train, y_train)
        fold_pred = model.predict(X_valid)
        if target_name.startswith("Delta Kin."):
            fold_pred = inverse_sign_log1p(fold_pred)
        oof_pred[valid_idx] = fold_pred
        fold_mae = mean_absolute_error(y.iloc[valid_idx], fold_pred)
        fold_rmse = float(np.sqrt(mean_squared_error(y.iloc[valid_idx], fold_pred)))
        log(
            f"fold={fold}/{splitter.get_n_splits()} "
            f"train_size={len(train_idx)} valid_size={len(valid_idx)} "
            f"mae={fold_mae:.4f} rmse={fold_rmse:.4f}"
        )

    final_model = CatBoostRegressor(
        iterations=700,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=6.0,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=RANDOM_STATE,
        allow_writing_files=False,
        verbose=False,
    )
    final_model.fit(X, transformed)
    final_model.save_model(str(output_dir / f"{sanitize_name(target_name)}.cbm"))
    log(f"saved model: {sanitize_name(target_name)}.cbm")

    full_pred = final_model.predict(X)
    if target_name.startswith("Delta Kin."):
        full_pred = inverse_sign_log1p(full_pred)

    return oof_pred, full_pred, final_model


def save_shap_artifacts(
    model: CatBoostRegressor,
    X: pd.DataFrame,
    target_name: str,
    output_dir: Path,
    property_label_map: dict[str, str],
) -> None:
    log(f"building SHAP artifacts for target: {target_name}")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    shap_values = np.asarray(shap_values)
    feature_display_names = [make_feature_display_name(column, property_label_map) for column in X.columns]
    feature_name_map = dict(zip(X.columns, feature_display_names))

    importance = pd.DataFrame(
        {
            "feature": X.columns,
            "feature_display_name": feature_display_names,
            "mean_abs_shap": np.abs(shap_values).mean(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False)
    importance.to_csv(output_dir / f"{sanitize_name(target_name)}_shap_importance.csv", index=False)
    top_features = ", ".join(importance.head(10)["feature_display_name"].tolist())
    log(f"top SHAP features for {target_name}: {top_features}")

    X_display = X.copy()
    X_display.columns = feature_display_names

    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_display, plot_type="bar", max_display=SHAP_MAX_DISPLAY, show=False)
    plt.subplots_adjust(left=0.35, right=0.98, top=0.98, bottom=0.08)
    plt.savefig(output_dir / f"{sanitize_name(target_name)}_shap_bar.png", dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(12, 14))
    shap.summary_plot(shap_values, X_display, max_display=SHAP_MAX_DISPLAY, show=False)
    plt.subplots_adjust(left=0.35, right=0.98, top=0.98, bottom=0.08)
    plt.savefig(output_dir / f"{sanitize_name(target_name)}_shap_beeswarm.png", dpi=200, bbox_inches="tight")
    plt.close()

    top_feature_names = importance.head(TOP_DEPENDENCE_PLOTS)["feature"].tolist()
    for feature_name in top_feature_names:
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(
            feature_name_map[feature_name],
            shap_values,
            X_display,
            interaction_index="auto",
            show=False,
        )
        plt.subplots_adjust(left=0.18, right=0.98, top=0.95, bottom=0.16)
        plt.savefig(
            output_dir / f"{sanitize_name(target_name)}_dependence_{safe_filename(feature_name)}.png",
            dpi=180,
            bbox_inches="tight",
        )
        plt.close()

    interaction_pairs = [
        (
            "scenario__температура_испытания_astm_d445_daimler_oxidation_test_dot_c",
            "scenario__количество_биотоплива_daimler_oxidation_test_dot_масс",
        ),
        (
            "component_share__базовое_масло_1",
            "component_share__детергент_1",
        ),
        (
            "prop_037__max",
            "prop_035__max",
        ),
        (
            "prop_042__wmean",
            "prop_019__coverage",
        ),
    ]
    existing_pairs = [
        pair for pair in interaction_pairs if pair[0] in X.columns and pair[1] in X.columns
    ][:TOP_INTERACTIONS]
    if existing_pairs:
        interaction_rows = []
        for left_feature, right_feature in existing_pairs:
            left_values = X[left_feature].fillna(X[left_feature].median())
            right_values = X[right_feature].fillna(X[right_feature].median())
            interaction_strength = float(
                np.corrcoef(shap_values[:, X.columns.get_loc(left_feature)], right_values)[0, 1]
            )
            interaction_rows.append(
                {
                    "left_feature": left_feature,
                    "left_feature_display_name": make_feature_display_name(left_feature, property_label_map),
                    "right_feature": right_feature,
                    "right_feature_display_name": make_feature_display_name(right_feature, property_label_map),
                    "proxy_strength": interaction_strength,
                }
            )

            plt.figure(figsize=(10, 6))
            shap.dependence_plot(
                feature_name_map[left_feature],
                shap_values,
                X_display,
                interaction_index=feature_name_map[right_feature],
                show=False,
            )
            plt.subplots_adjust(left=0.18, right=0.98, top=0.95, bottom=0.16)
            plt.savefig(
                output_dir
                / f"{sanitize_name(target_name)}_interaction_{safe_filename(left_feature)}__{safe_filename(right_feature)}.png",
                dpi=180,
                bbox_inches="tight",
            )
            plt.close()

        pd.DataFrame(interaction_rows).to_csv(
            output_dir / f"{sanitize_name(target_name)}_interaction_pairs.csv",
            index=False,
        )


def save_correlation_artifacts(
    X: pd.DataFrame,
    targets: pd.DataFrame,
    property_label_map: dict[str, str],
    output_dir: Path,
) -> None:
    log("building correlation heatmaps")
    correlation_dir = output_dir / "correlations"
    correlation_dir.mkdir(parents=True, exist_ok=True)

    top_features = X.var(numeric_only=True).sort_values(ascending=False).head(30).index.tolist()
    corr_features = X[top_features].copy()
    corr_features.columns = [make_feature_display_name(column, property_label_map) for column in corr_features.columns]
    corr_matrix = corr_features.corr()
    corr_matrix.to_csv(correlation_dir / "feature_correlation_top30.csv")

    plt.figure(figsize=(18, 14))
    sns.heatmap(corr_matrix, cmap="coolwarm", center=0)
    plt.title("Scenario Feature Correlation Heatmap (top 30 by variance)")
    plt.tight_layout()
    plt.savefig(correlation_dir / "feature_correlation_top30.png", dpi=180, bbox_inches="tight")
    plt.close()

    target_corr = targets.corr()
    target_corr.to_csv(correlation_dir / "target_correlation.csv")
    plt.figure(figsize=(4, 3))
    sns.heatmap(target_corr, annot=True, cmap="coolwarm", center=0, vmin=-1, vmax=1)
    plt.title("Target Correlation")
    plt.tight_layout()
    plt.savefig(correlation_dir / "target_correlation.png", dpi=180, bbox_inches="tight")
    plt.close()

    for target_name in targets.columns:
        target_feature_corr = (
            X.corrwith(targets[target_name])
            .dropna()
            .sort_values(key=lambda series: series.abs(), ascending=False)
            .head(20)
        )
        top_feature_names = target_feature_corr.index.tolist()
        target_matrix = X[top_feature_names].copy()
        target_matrix.columns = [make_feature_display_name(column, property_label_map) for column in top_feature_names]
        target_matrix.insert(0, target_name, targets[target_name])
        target_matrix_corr = target_matrix.corr()
        target_matrix_corr.to_csv(
            correlation_dir / f"{sanitize_name(target_name)}_top20_feature_corr.csv"
        )

        plt.figure(figsize=(18, 16))
        sns.heatmap(
            target_matrix_corr,
            cmap="coolwarm",
            center=0,
            annot=True,
            fmt=".2f",
        )
        plt.title(f"Correlation matrix: {target_name} + top 20 correlated features")
        plt.tight_layout()
        plt.savefig(
            correlation_dir / f"{sanitize_name(target_name)}_top20_feature_corr.png",
            dpi=180,
            bbox_inches="tight",
        )
        plt.close()


def save_scatter_artifacts(
    X: pd.DataFrame,
    targets: pd.DataFrame,
    shap_importance_paths: dict[str, Path],
    property_label_map: dict[str, str],
    output_dir: Path,
) -> None:
    log("building feature vs target scatter plots")
    for target_name, importance_path in shap_importance_paths.items():
        importance = pd.read_csv(importance_path)
        top_features = importance["feature"].head(TOP_SCATTER_PLOTS).tolist()
        target_slug = sanitize_name(target_name)
        target_dir = output_dir / f"{target_slug}_scatter"
        target_dir.mkdir(parents=True, exist_ok=True)
        for feature_name in top_features:
            plot_df = pd.DataFrame(
                {
                    "feature": X[feature_name],
                    "target": targets[target_name],
                }
            ).dropna()
            if plot_df.empty:
                continue
            plt.figure(figsize=(8, 6))
            sns.scatterplot(data=plot_df, x="feature", y="target", s=40)
            sns.regplot(data=plot_df, x="feature", y="target", scatter=False, lowess=True, color="crimson")
            plt.xlabel(make_feature_display_name(feature_name, property_label_map))
            plt.ylabel(target_name)
            plt.title("Feature vs target")
            plt.tight_layout()
            plt.savefig(target_dir / f"{safe_filename(feature_name)}.png", dpi=180, bbox_inches="tight")
            plt.close()


def save_condition_plots(
    scenario_features: pd.DataFrame,
    targets: pd.DataFrame,
    output_dir: Path,
) -> None:
    log("building boxplot and violin plots for scenario conditions")
    conditions = [
        "scenario__дозировка_катализатора_категория",
        "scenario__время_испытания_daimler_oxidation_test_dot_ч",
        "scenario__температура_испытания_astm_d445_daimler_oxidation_test_dot_c",
        "scenario__количество_биотоплива_daimler_oxidation_test_dot_масс",
    ]
    plot_df = scenario_features[conditions].join(targets)
    for target_name in TARGET_COLUMNS:
        target_slug = sanitize_name(target_name)
        for condition in conditions:
            display_name = condition.removeprefix("scenario__")

            plt.figure(figsize=(9, 6))
            sns.boxplot(data=plot_df, x=condition, y=target_name)
            sns.stripplot(data=plot_df, x=condition, y=target_name, color="black", alpha=0.45, size=3)
            plt.xlabel(display_name)
            plt.ylabel(target_name)
            plt.title("Boxplot by scenario condition")
            plt.tight_layout()
            plt.savefig(
                output_dir / f"{target_slug}_boxplot_{safe_filename(condition)}.png",
                dpi=180,
                bbox_inches="tight",
            )
            plt.close()

            plt.figure(figsize=(9, 6))
            sns.violinplot(data=plot_df, x=condition, y=target_name, inner="quartile")
            plt.xlabel(display_name)
            plt.ylabel(target_name)
            plt.title("Violin plot by scenario condition")
            plt.tight_layout()
            plt.savefig(
                output_dir / f"{target_slug}_violin_{safe_filename(condition)}.png",
                dpi=180,
                bbox_inches="tight",
            )
            plt.close()


def save_missingness_artifacts(
    props: pd.DataFrame,
    property_alias: dict[str, str],
    output_dir: Path,
) -> None:
    log("building missingness analysis")
    numeric_props = props.copy()
    numeric_props["value_num"] = pd.to_numeric(
        numeric_props["Значение показателя"].astype(str).str.replace(",", ".", regex=False),
        errors="coerce",
    )
    numeric_props["source_type"] = np.where(
        numeric_props["Наименование партии"].astype(str).str.lower() == "typical",
        "typical",
        "measured",
    )

    missing_summary = (
        numeric_props.groupby(["Наименование показателя", "source_type"])["value_num"]
        .agg(total="size", non_null="count")
        .reset_index()
    )
    missing_summary["missing"] = missing_summary["total"] - missing_summary["non_null"]
    missing_summary["missing_ratio"] = missing_summary["missing"] / missing_summary["total"]
    missing_summary["property_alias"] = missing_summary["Наименование показателя"].map(property_alias)
    missing_summary.to_csv(output_dir / "property_missingness_summary.csv", index=False)

    pivot_missing = missing_summary.pivot(
        index="Наименование показателя",
        columns="source_type",
        values="missing_ratio",
    ).fillna(1.0)
    pivot_missing = pivot_missing.sort_values(by=list(pivot_missing.columns), ascending=False).head(40)
    plt.figure(figsize=(10, 14))
    sns.heatmap(pivot_missing, cmap="Reds", vmin=0, vmax=1)
    plt.title("Property Missingness Heatmap")
    plt.tight_layout()
    plt.savefig(output_dir / "property_missingness_heatmap.png", dpi=180, bbox_inches="tight")
    plt.close()

    coverage_summary = (
        numeric_props.groupby(["Наименование показателя", "source_type"])["value_num"]
        .count()
        .reset_index(name="non_null")
    )
    plt.figure(figsize=(12, 8))
    sns.barplot(
        data=coverage_summary.sort_values("non_null", ascending=False).head(30),
        x="non_null",
        y="Наименование показателя",
        hue="source_type",
    )
    plt.title("Top 30 properties by non-null coverage")
    plt.tight_layout()
    plt.savefig(output_dir / "property_coverage_top30.png", dpi=180, bbox_inches="tight")
    plt.close()


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / "docs"
    output_dir = project_root / "baseline"
    output_dir.mkdir(parents=True, exist_ok=True)

    train, test, props = load_data(data_dir)
    property_table, property_alias = build_property_table(props)
    property_label_map = build_property_label_map(property_alias)
    feature_config = build_feature_config(train)

    X_train = build_scenario_features(train[MIXTURE_COLUMNS + SCENARIO_COLUMNS], property_table, feature_config)
    X_test = build_scenario_features(test[MIXTURE_COLUMNS + SCENARIO_COLUMNS], property_table, feature_config)
    X_test = X_test.reindex(columns=X_train.columns)
    log(f"aligned feature matrices: train_shape={X_train.shape}, test_shape={X_test.shape}")

    targets = train.groupby("scenario_id")[TARGET_COLUMNS].first().sort_index()
    X_train = X_train.loc[targets.index]
    log(f"targets ready: shape={targets.shape}")

    metrics: dict[str, dict[str, float]] = {}
    predictions = pd.DataFrame(index=X_test.index)
    shap_importance_paths: dict[str, Path] = {}

    for target_name in TARGET_COLUMNS:
        oof_pred, _, model = fit_target_model(X_train, targets[target_name], target_name, output_dir)
        test_pred = model.predict(X_test)
        if target_name.startswith("Delta Kin."):
            test_pred = inverse_sign_log1p(test_pred)

        metrics[target_name] = {
            "mae": float(mean_absolute_error(targets[target_name], oof_pred)),
            "rmse": float(np.sqrt(mean_squared_error(targets[target_name], oof_pred))),
            "r2": float(r2_score(targets[target_name], oof_pred)),
        }
        log(
            f"cv summary for {target_name}: "
            f"mae={metrics[target_name]['mae']:.4f}, "
            f"rmse={metrics[target_name]['rmse']:.4f}, "
            f"r2={metrics[target_name]['r2']:.4f}"
        )
        predictions[target_name] = test_pred
        save_shap_artifacts(model, X_train, target_name, output_dir, property_label_map)
        shap_importance_paths[target_name] = output_dir / f"{sanitize_name(target_name)}_shap_importance.csv"

    predictions = predictions.reset_index().rename(columns={"index": "scenario_id"})
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    X_train.to_csv(output_dir / "train_features.csv")
    X_test.to_csv(output_dir / "test_features.csv")
    pd.DataFrame(
        [{"property_alias": alias, "source_name": source} for source, alias in property_alias.items()]
    ).to_csv(output_dir / "property_alias_map.csv", index=False)
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as fp:
        json.dump(metrics, fp, ensure_ascii=False, indent=2)

    save_correlation_artifacts(X_train, targets, property_label_map, output_dir)
    save_scatter_artifacts(X_train, targets, shap_importance_paths, property_label_map, output_dir)
    save_condition_plots(X_train, targets, output_dir)
    save_missingness_artifacts(props, property_alias, output_dir)

    print("Saved artifacts to:", output_dir)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
