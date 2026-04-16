from __future__ import annotations

import argparse
import copy
import json
import math
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset


ID_COL = "scenario_id"
TARGET_COLS = [
    "Delta Kin. Viscosity KV100 - relative | - Daimler Oxidation Test (DOT), %",
    "Oxidation EOT | DIN 51453 Daimler Oxidation Test (DOT), A/cm",
]
COMPONENT_NAME_COL = "Компонент"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def read_csv_strict(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")
    return pd.read_csv(path)


def validate_required_columns(df: pd.DataFrame, required_columns: list[str], df_name: str) -> None:
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"В {df_name} отсутствуют обязательные колонки: {missing_columns}")


def normalize_string(value: Any) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).strip())


def parse_numeric_like(value: Any) -> float:
    if pd.isna(value):
        return float("nan")

    text = normalize_string(value)
    if not text:
        return float("nan")

    lowered = text.casefold()
    if lowered in {"нет", "none", "nan"}:
        return float("nan")

    if re.fullmatch(r"\d{4}-\d{2}-\d{2}(?: \d{2}:\d{2}:\d{2})?", text):
        return float("nan")

    cleaned = text.replace("−", "-").replace("–", "-").replace("—", "-")
    cleaned = cleaned.replace("≤", "").replace("≥", "").replace("<", "").replace(">", "").replace("~", "")
    cleaned = cleaned.replace("%", "").replace(" ", "")
    cleaned = cleaned.replace("°C", "").replace("°c", "").replace("⁰С", "").replace("⁰с", "")

    if cleaned.count(",") == 1 and cleaned.count(".") == 0:
        cleaned = cleaned.replace(",", ".")
    elif cleaned.count(",") > 1 and cleaned.count(".") == 0:
        return float("nan")
    elif cleaned.count(",") >= 1 and cleaned.count(".") >= 1:
        return float("nan")

    numeric_pattern = r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?$"
    if not re.fullmatch(numeric_pattern, cleaned.casefold()):
        return float("nan")

    return float(cleaned)


def detect_component_feature_columns(
    component_train_df: pd.DataFrame,
    scenario_train_df: pd.DataFrame,
) -> tuple[list[str], list[str], list[str]]:
    repeated_scenario_columns = [
        column
        for column in scenario_train_df.columns
        if column in component_train_df.columns and column not in {ID_COL, *TARGET_COLS}
    ]

    base_component_columns = [
        column
        for column in component_train_df.columns
        if column not in {ID_COL, *TARGET_COLS, *repeated_scenario_columns}
    ]

    numeric_columns = component_train_df[base_component_columns].select_dtypes(include=["number"]).columns.tolist()
    object_columns = [column for column in base_component_columns if column not in numeric_columns]

    auto_numeric_columns: list[str] = []
    categorical_columns: list[str] = []

    for column in object_columns:
        parsed_series = component_train_df[column].map(parse_numeric_like)
        non_null_mask = component_train_df[column].map(lambda value: normalize_string(value) != "")
        non_null_count = int(non_null_mask.sum())

        if non_null_count == 0:
            continue

        parsed_success_ratio = float(parsed_series.notna().sum()) / float(non_null_count)
        if parsed_success_ratio >= 0.8 and column != COMPONENT_NAME_COL:
            auto_numeric_columns.append(column)
        else:
            unique_count = component_train_df[column].fillna("").astype(str).nunique()
            if column == COMPONENT_NAME_COL or unique_count <= 32:
                categorical_columns.append(column)

    return numeric_columns, auto_numeric_columns, categorical_columns


def build_feature_spec(
    component_train_df: pd.DataFrame,
    scenario_train_df: pd.DataFrame,
) -> dict[str, Any]:
    validate_required_columns(component_train_df, [ID_COL, COMPONENT_NAME_COL], "component_train")
    validate_required_columns(scenario_train_df, [ID_COL, *TARGET_COLS], "scenario_train")

    numeric_columns, auto_numeric_columns, categorical_columns = detect_component_feature_columns(
        component_train_df=component_train_df,
        scenario_train_df=scenario_train_df,
    )

    numeric_arrays: list[np.ndarray] = []
    selected_numeric_columns: list[str] = []
    selected_auto_numeric_columns: list[str] = []

    for column in numeric_columns:
        values = pd.to_numeric(component_train_df[column], errors="coerce").to_numpy(dtype=np.float32)
        if np.isfinite(values).any():
            numeric_arrays.append(values.reshape(-1, 1))
            selected_numeric_columns.append(column)

    for column in auto_numeric_columns:
        values = component_train_df[column].map(parse_numeric_like).to_numpy(dtype=np.float32)
        if np.isfinite(values).any():
            numeric_arrays.append(values.reshape(-1, 1))
            selected_auto_numeric_columns.append(column)

    if not numeric_arrays:
        raise ValueError("После фильтрации не осталось компонентных числовых признаков.")

    component_numeric_matrix = np.concatenate(numeric_arrays, axis=1).astype(np.float32)
    component_numeric_mean = np.nanmean(component_numeric_matrix, axis=0)
    component_numeric_std = np.nanstd(component_numeric_matrix, axis=0)
    component_numeric_mean = np.where(np.isnan(component_numeric_mean), 0.0, component_numeric_mean)
    component_numeric_std = np.where(
        np.isnan(component_numeric_std) | (component_numeric_std < 1e-8),
        1.0,
        component_numeric_std,
    )

    global_columns = [
        column for column in scenario_train_df.columns if column not in {ID_COL, *TARGET_COLS}
    ]
    if not global_columns:
        raise ValueError("В scenario_train не осталось глобальных признаков.")

    global_matrix = scenario_train_df[global_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    global_mean = np.nanmean(global_matrix, axis=0)
    global_std = np.nanstd(global_matrix, axis=0)
    global_mean = np.where(np.isnan(global_mean), 0.0, global_mean)
    global_std = np.where(np.isnan(global_std) | (global_std < 1e-8), 1.0, global_std)

    target_matrix = scenario_train_df[TARGET_COLS].to_numpy(dtype=np.float32)
    target_mean = np.mean(target_matrix, axis=0)
    target_std = np.std(target_matrix, axis=0)
    target_std = np.where(target_std < 1e-8, 1.0, target_std)

    category_vocabularies: dict[str, dict[str, int]] = {}
    for column in categorical_columns:
        values = component_train_df[column].map(normalize_string)
        unique_values = sorted({value for value in values if value})
        vocabulary = {"__PAD__": 0, "__UNK__": 1, "__MISSING__": 2}
        for index, value in enumerate(unique_values, start=3):
            vocabulary[value] = index
        category_vocabularies[column] = vocabulary

    return {
        "component_numeric_columns": selected_numeric_columns,
        "component_auto_numeric_columns": selected_auto_numeric_columns,
        "component_categorical_columns": categorical_columns,
        "global_columns": global_columns,
        "category_vocabularies": category_vocabularies,
        "component_numeric_mean": component_numeric_mean.astype(np.float32),
        "component_numeric_std": component_numeric_std.astype(np.float32),
        "global_mean": global_mean.astype(np.float32),
        "global_std": global_std.astype(np.float32),
        "target_mean": target_mean.astype(np.float32),
        "target_std": target_std.astype(np.float32),
    }


def encode_component_numeric(component_df: pd.DataFrame, feature_spec: dict[str, Any]) -> np.ndarray:
    numeric_arrays: list[np.ndarray] = []

    for column in feature_spec["component_numeric_columns"]:
        values = pd.to_numeric(component_df[column], errors="coerce").to_numpy(dtype=np.float32)
        numeric_arrays.append(values.reshape(-1, 1))

    for column in feature_spec["component_auto_numeric_columns"]:
        values = component_df[column].map(parse_numeric_like).to_numpy(dtype=np.float32)
        numeric_arrays.append(values.reshape(-1, 1))

    numeric_matrix = np.concatenate(numeric_arrays, axis=1).astype(np.float32)
    numeric_matrix = np.where(
        np.isnan(numeric_matrix),
        feature_spec["component_numeric_mean"].reshape(1, -1),
        numeric_matrix,
    )
    numeric_matrix = (
        numeric_matrix - feature_spec["component_numeric_mean"].reshape(1, -1)
    ) / feature_spec["component_numeric_std"].reshape(1, -1)

    return numeric_matrix.astype(np.float32)


def encode_global_features(scenario_row: pd.Series, feature_spec: dict[str, Any]) -> np.ndarray:
    global_values = pd.to_numeric(
        scenario_row[feature_spec["global_columns"]],
        errors="coerce",
    ).to_numpy(dtype=np.float32)
    global_values = np.where(np.isnan(global_values), feature_spec["global_mean"], global_values)
    global_values = (global_values - feature_spec["global_mean"]) / feature_spec["global_std"]
    return global_values.astype(np.float32)


def encode_categorical_feature(values: pd.Series, vocabulary: dict[str, int]) -> np.ndarray:
    encoded: list[int] = []
    for value in values.map(normalize_string):
        if not value:
            encoded.append(vocabulary["__MISSING__"])
        else:
            encoded.append(vocabulary.get(value, vocabulary["__UNK__"]))
    return np.asarray(encoded, dtype=np.int64)


class ScenarioSetDataset(Dataset):
    def __init__(
        self,
        component_df: pd.DataFrame,
        scenario_df: pd.DataFrame,
        feature_spec: dict[str, Any],
        is_train: bool,
    ) -> None:
        component_groups = {
            scenario_id: group.reset_index(drop=True)
            for scenario_id, group in component_df.groupby(ID_COL, sort=False)
        }

        self.samples: list[dict[str, Any]] = []
        for _, scenario_row in scenario_df.iterrows():
            scenario_id = scenario_row[ID_COL]
            if scenario_id not in component_groups:
                raise ValueError(f"Для scenario_id={scenario_id} нет компонентных строк.")

            component_rows = component_groups[scenario_id]
            sample = {
                "scenario_id": scenario_id,
                "component_numeric": encode_component_numeric(component_rows, feature_spec),
                "component_categorical": {
                    column: encode_categorical_feature(
                        component_rows[column],
                        feature_spec["category_vocabularies"][column],
                    )
                    for column in feature_spec["component_categorical_columns"]
                },
                "global_features": encode_global_features(scenario_row, feature_spec),
            }

            if is_train:
                targets = scenario_row[TARGET_COLS].to_numpy(dtype=np.float32)
                targets = (targets - feature_spec["target_mean"]) / feature_spec["target_std"]
                sample["targets"] = targets.astype(np.float32)

            self.samples.append(sample)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.samples[index]


def make_collate_fn(feature_spec: dict[str, Any], is_train: bool):
    categorical_columns = feature_spec["component_categorical_columns"]
    numeric_dim = len(feature_spec["component_numeric_mean"])
    global_dim = len(feature_spec["global_columns"])

    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
        batch_size = len(batch)
        max_components = max(sample["component_numeric"].shape[0] for sample in batch)

        component_numeric = torch.zeros((batch_size, max_components, numeric_dim), dtype=torch.float32)
        component_mask = torch.zeros((batch_size, max_components), dtype=torch.bool)
        global_features = torch.zeros((batch_size, global_dim), dtype=torch.float32)
        component_categorical = {
            column: torch.zeros((batch_size, max_components), dtype=torch.long)
            for column in categorical_columns
        }

        if is_train:
            targets = torch.zeros((batch_size, len(TARGET_COLS)), dtype=torch.float32)

        scenario_ids: list[str] = []

        for batch_index, sample in enumerate(batch):
            num_components = sample["component_numeric"].shape[0]
            component_numeric[batch_index, :num_components] = torch.from_numpy(sample["component_numeric"])
            component_mask[batch_index, :num_components] = True
            global_features[batch_index] = torch.from_numpy(sample["global_features"])

            for column in categorical_columns:
                component_categorical[column][batch_index, :num_components] = torch.from_numpy(
                    sample["component_categorical"][column]
                )

            scenario_ids.append(sample["scenario_id"])

            if is_train:
                targets[batch_index] = torch.from_numpy(sample["targets"])

        output = {
            "scenario_id": scenario_ids,
            "component_numeric": component_numeric,
            "component_mask": component_mask,
            "component_categorical": component_categorical,
            "global_features": global_features,
        }
        if is_train:
            output["targets"] = targets

        return output

    return collate_fn


class ConditionedInteractionBlock(nn.Module):
    def __init__(self, model_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        pair_input_dim = model_dim * 5

        self.message_mlp = nn.Sequential(
            nn.Linear(pair_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, model_dim),
        )
        self.gate_mlp = nn.Sequential(
            nn.Linear(pair_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, model_dim),
            nn.Sigmoid(),
        )
        self.output_norm = nn.LayerNorm(model_dim)

    def forward(
        self,
        component_states: torch.Tensor,
        component_mask: torch.Tensor,
        global_state: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, max_components, model_dim = component_states.shape

        first = component_states.unsqueeze(2).expand(batch_size, max_components, max_components, model_dim)
        second = component_states.unsqueeze(1).expand(batch_size, max_components, max_components, model_dim)
        global_expanded = global_state.unsqueeze(1).unsqueeze(1).expand(
            batch_size,
            max_components,
            max_components,
            model_dim,
        )

        pair_inputs = torch.cat(
            [first, second, first * second, torch.abs(first - second), global_expanded],
            dim=-1,
        )

        messages = self.message_mlp(pair_inputs)
        gates = self.gate_mlp(pair_inputs)
        messages = messages * gates

        valid_pairs = component_mask.unsqueeze(2) & component_mask.unsqueeze(1)
        diagonal_mask = ~torch.eye(
            max_components,
            dtype=torch.bool,
            device=component_states.device,
        ).unsqueeze(0)
        valid_pairs = valid_pairs & diagonal_mask

        messages = messages * valid_pairs.unsqueeze(-1)
        neighbor_counts = valid_pairs.sum(dim=2).clamp_min(1).unsqueeze(-1)
        aggregated_messages = messages.sum(dim=2) / neighbor_counts

        updated_states = self.output_norm(component_states + aggregated_messages)
        updated_states = updated_states * component_mask.unsqueeze(-1)
        return updated_states


class InteractionSetRegressor(nn.Module):
    def __init__(
        self,
        component_numeric_dim: int,
        global_dim: int,
        categorical_cardinalities: dict[str, int],
        model_dim: int = 64,
        pair_hidden_dim: int = 128,
        dropout: float = 0.15,
        num_interaction_blocks: int = 2,
    ) -> None:
        super().__init__()

        self.component_numeric_encoder = nn.Sequential(
            nn.Linear(component_numeric_dim, model_dim),
            nn.ReLU(),
            nn.LayerNorm(model_dim),
            nn.Dropout(dropout),
        )

        self.global_encoder = nn.Sequential(
            nn.Linear(global_dim, model_dim),
            nn.ReLU(),
            nn.LayerNorm(model_dim),
            nn.Dropout(dropout),
        )

        self.categorical_embeddings = nn.ModuleDict()
        for column, cardinality in categorical_cardinalities.items():
            embedding_dim = min(16, max(4, int(math.sqrt(cardinality)) + 1))
            self.categorical_embeddings[column] = nn.Sequential(
                nn.Embedding(cardinality, embedding_dim, padding_idx=0),
                nn.Linear(embedding_dim, model_dim, bias=False),
            )

        self.interaction_blocks = nn.ModuleList(
            [
                ConditionedInteractionBlock(
                    model_dim=model_dim,
                    hidden_dim=pair_hidden_dim,
                    dropout=dropout,
                )
                for _ in range(num_interaction_blocks)
            ]
        )

        self.head = nn.Sequential(
            nn.Linear(model_dim * 3, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, len(TARGET_COLS)),
        )

    def forward(
        self,
        component_numeric: torch.Tensor,
        component_categorical: dict[str, torch.Tensor],
        component_mask: torch.Tensor,
        global_features: torch.Tensor,
    ) -> torch.Tensor:
        component_states = self.component_numeric_encoder(component_numeric)

        for column, embedding_stack in self.categorical_embeddings.items():
            component_states = component_states + embedding_stack(component_categorical[column])

        component_states = component_states * component_mask.unsqueeze(-1)
        global_state = self.global_encoder(global_features)

        for interaction_block in self.interaction_blocks:
            component_states = interaction_block(component_states, component_mask, global_state)

        mask_float = component_mask.unsqueeze(-1).float()
        component_count = component_mask.sum(dim=1).clamp_min(1).unsqueeze(-1).float()

        pooled_mean = (component_states * mask_float).sum(dim=1) / component_count

        masked_states = component_states.masked_fill(~component_mask.unsqueeze(-1), float("-inf"))
        pooled_max = masked_states.max(dim=1).values
        pooled_max = torch.where(torch.isfinite(pooled_max), pooled_max, torch.zeros_like(pooled_max))

        final_features = torch.cat([pooled_mean, pooled_max, global_state], dim=-1)
        return self.head(final_features)


def inverse_scale_targets(values: np.ndarray, feature_spec: dict[str, Any]) -> np.ndarray:
    return values * feature_spec["target_std"].reshape(1, -1) + feature_spec["target_mean"].reshape(1, -1)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    metrics: dict[str, Any] = {"per_target": {}}
    rmse_values: list[float] = []
    mae_values: list[float] = []
    r2_values: list[float] = []

    for index, target_name in enumerate(TARGET_COLS):
        rmse_value = float(np.sqrt(mean_squared_error(y_true[:, index], y_pred[:, index])))
        mae_value = float(mean_absolute_error(y_true[:, index], y_pred[:, index]))
        r2_value = float(r2_score(y_true[:, index], y_pred[:, index]))

        metrics["per_target"][target_name] = {
            "rmse": rmse_value,
            "mae": mae_value,
            "r2": r2_value,
        }
        rmse_values.append(rmse_value)
        mae_values.append(mae_value)
        r2_values.append(r2_value)

    metrics["mean_rmse"] = float(np.mean(rmse_values))
    metrics["mean_mae"] = float(np.mean(mae_values))
    metrics["mean_r2"] = float(np.mean(r2_values))
    return metrics


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    feature_spec: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    model.eval()

    scenario_ids: list[str] = []
    target_batches: list[np.ndarray] = []
    prediction_batches: list[np.ndarray] = []

    with torch.no_grad():
        for batch in dataloader:
            predictions = model(
                component_numeric=batch["component_numeric"].to(device),
                component_categorical={key: value.to(device) for key, value in batch["component_categorical"].items()},
                component_mask=batch["component_mask"].to(device),
                global_features=batch["global_features"].to(device),
            )
            target_batches.append(batch["targets"].cpu().numpy())
            prediction_batches.append(predictions.cpu().numpy())
            scenario_ids.extend(batch["scenario_id"])

    scaled_targets = np.concatenate(target_batches, axis=0)
    scaled_predictions = np.concatenate(prediction_batches, axis=0)

    targets = inverse_scale_targets(scaled_targets, feature_spec)
    predictions = inverse_scale_targets(scaled_predictions, feature_spec)

    metrics = calculate_metrics(targets, predictions)

    output_df = pd.DataFrame({ID_COL: scenario_ids})
    for index, target_name in enumerate(TARGET_COLS):
        output_df[target_name] = targets[:, index]
        output_df[f"pred::{target_name}"] = predictions[:, index]

    return metrics, output_df


def predict_for_test(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    feature_spec: dict[str, Any],
) -> pd.DataFrame:
    model.eval()

    scenario_ids: list[str] = []
    prediction_batches: list[np.ndarray] = []

    with torch.no_grad():
        for batch in dataloader:
            predictions = model(
                component_numeric=batch["component_numeric"].to(device),
                component_categorical={key: value.to(device) for key, value in batch["component_categorical"].items()},
                component_mask=batch["component_mask"].to(device),
                global_features=batch["global_features"].to(device),
            )
            prediction_batches.append(predictions.cpu().numpy())
            scenario_ids.extend(batch["scenario_id"])

    scaled_predictions = np.concatenate(prediction_batches, axis=0)
    predictions = inverse_scale_targets(scaled_predictions, feature_spec)

    output_df = pd.DataFrame({ID_COL: scenario_ids})
    for index, target_name in enumerate(TARGET_COLS):
        output_df[target_name] = predictions[:, index]

    return output_df


def save_json(data: dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Обучение set-модели с явным interaction block для Daimler DOT.")
    parser.add_argument("--component-train", required=True, type=Path)
    parser.add_argument("--component-test", required=True, type=Path)
    parser.add_argument("--scenario-train", required=True, type=Path)
    parser.add_argument("--scenario-test", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--epochs", default=300, type=int)
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--learning-rate", default=1e-3, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--patience", default=40, type=int)
    parser.add_argument("--val-size", default=0.2, type=float)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    if not 0.0 < args.val_size < 1.0:
        raise ValueError("--val-size должен быть в диапазоне (0, 1).")

    set_seed(args.seed)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    component_train_df = read_csv_strict(args.component_train)
    component_test_df = read_csv_strict(args.component_test)
    scenario_train_df = read_csv_strict(args.scenario_train)
    scenario_test_df = read_csv_strict(args.scenario_test)

    validate_required_columns(component_train_df, [ID_COL, COMPONENT_NAME_COL], "component_train")
    validate_required_columns(component_test_df, [ID_COL, COMPONENT_NAME_COL], "component_test")
    validate_required_columns(scenario_train_df, [ID_COL, *TARGET_COLS], "scenario_train")
    validate_required_columns(scenario_test_df, [ID_COL], "scenario_test")

    train_ids, valid_ids = train_test_split(
        scenario_train_df[ID_COL].tolist(),
        test_size=args.val_size,
        random_state=args.seed,
    )

    scenario_train_split_df = scenario_train_df[
        scenario_train_df[ID_COL].isin(train_ids)
    ].copy().reset_index(drop=True)
    scenario_valid_split_df = scenario_train_df[
        scenario_train_df[ID_COL].isin(valid_ids)
    ].copy().reset_index(drop=True)

    component_train_split_df = component_train_df[
        component_train_df[ID_COL].isin(train_ids)
    ].copy().reset_index(drop=True)
    component_valid_split_df = component_train_df[
        component_train_df[ID_COL].isin(valid_ids)
    ].copy().reset_index(drop=True)

    feature_spec = build_feature_spec(
        component_train_df=component_train_split_df,
        scenario_train_df=scenario_train_split_df,
    )

    train_dataset = ScenarioSetDataset(
        component_df=component_train_split_df,
        scenario_df=scenario_train_split_df,
        feature_spec=feature_spec,
        is_train=True,
    )
    valid_dataset = ScenarioSetDataset(
        component_df=component_valid_split_df,
        scenario_df=scenario_valid_split_df,
        feature_spec=feature_spec,
        is_train=True,
    )
    test_dataset = ScenarioSetDataset(
        component_df=component_test_df,
        scenario_df=scenario_test_df,
        feature_spec=feature_spec,
        is_train=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=make_collate_fn(feature_spec, is_train=True),
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=make_collate_fn(feature_spec, is_train=True),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=make_collate_fn(feature_spec, is_train=False),
    )

    categorical_cardinalities = {
        column: len(vocabulary)
        for column, vocabulary in feature_spec["category_vocabularies"].items()
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = InteractionSetRegressor(
        component_numeric_dim=len(feature_spec["component_numeric_mean"]),
        global_dim=len(feature_spec["global_columns"]),
        categorical_cardinalities=categorical_cardinalities,
        model_dim=64,
        pair_hidden_dim=128,
        dropout=0.15,
        num_interaction_blocks=2,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    loss_fn = nn.SmoothL1Loss(beta=1.0)

    best_state: dict[str, torch.Tensor] | None = None
    best_metrics: dict[str, Any] | None = None
    best_epoch = -1
    best_score = float("inf")
    no_improvement_epochs = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        sample_count = 0

        for batch in train_loader:
            optimizer.zero_grad()

            predictions = model(
                component_numeric=batch["component_numeric"].to(device),
                component_categorical={key: value.to(device) for key, value in batch["component_categorical"].items()},
                component_mask=batch["component_mask"].to(device),
                global_features=batch["global_features"].to(device),
            )
            targets = batch["targets"].to(device)

            loss = loss_fn(predictions, targets)
            if not torch.isfinite(loss):
                raise RuntimeError("Loss стал NaN/Inf. Проверьте входные данные и масштабирование.")

            loss.backward()
            optimizer.step()

            batch_size = targets.shape[0]
            running_loss += float(loss.item()) * batch_size
            sample_count += batch_size

        train_loss = running_loss / max(sample_count, 1)
        valid_metrics, _ = evaluate_model(model, valid_loader, device, feature_spec)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "valid_mean_rmse": valid_metrics["mean_rmse"],
                "valid_mean_mae": valid_metrics["mean_mae"],
                "valid_mean_r2": valid_metrics["mean_r2"],
            }
        )

        print(
            f"epoch={epoch} "
            f"train_loss={train_loss:.6f} "
            f"valid_mean_rmse={valid_metrics['mean_rmse']:.6f} "
            f"valid_mean_r2={valid_metrics['mean_r2']:.6f}"
        )

        if valid_metrics["mean_rmse"] < best_score:
            best_score = valid_metrics["mean_rmse"]
            best_metrics = valid_metrics
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            no_improvement_epochs = 0
        else:
            no_improvement_epochs += 1

        if no_improvement_epochs >= args.patience:
            break

    if best_state is None or best_metrics is None:
        raise RuntimeError("Не удалось сохранить лучшую модель.")

    model.load_state_dict(best_state)

    final_valid_metrics, valid_predictions_df = evaluate_model(model, valid_loader, device, feature_spec)
    test_predictions_df = predict_for_test(model, test_loader, device, feature_spec)

    metrics_output = {
        "best_epoch": best_epoch,
        "best_validation_metrics": best_metrics,
        "final_validation_metrics": final_valid_metrics,
        "selected_component_numeric_columns": (
            feature_spec["component_numeric_columns"] + feature_spec["component_auto_numeric_columns"]
        ),
        "selected_component_categorical_columns": feature_spec["component_categorical_columns"],
        "global_columns": feature_spec["global_columns"],
        "history": history,
    }

    valid_predictions_df.to_csv(out_dir / "validation_predictions_interaction_model.csv", index=False)
    test_predictions_df.to_csv(out_dir / "test_predictions_interaction_model.csv", index=False)
    save_json(metrics_output, out_dir / "validation_metrics_interaction_model.json")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_spec": {
                "component_numeric_columns": feature_spec["component_numeric_columns"],
                "component_auto_numeric_columns": feature_spec["component_auto_numeric_columns"],
                "component_categorical_columns": feature_spec["component_categorical_columns"],
                "global_columns": feature_spec["global_columns"],
                "category_vocabularies": feature_spec["category_vocabularies"],
                "component_numeric_mean": feature_spec["component_numeric_mean"].tolist(),
                "component_numeric_std": feature_spec["component_numeric_std"].tolist(),
                "global_mean": feature_spec["global_mean"].tolist(),
                "global_std": feature_spec["global_std"].tolist(),
                "target_mean": feature_spec["target_mean"].tolist(),
                "target_std": feature_spec["target_std"].tolist(),
            },
        },
        out_dir / "interaction_model.pt",
    )

    print("Обучение завершено.")
    print(f"Лучшая эпоха: {best_epoch}")
    print(json.dumps(best_metrics, ensure_ascii=False, indent=2))
    print(f"Validation predictions: {(out_dir / 'validation_predictions_interaction_model.csv').resolve()}")
    print(f"Test predictions: {(out_dir / 'test_predictions_interaction_model.csv').resolve()}")
    print(f"Metrics: {(out_dir / 'validation_metrics_interaction_model.json').resolve()}")
    print(f"Model: {(out_dir / 'interaction_model.pt').resolve()}")


if __name__ == "__main__":
    main()