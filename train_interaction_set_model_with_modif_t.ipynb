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
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset


ID_COL = "scenario_id"
TARGET_COLS = [
    "Delta Kin. Viscosity KV100 - relative | - Daimler Oxidation Test (DOT), %",
    "Oxidation EOT | DIN 51453 Daimler Oxidation Test (DOT), A/cm",
]
COMPONENT_NAME_COL = "Компонент"
TARGET_TRAINING_CONFIGS: dict[int, dict[str, Any]] = {
    0: {
        "top_k": 10,
        "learning_rate": 5e-4,
        "weight_decay": 5e-4,
        "dropout": 0.25,
        "model_dim": 64,
        "pair_hidden_dim": 128,
        "num_interaction_blocks": 2,
        "scheduler_factor": 0.5,
        "scheduler_patience": 8,
        "min_learning_rate": 1e-5,
        "grad_clip_norm": 1.0,
    },
    1: {
        "top_k": 15,
        "learning_rate": 8e-4,
        "weight_decay": 1e-4,
        "dropout": 0.15,
        "model_dim": 64,
        "pair_hidden_dim": 128,
        "num_interaction_blocks": 2,
        "scheduler_factor": 0.5,
        "scheduler_patience": 8,
        "min_learning_rate": 1e-5,
        "grad_clip_norm": 1.0,
    },
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def fit_target_transform(values: np.ndarray, target_index: int, mode: str) -> dict[str, Any]:
    clean_values = np.asarray(values, dtype=np.float32)
    if mode == "none":
        return {"type": "identity", "shift": 0.0}
    if mode == "auto" and target_index == 0:
        min_value = float(np.nanmin(clean_values))
        shift = max(0.0, 1.0 - min_value)
        return {"type": "log1p_shift", "shift": shift}
    return {"type": "identity", "shift": 0.0}


def apply_target_transform(values: np.ndarray, transform_spec: dict[str, Any]) -> np.ndarray:
    values_array = np.asarray(values, dtype=np.float32)
    transform_type = transform_spec["type"]

    if transform_type == "identity":
        return values_array
    if transform_type == "log1p_shift":
        shifted = np.maximum(values_array + float(transform_spec["shift"]), 0.0)
        return np.log1p(shifted).astype(np.float32)

    raise ValueError(f"Неизвестный тип target transform: {transform_type}")


def inverse_target_transform(values: np.ndarray, transform_spec: dict[str, Any]) -> np.ndarray:
    values_array = np.asarray(values, dtype=np.float32)
    transform_type = transform_spec["type"]

    if transform_type == "identity":
        return values_array
    if transform_type == "log1p_shift":
        restored = np.expm1(values_array) - float(transform_spec["shift"])
        return restored.astype(np.float32)

    raise ValueError(f"Неизвестный тип target transform: {transform_type}")


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
    target_transform_mode: str = "none",
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
    target_transform_specs: list[dict[str, Any]] = []
    transformed_target_columns: list[np.ndarray] = []
    for target_index in range(len(TARGET_COLS)):
        transform_spec = fit_target_transform(
            target_matrix[:, target_index],
            target_index,
            mode=target_transform_mode,
        )
        target_transform_specs.append(transform_spec)
        transformed_column = apply_target_transform(target_matrix[:, target_index], transform_spec)
        transformed_target_columns.append(transformed_column.reshape(-1, 1))

    transformed_target_matrix = np.concatenate(transformed_target_columns, axis=1)
    target_mean = np.mean(transformed_target_matrix, axis=0)
    target_std = np.std(transformed_target_matrix, axis=0)
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
        "target_transform_specs": target_transform_specs,
        "target_mean": target_mean.astype(np.float32),
        "target_std": target_std.astype(np.float32),
    }

def select_features_per_target(
    component_df: pd.DataFrame,
    scenario_df: pd.DataFrame,
    feature_spec: dict[str, Any],
    target_index: int,
    top_k: int = 5,
) -> dict[str, Any]:
    """
    Отбираем наиболее важные признаки под конкретный таргет
    """
    # агрегируем компонентные признаки (mean pooling)
    grouped = component_df.groupby(ID_COL)

    X_list = []
    ids = []
    for scenario_id, group in grouped:
        numeric = encode_component_numeric(group, feature_spec)
        pooled = np.mean(numeric, axis=0)
        X_list.append(pooled)
        ids.append(scenario_id)

    X = np.array(X_list)

    y_df = scenario_df.set_index(ID_COL).loc[ids]
    y = y_df[TARGET_COLS[target_index]].to_numpy(dtype=np.float32)
    y = apply_target_transform(y, feature_spec["target_transform_specs"][target_index])

    # mutual information
    mi = mutual_info_regression(X, y, random_state=0)
    top_idx = np.argsort(mi)[-top_k:]

    selected_columns = [
        feature_spec["component_numeric_columns"][i]
        if i < len(feature_spec["component_numeric_columns"])
        else feature_spec["component_auto_numeric_columns"][i - len(feature_spec["component_numeric_columns"])]
        for i in top_idx
    ]

    # создаем копию feature_spec
    new_spec = copy.deepcopy(feature_spec)

    new_spec["component_numeric_columns"] = [
        col for col in feature_spec["component_numeric_columns"] if col in selected_columns
    ]
    new_spec["component_auto_numeric_columns"] = [
        col for col in feature_spec["component_auto_numeric_columns"] if col in selected_columns
    ]
    # пересчитываем mean/std под новые фичи
    filtered_df = component_df.copy()

    numeric_arrays = []
    for col in new_spec["component_numeric_columns"]:
        values = pd.to_numeric(filtered_df[col], errors="coerce").to_numpy(dtype=np.float32)
        numeric_arrays.append(values.reshape(-1, 1))

    for col in new_spec["component_auto_numeric_columns"]:
        values = filtered_df[col].map(parse_numeric_like).to_numpy(dtype=np.float32)
        numeric_arrays.append(values.reshape(-1, 1))

    numeric_matrix = np.concatenate(numeric_arrays, axis=1).astype(np.float32)

    mean = np.nanmean(numeric_matrix, axis=0)
    std = np.nanstd(numeric_matrix, axis=0)

    mean = np.where(np.isnan(mean), 0.0, mean)
    std = np.where((np.isnan(std)) | (std < 1e-8), 1.0, std)

    new_spec["component_numeric_mean"] = mean.astype(np.float32)
    new_spec["component_numeric_std"] = std.astype(np.float32)

    return new_spec

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
        target_index: int,
        is_train: bool,
    ) -> None:
        self.target_index = target_index
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
                targets = np.array([scenario_row[TARGET_COLS[self.target_index]]], dtype=np.float32)
                targets = apply_target_transform(
                    targets,
                    feature_spec["target_transform_specs"][self.target_index],
                )
                target_mean = feature_spec["target_mean"][self.target_index]
                target_std = feature_spec["target_std"][self.target_index]

                targets = (targets - target_mean) / target_std
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
            targets = torch.zeros((batch_size, 1), dtype=torch.float32)

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
            nn.Linear(64, 1),
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


def inverse_scale_target(values: np.ndarray, feature_spec: dict, target_index: int):
    mean = feature_spec["target_mean"][target_index]
    std = feature_spec["target_std"][target_index]
    transformed_values = values * std + mean
    return inverse_target_transform(
        transformed_values,
        feature_spec["target_transform_specs"][target_index],
    )


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    y_true = y_true.squeeze(-1)
    y_pred = y_pred.squeeze(-1)

    finite_mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not finite_mask.any():
        raise ValueError("Не удалось посчитать метрики: все значения y_true/y_pred невалидны.")

    y_true = y_true[finite_mask]
    y_pred = y_pred[finite_mask]

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred)) if y_true.shape[0] >= 2 else float("nan")

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    feature_spec: dict[str, Any],
    target_index: int
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

    targets = inverse_scale_target(scaled_targets, feature_spec, target_index=target_index)
    predictions = inverse_scale_target(scaled_predictions, feature_spec, target_index=target_index)

    metrics = calculate_metrics(targets, predictions)

    output_df = pd.DataFrame({ID_COL: scenario_ids})
    target_name = TARGET_COLS[target_index]

    output_df[target_name] = targets.squeeze(-1)
    output_df[f"{target_name}_pred"] = predictions.squeeze(-1)
    return metrics, output_df


def predict_for_test(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    feature_spec: dict[str, Any],
    target_index:int
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
    predictions = inverse_scale_target(scaled_predictions, feature_spec, target_index)

    output_df = pd.DataFrame({ID_COL: scenario_ids})
    target_name = TARGET_COLS[target_index]

    output_df[target_name] = predictions.squeeze(-1)

    return output_df


def save_json(data: dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def serialize_feature_spec(feature_spec: dict[str, Any]) -> dict[str, Any]:
    serialized: dict[str, Any] = {}
    for key, value in feature_spec.items():
        if isinstance(value, np.ndarray):
            serialized[key] = value.tolist()
        else:
            serialized[key] = value
    return serialized


def get_target_training_config(args: argparse.Namespace, target_index: int) -> dict[str, Any]:
    defaults = {
        "top_k": 15,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "dropout": 0.15,
        "model_dim": 64,
        "pair_hidden_dim": 128,
        "num_interaction_blocks": 2,
        "scheduler_factor": 0.5,
        "scheduler_patience": 8,
        "min_learning_rate": 1e-5,
        "grad_clip_norm": 1.0,
    }
    defaults.update(TARGET_TRAINING_CONFIGS.get(target_index, {}))
    return defaults


def summarize_fold_metrics(fold_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for metric_name in ("rmse", "mae", "r2"):
        values = [float(metrics[metric_name]) for metrics in fold_metrics if np.isfinite(metrics[metric_name])]
        summary[f"mean_{metric_name}"] = float(np.mean(values)) if values else float("nan")
        summary[f"std_{metric_name}"] = float(np.std(values)) if values else float("nan")
    return summary


def make_regression_stratification_labels(
    target_values: np.ndarray,
    min_count_per_bin: int,
    max_bins: int = 10,
) -> np.ndarray | None:
    values = np.asarray(target_values, dtype=np.float32)
    unique_count = int(pd.Series(values).nunique(dropna=True))
    upper_bins = min(max_bins, unique_count)

    for num_bins in range(upper_bins, 1, -1):
        labels = pd.qcut(
            pd.Series(values).rank(method="first"),
            q=num_bins,
            labels=False,
            duplicates="drop",
        )
        counts = pd.Series(labels).value_counts()
        if counts.shape[0] >= 2 and int(counts.min()) >= min_count_per_bin:
            return labels.to_numpy(dtype=np.int64)

    return None


def build_train_valid_splits(
    scenario_train_df: pd.DataFrame,
    target_index: int,
    num_folds: int,
    val_size: float,
    seed: int,
    use_regression_stratify: bool,
) -> list[tuple[list[str], list[str]]]:
    scenario_ids = scenario_train_df[ID_COL].tolist()
    target_values = scenario_train_df[TARGET_COLS[target_index]].to_numpy(dtype=np.float32)

    if num_folds <= 1:
        stratify_labels = (
            make_regression_stratification_labels(target_values, min_count_per_bin=2)
            if use_regression_stratify
            else None
        )
        train_ids, valid_ids = train_test_split(
            scenario_ids,
            test_size=val_size,
            random_state=seed,
            stratify=stratify_labels,
        )
        return [(train_ids, valid_ids)]

    if num_folds > len(scenario_ids):
        raise ValueError("--num-folds не может быть больше числа train-сценариев.")

    stratify_labels = (
        make_regression_stratification_labels(target_values, min_count_per_bin=num_folds)
        if use_regression_stratify
        else None
    )
    if stratify_labels is None:
        kfold = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
        splits: list[tuple[list[str], list[str]]] = []
        scenario_ids_array = np.asarray(scenario_ids)
        for train_idx, valid_idx in kfold.split(scenario_ids_array):
            splits.append((
                scenario_ids_array[train_idx].tolist(),
                scenario_ids_array[valid_idx].tolist(),
            ))
        return splits

    kfold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
    splits: list[tuple[list[str], list[str]]] = []
    scenario_ids_array = np.asarray(scenario_ids)
    for train_idx, valid_idx in kfold.split(scenario_ids_array, stratify_labels):
        splits.append((
            scenario_ids_array[train_idx].tolist(),
            scenario_ids_array[valid_idx].tolist(),
        ))
    return splits


def order_by_reference_ids(df: pd.DataFrame, reference_ids: list[str], value_columns: list[str]) -> pd.DataFrame:
    ordered = pd.DataFrame({ID_COL: reference_ids})
    return ordered.merge(df[[ID_COL, *value_columns]], on=ID_COL, how="left")


def average_test_predictions(fold_test_predictions: list[pd.DataFrame], target_name: str) -> pd.DataFrame:
    combined = pd.concat(fold_test_predictions, ignore_index=True)
    return combined.groupby(ID_COL, as_index=False)[target_name].mean()


def train_single_fold(
    *,
    component_train_df: pd.DataFrame,
    component_valid_df: pd.DataFrame,
    component_test_df: pd.DataFrame,
    scenario_train_df: pd.DataFrame,
    scenario_valid_df: pd.DataFrame,
    scenario_test_df: pd.DataFrame,
    args: argparse.Namespace,
    target_index: int,
    fold_index: int,
    device: torch.device,
) -> dict[str, Any]:
    fold_seed = args.seed + target_index * 100 + fold_index
    set_seed(fold_seed)

    base_feature_spec = build_feature_spec(
        component_train_df=component_train_df,
        scenario_train_df=scenario_train_df,
        target_transform_mode=args.target_transform_mode,
    )
    target_config = get_target_training_config(args, target_index)
    target_feature_spec = select_features_per_target(
        component_train_df,
        scenario_train_df,
        base_feature_spec,
        target_index,
        top_k=target_config["top_k"],
    )

    train_dataset = ScenarioSetDataset(
        component_df=component_train_df,
        scenario_df=scenario_train_df,
        feature_spec=target_feature_spec,
        is_train=True,
        target_index=target_index,
    )
    valid_dataset = ScenarioSetDataset(
        component_df=component_valid_df,
        scenario_df=scenario_valid_df,
        feature_spec=target_feature_spec,
        is_train=True,
        target_index=target_index,
    )
    test_dataset = ScenarioSetDataset(
        component_df=component_test_df,
        scenario_df=scenario_test_df,
        feature_spec=target_feature_spec,
        is_train=False,
        target_index=target_index,
    )

    generator = torch.Generator()
    generator.manual_seed(fold_seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=make_collate_fn(target_feature_spec, True),
        generator=generator,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=make_collate_fn(target_feature_spec, True),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=make_collate_fn(target_feature_spec, False),
    )

    categorical_cardinalities = {
        column: len(vocabulary)
        for column, vocabulary in target_feature_spec["category_vocabularies"].items()
    }
    model_init_kwargs = {
        "component_numeric_dim": len(target_feature_spec["component_numeric_mean"]),
        "global_dim": len(target_feature_spec["global_columns"]),
        "categorical_cardinalities": categorical_cardinalities,
        "model_dim": target_config["model_dim"],
        "pair_hidden_dim": target_config["pair_hidden_dim"],
        "dropout": target_config["dropout"],
        "num_interaction_blocks": target_config["num_interaction_blocks"],
    }

    model = InteractionSetRegressor(**model_init_kwargs).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=target_config["learning_rate"],
        weight_decay=target_config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=target_config["scheduler_factor"],
        patience=target_config["scheduler_patience"],
        min_lr=target_config["min_learning_rate"],
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
            optimizer.zero_grad(set_to_none=True)

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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=target_config["grad_clip_norm"])
            optimizer.step()

            batch_size = targets.shape[0]
            running_loss += float(loss.item()) * batch_size
            sample_count += batch_size

        train_loss = running_loss / max(sample_count, 1)
        valid_metrics, _ = evaluate_model(
            model,
            valid_loader,
            device,
            target_feature_spec,
            target_index=target_index,
        )
        scheduler.step(valid_metrics["rmse"])
        current_lr = float(optimizer.param_groups[0]["lr"])

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "valid_rmse": valid_metrics["rmse"],
                "valid_mae": valid_metrics["mae"],
                "valid_r2": valid_metrics["r2"],
                "learning_rate": current_lr,
            }
        )

        print(
            f"fold={fold_index} "
            f"epoch={epoch} "
            f"train_loss={train_loss:.6f} "
            f"valid_rmse={valid_metrics['rmse']:.6f} "
            f"valid_r2={valid_metrics['r2']:.6f} "
            f"lr={current_lr:.6g}"
        )

        if valid_metrics["rmse"] < best_score:
            best_score = valid_metrics["rmse"]
            best_metrics = valid_metrics
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            no_improvement_epochs = 0
        else:
            no_improvement_epochs += 1

        if no_improvement_epochs >= args.patience:
            break

    if best_state is None or best_metrics is None:
        raise RuntimeError("Не удалось сохранить лучшую модель для fold.")

    model.load_state_dict(best_state)

    final_valid_metrics, valid_df = evaluate_model(
        model,
        valid_loader,
        device,
        target_feature_spec,
        target_index=target_index,
    )
    test_df = predict_for_test(
        model,
        test_loader,
        device,
        target_feature_spec,
        target_index=target_index,
    )

    return {
        "fold_index": fold_index,
        "training_config": target_config,
        "feature_spec": serialize_feature_spec(target_feature_spec),
        "model_init_kwargs": model_init_kwargs,
        "best_epoch": best_epoch,
        "best_validation_metrics": best_metrics,
        "final_validation_metrics": final_valid_metrics,
        "selected_component_numeric_columns": (
            target_feature_spec["component_numeric_columns"]
            + target_feature_spec["component_auto_numeric_columns"]
        ),
        "selected_component_categorical_columns": target_feature_spec["component_categorical_columns"],
        "global_columns": target_feature_spec["global_columns"],
        "history": history,
        "valid_df": valid_df,
        "test_df": test_df,
        "model_state_dict": best_state,
    }


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
    parser.add_argument("--num-folds", default=3, type=int)
    parser.add_argument("--target-transform-mode", choices=["none", "auto"], default="none")
    parser.add_argument("--regression-stratify", action="store_true")
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    if not 0.0 < args.val_size < 1.0:
        raise ValueError("--val-size должен быть в диапазоне (0, 1).")
    if args.num_folds < 1:
        raise ValueError("--num-folds должен быть >= 1.")

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_test_predictions: list[pd.DataFrame] = []
    all_valid_predictions: list[pd.DataFrame] = []
    all_metrics: dict[str, Any] = {}
    trained_target_artifacts: dict[str, Any] = {}

    for target_index, target_name in enumerate(TARGET_COLS):
        print(f"\n===== TRAINING FOR TARGET: {target_name} =====")
        train_valid_splits = build_train_valid_splits(
            scenario_train_df=scenario_train_df,
            target_index=target_index,
            num_folds=args.num_folds,
            val_size=args.val_size,
            seed=args.seed,
            use_regression_stratify=args.regression_stratify,
        )

        fold_results: list[dict[str, Any]] = []
        fold_valid_predictions: list[pd.DataFrame] = []
        fold_test_predictions: list[pd.DataFrame] = []

        for fold_index, (train_ids, valid_ids) in enumerate(train_valid_splits, start=1):
            print(f"\n--- fold {fold_index}/{len(train_valid_splits)} ---")

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

            fold_result = train_single_fold(
                component_train_df=component_train_split_df,
                component_valid_df=component_valid_split_df,
                component_test_df=component_test_df,
                scenario_train_df=scenario_train_split_df,
                scenario_valid_df=scenario_valid_split_df,
                scenario_test_df=scenario_test_df,
                args=args,
                target_index=target_index,
                fold_index=fold_index,
                device=device,
            )
            fold_results.append(fold_result)
            fold_valid_predictions.append(fold_result["valid_df"])
            fold_test_predictions.append(fold_result["test_df"])

        oof_valid_df = pd.concat(fold_valid_predictions, ignore_index=True)
        oof_valid_df = order_by_reference_ids(
            oof_valid_df,
            scenario_train_df[ID_COL].tolist(),
            [target_name, f"{target_name}_pred"],
        )

        test_df = average_test_predictions(fold_test_predictions, target_name)
        test_df = order_by_reference_ids(test_df, scenario_test_df[ID_COL].tolist(), [target_name])

        oof_metrics = calculate_metrics(
            oof_valid_df[[target_name]].to_numpy(dtype=np.float32),
            oof_valid_df[[f"{target_name}_pred"]].to_numpy(dtype=np.float32),
        )
        fold_metric_summary = summarize_fold_metrics(
            [fold_result["final_validation_metrics"] for fold_result in fold_results]
        )
        best_fold_result = min(
            fold_results,
            key=lambda result: result["best_validation_metrics"]["rmse"],
        )

        all_metrics[target_name] = {
            "num_folds": len(train_valid_splits),
            "training_config": fold_results[0]["training_config"],
            "oof_validation_metrics": oof_metrics,
            "fold_metric_summary": fold_metric_summary,
            "best_fold": {
                "fold_index": best_fold_result["fold_index"],
                "best_epoch": best_fold_result["best_epoch"],
                "best_validation_metrics": best_fold_result["best_validation_metrics"],
            },
            "folds": [
                {
                    "fold_index": fold_result["fold_index"],
                    "best_epoch": fold_result["best_epoch"],
                    "best_validation_metrics": fold_result["best_validation_metrics"],
                    "final_validation_metrics": fold_result["final_validation_metrics"],
                    "selected_component_numeric_columns": fold_result["selected_component_numeric_columns"],
                    "selected_component_categorical_columns": fold_result["selected_component_categorical_columns"],
                    "global_columns": fold_result["global_columns"],
                    "history": fold_result["history"],
                }
                for fold_result in fold_results
            ],
        }

        all_valid_predictions.append(oof_valid_df[[ID_COL, target_name, f"{target_name}_pred"]])
        all_test_predictions.append(test_df[[ID_COL, target_name]])

        trained_target_artifacts[target_name] = {
            "target_index": target_index,
            "num_folds": len(train_valid_splits),
            "training_config": fold_results[0]["training_config"],
            "oof_validation_metrics": oof_metrics,
            "folds": [
                {
                    "fold_index": fold_result["fold_index"],
                    "best_epoch": fold_result["best_epoch"],
                    "best_validation_metrics": fold_result["best_validation_metrics"],
                    "model_init_kwargs": fold_result["model_init_kwargs"],
                    "feature_spec": fold_result["feature_spec"],
                    "model_state_dict": fold_result["model_state_dict"],
                }
                for fold_result in fold_results
            ],
        }

    #    merge predictions
    valid_final = all_valid_predictions[0]
    for df in all_valid_predictions[1:]:
        valid_final = valid_final.merge(df, on=ID_COL)

    test_final = all_test_predictions[0]
    for df in all_test_predictions[1:]:
        test_final = test_final.merge(df, on=ID_COL)

    valid_final.to_csv(out_dir / "validation_predictions_interaction_model.csv", index=False)
    test_final.to_csv(out_dir / "test_predictions_interaction_model.csv", index=False)

    save_json(all_metrics, out_dir / "validation_metrics_interaction_model.json")

    torch.save(
        {
            "target_columns": TARGET_COLS,
            "targets": trained_target_artifacts,
        },
        out_dir / "interaction_models.pt",
    )

    print("Обучение завершено.")
    print(json.dumps(all_metrics, ensure_ascii=False, indent=2))
    print(f"Validation predictions: {(out_dir / 'validation_predictions_interaction_model.csv').resolve()}")
    print(f"Test predictions: {(out_dir / 'test_predictions_interaction_model.csv').resolve()}")
    print(f"Metrics: {(out_dir / 'validation_metrics_interaction_model.json').resolve()}")
    print(f"Models: {(out_dir / 'interaction_models.pt').resolve()}")


if __name__ == "__main__":
    main()
