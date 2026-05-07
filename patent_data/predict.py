from __future__ import annotations

import sys
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader


THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.train_hierarchical_model import (
    HierarchicalScenarioRegressor,
    ScenarioHierarchicalDataset,
    make_collate_fn,
    predict_for_test,
)
from patent_data.train import build_training_tables


CHECKPOINT_PATH = THIS_DIR / "model" / "trained_model_with_patents.pt"
FINAL_PREDICTIONS_PATH = THIS_DIR / "predictions_with_patents.csv"


def load_checkpoint_feature_spec(raw_feature_spec: dict) -> dict:
    feature_spec = dict(raw_feature_spec)
    for key in (
        "component_numeric_mean",
        "component_numeric_std",
        "global_mean",
        "global_std",
        "target_mean",
        "target_std",
    ):
        feature_spec[key] = np.asarray(feature_spec[key], dtype=np.float32)
    return feature_spec


def main() -> None:
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"{CHECKPOINT_PATH} не найден. Сначала запустите patent_data/train.py"
        )

    _, component_test_path, _, scenario_test_path = build_training_tables()
    component_test_df = pd.read_csv(component_test_path)
    scenario_test_df = pd.read_csv(scenario_test_path)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    feature_spec = load_checkpoint_feature_spec(checkpoint["feature_spec"])

    categorical_cardinalities = {
        column: len(vocabulary)
        for column, vocabulary in feature_spec["category_vocabularies"].items()
    }

    model = HierarchicalScenarioRegressor(
        component_numeric_dim=len(feature_spec["component_numeric_mean"]),
        global_dim=len(feature_spec["global_columns"]),
        categorical_cardinalities=categorical_cardinalities,
        family_cardinality=len(feature_spec["family_vocabulary"]),
        model_dim=64,
        dropout=0.15,
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    test_dataset = ScenarioHierarchicalDataset(
        component_df=component_test_df,
        scenario_df=scenario_test_df,
        feature_spec=feature_spec,
        is_train=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=make_collate_fn(feature_spec, is_train=False),
    )

    predictions_df = predict_for_test(model, test_loader, torch.device("cpu"), feature_spec)
    predictions_df.to_csv(FINAL_PREDICTIONS_PATH, index=False)
    print(f"Predictions with patents saved to: {FINAL_PREDICTIONS_PATH.resolve()}")


if __name__ == "__main__":
    main()
