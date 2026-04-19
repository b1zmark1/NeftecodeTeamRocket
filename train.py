from __future__ import annotations

import shutil
import sys
from pathlib import Path

from src.feature_engineering import build_o2_feature_tables
from src.train_hierarchical_model import main as train_hierarchical_main


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RUNTIME_DIR = ROOT / "_notebook_runtime_o2"
OUT_DIR = RUNTIME_DIR / "train_out"
MODEL_DIR = ROOT / "model"

TRAIN_PATH = DATA_DIR / "daimler_mixtures_train.csv"
TEST_PATH = DATA_DIR / "daimler_mixtures_test.csv"
PROPS_PATH = DATA_DIR / "daimler_component_properties.csv"


def build_training_tables() -> tuple[Path, Path, Path, Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return build_o2_feature_tables(TRAIN_PATH, TEST_PATH, PROPS_PATH, RUNTIME_DIR)


def train_model() -> Path:
    component_train_path, component_test_path, scenario_train_path, scenario_test_path = build_training_tables()

    argv_backup = list(sys.argv)
    sys.argv = [
        "train.py",
        "--component-train",
        str(component_train_path),
        "--component-test",
        str(component_test_path),
        "--scenario-train",
        str(scenario_train_path),
        "--scenario-test",
        str(scenario_test_path),
        "--out-dir",
        str(OUT_DIR),
        "--seed",
        "42",
    ]
    try:
        train_hierarchical_main()
    finally:
        sys.argv = argv_backup

    fresh_prediction_path = OUT_DIR / "test_predictions_hierarchical_model.csv"
    if not fresh_prediction_path.exists():
        raise FileNotFoundError(fresh_prediction_path)
    return fresh_prediction_path


def main() -> None:
    fresh_prediction_path = train_model()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    trained_checkpoint_path = OUT_DIR / "hierarchical_model.pt"
    if not trained_checkpoint_path.exists():
        raise FileNotFoundError(trained_checkpoint_path)
    shutil.copyfile(trained_checkpoint_path, MODEL_DIR / "trained_model.pt")
    shutil.copyfile(fresh_prediction_path, ROOT / "prediction_fresh_retrain.csv")
    print(f"Trained checkpoint saved to: {(MODEL_DIR / 'trained_model.pt').resolve()}")
    print(f"Fresh retrain prediction saved to: {(ROOT / 'prediction_fresh_retrain.csv').resolve()}")


if __name__ == "__main__":
    main()
