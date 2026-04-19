from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.feature_engineering import build_o2_feature_tables
from src.train_hierarchical_model import main as train_hierarchical_main


DATA_DIR = ROOT / "data"
MERGED_DATA_DIR = THIS_DIR / "data_with_patents"
RUNTIME_DIR = THIS_DIR / "_runtime"
OUT_DIR = RUNTIME_DIR / "train_out"
MODEL_DIR = THIS_DIR / "model"

TRAIN_PATH = DATA_DIR / "daimler_mixtures_train.csv"
TEST_PATH = DATA_DIR / "daimler_mixtures_test.csv"
PROPS_PATH = DATA_DIR / "daimler_component_properties.csv"

PATENT_TRAIN_PATH = THIS_DIR / "daimler_mixtures_train_patent_attach.csv"
PATENT_PROPS_PATH = THIS_DIR / "daimler_component_properties_patent_attach.csv"


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def build_merged_data() -> tuple[Path, Path, Path]:
    mixtures_train = read_csv(TRAIN_PATH)
    mixtures_test = read_csv(TEST_PATH)
    component_props = read_csv(PROPS_PATH)

    patent_train = read_csv(PATENT_TRAIN_PATH)
    patent_props = read_csv(PATENT_PROPS_PATH)

    merged_train = pd.concat([mixtures_train, patent_train], ignore_index=True)
    merged_props = pd.concat([component_props, patent_props], ignore_index=True)

    MERGED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    merged_train_path = MERGED_DATA_DIR / "daimler_mixtures_train.csv"
    merged_test_path = MERGED_DATA_DIR / "daimler_mixtures_test.csv"
    merged_props_path = MERGED_DATA_DIR / "daimler_component_properties.csv"

    merged_train.to_csv(merged_train_path, index=False)
    mixtures_test.to_csv(merged_test_path, index=False)
    merged_props.to_csv(merged_props_path, index=False)

    return merged_train_path, merged_test_path, merged_props_path


def build_training_tables() -> tuple[Path, Path, Path, Path]:
    merged_train_path, merged_test_path, merged_props_path = build_merged_data()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return build_o2_feature_tables(
        merged_train_path,
        merged_test_path,
        merged_props_path,
        RUNTIME_DIR,
    )


def train_model() -> Path:
    component_train_path, component_test_path, scenario_train_path, scenario_test_path = build_training_tables()

    argv_backup = list(sys.argv)
    sys.argv = [
        "patent_data/train.py",
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

    shutil.copyfile(trained_checkpoint_path, MODEL_DIR / "trained_model_with_patents.pt")
    shutil.copyfile(fresh_prediction_path, THIS_DIR / "prediction_fresh_retrain_with_patents.csv")

    print(f"Trained checkpoint saved to: {(MODEL_DIR / 'trained_model_with_patents.pt').resolve()}")
    print(
        f"Fresh retrain prediction saved to: "
        f"{(THIS_DIR / 'prediction_fresh_retrain_with_patents.csv').resolve()}"
    )


if __name__ == "__main__":
    main()
