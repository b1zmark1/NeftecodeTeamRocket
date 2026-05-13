from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from src.train_hierarchical_model import main as train_hierarchical_main
from src.feature_engineering import build_o2_feature_tables


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RUNTIME_DIR = ROOT / "_notebook_runtime_o2"
OUT_DIR = RUNTIME_DIR / "train_out"
MODEL_DIR = ROOT / "model"

TRAIN_PATH = DATA_DIR / "daimler_mixtures_train.csv"
TEST_PATH = DATA_DIR / "daimler_mixtures_test.csv"
PROPS_PATH = DATA_DIR / "daimler_component_properties.csv"


def validate_optional_dependencies(args: argparse.Namespace) -> None:
    if not args.enable_mlflow:
        return
    try:
        import mlflow  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "MLflow logging requested, but mlflow is not installed. "
            "Install dev dependencies with: pip install -r requirements-dev.txt"
        ) from exc


def build_training_tables() -> tuple[Path, Path, Path, Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return build_o2_feature_tables(TRAIN_PATH, TEST_PATH, PROPS_PATH, RUNTIME_DIR)


def train_model(args: argparse.Namespace) -> Path:
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
        str(args.seed),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--learning-rate",
        str(args.learning_rate),
        "--weight-decay",
        str(args.weight_decay),
        "--patience",
        str(args.patience),
        "--val-size",
        str(args.val_size),
    ]
    if args.enable_mlflow:
        sys.argv.extend(
            [
                "--enable-mlflow",
                "--mlflow-experiment",
                args.mlflow_experiment,
            ]
        )
        if args.mlflow_tracking_uri:
            sys.argv.extend(["--mlflow-tracking-uri", args.mlflow_tracking_uri])
        if args.mlflow_run_name:
            sys.argv.extend(["--mlflow-run-name", args.mlflow_run_name])
    if args.init_checkpoint:
        sys.argv.extend(["--init-checkpoint", str(args.init_checkpoint)])
    try:
        train_hierarchical_main()
    finally:
        sys.argv = argv_backup

    fresh_prediction_path = OUT_DIR / "test_predictions_hierarchical_model.csv"
    if not fresh_prediction_path.exists():
        raise FileNotFoundError(fresh_prediction_path)
    return fresh_prediction_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build feature tables from data/ and train the hierarchical Daimler DOT model."
        )
    )
    parser.add_argument("--epochs", default=150, type=int)
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--learning-rate", default=1e-3, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--patience", default=30, type=int)
    parser.add_argument("--val-size", default=0.2, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--enable-mlflow", action="store_true")
    parser.add_argument("--mlflow-tracking-uri", default=None)
    parser.add_argument("--mlflow-experiment", default="neftecode-daimler-dot")
    parser.add_argument("--mlflow-run-name", default=None)
    parser.add_argument(
        "--init-checkpoint",
        default=None,
        type=Path,
        help="Checkpoint for warm-start retraining. Matching tensors are reused.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_optional_dependencies(args)
    fresh_prediction_path = train_model(args)
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
