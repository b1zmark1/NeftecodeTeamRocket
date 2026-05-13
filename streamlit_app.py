from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from src.feature_engineering import build_o2_feature_tables


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = ROOT / "data"
DEFAULT_MODEL_PATH = ROOT / "model" / "hierarchical_o2_baseline.pt"
RUNS_DIR = ROOT / "streamlit_runs"


@dataclass
class TrainingConfig:
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    patience: int
    val_size: float
    seed: int


def configure_page() -> None:
    st.set_page_config(
        page_title="Neftecode DOT Trainer",
        page_icon="docks/logo.png" if (ROOT / "docks" / "logo.png").exists() else None,
        layout="wide",
    )


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --rocket-bg: #0f080f;
            --rocket-bg-soft: #170d19;
            --rocket-panel: #241029;
            --rocket-panel-strong: #35183f;
            --rocket-purple: #542d68;
            --rocket-purple-light: #8f5aa9;
            --rocket-lilac: #d8b8ea;
            --rocket-text: #f6eff8;
            --rocket-muted: #bfa8c9;
            --rocket-border: rgba(216, 184, 234, 0.22);
            --rocket-font: "PP Neue Machina", "PP Neue Machina Inktrap", "PP Neue Machina Plain",
                "Neue Machina", "Inter", "Segoe UI", sans-serif;
        }

        html, body, [class*="css"] {
            font-family: var(--rocket-font);
        }

        .stApp {
            color: var(--rocket-text);
            background:
                radial-gradient(circle at 18% 8%, rgba(84, 45, 104, 0.52), transparent 34rem),
                linear-gradient(135deg, var(--rocket-bg) 0%, #160b18 46%, #25102b 100%);
        }

        .block-container {
            max-width: 1180px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        h1, h2, h3, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
            color: var(--rocket-text);
            font-family: var(--rocket-font);
            font-weight: 700;
            letter-spacing: 0;
        }

        p, label, span, div {
            font-family: var(--rocket-font);
        }

        [data-testid="stCaptionContainer"] {
            color: var(--rocket-muted);
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #170d19 0%, #0f080f 100%);
            border-right: 1px solid var(--rocket-border);
        }

        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] label {
            color: var(--rocket-text);
        }

        [data-testid="stSidebar"] [data-baseweb="radio"] {
            gap: 0.35rem;
        }

        [data-testid="stMetric"] {
            background: rgba(36, 16, 41, 0.86);
            border: 1px solid var(--rocket-border);
            border-radius: 8px;
            padding: 1rem;
        }

        [data-testid="stMetricLabel"] {
            color: var(--rocket-muted);
        }

        [data-testid="stMetricValue"] {
            color: var(--rocket-lilac);
        }

        .stButton > button,
        .stDownloadButton > button {
            background: linear-gradient(135deg, var(--rocket-purple) 0%, #7a4591 100%);
            border: 1px solid rgba(216, 184, 234, 0.36);
            border-radius: 8px;
            color: var(--rocket-text);
            font-family: var(--rocket-font);
            font-weight: 700;
            min-height: 2.75rem;
        }

        .stButton > button:hover,
        .stDownloadButton > button:hover {
            background: linear-gradient(135deg, #6a3980 0%, #9b63b5 100%);
            border-color: var(--rocket-lilac);
            color: #ffffff;
        }

        .stButton > button:disabled {
            background: #2a1a2f;
            border-color: rgba(216, 184, 234, 0.14);
            color: rgba(246, 239, 248, 0.46);
        }

        [data-baseweb="input"],
        [data-baseweb="select"],
        [data-baseweb="textarea"] {
            background-color: rgba(15, 8, 15, 0.74);
            border-color: var(--rocket-border);
            color: var(--rocket-text);
        }

        [data-baseweb="input"] input,
        [data-baseweb="textarea"] textarea {
            color: var(--rocket-text);
            font-family: var(--rocket-font);
        }

        [data-testid="stFileUploader"] section {
            background: rgba(36, 16, 41, 0.78);
            border: 1px dashed rgba(216, 184, 234, 0.36);
            border-radius: 8px;
        }

        [data-testid="stFileUploader"] section:hover {
            border-color: var(--rocket-purple-light);
        }

        [data-testid="stExpander"],
        [data-testid="stDataFrame"],
        .stAlert {
            border-radius: 8px;
        }

        [data-testid="stExpander"] {
            background: rgba(36, 16, 41, 0.72);
            border: 1px solid var(--rocket-border);
        }

        div[data-testid="stCodeBlock"] {
            border: 1px solid var(--rocket-border);
            border-radius: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def save_uploaded_file(uploaded_file, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_bytes(uploaded_file.getbuffer())


def copy_default_data(target_dir: Path) -> tuple[Path, Path, Path]:
    target_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "train": DEFAULT_DATA_DIR / "daimler_mixtures_train.csv",
        "test": DEFAULT_DATA_DIR / "daimler_mixtures_test.csv",
        "properties": DEFAULT_DATA_DIR / "daimler_component_properties.csv",
    }
    for source in paths.values():
        if not source.exists():
            raise FileNotFoundError(source)
        shutil.copyfile(source, target_dir / source.name)

    return (
        target_dir / "daimler_mixtures_train.csv",
        target_dir / "daimler_mixtures_test.csv",
        target_dir / "daimler_component_properties.csv",
    )


def save_uploaded_data(target_dir: Path, train_file, test_file, props_file) -> tuple[Path, Path, Path]:
    target_dir.mkdir(parents=True, exist_ok=True)
    train_path = target_dir / "daimler_mixtures_train.csv"
    test_path = target_dir / "daimler_mixtures_test.csv"
    props_path = target_dir / "daimler_component_properties.csv"

    save_uploaded_file(train_file, train_path)
    save_uploaded_file(test_file, test_path)
    save_uploaded_file(props_file, props_path)
    return train_path, test_path, props_path


def run_training(
    train_path: Path,
    test_path: Path,
    props_path: Path,
    out_dir: Path,
    config: TrainingConfig,
    init_checkpoint: Path | None,
) -> tuple[str, Path]:
    runtime_dir = out_dir / "runtime"
    component_train_path, component_test_path, scenario_train_path, scenario_test_path = build_o2_feature_tables(
        train_path=train_path,
        test_path=test_path,
        props_path=props_path,
        runtime_dir=runtime_dir,
    )

    argv = [
        sys.executable,
        str(ROOT / "src" / "train_hierarchical_model.py"),
        "--component-train",
        str(component_train_path),
        "--component-test",
        str(component_test_path),
        "--scenario-train",
        str(scenario_train_path),
        "--scenario-test",
        str(scenario_test_path),
        "--out-dir",
        str(out_dir),
        "--epochs",
        str(config.epochs),
        "--batch-size",
        str(config.batch_size),
        "--learning-rate",
        str(config.learning_rate),
        "--weight-decay",
        str(config.weight_decay),
        "--patience",
        str(config.patience),
        "--val-size",
        str(config.val_size),
        "--seed",
        str(config.seed),
    ]
    if init_checkpoint is not None:
        argv.extend(["--init-checkpoint", str(init_checkpoint)])

    completed = subprocess.run(
        argv,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    logs = "\n".join(part for part in (completed.stdout, completed.stderr) if part)
    if completed.returncode != 0:
        raise RuntimeError(logs or f"Training process failed with exit code {completed.returncode}")

    return logs, out_dir


def read_metrics(metrics_path: Path) -> dict:
    with metrics_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def show_metrics(metrics: dict) -> None:
    final_metrics = metrics["final_validation_metrics"]
    best_metrics = metrics["best_validation_metrics"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Best epoch", metrics["best_epoch"])
    col2.metric("Final RMSE", f"{final_metrics['mean_rmse']:.4f}")
    col3.metric("Final MAE", f"{final_metrics['mean_mae']:.4f}")
    col4.metric("Final R2", f"{final_metrics['mean_r2']:.4f}")

    with st.expander("Метрики по таргетам", expanded=False):
        rows = []
        for target_name, values in final_metrics["per_target"].items():
            rows.append({"target": target_name, **values})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    warm_start = metrics.get("warm_start")
    if warm_start:
        st.info(
            "Дообучение стартовало из checkpoint: "
            f"загружено тензоров {warm_start['loaded_tensors']}, "
            f"пропущено {warm_start['skipped_tensors']}."
        )

    with st.expander("Лучшая validation-метрика", expanded=False):
        st.json(best_metrics)


def add_download_button(label: str, path: Path, mime: str) -> None:
    if path.exists():
        st.download_button(
            label=label,
            data=path.read_bytes(),
            file_name=path.name,
            mime=mime,
            use_container_width=True,
        )


def main() -> None:
    configure_page()
    apply_theme()

    st.title("Neftecode DOT Trainer")
    st.caption("Загрузка CSV, обучение и дообучение иерархической модели Daimler DOT.")

    with st.sidebar:
        st.header("Данные")
        data_source = st.radio(
            "Источник",
            ["CSV из проекта", "Загрузить CSV"],
            horizontal=False,
        )

        train_file = test_file = props_file = None
        if data_source == "Загрузить CSV":
            train_file = st.file_uploader("daimler_mixtures_train.csv", type="csv")
            test_file = st.file_uploader("daimler_mixtures_test.csv", type="csv")
            props_file = st.file_uploader("daimler_component_properties.csv", type="csv")

        st.header("Режим")
        mode = st.radio("Запуск", ["Обучить с нуля", "Дообучить checkpoint"])

        checkpoint_file = None
        checkpoint_source = "Baseline из model/"
        if mode == "Дообучить checkpoint":
            checkpoint_source = st.radio(
                "Checkpoint",
                ["Baseline из model/", "Загрузить .pt"],
            )
            if checkpoint_source == "Загрузить .pt":
                checkpoint_file = st.file_uploader("checkpoint .pt", type="pt")

        st.header("Параметры")
        config = TrainingConfig(
            epochs=st.number_input("Epochs", min_value=1, max_value=500, value=30, step=1),
            batch_size=st.number_input("Batch size", min_value=1, max_value=128, value=8, step=1),
            learning_rate=st.number_input("Learning rate", min_value=1e-6, max_value=1.0, value=1e-3, format="%.6f"),
            weight_decay=st.number_input("Weight decay", min_value=0.0, max_value=1.0, value=1e-4, format="%.6f"),
            patience=st.number_input("Patience", min_value=1, max_value=200, value=15, step=1),
            val_size=st.slider("Validation size", min_value=0.05, max_value=0.5, value=0.2, step=0.05),
            seed=st.number_input("Seed", min_value=0, max_value=100000, value=42, step=1),
        )

    upload_ready = data_source == "CSV из проекта" or all([train_file, test_file, props_file])
    checkpoint_ready = mode == "Обучить с нуля" or checkpoint_source == "Baseline из model/" or checkpoint_file is not None
    run_button = st.button("Запустить обучение", type="primary", disabled=not (upload_ready and checkpoint_ready))

    if not upload_ready:
        st.warning("Загрузите train, test и properties CSV.")
    if mode == "Дообучить checkpoint" and checkpoint_source == "Baseline из model/" and not DEFAULT_MODEL_PATH.exists():
        st.warning(f"Baseline checkpoint не найден: {DEFAULT_MODEL_PATH}")

    if run_button:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = RUNS_DIR / run_id
        uploads_dir = run_dir / "uploads"
        out_dir = run_dir / "train_out"
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            if data_source == "CSV из проекта":
                train_path, test_path, props_path = copy_default_data(uploads_dir)
            else:
                train_path, test_path, props_path = save_uploaded_data(uploads_dir, train_file, test_file, props_file)

            init_checkpoint = None
            if mode == "Дообучить checkpoint":
                if checkpoint_source == "Baseline из model/":
                    init_checkpoint = DEFAULT_MODEL_PATH
                else:
                    init_checkpoint = run_dir / "init_checkpoint.pt"
                    save_uploaded_file(checkpoint_file, init_checkpoint)

            with st.spinner("Идет обучение модели..."):
                logs, completed_out_dir = run_training(
                    train_path=train_path,
                    test_path=test_path,
                    props_path=props_path,
                    out_dir=out_dir,
                    config=config,
                    init_checkpoint=init_checkpoint,
                )

            st.session_state["last_run"] = {
                "run_dir": str(run_dir),
                "out_dir": str(completed_out_dir),
                "logs": logs,
            }
            st.success(f"Обучение завершено. Run: {run_id}")
        except Exception as exc:
            st.error(f"Не удалось выполнить обучение: {exc}")
            st.stop()

    last_run = st.session_state.get("last_run")
    if last_run:
        out_dir = Path(last_run["out_dir"])
        metrics_path = out_dir / "validation_metrics_hierarchical_model.json"
        predictions_path = out_dir / "test_predictions_hierarchical_model.csv"
        validation_predictions_path = out_dir / "validation_predictions_hierarchical_model.csv"
        model_path = out_dir / "hierarchical_model.pt"

        if metrics_path.exists():
            st.subheader("Результаты последнего запуска")
            show_metrics(read_metrics(metrics_path))

        col1, col2, col3 = st.columns(3)
        with col1:
            add_download_button("Скачать predictions CSV", predictions_path, "text/csv")
        with col2:
            add_download_button("Скачать validation CSV", validation_predictions_path, "text/csv")
        with col3:
            add_download_button("Скачать checkpoint", model_path, "application/octet-stream")

        if predictions_path.exists():
            st.subheader("Предсказания")
            st.dataframe(pd.read_csv(predictions_path), use_container_width=True, hide_index=True)

        with st.expander("Лог обучения", expanded=False):
            st.code(last_run["logs"] or "Нет вывода.", language="text")


if __name__ == "__main__":
    main()
