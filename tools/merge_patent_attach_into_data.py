from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ATTACH_DIR = ROOT / "patent_data"
OUT_DIR = ROOT / "data_with_patents"


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def main() -> None:
    mixtures_train = read_csv(DATA_DIR / "daimler_mixtures_train.csv")
    mixtures_test = read_csv(DATA_DIR / "daimler_mixtures_test.csv")
    component_props = read_csv(DATA_DIR / "daimler_component_properties.csv")

    patent_train = read_csv(ATTACH_DIR / "daimler_mixtures_train_patent_attach.csv")
    patent_props = read_csv(ATTACH_DIR / "daimler_component_properties_patent_attach.csv")

    merged_train = pd.concat([mixtures_train, patent_train], ignore_index=True)
    merged_props = pd.concat([component_props, patent_props], ignore_index=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    merged_train.to_csv(OUT_DIR / "daimler_mixtures_train.csv", index=False)
    mixtures_test.to_csv(OUT_DIR / "daimler_mixtures_test.csv", index=False)
    merged_props.to_csv(OUT_DIR / "daimler_component_properties.csv", index=False)

    print(f"Merged files saved to: {OUT_DIR.resolve()}")
    print(f"Train rows: {merged_train.shape[0]}")
    print(f"Properties rows: {merged_props.shape[0]}")


if __name__ == "__main__":
    main()
