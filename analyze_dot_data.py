from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from dot_data import load_scenarios, mechanism_aggregates, scenario_summary


ROOT = Path(__file__).resolve().parent
TRAIN_PATH = ROOT / "daimler_mixtures_train.csv"
TEST_PATH = ROOT / "daimler_mixtures_test.csv"
PROPERTY_PATH = ROOT / "daimler_component_properties.csv"


def describe_dataset(name: str, scenarios):
    component_counts = [len(s.components) for s in scenarios]
    families = Counter()
    condition_combos = Counter()
    uncertainty = Counter()

    for scenario in scenarios:
        condition_combos[(scenario.temperature_c, scenario.time_h, scenario.biofuel_pct, scenario.catalyst_category)] += 1
        summary = scenario_summary(scenario)
        for family in summary["family_mass_share"]:
            families[family] += 1
        for prop_name, count in summary["property_uncertainty"].items():
            uncertainty[prop_name] += count

    print(f"DATASET: {name}")
    print(f"scenarios={len(scenarios)}")
    print(
        "components_per_scenario="
        f"min={min(component_counts)} "
        f"max={max(component_counts)} "
        f"avg={sum(component_counts)/len(component_counts):.2f}"
    )
    print(f"families={families.most_common()}")
    print(f"condition_combos={condition_combos}")
    print(f"most_uncertain_properties={uncertainty.most_common(15)}")
    print()


def preview_mechanisms(scenarios):
    print("MECHANISM PREVIEW")
    for scenario in scenarios[:3]:
        print(scenario.scenario_id)
        payload = mechanism_aggregates(scenario)
        compact = {}
        for group, group_payload in payload.items():
            compact[group] = {
                "token_count": group_payload["token_count"],
                "mass_share_total": round(group_payload["mass_share_total"], 3),
                "property_count": len(group_payload["properties"]),
                "sample_properties": list(group_payload["properties"].keys())[:5],
            }
        print(json.dumps(compact, ensure_ascii=False, indent=2))
        print()


def main():
    train = load_scenarios(TRAIN_PATH, PROPERTY_PATH, include_targets=True)
    test = load_scenarios(TEST_PATH, PROPERTY_PATH, include_targets=False)

    describe_dataset("train", train)
    describe_dataset("test", test)
    preview_mechanisms(train)


if __name__ == "__main__":
    main()
