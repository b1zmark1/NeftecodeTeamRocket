# Training Data Artifacts

## Files

- `train_flat_features_v1.csv` / `test_flat_features_v1.csv`
  - wide scenario-level tables
  - one row per `scenario_id`
  - contain numeric baseline features for linear models / MLP / Bayesian regression

- `train_flat_features_v1.npz` / `test_flat_features_v1.npz`
  - compressed NumPy version of the same flat features
  - fields:
    - `X`: feature matrix
    - `y`: targets, only in train
    - `feature_names`
    - `scenario_ids`

- `train_tokens_v1.jsonl` / `test_tokens_v1.jsonl`
  - tokenized scenario representation for set models
  - one JSON object per `scenario_id`
  - each object contains:
    - `conditions`
    - `tokens`
    - `targets`, only in train

- `feature_schema_v1.json`
  - schema of family blocks and feature names

## Flat feature design

The flat matrix contains:

- scenario conditions and severity features
- total dose / count / presence per family
- weighted mean / max / count for selected numeric properties inside each family
- explicit chemistry-informed interaction terms

Total: `238` numeric features.

## Token design

Each token corresponds to a unique `(component_id, batch_id)` inside a scenario after duplicate merge.

Each token stores:

- `component_id`
- `component_family`
- `batch_id`
- `dose_transformed`
- `dose_rank_in_scenario`
- `dose_share_of_total_transformed`
- `row_count_after_merge`
- family-specific numeric properties
- categorical properties
- masks:
  - `property_present`
  - `property_parse_kind`
  - `property_source`

## Notes

- `batch_id` is preserved for traceability and joins, but should not be used as a primary categorical predictor.
- Missing values are intentional and should be modeled with masks, not blindly imputed away.
- Doses are transformed values from the competition data and should not be interpreted as true mass fractions.
