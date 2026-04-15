# DOT Modeling Strategy

## What the raw data tells us

- Train contains 167 scenarios and test contains 40 scenarios.
- A scenario has 6-20 component rows in train and 7-18 in test.
- Nine chemical families are present: base oils, antioxidants, detergents, dispersants, antiwear additives, viscosity modifiers, pour point depressants, antifoam additives, and molybdenum compounds.
- The same component can appear multiple times inside one scenario. In these files that happens often for base oils. For modeling, identical `(scenario_id, component_id, batch_id)` rows should be merged by summing the transformed dosage.
- The operating space is small but structured: temperature has values `150`, `154`, `160`; time `168` or `216`; biofuel `0`, `5`, `7`; catalyst category `1` or `2`.
- The viscosity target is heavy-tailed and can be negative, while oxidation is strictly positive and moderately skewed. This argues for robust scaling and a robust regression loss, not plain MSE on raw targets.

## Parsing and cleaning rules

- Join component properties on both `component_id` and `batch_id`.
- Use measured batch properties first, then fill missing properties from the same component with batch `typical`.
- Preserve missingness explicitly. Missing values are informative because different families expose different subsets of properties.
- Parse property values with typed uncertainty:
  - exact numeric values
  - interval midpoint for ranges like `300-340`
  - boundary value for `<0.3`, `<=0.15`, `≥1.2`
  - fallback first numeric token for mixed strings such as `9,0 (Mg)`
  - missing for clearly non-numeric or malformed values
- Keep the parse mode as a feature or mask. It is part of the uncertainty model.

## Best feature representation

The final representation should be **hierarchical** instead of flat.

### 1. Scenario-level features

- `temperature_c`, `time_h`, `biofuel_pct`, `catalyst_category`
- exposure proxies such as `time_h * exp((temperature_c - 150) / 10)`
- pair interactions: `temperature x biofuel`, `temperature x catalyst`, `time x biofuel`

These features capture the test severity and should condition the component interaction module.

### 2. Component-token features

Each unique `(component_id, batch_id)` inside a scenario becomes one token with:

- transformed dosage
- component family
- component id embedding
- cleaned numeric properties
- property-presence mask
- parse-quality mask
- source mask: measured or typical

This is the right level to preserve variable component count and batch-specific behavior.

### 3. Mechanistic aggregate features

On top of the token sequence, build explicit chemistry-aware aggregates:

- **Base oil structure**: KV40, KV100, VI, CCS, pour point, NOACK, density, saturates, sulfur
- **Antioxidant activity**: AO type, bond dissociation energy, ionization potential, HOMO/LUMO, dipole moment, active N/O
- **Detergency reserve**: TBN, Ca/Mg content, soap/base ratio, micelle size, substrate class
- **Dispersancy/polarity**: N content, B content, polyamine class, succinimide type, hydrophobic tail mass, PDI
- **Antiwear/redox system**: P, Zn, S, P:Zn, Mo, S:Mo, ligand type
- **Polymer rheology**: polymer type, polymer content, molecular weight, monomer ratio, shear stability

For each group use weighted sum, weighted mean, max, count, and total contributing dosage.

### 4. Interaction features

DOT is dominated by nonlinear synergy and antagonism. That should be encoded explicitly.

- family-family dosage products: `AO x ZDDP`, `AO x detergent`, `detergent x dispersant`, `Mo x ZDDP`, `base oil x VII`
- chemistry-condition products: `AO_strength x temperature`, `biofuel x unsaturation_proxy`, `catalyst x sulfur/phosphorus`, `time x volatility`
- latent token-token interactions from the model itself

## Recommended model architecture

Given the small dataset, the best main model is a **hybrid mechanistic Deep Sets model**:

1. Token encoder for each component:
   - numeric properties
   - family embedding
   - component embedding
   - dosage
   - masks
2. Conditioning block from scenario-level test conditions.
3. Set aggregation with attention or gated pooling.
4. Small explicit pair-interaction head on top of family or latent token representations.
5. Shared trunk with two regression heads for:
   - `Delta Kin. Viscosity KV100 - relative`
   - `Oxidation EOT`

Why this is the best tradeoff here:

- Deep Sets is permutation-invariant and naturally handles variable component count.
- It is much less data-hungry than a full transformer.
- Explicit mechanistic aggregates reduce overfitting and make interpretation easier.
- A multitask head is justified because the targets are related but not identical.

Set Transformer can be tested as a secondary model, but on 167 train scenarios it is more likely to overfit unless heavily regularized.

## Feature selection strategy

- Drop raw batch id as a direct categorical feature from the main model. Use batch through measured properties instead.
- Keep component id embedding because anonymized ids still encode formulation role and repeated empirical behavior.
- Use grouped feature selection:
  - scenario conditions
  - family dosage totals
  - mechanistic aggregates
  - selected raw properties with enough support
- Prefer stability selection across repeated folds over single-run importance.
- Any property with very low support should enter only through family-specific modules or masks, not as a global dense column.

## Validation strategy

- Split only at `scenario_id`.
- Use repeated grouped cross-validation with balancing by condition combination.
- Track performance separately for:
  - high-biofuel scenarios
  - high-severity scenarios (`150C`/`216 h`)
  - scenarios containing unseen component ids in test-like simulation
- Compare at least three ablations:
  - scenario features only
  - scenario + flat aggregated chemistry features
  - full hybrid set model

## Interpretation plan

- Global: permutation importance or SHAP on the explicit aggregate branch.
- Local: attention weights and token contribution analysis for each scenario.
- Chemistry-level: report positive and negative interactions for the main mechanistic pairs.
- Uncertainty-level: report sensitivity to missing properties and to `typical` fallback.

## Literature-backed hypothesis table

| Factor | Mechanism affecting DOT | How to encode it in the model |
| --- | --- | --- |
| Temperature and time | Oxidation follows free-radical chain chemistry and accelerates strongly with test severity; oxidation products increase polarity and can raise viscosity. | Scenario severity features and severity-conditioned interaction block. |
| Biofuel contamination | FAME dilution changes oxidation behavior and can accelerate viscosity growth at relevant temperatures; effect depends on temperature and additive package. | Explicit `biofuel_pct` feature and interactions with antioxidant, base-oil and antiwear descriptors. |
| Base oil structure | Higher saturates and lower aromatics generally improve oxidative stability; volatility and viscosity profile also change thickening behavior. | Base-oil token properties and aggregated base-oil structure descriptors. |
| Antioxidant chemistry | Radical scavengers and peroxide decomposers inhibit different oxidation stages; molecular structure controls effectiveness. | Antioxidant family embeddings plus quantum-chemical and compositional descriptors. |
| ZDDP / antiwear chemistry | ZDDP acts as both radical scavenger and peroxide decomposer; P/Zn/S ratios matter. | Antiwear-redox aggregate block and interaction terms with antioxidants and biofuel. |
| Detergent reserve | Overbased detergent chemistry changes alkalinity, colloidal structure and interaction with oxidation products and water. | Detergency-reserve descriptors: TBN, Ca/Mg, soap/base ratio, micelle size. |
| Dispersants | Dispersants control sludge/insoluble stabilization and influence viscosity growth through soot and oxidation-product suspension. | Dispersancy/polarity descriptors and interaction terms with detergent and base oil. |
| Additive synergy and antagonism | Literature reports non-monotonic interactions among antioxidant, detergent, dispersant and base oil chemistry. | Pairwise family interaction head on top of Deep Sets representation. |

## Sources

- [DIN 51453 standard overview](https://store.accuristech.com/standards/din-51453?product_id=2913854)
- [Research Progress of Antioxidant Additives for Lubricating Oils](https://doi.org/10.3390/lubricants12040115)
- [A review of zinc dialkyldithiophosphates (ZDDPs): characterisation and role in the lubricating oil](https://doi.org/10.1016/S0301-679X(01)00028-7)
- [Effect of biodiesel on the autoxidation of lubricant base fluids](https://doi.org/10.1016/j.fuel.2014.01.039)
- [Impact of pure biodiesel fuel on the service life of engine-lubricant: A case study](https://doi.org/10.1016/j.fuel.2019.116418)
- [Oxidation of Soybean Biodiesel Fuel in Diesel Engine Oils](https://doi.org/10.4271/04-12-03-0015)
- [Interactions of Additives and Lubricating Base Oils](https://trid.trb.org/View/204409)
