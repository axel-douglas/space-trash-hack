# Rex-AI Material & Policy Model Card

## Overview

The Rex-AI pipeline couples a gradient boosted regressor with rule-based policy
logic to score in-orbit waste transformation opportunities. The regressor
predicts mission deltas (crew time, energy, sealing, rigidity, water balance)
using engineered features derived from NASA logistics studies, polymer
references and regolith characterisation. The policy engine then converts the
predictions into actionable manifests, recommendations and material passports
with compatibility justifications.

## Datasets and provenance

* **NASA logistics tables** – preprocessed CSVs derived from the Logistics to
  Living reports, Trash-to-Gas trade studies and propellant budgets. The
  preprocessing pipeline converts textual ranges to numeric features and keeps
  regeneration scripts under version control for reproducibility.【F:datasets/README.md†L1-L165】
* **Zenodo material bundle** – laminate workbooks, spectral libraries and
  polymer datasheets mirrored from Zenodo and Mendeley. The bundle records DOI
  and licence metadata, normalises mechanical units and surfaces the
  compatibility/spectral artefacts consumed by the generator.【F:datasets/README.md†L70-L115】
* **MGS-1 regolith suite** – grain size, spectra and thermogravimetry from the
  Cannon et al. simulant release. These files seed the regolith composition
  features and policy heuristics for mineral compatibility.【F:datasets/README.md†L167-L266】

The final feature matrix used for training is stored in `data/models/` alongside
metadata describing feature statistics, target columns and residual summaries to
preserve full lineage.【F:data/models/metadata.json†L1-L28】

## Evaluation

Offline validation uses the scenario and ablation benchmarks in
`data/benchmarks/`. Scenario runs represent mission configurations the policy
engine is expected to handle in production, while the ablation runs remove
critical feature groups (e.g. logistics indices, regolith chemistry) to stress
check model reliance on engineered inputs.【F:data/benchmarks/scenario_metrics.csv†L1-L10】【F:data/benchmarks/ablation_metrics.csv†L1-L20】

| Target | Scenario MAE | Scenario RMSE | Ablation MAE | Ablation RMSE |
| ------ | ------------:| -------------:| ------------:| --------------:|
| Crew time (crew_min) | 4.18e3 | 4.18e3 | 9.83e5 | 9.83e5 |
| Energy (energy_kwh) | 7.60e4 | 7.60e4 | 1.60e4 | 1.60e4 |
| Sealing (estanqueidad) | 4.38e-1 | 4.42e-1 | 9.2e-2 | 9.6e-2 |
| Rigidity (rigidez) | 7.5e-2 | 8.0e-2 | 1.95e-1 | 1.98e-1 |
| Water balance (water_l) | 4.35e2 | 4.36e2 | 1.16e2 | 1.22e2 |
| Aggregate (overall) | 1.61e4 | 3.41e4 | 1.99e5 | 4.40e5 |

Compared to ablations the scenario bundle shows a 200× reduction in mean crew
MAE and two orders of magnitude better aggregate RMSE, underscoring the value of
retaining regolith composition and logistics context in the feature space.

## Post-inference diagnostics

The model registry tracks per-target MAE/RMSE and residual standard deviations.
`python -m scripts.verify_model_ready` now exports these metrics together with
per-classifier precision/recall diagnostics, ensuring release candidates include
quantitative evidence of post-inference behaviour.【F:scripts/verify_model_ready.py†L1-L149】

Precision/recall and ROC curves generated for optional classifiers must be
packaged next to the joblib artefacts; the verification script asserts their
presence when paths are declared in `metadata.json`.【F:scripts/verify_model_ready.py†L112-L147】【F:data/models/metadata.json†L17-L36】

## Safety assumptions

* **Manual review of compatibility evidence** – the Zenodo laminate workbook and
  regolith heuristics surface partner materials and mixing rules. Operators must
  confirm the evidences before executing substitutions, especially when vendor
  datasheets impose proprietary restrictions.【F:datasets/README.md†L88-L115】【F:app/modules/policy_engine.py†L281-L338】
* **Inference bounds** – the policy engine applies deterministic penalties and
  quotas, assuming manifests fall within the NASA waste taxonomy. Inputs lacking
  recognised categories may bypass safeguards and should be filtered upstream.【F:app/modules/policy_engine.py†L41-L323】
* **Spectral heuristics** – spectral slopes and FTIR fingerprints are used only
  for coarse compatibility scoring; they are not a substitute for destructive
  testing and are documented as such in the bundle metadata.【F:app/modules/data_sources.py†L260-L336】【F:app/modules/data_sources.py†L1020-L1080】

## Limitations of the policy engine

* The compatibility matrix currently covers polymers listed in the Zenodo bundle
  and a small set of regolith fillers. Off-nominal materials fall back to neutral
  scores, so novel composites require manual curation before running the policy
  audit.【F:app/modules/data_sources.py†L697-L1114】【F:tests/test_data_sources.py†L68-L139】
* Material passports embed timestamped manifests. Although the JSON layout is
  deterministic when timestamps are fixed, downstream consumers should treat the
  output as append-only to avoid breaking audit trails.【F:app/modules/policy_engine.py†L568-L626】
* Classifier outputs (rigidity/tightness) rely on minimal training examples and
  serve as heuristics. Operators should combine them with manual inspection until
  larger labelled corpora become available.【F:data/models/metadata.json†L29-L56】

## Responsible release checklist

1. Regenerate processed datasets from raw NASA and Zenodo sources using the
   documented scripts to maintain reproducibility.【F:datasets/README.md†L117-L133】
2. Train the model and export metadata via the pipeline in `app/modules/ml_models`
   so feature statistics and residuals are captured.【F:app/modules/ml_models.py†L313-L664】
3. Run `make verify` (pytest + model verification) to ensure passports,
   compatibility exports and diagnostics remain consistent before publishing a
   release bundle.【F:Makefile†L1-L8】
