# GENEB Explorer

A GENEB-inspired university project comparing four conventional classifiers and their modified configurations on four DNA classification tasks using 64 normalized 3-mer frequencies.

## Run locally

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Run from the repository root so Streamlit loads `.streamlit/config.toml`. The supported dark theme keeps inputs, labels, alerts and the sidebar consistent. Python 3.12 and the pinned scikit-learn 1.6.1 were used for local validation.

Models use Git LFS. Run `git lfs pull` in a trusted checkout before real inference. Only load artifacts from the project owner: joblib/pickle files can execute code. This update does not retrain or modify any project models or result CSVs.

## Training evidence

Reviewed the owner's `GENEB_ML_PROJECT.ipynb` and `GENEB_Website.ipynb` without executing their cells. The website notebook packages the saved artifacts and builds the earlier app. The training notebook:

- Downloads task-specific train/test CSVs from `darlednik/geneb-tasks`; numeric labels are used directly.
- Extracts overlapping 3-mers and L1-normalizes them. Promoters use a learned training vocabulary; the other tasks use a fixed vocabulary. The inspected promoter training snapshot contains all 64 3-mers in the deployed order.
- Keeps StandardScaler and supervised feature selection within model pipelines during GridSearchCV. Modified logistic regression and SVM use three-fold CV scored by MCC. No supplied test-label use for tuning was found in the shown code; source split homology or chromosome overlap has not been audited.
- Uses random_state=42 on stochastic estimators. Modified MLP has early_stopping=True and validation_fraction=0.15.
- Computes macro-F1, accuracy and MCC on the supplied test split.
- Measures only model.fit, including CV search and refit where present. Excludes feature extraction and test prediction; hardware was not recorded.
- Contains a convergence warning during modified MLP training on Human vs Worm. Combined changes and one seed do not support causal or statistical-significance claims.

“Baseline” in the interface corresponds to “Original” in CSV rows and filenames. There are 32 reported model–dataset evaluations, with extra fits during tuning.

## Data provenance

`training_metadata.json` summarizes source CSVs downloaded at revision `005717000cfffd8d506d1bdd9c5c3c458895d428`. Counts, class counts, lengths and SHA-256 checksums were computed directly. The original training notebook did not pin its download revision, so the current snapshot is distinguished from the historical training input.

| Task | Train | Test | Length (bases) |
| --- | ---: | ---: | ---: |
| Non-TATA promoters | 27,097 | 9,034 | 251 |
| Enhancers (Cohn) | 20,843 | 6,948 | 500 |
| Coding versus intergenic | 75,000 | 25,000 | 200 |
| Human versus worm | 75,000 | 25,000 | 200 |

`demo_sequences.json` retains one current-source test row per numeric class per task, with its row index. They are benchmark examples, not proof of accuracy. Synthetic examples have no known biological class.

The enhancer dataset owner's clarification identifies 0 as background and 1 as enhancer. That biological mapping is shown with its source. The existing main-branch class mappings added in commit `944b2a6` are preserved for the other tasks. The UI marks them as project-configured mappings pending independent source verification: neither the training notebook nor the inspected task CSVs documents the correspondence. Do not treat alphabetical folder order alone as proof; the upstream PyTorch loader enumerates directories without sorting.

## Validation

```bash
python -m pip install pytest
python -m pytest -q tests
```

The 12 regression tests compare refactored features with the original deployed transformation, check invalid inputs and the worked example, exercise all five sections, all four laboratory choices and all 16 dataset/metric combinations, and check prediction state persistence and missing-artifact handling. UI tests use a synthetic dummy model, not project weights.

Separately smoke-tested all eight real logistic-regression artifacts under scikit-learn 1.6.1 on benchmark example sequences. These are functionality checks, not a new accuracy evaluation. Random forest, SVM and MLP artifact inference and visual layout still require a separate preview deployment before merging.

## Remaining checks before publishing

- Independently verify the preserved main-branch biological label mappings for promoters, coding and species.
- Visually review a separate Streamlit preview and retest the remaining real model families.
- Confirm exact historical data revisions and hardware if recoverable; never claim the current source hash is the historical training hash.
- Add confusion matrices only from actual saved true/predicted labels. Aggregate scores cannot reconstruct them.
- Consider multiple seeds, convergence checks, ablations and a source-split similarity audit as future experiments.

## References

- [GENEB paper](https://proceedings.mlr.press/v306/ledneva26b.html)
- [GENEB task collection](https://huggingface.co/datasets/darlednik/geneb-tasks)
- [Enhancer label clarification by dataset owner](https://huggingface.co/datasets/katarinagresova/Genomic_Benchmarks_human_enhancers_cohn/discussions/1)
- [Genomic Benchmarks](https://github.com/ML-Bioinfo-CEITEC/genomic_benchmarks)
