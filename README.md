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

Biological mappings were checked by exact sequence matching between the downloaded GENEB task snapshot and the original upstream Zenodo release. Promoter sequences were matched to named negative/positive archive folders (36,131 records); coding and species sequences were joined to the semantic labels in the original test CSVs (24,997 and 24,357 unique matched sequences respectively). All matched sequences agreed on the mappings below. Enhancer labels were checked against the dataset owner's clarification and the original test CSV.

| Task | Class 0 | Class 1 |
| --- | --- | --- |
| Promoters | Non-promoter | Non-TATA promoter |
| Enhancers | Background | Enhancer |
| Coding | Coding | Intergenic |
| Species | Worm | Human |

**The earlier main-branch change used the reverse species mapping. This update corrects the displayed meanings without changing numeric model outputs or retraining.** Do not infer encodings from alphabetical class names; the GENEB distribution's numeric labels must be checked against the actual source sequences. Per-task evidence and source URLs are recorded in `training_metadata.json`.


## Validation

```bash
python -m pip install pytest
python -m pytest -q tests
```

The 13 regression tests compare refactored features with the original deployed transformation, check invalid inputs and the worked example, exercise all five sections, all four laboratory choices and all 16 dataset/metric combinations, and check prediction state persistence and missing-artifact handling. UI tests use a synthetic dummy model, not project weights.

Separately smoke-tested all 32 real saved artifacts under scikit-learn 1.6.1 on two benchmark example sequences each (64 predictions). These are functionality checks, not a new accuracy evaluation; some predictions differ from example labels, as expected for imperfect models. The cloud browser could not connect to the local Streamlit preview, so browser visual QA remains a deployment check.

## Remaining checks before publishing

- Visually review a Streamlit preview or verify the deployed changes after authorized publication.
- Confirm exact historical data revisions and hardware if recoverable; never claim the current source hash is the historical training hash.
- Add confusion matrices only from actual saved true/predicted labels. Aggregate scores cannot reconstruct them.
- Consider multiple seeds, convergence checks, ablations and a source-split similarity audit as future experiments.

## References

- [GENEB paper](https://proceedings.mlr.press/v306/ledneva26b.html)
- [GENEB task collection](https://huggingface.co/datasets/darlednik/geneb-tasks)
- [Enhancer label clarification by dataset owner](https://huggingface.co/datasets/katarinagresova/Genomic_Benchmarks_human_enhancers_cohn/discussions/1)
- [Genomic Benchmarks](https://github.com/ML-Bioinfo-CEITEC/genomic_benchmarks)
