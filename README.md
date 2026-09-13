# NSL-KDD baseline runs

Joseph Stokes and Yoma Ogueh - undergraduate research partners working with Dr. Mustafa Abdallah.

This repository documents our joint research and the code used to train seven model types on NSL-KDD in September 2026. Our aim was to become familiar with the group's existing implementation and preserve preliminary baselines for discussion before XAI-Guided Evasion Attack (XGEA) experiments. We have not implemented XGEA experiments here yet.

## Start here: which code was used?

- `*_ALL_FINAL_original.py`: downloaded copies of the authors' [XAI_NIDS NSL-KDD scripts](https://github.com/ogarreche/XAI_NIDS/tree/main/NSL-KDD). These original files were not edited.
- `save_original_*_run.py`: AI-assisted launchers prepared with Codex for our baseline runs. They execute the original training/evaluation sections, skip explanation experiments and save models, processed inputs, predictions and logs. Three launchers apply explicit execution fixes described below.
- `reports/result_manifest.json`: metrics calculated from saved test predictions, with the exact run folders used.
- `reports/source_provenance.json`: upstream URLs, original-source hashes, dataset hashes, recorded package versions and execution changes for those runs.
- `requirements-lock.txt`: package versions exported from the working Python environment when this repository was prepared. Some dependencies were installed between runs; per-run metadata provides the versions recorded at each run.

This is not a new implementation of the seven algorithms and not an exact reproduction of the paper's result tables. The original algorithms/code should be attributed to their authors; the wrappers and documentation are local research support work. Early alternative baseline drafts were not used for these results and are excluded from this repository.

## Data and task

Data source: [HoaNP/NSL-KDD-DataSet](https://github.com/HoaNP/NSL-KDD-DataSet), a mirror of [UNB's NSL-KDD dataset](https://www.unb.ca/cic/datasets/nsl.html).

- `KDDTrain+.txt`: 125,973 source training records.
- `KDDTest+.txt`: 22,544 test records.
- Five categories: Normal, DoS, Probe, R2L and U2R.
- Some original scripts oversample the training set. Preprocessing is not standardized across scripts.

The datasets, virtual environment and large trained-model files remain local and are excluded from Git. Their hashes and result summaries are included for traceability.

## Windows setup

Run these from the repository folder in a PowerShell terminal. Python 3.11 was used; the recorded runs used Python 3.11.9 on Windows.

Create an isolated environment and install the recorded packages:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
```

Download the two input files:

```powershell
New-Item -ItemType Directory -Force data
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/HoaNP/NSL-KDD-DataSet/master/KDDTrain%2B.txt" -OutFile "data\KDDTrain+.txt"
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/HoaNP/NSL-KDD-DataSet/master/KDDTest%2B.txt" -OutFile "data\KDDTest+.txt"
```

These download URLs track the mirror's current branch. Compare downloaded file hashes against `reports/source_provenance.json` when reproducing the recorded runs.

## Train and save

Run one command at a time. Each launcher locates its own data folder, writes to a new timestamped folder under `results/`, and leaves the downloaded original script unchanged.

```powershell
.\.venv\Scripts\python.exe save_original_rf_run.py
.\.venv\Scripts\python.exe save_original_ada_run.py
.\.venv\Scripts\python.exe save_original_knn_run.py
.\.venv\Scripts\python.exe save_original_light_run.py
.\.venv\Scripts\python.exe save_original_svm_run.py
.\.venv\Scripts\python.exe save_original_mlp_run.py
.\.venv\Scripts\python.exe save_original_dnn_run.py
```

MLP and DNN can remain quiet during training. DNN retains two consecutive 50-epoch fits; the first training-time message does not indicate completion. Wait for the saved-model message and the terminal prompt. Several original estimators have no fixed seed, so repeated scores need not be identical.

## Our recorded results

Overall accuracy is computed from the saved five-category predictions, not the per-class one-versus-rest accuracy printed by the original scripts.

| Model | Overall accuracy | Macro F1 |
|---|---:|---:|
| Random Forest, evaluated second configuration | 69.30% | 42.82% |
| AdaBoost | 49.34% | 23.10% |
| KNN | 75.32% | 50.85% |
| LightGBM | 60.28% | 36.76% |
| SVM via SGD hinge loss | 71.99% | 43.50% |
| MLP | 75.98% | 56.13% |
| DNN | 68.81% | 42.52% |

Macro F1 assigns equal weight to the five classes, with undefined F1 values treated as zero. Full per-class recall and grouped attack metrics are in the result manifest. These are preliminary observations from different configurations, not evidence that one algorithm is inherently better.

## Original settings and documented fixes

| Model | Configuration actually run | Launcher-specific execution fix |
|---|---|---|
| RF | First: 200 trees/depth 15; evaluated second: 10 trees/depth 5 | None; both models saved |
| AdaBoost | 1 estimator, learning rate 1.0; minority oversampling | None |
| KNN | 3 neighbors; minority oversampling | None |
| LightGBM | `LGBMClassifier(random_state=0)` | Skip two references to an undefined optional SHAP feature list; retain all features |
| SVM | `SGDClassifier(loss='hinge')`, multioutput wrapper; minority oversampling | Align test columns by name to training-column order |
| MLP | Hidden layer `(100,)`, `max_iter=200`, `random_state=1`; minority oversampling | None |
| DNN | Dense 128/64 ReLU, 5 softmax; Adam; batch size 32; two 50-epoch fits | Restore commented scaler initialization and function declaration in the executed text |

All launchers skip XAI-only sections and add saving/logging. Source snapshots saved with each run are the unmodified originals; the launcher explains which sections and fixes were executed.

## Limitations to discuss before XGEA

- The repository settings differ from the paper in several places. For example, Appendix C lists RF with 100 trees/depth 10, AdaBoost with 50 estimators and KNN with 5 neighbors.
- Original scripts fit some preprocessing separately on test data. This behavior was retained rather than silently redesigned.
- DNN retains the original test-column order without an alignment correction; this needs checking before interpreting its score as a validated baseline.
- Most scripts choose a category by applying `argmax` to multioutput hard predictions. DNN instead uses softmax probabilities. These procedures are not equivalent.
- Original printed AUC values use hard labels. They should not be described as probability-based ROC AUC. They are not the overall accuracy values above.
- RF's printed timing belongs to its first model. DNN's printed training time covers only its first 50 epochs. ADA/KNN mislabel prediction time as a second training time.
- Saved processed inputs are not a reusable transformation pipeline for raw traffic. Input transformations and valid feature constraints still need to be defined before XGEA.
- Library defaults may differ from those used by the authors. This environment has not been verified in a fresh installation or against the authors' original environment.

## Attribution

Original code: [ogarreche/XAI_NIDS](https://github.com/ogarreche/XAI_NIDS).

Paper: Osvaldo Arreche, Tanish Guntur and Mustafa Abdallah. "XAI-IDS: Toward Proposing an Explainable Artificial Intelligence Framework for Enhancing Network Intrusion Detection Systems." Applied Sciences 2024, 14, 4170. [DOI: 10.3390/app14104170](https://doi.org/10.3390/app14104170).

Dataset reference: M. Tavallaee, E. Bagheri, W. Lu and A. Ghorbani, "A Detailed Analysis of the KDD CUP 99 Data Set," CISDA 2009. Refer to the dataset provider and upstream repository for applicable terms. This repository does not claim ownership of or assign a new license to the upstream code or dataset.
