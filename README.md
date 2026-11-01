# Leakage-Controlled, Explainable Machine Learning for Self-Reported Stroke History (BRFSS 2023)

Code for the book chapter:

> Suri, Divya, and Samta Rani. "An Explainable Machine Learning Framework for Identifying Self-Reported Stroke History with Comprehensive Data Preprocessing." In *Machine Learning Applications in Healthcare and Bioinformatics*. Cambridge Scholars Publishing. Accepted, in revision.

The pipeline classifies **self-reported stroke history** (BRFSS item CVDSTRK3) from the 2023 Behavioral Risk Factor Surveillance System. The design priorities are:

- **No leakage.** The data are split into training, validation and test partitions (70/15/15, stratified) before anything is fitted. All imputation, encoding and scaling happens inside scikit-learn pipelines fitted on training data. Calibration and decision thresholds are chosen on the validation partition. The test partition is used once.
- **Codebook-verified preprocessing.** Missing-value codes are audited against the CDC codebook, and skip-logic fills are tested against the data. The analysis is restricted to core questionnaire items, and the same completeness rule is applied to respondents with and without stroke.
- **Honest evaluation.** Results use calibrated probabilities with 95% confidence intervals and paired DeLong tests with Holm adjustment. They also include a factorial ablation of imputation × feature selection, robustness checks, checks of SHAP stability and LIME fidelity, and subgroup fairness.

## Main result

| Feature set | Model | Test AUC (95% CI) | Brier (calibrated) |
|---|---|---|---|
| B: 51 predictors, ten likely post-stroke items removed (primary) | Stacking ensemble | 0.817 (0.810–0.825) | 0.0376 |
| A: all 61 core predictors | Stacking ensemble | 0.834 (0.827–0.841) | 0.0371 |

Test partition: 63,523 respondents, 2,703 with stroke history.

The model classifies *prevalent* stroke from cross-sectional survey data. It does **not** predict future stroke and is not intended for clinical use.

## Notebooks

Run them in order. Each one reads the outputs of the previous ones.

| Notebook | Purpose | Main outputs |
|---|---|---|
| `01_cleaning` | Load BRFSS 2023, convert codebook-verified missing codes, recode, build the outcome | `data/processed/brfss_cleaned_v2.csv`, `outputs/cleaning_flow_v2.csv` |
| `01b_core_vs_module` | Download the CDC codebook; tag each item as core or optional module; missingness analysis | `outputs/variable_dictionary.csv` |
| `02_split_and_features` | Decode composite items, verified skip-logic fills, feature sets A/B, partial-interview rule, data split | `data/processed/brfss_model_ready_v2.csv`, `outputs/feature_config_v2.json` |
| `03_model_training` | Tune and train six classifiers on both feature sets (class weights, no SMOTE) | `models/v2/*.joblib`, validation AUC |
| `03b_feature_selection_factorial` | Five-method voting feature selection on training data, stability, imputation comparison, 2×2 factorial ablation | selected features, factorial results |
| `04_final_evaluation` | Platt calibration, thresholds, single test-set evaluation, bootstrap CIs, DeLong tests, sensitivity analyses | `outputs/v2_test_*.csv`, figures |
| `04b_robustness_checks` | Five robustness checks (R1–R5), each re-tuned | `outputs/v2_robustness.csv` |
| `05_explainability_fairness` | TreeSHAP, linear SHAP, SHAP stability, LIME fidelity, subgroup performance | `outputs/v2_shap_*.csv`, `outputs/v2_fairness.csv` |

`src/stroke_utils.py` contains the MissForest-style imputer used in notebooks 03b and 04. Notebook 03b also writes this file to Drive.

## How to run

The notebooks are written for **Google Colab** and expect this folder in Google Drive:

```
MyDrive/stroke_prediction/
├── data/raw/LLCP2023.XPT      ← BRFSS 2023 data file (download below)
├── data/processed/            ← created by the notebooks
├── models/                    ← created by the notebooks
├── outputs/                   ← created by the notebooks
└── src/                       ← created by notebook 03b
```

1. Download the 2023 BRFSS data (SAS Transport format) from the [CDC BRFSS 2023 page](https://www.cdc.gov/brfss/annual_data/annual_2023.html). Unzip it and place `LLCP2023.XPT` in `data/raw/`.
2. Open each notebook in Colab, in the order above, and run all cells. Notebook 01b downloads the codebook itself.
3. Approximate run times on free Colab: notebook 03 takes 3–5 hours, 03b 1.5–2 hours, and the others under 45 minutes each. Long notebooks resume from saved models after a disconnect.

To run elsewhere, change `BASE_DIR` at the top of each notebook and remove the `drive.mount` cell.

## Software

Python 3 (Google Colab), scikit-learn 1.6.1, XGBoost 3.4.1, LightGBM 4.6.0, SHAP, LIME and pyreadstat. Packages missing from Colab are installed by the notebooks. All analyses use a fixed random seed of 42.

## Data

BRFSS data are public and are not included in this repository. Please cite the CDC: Centers for Disease Control and Prevention, *Behavioral Risk Factor Surveillance System Survey Data*, 2023.
