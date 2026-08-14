# UC10 — Healthcare Claims Anomaly Detection System

> **End-to-end data quality monitoring, multi-method anomaly detection, risk scoring, and investigator-ready reporting for healthcare claims data.**

---

## Table of Contents

- [Overview](#overview)
- [Pipeline Architecture](#pipeline-architecture)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [How to Run Each Stage](#how-to-run-each-stage)
- [Stage Details](#stage-details)
  - [Stage 1 — Data Quality Monitoring](#stage-1--data-quality-monitoring)
  - [Stage 2 — Data Cleaning](#stage-2--data-cleaning)
  - [Stage 3 — Statistical Analysis](#stage-3--statistical-analysis)
  - [Stage 4 — Rule-Based Anomaly Detection](#stage-4--rule-based-anomaly-detection)
  - [Stage 5 — ML Anomaly Detection (Isolation Forest)](#stage-5--ml-anomaly-detection-isolation-forest)
  - [Stage 6 — ML Model Validation](#stage-6--ml-model-validation)
  - [Stage 7 — Risk Scoring & Investigation Queue](#stage-7--risk-scoring--investigation-queue)
  - [Stage 8 — Investigation & Explanation](#stage-8--investigation--explanation)
  - [Stage 9 — Final Risk Scoring & Provider Prioritization](#stage-9--final-risk-scoring--provider-prioritization)
- [Scoring Formula](#scoring-formula)
- [Web Dashboard](#web-dashboard)
- [Key Output Files](#key-output-files)
- [Assumptions & Limitations](#assumptions--limitations)
- [Tech Stack](#tech-stack)

---

## Overview

This project implements a **multi-layered healthcare claims anomaly detection system** built for the CTS Hackathon. It processes a combined dataset of **10,000 records** spanning three claim types:

| Record Type | Prefix | Count |
|---|---|---|
| **MEDICAL_CLAIM** | `MC` | ~5,000 |
| **PHARMACY_CLAIM** | `PH` | ~3,000 |
| **PRIOR_AUTH** | `PA` | ~2,000 |

The system applies **three independent anomaly detection methods** — Statistical Analysis, Rule-Based Detection, and Isolation Forest ML — then consolidates their outputs into a unified risk score per record and a prioritized investigation queue for auditors.

---

## Pipeline Architecture

```
┌───────────────────────────────────────────────────────────────────────────┐
│                          RAW DATASET (10,000 records)                    │
│               data/claims_pharmacy_auth_monitor_dataset_features.csv     │
└──────────────────────────────────┬────────────────────────────────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │  Stage 1: Data Quality       │
                    │  Profiling & SLA Risk         │
                    └──────────────┬───────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │  Stage 2: Data Cleaning      │
                    │  (3 separate cleaned CSVs)   │
                    └──────────────┬───────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              ▼                    ▼                     ▼
   ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
   │ Stage 3:         │ │ Stage 4:         │ │ Stage 5:         │
   │ Statistical      │ │ Rule-Based       │ │ Isolation Forest │
   │ Analysis         │ │ Anomaly Det.     │ │ ML Detection     │
   └────────┬─────────┘ └────────┬─────────┘ └────────┬─────────┘
            │                    │                     │
            │                    │          ┌──────────▼──────────┐
            │                    │          │ Stage 6: ML Model   │
            │                    │          │ Validation           │
            │                    │          └──────────┬──────────┘
            │                    │                     │
            └────────────────────┼─────────────────────┘
                                 │
                    ┌────────────▼─────────────┐
                    │ Stage 7: Risk Scoring &   │
                    │ Investigation Queue        │
                    └────────────┬──────────────┘
                                 │
                    ┌────────────▼─────────────┐
                    │ Stage 8: Investigation &  │
                    │ Explanation Layer          │
                    └────────────┬──────────────┘
                                 │
                    ┌────────────▼─────────────┐
                    │ Stage 9: Final Risk       │
                    │ Scoring & Provider         │
                    │ Prioritization             │
                    └──────────────────────────┘
```

---

## Project Structure

```
UC10-data-quality-monitor/
│
├── data/                                    # Raw input dataset
│   └── claims_pharmacy_auth_monitor_dataset_features.csv
│
├── src/                                     # Stage 1 & 2: Data Quality + Cleaning
│   ├── main.py                              # Orchestrator for profiling + quality engine
│   ├── app.py                               # Streamlit web dashboard
│   ├── profiler.py                          # Data profiling & baseline statistics
│   ├── rule_catalog.py                      # 21 configurable data quality rules
│   ├── quality_engine.py                    # Rule execution engine
│   ├── scoring.py                           # Dimension scores & SLA risk calculation
│   ├── medical_claim_cleaner.py             # Cleaning logic for MEDICAL_CLAIM records
│   └── statistical_analysis.py             # Statistical analysis runner
│
├── outputs/                                 # Cleaned datasets & quality reports
│   ├── data_profile.json                    # Data profiling results
│   ├── quality_report.json                  # Quality rule execution results
│   ├── batch_sla_risk.json                  # SLA risk assessment
│   ├── medical_claim_cleaned.csv            # Cleaned MEDICAL_CLAIM dataset
│   ├── pharmacy_claim_cleaned.csv           # Cleaned PHARMACY_CLAIM dataset
│   └── authorization_cleaned.csv            # Cleaned PRIOR_AUTH dataset
│
├── statistical_analysis/                    # Stage 3: Statistical Analysis
│   ├── medical/                             # Medical claim statistics & outliers
│   ├── pharmacy/                            # Pharmacy claim statistics & outliers
│   └── authorization/                       # Prior auth statistics & outliers
│
├── anomaly_detection/                       # Stage 4: Rule-Based Anomaly Detection
│   ├── rule_config.py                       # Rule definitions & severity levels
│   ├── rule_engine.py                       # Rule execution engine
│   ├── medical_rules.py                     # Medical claim-specific rules
│   ├── pharmacy_rules.py                    # Pharmacy claim-specific rules
│   ├── authorization_rules.py               # Prior auth-specific rules
│   ├── anomaly_utils.py                     # Shared utility functions
│   ├── test_rules.py                        # Unit tests for rule engine
│   └── outputs/                             # Rule violation CSVs
│       └── combined_rule_anomalies.csv
│
├── ml_anomaly_detection/                    # Stage 5: ML Anomaly Detection
│   ├── config.py                            # Shared ML configuration
│   ├── preprocessing.py                     # Feature engineering & encoding
│   ├── medical_model.py                     # Isolation Forest for MEDICAL_CLAIM
│   ├── pharmacy_model.py                    # Isolation Forest for PHARMACY_CLAIM
│   ├── prior_auth_model.py                  # Isolation Forest for PRIOR_AUTH
│   ├── train_models.py                      # Model training orchestrator
│   ├── predict.py                           # Prediction & anomaly scoring
│   ├── validation.py                        # Model validation utilities
│   ├── models/                              # Saved model .pkl files
│   ├── outputs/                             # ML prediction CSVs
│   │   ├── *_ml_predictions.csv             # Per-record anomaly scores
│   │   └── *_ml_anomalies.csv               # Flagged anomalies with explanations
│   ├── reports/                             # Training reports
│   └── model_validation/                    # Stage 6: Model Validation
│       ├── run_validation.py                # Comprehensive validation suite
│       ├── FINAL_MODEL_VALIDATION_REPORT.md # Human-readable validation report
│       ├── cross_method_consensus.csv       # ML vs Rule vs Statistical agreement
│       ├── contamination_sensitivity.csv    # Contamination parameter sweep
│       ├── model_stability_report.csv       # Seed stability results
│       ├── data_leakage_check.txt           # Train/test leakage analysis
│       ├── anomaly_score_statistics.csv     # Score distribution analysis
│       ├── model_feature_summary.csv        # Feature importance rankings
│       └── plots/                           # Visualization plots
│
├── risk_scoring/                            # Stage 7: Risk Scoring & Investigation Queue
│   ├── run_risk_scoring.py                  # Main risk scoring orchestrator
│   ├── risk_scoring_engine.py               # Core scoring logic
│   ├── investigation_queue.py               # Investigation queue builder
│   ├── provider_risk.py                     # Provider-level risk aggregation
│   ├── config.py                            # Scoring weights & thresholds
│   ├── validate_risk_scoring.py             # Validation suite
│   ├── RISK_SCORING_REPORT.md               # Human-readable scoring report
│   ├── record_risk_scores.csv               # Per-record risk scores
│   ├── investigation_queue.csv              # Prioritized investigation queue
│   ├── provider_risk_summary.csv            # Provider-level risk summary
│   └── risk_scoring_validation.csv          # Validation check results
│
├── investigation_and_explanation/           # Stage 8: Investigation & Explanation
│   ├── run_investigation.py                 # Explanation generator
│   ├── INVESTIGATION_REPORT.md              # Human-readable investigation report
│   ├── final_investigation_queue.csv        # Enriched investigation queue
│   ├── provider_investigation_summary.csv   # Provider investigation details
│   ├── dashboard_data.csv                   # Dashboard-ready data export
│   └── validation_report.csv               # Validation check results
│
├── final_risk_scoring/                      # Stage 9: Final Risk Scoring
│   ├── run_final_scoring.py                 # Master scoring & consolidation
│   ├── run_validation.py                    # 10-check validation suite
│   ├── test_final_risk_scoring.py           # 11 unit/integration tests
│   ├── FINAL_RISK_SCORING_REPORT.md         # Final report with weights & examples
│   ├── final_risk_validation_report.md      # Validation PASS/FAIL results
│   ├── final_risk_scores.csv                # All 10,000 scored records
│   ├── investigation_queue.csv              # 7,211 flagged records (prioritized)
│   ├── provider_risk_summary.csv            # Provider NPI risk aggregation
│   ├── provider_cross_record_type_summary.csv # Cross-domain provider patterns
│   ├── overall_risk_summary.csv             # Dashboard summary metrics
│   ├── record_type_risk_summary.csv         # Per-record-type breakdowns
│   ├── detection_method_summary.csv         # Per-detection-method statistics
│   └── risk_distribution.csv               # Score distribution by priority
│
├── requirements.txt                         # Python dependencies
└── README.md                                # This file
```

---

## Setup Instructions

### Prerequisites
- Python 3.11+
- pip

### 1. Install Dependencies
```bash
pip install -r requirements.txt
pip install scikit-learn matplotlib seaborn streamlit
```

### 2. Place the Dataset
Ensure the raw dataset is located at:
```
data/claims_pharmacy_auth_monitor_dataset_features.csv
```

---

## How to Run Each Stage

Each stage is self-contained and can be run independently. Stages must be run in order for the first time since later stages depend on earlier outputs.

```bash
# Stage 1: Data Quality Monitoring + Data Profiling
python src/main.py

# Stage 2: Data Cleaning (produces 3 cleaned CSVs in outputs/)
# (Runs as part of src/main.py or via the cleaning scripts)

# Stage 3: Statistical Analysis
python src/statistical_analysis.py

# Stage 4: Rule-Based Anomaly Detection
python anomaly_detection/rule_engine.py

# Stage 5: ML Anomaly Detection (Train + Predict)
python ml_anomaly_detection/train_models.py

# Stage 6: ML Model Validation
python ml_anomaly_detection/model_validation/run_validation.py

# Stage 7: Risk Scoring & Investigation Queue
python risk_scoring/run_risk_scoring.py

# Stage 8: Investigation & Explanation Layer
python investigation_and_explanation/run_investigation.py

# Stage 9: Final Risk Scoring & Provider Prioritization
python final_risk_scoring/run_final_scoring.py
python final_risk_scoring/run_validation.py

# Run Unit Tests for Final Scoring
python -m pytest final_risk_scoring/test_final_risk_scoring.py -v

# Launch the Web Dashboard
streamlit run src/app.py
```

---

## Stage Details

### Stage 1 — Data Quality Monitoring

Profiles the raw dataset to compute baseline statistics (row counts, data types, null rates, unique values) and evaluates **21 configurable data quality rules** across five quality dimensions:

| Dimension | Description |
|---|---|
| **Completeness** | Missing/null value checks |
| **Validity** | Format, range, and enum validation |
| **Consistency** | Cross-field logical checks |
| **Timeliness** | Date freshness & SLA compliance |
| **Uniqueness** | Duplicate detection |

Outputs an overall quality score (0–100), dimension scores, and SLA breach risk assessment.

### Stage 2 — Data Cleaning

Splits the combined dataset into three cleaned CSVs by `Record_Type`:
- `outputs/medical_claim_cleaned.csv` (MEDICAL_CLAIM)
- `outputs/pharmacy_claim_cleaned.csv` (PHARMACY_CLAIM)
- `outputs/authorization_cleaned.csv` (PRIOR_AUTH)

Applies type coercion, date parsing, and null handling. Original data is **never modified**.

### Stage 3 — Statistical Analysis

For each record type independently, computes:
- **Continuous variable statistics**: mean, std, min, max, percentiles, skewness, kurtosis
- **Categorical frequency distributions**: value counts and proportions
- **Z-score outlier detection**: flags records with |Z| > 3.0 as statistical outliers

Outputs are stored per record type in `statistical_analysis/{medical,pharmacy,authorization}/`.

### Stage 4 — Rule-Based Anomaly Detection

Applies **domain-specific business rules** to each record type:
- **Medical Claims**: billing amount thresholds, date consistency, duplicate billing, modifier validation
- **Pharmacy Claims**: NDC format checks, quantity limits, refill frequency, pricing anomalies
- **Prior Auth**: decision date logic, status transitions, linked claim validation

Each violation is tagged with a **Rule ID**, **Severity** (High/Medium/Low), and human-readable description. Combined results are written to `anomaly_detection/outputs/combined_rule_anomalies.csv`.

### Stage 5 — ML Anomaly Detection (Isolation Forest)

Trains **three separate Isolation Forest models**, one per record type, using the following configuration:

| Parameter | Value |
|---|---|
| `n_estimators` | 300 |
| `contamination` | 0.02 (2%) |
| `random_state` | 42 |

Each model:
1. Selects record-type-specific numerical features
2. Applies `StandardScaler` normalization
3. Trains an Isolation Forest on the scaled features
4. Computes anomaly scores, percentile rankings, and top contributing features
5. Saves the trained model as a `.pkl` file

### Stage 6 — ML Model Validation

Validates each trained model across **five dimensions**:

1. **Train/Test Split Validation**: 80/20 split, compares anomaly rates between train and test sets to detect overfitting
2. **Seed Stability Analysis**: Retrains with 5 different random seeds, measures anomaly rate standard deviation
3. **Contamination Sensitivity Sweep**: Tests contamination values [0.01, 0.02, 0.03, 0.05, 0.10] and compares score distributions
4. **Data Leakage Check**: Verifies no test samples appear in training data
5. **Cross-Method Consensus**: Compares ML flags against Rule-Based and Statistical flags

Results documented in `model_validation/FINAL_MODEL_VALIDATION_REPORT.md`.

### Stage 7 — Risk Scoring & Investigation Queue

Consolidates signals from all three detection methods into a per-record risk score and builds a prioritized investigation queue. Also computes provider-level NPI risk aggregations.

### Stage 8 — Investigation & Explanation

Generates **plain-language, investigator-friendly explanations** for every flagged record. Explains *why* each record was flagged by each method and what specific violations or deviations were found.

### Stage 9 — Final Risk Scoring & Provider Prioritization

The master consolidation layer that produces the final deliverables: scored records, prioritized investigation queue, and provider risk summaries. See [Scoring Formula](#scoring-formula) below.

---

## Scoring Formula

The final risk score is computed on a **0–100 scale** using a weighted combination of three independent signals:

```
Final_Risk_Score = ML_Contribution + Rule_Contribution + Statistical_Contribution
```

| Signal | Weight | Scoring Logic |
|---|---|---|
| **ML (Isolation Forest)** | 40% | `(Anomaly_Percentile / 100) × 40.0` when flagged; else `0.0` |
| **Rule-Based** | 35% | High severity → 35.0; Medium → 20.0; Low → 10.0; None → 0.0 |
| **Statistical** | 25% | Max Z > 3.0 → 25.0; 1.5 < Max Z ≤ 3.0 → 15.0; else 0.0 |

### Risk Priority Brackets

| Priority | Score Range | Count | Action |
|---|---|---|---|
| **CRITICAL** | 80.0 – 100.0 | 84 | Immediate priority audit |
| **HIGH** | 60.0 – 79.99 | 235 | High priority review |
| **MEDIUM** | 40.0 – 59.99 | 1,317 | Standard review queue |
| **LOW** | 0.0 – 39.99 | 8,364 | Normal billing baseline |

### Consensus Levels

- **Consensus 3**: All three methods agree the record is anomalous (highest confidence)
- **Consensus 2**: Two of three methods flag the record
- **Consensus 1**: Only one method flags the record

---

## Web Dashboard

A **Streamlit** dashboard is included for interactive visualization:

```bash
streamlit run src/app.py
```

The dashboard displays:
- **Executive Summary**: total records, overall quality score, risk level, critical issues
- **Dimension Scores**: completeness, validity, consistency, timeliness, uniqueness
- **SLA Risk Assessment**: batch SLA breach rates, retry counts
- **Rule Execution Details**: failed rules by severity, top issues, full rules table

---

## Key Output Files

| File | Description |
|---|---|
| `final_risk_scoring/final_risk_scores.csv` | All 10,000 records with final risk scores (0–100) |
| `final_risk_scoring/investigation_queue.csv` | 7,211 flagged records sorted by priority and score |
| `final_risk_scoring/provider_risk_summary.csv` | Provider NPI-level risk aggregations |
| `final_risk_scoring/provider_cross_record_type_summary.csv` | Cross-domain provider behavior patterns |
| `anomaly_detection/outputs/combined_rule_anomalies.csv` | All rule-based violations |
| `ml_anomaly_detection/outputs/*_ml_predictions.csv` | ML anomaly scores per record type |
| `model_validation/FINAL_MODEL_VALIDATION_REPORT.md` | ML model validation report |
| `outputs/quality_report.json` | Data quality rule execution results |

---

## Assumptions & Limitations

1. **Read-Only Pipeline**: The original dataset is never modified, cleaned, or overwritten. All cleaning produces new output files.
2. **Date Parsing**: Dates are parsed using `errors="coerce"`. Unparseable strings become `NaT` and are flagged as missing.
3. **No Fraud Confirmation**: The system ranks records and providers by anomalous deviation and rule violations. These are prioritizations for investigation, **not confirmations of billing fraud**.
4. **Baseline Sensitivity**: Statistical baselines represent training-split averages. Major changes to billing structures may require recomputing baselines.
5. **Single-Batch Scope**: Referential integrity checks (e.g., `Auth_Linked_ID` lookups) are performed within the same batch. Historical lookups would require database access.
6. **Source System Strings**: Source systems must exactly match `CARRIER_CLAIMS_SYS`, `PHARMACY_ADJ_SYS`, and `AUTH_MGMT_SYS`. Any variation triggers a failure.

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.11 |
| Data Processing | Pandas, NumPy |
| ML Models | scikit-learn (Isolation Forest) |
| Visualization | Matplotlib, Seaborn |
| Web Dashboard | Streamlit |
| Version Control | Git / GitHub |
