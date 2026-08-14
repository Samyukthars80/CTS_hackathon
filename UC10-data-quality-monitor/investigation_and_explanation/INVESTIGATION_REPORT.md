# Investigator-Friendly Final Queue & Pattern Explanation Report

This document serves as the human-readable **Investigator View** for the unified anomaly detection and risk scoring framework. It provides key high-risk patterns, detailed summaries of providers requiring investigation, and traceabilities to support case-level audits.

---

## A. System Overview
The claims investigation system consolidates multiple detection signals to analyze and score healthcare transactions across three record types:
1.  **Medical Claims** (`MEDICAL_CLAIM`)
2.  **Pharmacy Claims** (`PHARMACY_CLAIM`)
3.  **Prior Authorizations** (`PRIOR_AUTH`)

By combining deterministic rule compliance, statistical distribution benchmarking, and unsupervised machine learning (Isolation Forest), the consolidated pipeline produces an explainable overall risk score (0-100) and Recommended Priority (`P1` to `P4`) for every record.

---

## B. Core Queue Analytics

### 1. Analysis Summary
*   **Total Records Scored & Analyzed**: 10,000
*   **Total Flagged Records (Score >= 25.0)**: 5,140 (51.4%)
*   **Unflagged Records (Score < 25.0)**: 4,860 (48.6%)

### 2. Risk Category Distribution
| Risk Category | Score Range | Record Count | Percentage |
| :--- | :--- | :--- | :--- |
| **CRITICAL** | 75.0 – 100.0 | 85 | 0.85% |
| **HIGH** | 50.0 – 74.99 | 1,157 | 11.57% |
| **MEDIUM** | 25.0 – 49.99 | 3,898 | 38.98% |
| **LOW** | 0.0 – 24.99 | 4,860 | 48.60% |

### 3. Investigation Priority Distribution
*   **P1 - Critical Priority**: 85 records (Immediate investigation required)
*   **P2 - High Priority**: 1,157 records (Expedited review queue)
*   **P3 - Medium Priority**: 3,898 records (Standard audit review)
*   **P4 - Low Priority**: 4,860 records (Routine billing validation)

### 4. Detection Method Contribution
*   **Rule-Engine Flagged (>=1 business violation)**: 4,482 records
*   **Statistical-Outlier Flagged (>=1 column Z-score > 1.5)**: 1,848 records
*   **ML-Anomaly Flagged (IF Anomaly)**: 200 records
*   **Consensus Levels Distribution**:
    *   **Level 3 Consensus** (Flagged by ML + Rule + Stat): 99 records (0.99%)
    *   **Level 2 Consensus** (Flagged by any two): 1,538 records (15.38%)
    *   **Level 1 Consensus** (Flagged by only one): 3,503 records (35.03%)
    *   **Level 0 Consensus** (Unflagged): 4,860 records (48.60%)

---

## C. Top 20 High-Risk Investigation Cases
These are the top 20 prioritized records from the queue (`final_investigation_queue.csv`):

| Rank | Record ID | Record Type | Score | Risk Category | Priority | Consensus | Rule Violations | Statistical Flag | ML Flag | Primary Investigation Reason |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | `MC103230` | `MEDICAL_CLAIM` | 100.00 | `CRITICAL` | `P1 - Critical` | 3 | 3 | 1 | 1 | High-severity business rule violations (Count: 2) |
| **2** | `MC102394` | `MEDICAL_CLAIM` | 100.00 | `CRITICAL` | `P1 - Critical` | 3 | 3 | 1 | 1 | High-severity business rule violations (Count: 2) |
| **3** | `MC100962` | `MEDICAL_CLAIM` | 100.00 | `CRITICAL` | `P1 - Critical` | 3 | 2 | 1 | 1 | High-severity business rule violations (Count: 2) |
| **4** | `MC104591` | `MEDICAL_CLAIM` | 100.00 | `CRITICAL` | `P1 - Critical` | 3 | 2 | 1 | 1 | High-severity business rule violations (Count: 2) |
| **5** | `MC104693` | `MEDICAL_CLAIM` | 100.00 | `CRITICAL` | `P1 - Critical` | 3 | 3 | 1 | 1 | High-severity business rule violations (Count: 2) |
| **6** | `MC101608` | `MEDICAL_CLAIM` | 100.00 | `CRITICAL` | `P1 - Critical` | 3 | 2 | 1 | 1 | High-severity business rule violations (Count: 2) |
| **7** | `MC101860` | `MEDICAL_CLAIM` | 100.00 | `CRITICAL` | `P1 - Critical` | 3 | 3 | 1 | 1 | High-severity business rule violations (Count: 2) |
| **8** | `MC103595` | `MEDICAL_CLAIM` | 100.00 | `CRITICAL` | `P1 - Critical` | 3 | 3 | 1 | 1 | High-severity business rule violations (Count: 2) |
| **9** | `MC102816` | `MEDICAL_CLAIM` | 100.00 | `CRITICAL` | `P1 - Critical` | 3 | 3 | 1 | 1 | High-severity business rule violations (Count: 2) |
| **10** | `MC100903` | `MEDICAL_CLAIM` | 100.00 | `CRITICAL` | `P1 - Critical` | 3 | 3 | 1 | 1 | High-severity business rule violations (Count: 2) |
| **11** | `MC102717` | `MEDICAL_CLAIM` | 100.00 | `CRITICAL` | `P1 - Critical` | 3 | 3 | 1 | 1 | High-severity business rule violations (Count: 2) |
| **12** | `MC104392` | `MEDICAL_CLAIM` | 100.00 | `CRITICAL` | `P1 - Critical` | 3 | 3 | 1 | 1 | High-severity business rule violations (Count: 2) |
| **13** | `MC102434` | `MEDICAL_CLAIM` | 100.00 | `CRITICAL` | `P1 - Critical` | 3 | 3 | 1 | 1 | High-severity business rule violations (Count: 2) |
| **14** | `MC102377` | `MEDICAL_CLAIM` | 100.00 | `CRITICAL` | `P1 - Critical` | 3 | 3 | 1 | 1 | High-severity business rule violations (Count: 2) |
| **15** | `MC103289` | `MEDICAL_CLAIM` | 100.00 | `CRITICAL` | `P1 - Critical` | 3 | 3 | 1 | 1 | High-severity business rule violations (Count: 2) |
| **16** | `MC100185` | `MEDICAL_CLAIM` | 100.00 | `CRITICAL` | `P1 - Critical` | 3 | 3 | 1 | 1 | High-severity business rule violations (Count: 2) |
| **17** | `MC102919` | `MEDICAL_CLAIM` | 100.00 | `CRITICAL` | `P1 - Critical` | 3 | 3 | 1 | 1 | High-severity business rule violations (Count: 2) |
| **18** | `MC104473` | `MEDICAL_CLAIM` | 100.00 | `CRITICAL` | `P1 - Critical` | 3 | 3 | 1 | 1 | High-severity business rule violations (Count: 2) |
| **19** | `MC100473` | `MEDICAL_CLAIM` | 100.00 | `CRITICAL` | `P1 - Critical` | 3 | 3 | 1 | 1 | High-severity business rule violations (Count: 2) |
| **20** | `MC103681` | `MEDICAL_CLAIM` | 100.00 | `CRITICAL` | `P1 - Critical` | 3 | 3 | 1 | 1 | High-severity business rule violations (Count: 2) |

---

## D. Top 20 Providers Requiring Investigation
Aggregated from `provider_investigation_summary.csv` and ranked based on total critical-risk claims, high-risk claims, and flagged claim percentage:

| NPI Group | Predominant Claim Type | Total Claims | Flagged Claims | Flagged % | High Risk Claims | Critical Risk Claims | Avg Risk Score | Provider Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `1215210273` | `MEDICAL_CLAIM` | 554 | 512 | 92.42% | 210 | 13 | 44.33 | PROVIDER REQUIRING REVIEW |
| `1609820125` | `MEDICAL_CLAIM` | 44 | 25 | 56.82% | 8 | 9 | 38.85 | PROVIDER REQUIRING REVIEW |
| `1841366812` | `MEDICAL_CLAIM` | 113 | 100 | 88.50% | 36 | 4 | 45.24 | PROVIDER REQUIRING REVIEW |
| `1386952034` | `MEDICAL_CLAIM` | 145 | 104 | 71.72% | 36 | 2 | 38.51 | PROVIDER REQUIRING REVIEW |
| `1225528151` | `MEDICAL_CLAIM` | 12 | 6 | 50.00% | 2 | 2 | 32.76 | PROVIDER REQUIRING REVIEW |
| `1972196541` | `MEDICAL_CLAIM` | 12 | 9 | 75.00% | 1 | 2 | 37.36 | PROVIDER REQUIRING REVIEW |
| `1689688988` | `MEDICAL_CLAIM` | 28 | 28 | 100.00% | 16 | 1 | 50.22 | PROVIDER REQUIRING REVIEW |
| `1558308825` | `MEDICAL_CLAIM` | 111 | 84 | 75.68% | 10 | 1 | 35.36 | PROVIDER REQUIRING REVIEW |
| `1598053472` | `MEDICAL_CLAIM` | 17 | 15 | 88.24% | 7 | 1 | 50.61 | PROVIDER REQUIRING REVIEW |
| `1225684368` | `MEDICAL_CLAIM` | 16 | 16 | 100.00% | 7 | 1 | 50.25 | PROVIDER REQUIRING REVIEW |
| `1164451613` | `MEDICAL_CLAIM` | 21 | 18 | 85.71% | 6 | 1 | 42.34 | PROVIDER REQUIRING REVIEW |
| `1700237633` | `MEDICAL_CLAIM` | 77 | 44 | 57.14% | 5 | 1 | 31.38 | PROVIDER REQUIRING REVIEW |
| `1326288143` | `MEDICAL_CLAIM` | 5 | 5 | 100.00% | 4 | 1 | 74.49 | PROVIDER REQUIRING REVIEW |
| `1659927556` | `MEDICAL_CLAIM` | 7 | 7 | 100.00% | 3 | 1 | 52.56 | PROVIDER REQUIRING REVIEW |
| `1134606403` | `MEDICAL_CLAIM` | 44 | 27 | 61.36% | 3 | 1 | 32.85 | PROVIDER REQUIRING REVIEW |
| `1245283530` | `MEDICAL_CLAIM` | 6 | 5 | 83.33% | 2 | 1 | 45.28 | PROVIDER REQUIRING REVIEW |
| `1821006867` | `MEDICAL_CLAIM` | 21 | 17 | 80.95% | 2 | 1 | 41.77 | PROVIDER REQUIRING REVIEW |
| `1154891562` | `MEDICAL_CLAIM` | 10 | 7 | 70.00% | 2 | 1 | 38.62 | PROVIDER REQUIRING REVIEW |
| `1730144593` | `MEDICAL_CLAIM` | 20 | 19 | 95.00% | 2 | 1 | 38.19 | PROVIDER REQUIRING REVIEW |
| `1972509701` | `MEDICAL_CLAIM` | 2 | 2 | 100.00% | 1 | 1 | 68.14 | PROVIDER REQUIRING REVIEW |

---

## E. Plain-Language Explanation Case Studies

### 1. Typical P1-Critical Medical Case (`MC103230`)
*   **Risk Category**: `CRITICAL` (Score: `100.0/100`)
*   **Priority**: `P1 - Critical`
*   **Score Breakdown**: `Rule: 40.0/40; Statistical: 25.0/25; ML: 24.7/25; Consensus: 10.0/10`
*   **Explanation**: This medical claim was classified as CRITICAL risk (100.0/100) because it violated explicit business rule validations, it was identified as statistically unusual compared to normal claims distributions, and the machine learning model flagged it as highly anomalous in multi-dimensional space. It violated 3 rules (IDs: R_FIN_002; R_FIN_005; R_DATE_001) with explanations: Allowed amount ($-1551.61) is negative. Under financial semantics, allowed amounts must be non-negative.; Paid amount ($1551.61) exceeds the allowed amount ($-1551.61). This is financially inconsistent as payments must not exceed allowed benefits.; Claim was processed (2016-08-05) before its submission date (2016-08-13). Processing latency is negative (-8.0 days). This indicates an invalid processing timeline. Statistically anomalous columns include: Allowed_Amount; Provider_Total_Records. ML contributing factors: Allowed_to_Billed_Ratio is unusual (scaled value: -7.08); Provider_State (AZ) is unusual; Processing_Latency_Days (-8.00) is unusual. This record requires investigation to verify claims integrity.

### 2. High-Risk Provider Pattern Explanation (`Provider NPI: 1215210273`)
*   **Status**: `PROVIDER REQUIRING REVIEW`
*   **Explanation**: Provider NPI 1215210273 submitted 554 claims, of which 512 (92.4%) were flagged as high-risk. This provider is flagged for REVIEW because of: 13 critical-risk records, 210 high-risk records, flagged claim proportion of 92.4% exceeds 10%, average claim risk score is 44.3. Pattern-level investigation is recommended.

---

## F. Traceability & Source Maps
Every piece of compiled evidence is mapped directly to its origin file:
*   **Rules Evidence**: Verified against [combined_rule_anomalies.csv](file:///c:/Users/pk404/OneDrive/Desktop/CTS_hackathon/UC10-data-quality-monitor/anomaly_detection/outputs/combined_rule_anomalies.csv) (includes rule constraints and explanations).
*   **Statistical Deviations**: Verified against descriptive means and standard deviations calculated strictly from baseline training splits ($80\%$) of clean datasets.
*   **Machine Learning Scores**: Sourced from model predictions in `ml_anomaly_detection/outputs/*_ml_predictions.csv` and contributions from `outputs/*_ml_anomalies.csv`.
*   **Risk Scores**: Sourced from consolidated calculations in `risk_scoring/record_risk_scores.csv`.

---

## G. Methodology & Validation Checks
*   Validation check suites executed during run time confirm:
    1.  No duplicate Record_IDs exist.
    2.  Risk scores are bound in $[0.0, 100.0]$.
    3.  Consensus levels match active detection flags count exactly.
    4.  All P1 claims contain valid scores.
    5.  All previous-stage pipeline source files remain untouched.
*   All tests reported **PASS** in [validation_report.csv](file:///c:/Users/pk404/OneDrive/Desktop/CTS_hackathon/UC10-data-quality-monitor/investigation_and_explanation/validation_report.csv).

---

## H. Important Limitations
1.  **Baseline Drift**: The statistical means and standard deviations are computed from the historical training split. Operational adjustments in billing practices over time may require updating the baseline.
2.  **Chronological Split**: The ML models were trained on the initial 80% chronological split. Claims submitted in later periods may exhibit features not seen during training.
3.  **Pattern Flagging**: Flagged records and providers indicate unusual patterns requiring audit review. They do not constitute a definitive confirmation of fraud.
