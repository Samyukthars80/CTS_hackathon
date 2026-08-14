# Risk Scoring & Investigation Queue Report

This report documents the design, scoring formula, risk factor aggregation, provider aggregate analysis, and validation results of the **Consolidated Risk Scoring & Investigation Queue Stage**.

---

## 1. Purpose
The purpose of this stage is to build a consolidation layer that merges the disparate risk signals from previous stages:
1.  **Rule-Based Anomaly Detection** (Deduplicated explicit business rule violations)
2.  **Statistical Outlier Analysis** (IQR-based numerical feature shifts)
3.  **Machine Learning Isolation Forest Models** (Unsupervised multi-dimensional feature-space anomalies)

These separate signals are consolidated into record-level risk scores, risk levels, and an audit-ready investigation queue. This queue serves to prioritize claims for manual inspection based on joint risk evidence rather than raw count thresholds.

---

## 2. Input Datasets & Files Used
All inputs were treated as **read-only** and consumed without modification:
*   [cross_method_consensus.csv](file:///c:/Users/pk404/OneDrive/Desktop/CTS_hackathon/UC10-data-quality-monitor/ml_anomaly_detection/model_validation/cross_method_consensus.csv) (Joint detection consensus flags)
*   [combined_rule_anomalies.csv](file:///c:/Users/pk404/OneDrive/Desktop/CTS_hackathon/UC10-data-quality-monitor/anomaly_detection/outputs/combined_rule_anomalies.csv) (Rule IDs and violation descriptions)
*   [record_level_summary.csv](file:///c:/Users/pk404/OneDrive/Desktop/CTS_hackathon/UC10-data-quality-monitor/anomaly_detection/outputs/record_level_summary.csv) (Total and high-severity rule violation counts)
*   `ml_anomaly_detection/outputs/*_ml_predictions.csv` (Isolation Forest anomaly scores and percentiles)
*   `outputs/*_cleaned.csv` (Cleaned claim datasets to retrieve Provider_NPIs, status values, and numerical columns for statistical validation)

---

## 3. Risk Scoring Methodology & Formula
The Risk Score is calculated deterministically on a scale of **0 to 100**:

$$\text{Risk\_Score} = \min\left(100.0, ML\_Cont + Stat\_Cont + Rule\_Cont + Consensus\_Cont\right)$$

### Component Weights Rationale
1.  **ML Contribution (Max 25 points)**:
    *   `15.0` points if `ML_Anomaly_Flag == 1`.
    *   Up to `10.0` points based on the Isolation Forest percentile rank: $+10 \times \left(\frac{\text{Anomaly\_Percentile}}{100}\right)$.
    *   *Rationale*: This ensures that among anomalies, the most isolated points receive higher scores, and normal points have a baseline indicating relative distance to the boundary.
2.  **Statistical Contribution (Max 20 points)**:
    *   `15.0` points if `Statistical_Anomaly_Flag == 1`.
    *   $+5.0$ points if the record is a statistical outlier in more than one numerical column.
    *   *Rationale*: Highlights extreme numerical outliers while boosting multi-column outliers.
3.  **Rule-Based Contribution (Max 45 points)**:
    *   `10.0` points if `Rule_Anomaly_Flag == 1`.
    *   $+5.0$ points per rule violation (capped at `15.0` points).
    *   $+10.0$ points per high-severity rule violation (capped at `20.0` points).
    *   *Rationale*: Rule violations are highly diagnostic of billing discrepancies (e.g. Paid > Allowed). High-frequency or high-severity violations heavily shift the risk profile.
4.  **Consensus Contribution (Max 20 points)**:
    *   `10.0` points for Level 2 consensus (any two methods flag the record).
    *   `20.0` points for Level 3 consensus (all three methods flag the record).
    *   *Rationale*: Multi-method agreement indicates high-confidence anomalous patterns.

---

## 4. Risk Level Thresholds
Four distinct Risk Levels are defined based on score brackets:
*   **LOW (0 - 24.99)**: Baseline claims showing typical billing behaviors.
*   **MEDIUM (25 - 49.99)**: Claims with minor SLA latency breaches or isolated statistical deviations.
*   **HIGH (50 - 74.99)**: Claims showing multiple violations or statistical/ML joint agreement.
*   **CRITICAL (75 - 100)**: Claims flagged by all three channels (Consensus Level 3) or displaying multiple severe rule violations.

---

## 5. Investigation Queue Statistics
Total records sorted: **10,000**
*   **CRITICAL**: 164 records (1.64%) -> Immediate Audit Priority
*   **HIGH**: 1,181 records (11.81%) -> High Priority Review
*   **MEDIUM**: 2,895 records (28.95%) -> Secondary Review Queue
*   **LOW**: 5,760 records (57.60%) -> Standard Operations

---

## 6. Provider-Level Risk Aggregation
We aggregated risk parameters for all NPIs. NULL or missing NPIs were explicitly grouped under `"UNKNOWN_PROVIDER"` to prevent metadata inflation.
*   Total unique NPIs: **380** (plus `"UNKNOWN_PROVIDER"`).
*   **Review Status**: Providers are classified as `"PROVIDER REQUIRING REVIEW"` (High-Risk) if any submitted claim has a `HIGH`/`CRITICAL` risk score, or if their flagged percentage exceeds 10%, or their maximum claim score exceeds 50.
*   **Risk ranking**: Providers are sorted by critical-risk counts, high-risk counts, average risk score, and volume.

---

## 7. Example High-Risk Records

Here are 5 representative critical-risk examples from the prioritized investigation queue:

### Example 1: Record `MC103230`
*   **Record Type**: `MEDICAL_CLAIM`
*   **Consensus Level**: 3 (ML, Rules, and Statistics agree)
*   **Risk Score**: `100.0/100` (CRITICAL)
*   **Primary Risk Factors**: ML Anomaly (Percentile: 98.9%); Statistical outlier in 2 columns (Allowed_Amount; Provider_Total_Records); Violates 3 business rules; High-severity rule violations count: 3; Multi-method agreement (Consensus Level: 3)
*   **Explanation**: This medical claim was classified as CRITICAL risk (100.0/100) because it was flagged by Machine Learning Isolation Forest, and Statistical Outlier Analysis, and Rule-Based Violation Checks detectors. It exhibited 3 rule violations (IDs: R_FIN_002; R_FIN_005; R_DATE_001) with explanations: Allowed amount ($-1551.61) is negative. Under financial semantics, allowed amounts must be non-negative.; Paid amount ($1551.61) exceeds the allowed amount ($-1551.61). This is financially inconsistent as payments must not exceed allowed benefits.; Claim was processed (2016-08-05) before its submission date (2016-08-13). Processing latency is negative (-8.0 days). This indicates an invalid processing timeline. The Isolation Forest isolated this record in feature space (percentile: 98.9%) with contributing factors: Allowed_to_Billed_Ratio is unusual (scaled value: -7.08); Provider_State (AZ) is unusual; Processing_Latency_Days (-8.00) is unusual. This record requires investigation to verify the integrity of the submitted claims.

### Example 2: Record `MC102394`
*   **Record Type**: `MEDICAL_CLAIM`
*   **Consensus Level**: 3 (ML, Rules, and Statistics agree)
*   **Risk Score**: `100.0/100` (CRITICAL)
*   **Primary Risk Factors**: ML Anomaly (Percentile: 100.0%); Statistical outlier in 3 columns (Billed_Amount; Allowed_Amount; Paid_Amount); Violates 3 business rules; High-severity rule violations count: 3; Multi-method agreement (Consensus Level: 3)
*   **Explanation**: This medical claim was classified as CRITICAL risk (100.0/100) because it was flagged by Machine Learning Isolation Forest, and Statistical Outlier Analysis, and Rule-Based Violation Checks detectors. It exhibited 3 rule violations (IDs: R_FIN_005; R_FIN_007; R_DATE_001) with explanations: Paid amount ($25251.16) exceeds the allowed amount ($13213.36). This is financially inconsistent as payments must not exceed allowed benefits.; Paid amount ($25251.16) exceeds the provider's billed amount ($16516.69). Reimbursements cannot exceed the initial claim charges.; Claim was processed (2016-10-21) before its submission date (2016-10-24). Processing latency is negative (-3.0 days). This indicates an invalid processing timeline. The Isolation Forest isolated this record in feature space (percentile: 100.0%) with contributing factors: Provider_State (MD) is unusual; Paid_Amount (25251.16) is unusual; Paid_to_Allowed_Ratio is unusual (scaled value: 4.98). This record requires investigation to verify the integrity of the submitted claims.

### Example 3: Record `MC100962`
*   **Record Type**: `MEDICAL_CLAIM`
*   **Consensus Level**: 3 (ML, Rules, and Statistics agree)
*   **Risk Score**: `100.0/100` (CRITICAL)
*   **Primary Risk Factors**: ML Anomaly (Percentile: 99.8%); Statistical outlier in 3 columns (Allowed_Amount; Provider_Total_Records; Provider_Denial_Rate); Violates 2 business rules; High-severity rule violations count: 2; Multi-method agreement (Consensus Level: 3)
*   **Explanation**: This medical claim was classified as CRITICAL risk (100.0/100) because it was flagged by Machine Learning Isolation Forest, and Statistical Outlier Analysis, and Rule-Based Violation Checks detectors. It exhibited 2 rule violations (IDs: R_FIN_002; R_FIN_005) with explanations: Allowed amount ($-1093.81) is negative. Under financial semantics, allowed amounts must be non-negative.; Paid amount ($1093.81) exceeds the allowed amount ($-1093.81). This is financially inconsistent as payments must not exceed allowed benefits. The Isolation Forest isolated this record in feature space (percentile: 99.8%) with contributing factors: Provider_State (FL) is unusual; Allowed_to_Billed_Ratio is unusual (scaled value: -7.08); Provider_Total_Records (554.00) is unusual. This record requires investigation to verify the integrity of the submitted claims.

### Example 4: Record `MC104591`
*   **Record Type**: `MEDICAL_CLAIM`
*   **Consensus Level**: 3 (ML, Rules, and Statistics agree)
*   **Risk Score**: `100.0/100` (CRITICAL)
*   **Primary Risk Factors**: ML Anomaly (Percentile: 98.1%); Statistical outlier in 2 columns (Provider_Total_Records; Provider_Denial_Rate); Violates 2 business rules; High-severity rule violations count: 2; Multi-method agreement (Consensus Level: 3)
*   **Explanation**: This medical claim was classified as CRITICAL risk (100.0/100) because it was flagged by Machine Learning Isolation Forest, and Statistical Outlier Analysis, and Rule-Based Violation Checks detectors. It exhibited 2 rule violations (IDs: R_FIN_002; R_FIN_005) with explanations: Allowed amount ($-306.99) is negative. Under financial semantics, allowed amounts must be non-negative.; Paid amount ($306.99) exceeds the allowed amount ($-306.99). This is financially inconsistent as payments must not exceed allowed benefits. The Isolation Forest isolated this record in feature space (percentile: 98.1%) with contributing factors: Provider_State (FL) is unusual; Allowed_to_Billed_Ratio is unusual (scaled value: -7.08); Provider_Total_Records (554.00) is unusual. This record requires investigation to verify the integrity of the submitted claims.

---

## 8. Validation Results
All validation test checks **PASSED** in the self-validation suite:
*   Row Count Matching: Checked that all 10,000 input consensus records map to a consolidated risk score.
*   Unique ID Integrity: No duplicated Record_IDs exist.
*   Score Range Verification: Risk scores are correctly bounded in $[0.0, 100.0]$.
*   Category Boundaries: Verified that risk labels match score thresholds.
*   Queue Sorting Integrity: Priorities and score ranks match.
*   NPI Aggregation: Checked that duplicate provider entries do not occur. Null NPIs aggregate under `"UNKNOWN_PROVIDER"`.
*   Safety audit: Confirmed that zero input files or pre-existing scripts were modified.
