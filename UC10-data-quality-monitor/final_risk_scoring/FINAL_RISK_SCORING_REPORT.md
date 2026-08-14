# Consolidated Final Risk Scoring & Prioritized Investigation Queue Report

This report documents the design, scoring formula, weights, validation results, and top prioritized findings for the **Final Risk Scoring and Provider Prioritization Stage**.

---

## 1. Architecture
The final risk scoring stage represents the consolidation layer of the Healthcare Claims Anomaly Detection Project. It aggregates three independent anomaly signals:
1.  **Rule-Based Anomaly Detection**: Compliance violations identified by explicit business rule validations.
2.  **Statistical Outlier Analysis**: Extremeness deviations ($Z$-score magnitudes) on continuous variables compared against baseline training distributions.
3.  **Isolation Forest ML Anomaly Detection**: Multi-dimensional feature space isolation path lengths.

---

## 2. Input Files Used
All inputs were treated as **read-only** and consumed without modification:
*   `outputs/*_cleaned.csv` (Source cleaned claim datasets to retrieve Provider_NPIs and compute baseline continuous variables shifts)
*   `anomaly_detection/outputs/combined_rule_anomalies.csv` (Rule IDs and violation descriptions)
*   `ml_anomaly_detection/outputs/*_ml_predictions.csv` (Isolation Forest anomaly scores and percentiles)
*   `ml_anomaly_detection/outputs/*_ml_anomalies.csv` (Top contributing feature-level explanations)

---

## 3. How the Three Detection Methods are Combined
The overall risk score is calculated on a 0-100 scale using the following formula:

$$\text{Final\_Risk\_Score} = \text{ML\_Contribution} + \text{Rule\_Contribution} + \text{Statistical\_Contribution}$$

*   **ML Contribution** (Weight: 40%):
    *   If the record is flagged by ML (`ML_Anomaly_Flag == 1`), the contribution is:
        $$\text{ML\_Contribution} = \frac{\text{Anomaly\_Percentile}}{100} \times 40.0$$
    *   If not flagged, contribution is `0.0`.
*   **Rule Contribution** (Weight: 35%):
    *   Highest Severity `"High"` -> 35.0 points.
    *   Highest Severity `"Medium"` -> 20.0 points.
    *   Highest Severity `"Low"` -> 10.0 points.
    *   No violations -> 0.0 points.
*   **Statistical Contribution** (Weight: 25%):
    *   Compare numerical columns against training split statistics to calculate standard deviation shifts ($Z = |x - \mu|/\sigma$):
        *   If max $Z > 3.0$: 25.0 points (severe deviation).
        *   If $1.5 < \text{max } Z \le 3.0$: 15.0 points (moderate deviation).
        *   Otherwise: 0.0 points.

---

## 4. Consensus & Risk Priority Thresholds
*   **Consensus Level**: Defined as the number of methods that flagged the record (`ML_Flag + Rule_Flag + Statistical_Flag`), ranging from `0` (unflagged) to `3` (all three channels agree).
*   **Risk Priority Brackets**:
    *   **CRITICAL**: Score 80.0 to 100.0 (84 records) -> Immediate Priority Audit
    *   **HIGH**: Score 60.0 to 79.99 (235 records) -> High Priority Review
    *   **MEDIUM**: Score 40.0 to 59.99 (1,317 records) -> Standard Review Queue
    *   **LOW**: Score 0.0 to 39.99 (8,364 records) -> Normal billing baseline

---

## 5. Provider Aggregations & Cross-Record-Type Patterns
*   **Provider Status Classification**:
    *   `CRITICAL` provider risk: average claim score $\ge 45.0$ OR $\ge 2$ critical claims.
    *   `HIGH` provider risk: average claim score $\ge 30.0$ OR $\ge 3$ high claims OR $\ge 1$ critical claim.
    *   `MEDIUM` provider risk: average claim score $\ge 15.0$ OR flagged rate $\ge 10\%$.
    *   `LOW` provider risk: otherwise.
*   **Cross-Record-Type Analysis**:
    Identifies providers submitting claims across multiple domains (Medical, Pharmacy, and Prior Auth) where high flagged rates exist. For example, Provider NPI 1215210273 submitted 554 claims (92.4% flagged) with 13 critical claims across domains.

---

## 6. Examples of Plain-Language Explanations

### Claim MC103230
*   **Risk Priority**: `CRITICAL` (Score: 100.0/100)
*   **Consensus**: 3 methods
*   **Explanation**: Potentially suspicious claim prioritized as CRITICAL priority (score: 100.00/100) because the rule engine flagged 3 compliance violations (IDs: R_FIN_002; R_FIN_005; R_DATE_001), statistical analysis identified 2 unusual continuous features (Outlier: Allowed_Amount; Outlier: Provider_Total_Records), and the machine learning model classified the claim as highly anomalous in multi-dimensional space (percentile: 98.8%). This indicates an unusual billing pattern requiring investigation to verify claims integrity.

---

## 7. Validation Results
All 10 checks reported **PASS** in [final_risk_validation_report.md](file:///c:/Users/pk404/OneDrive/Desktop/CTS_hackathon/UC10-data-quality-monitor/final_risk_scoring/final_risk_validation_report.md):
*   Record counts match input (10,000)
*   All Record_IDs are properly formatted and traceable (MC/PH/PA prefix)
*   No duplicate Record_IDs exist in the final scoring file
*   Consensus levels align exactly with active flags count
*   Scored bounds are within $[0, 100]$
*   No previous stages were modified.

---

## 8. Limitations
1.  **Baseline Changes**: Baselines represent training split averages. Major updates to billing structures may require retraining features.
2.  **No Fraud Confirmation**: The system ranks records and providers based on anomalous deviations and rule violations. These are prioritizations, not a confirmation of billing fraud.
