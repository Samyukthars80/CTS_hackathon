# ML Anomaly Detection Model Summary

This report documents the design, features, preprocessing, training, and evaluation results of the **ML-Based Anomaly Detection Stage** using Isolation Forest models.

---

## 1. What is Isolation Forest?
Isolation Forest is an unsupervised machine learning algorithm designed specifically for anomaly detection. Unlike traditional clustering or distance-based anomaly detectors that define a "normal" region and flag anything outside it, Isolation Forest explicitly isolates anomalies.
*   **Mechanism**: It recursively partitions the dataset by randomly selecting a feature and then randomly selecting a split value between the minimum and maximum values of that feature. 
*   **Isolation Path Length**: Because anomalies have unusual values or combinations of values, they require fewer random partitions to isolate. In other words, they have a shorter average path length in the resulting decision trees.

---

## 2. Why Three Separate Models?
Our dataset contains three distinct types of claims:
1.  **Medical Claims (`MEDICAL_CLAIM`)**
2.  **Pharmacy Claims (`PHARMACY_CLAIM`)**
3.  **Prior Authorizations (`PRIOR_AUTH`)**

We train **three independent Isolation Forest models** rather than a single combined model because:
*   **Feature Semantics**: The columns have completely different structures, meanings, and scale ranges (e.g. pharmacy claims have `Days_Supply` and `Quantity_Dispensed`, which do not exist in prior authorizations).
*   **Distribution Shift**: Mixing these records into a single training set would corrupt the split trees, as the partition paths would isolate records based on record type rather than true behavioral anomalies.

---

## 3. Preprocessing and Feature Engineering Pipeline
A custom preprocessing class, [ClaimsPreprocessor](file:///c:/Users/pk404/OneDrive/Desktop/CTS_hackathon/UC10-data-quality-monitor/ml_anomaly_detection/preprocessing.py), was built to handle each dataset:

### A. Preprocessing Steps
1.  **Derived Ratios**: Before any imputation, the preprocessor calculates behavioral ratios where denominators are valid and non-zero:
    *   `Paid_to_Allowed_Ratio` = `Paid_Amount / Allowed_Amount`
    *   `Paid_to_Billed_Ratio` = `Paid_Amount / Billed_Amount`
    *   `Allowed_to_Billed_Ratio` = `Allowed_Amount / Billed_Amount`
    *   `Quantity_to_Days_Ratio` = `Quantity_Dispensed / Days_Supply`
2.  **Numerical Imputation**: Missing values (such as processing latencies for pending claims) are imputed using the **median of the training set only** to prevent leakage.
3.  **Missingness Indicators**: For columns that contain missing values, a binary column (e.g., `Processing_Latency_Days_is_missing`) is created. This allows the model to learn if the absence of information itself represents an anomaly.
4.  **Categorical Encoding**: Low-cardinality categorical features are converted to integer codes using `OrdinalEncoder`. Any category not seen during training is mapped to a standard `-1` value during prediction.
5.  **Feature Filtering**: Drop constant features (zero standard deviation in training).
6.  **Feature Scaling**: Scale all continuous features using `StandardScaler` fitted on the training split.

### B. Feature Selection Summary

| Model / Record Type | Numerical Features | Categorical Features | Ratios Features | Excluded Columns |
| :--- | :--- | :--- | :--- | :--- |
| **Medical Claims** | `Billed_Amount`, `Allowed_Amount`, `Paid_Amount`, `Patient_Responsibility`, `Retry_Count`, `Processing_Latency_Days`, `SLA_Target_Days`, `Beneficiary_Record_Count`, `Provider_Total_Records`, `Provider_Denial_Rate` | `Provider_State`, `Status`, `Urgency_Flag` | `Paid_to_Allowed_Ratio`, `Paid_to_Billed_Ratio`, `Allowed_to_Billed_Ratio` | Unique IDs (`Record_ID`, `BENE_ID`, `Provider_NPI`), raw dates, constant columns, statistical anomaly flags. |
| **Pharmacy Claims** | `Billed_Amount`, `Allowed_Amount`, `Paid_Amount`, `Patient_Responsibility`, `Retry_Count`, `Processing_Latency_Days`, `SLA_Target_Days`, `Days_Supply`, `Quantity_Dispensed`, `Beneficiary_Record_Count`, `Provider_Total_Records`, `Provider_Denial_Rate` | `Provider_State`, `Status` | `Paid_to_Allowed_Ratio`, `Paid_to_Billed_Ratio`, `Allowed_to_Billed_Ratio`, `Quantity_to_Days_Ratio` | Unique IDs (`Record_ID`, `BENE_ID`, `Provider_NPI`, `NDC_Code`), raw dates, constant columns, rule anomalies. |
| **Prior Auths** | `Retry_Count`, `Processing_Latency_Days`, `SLA_Target_Days`, `Provider_Total_Records`, `Provider_Denial_Rate`, `Beneficiary_Record_Count` | `Provider_State`, `Status`, `Urgency_Flag` | None | Unique IDs (`Record_ID`, `BENE_ID`, `Provider_NPI`, `Auth_Linked_ID`), raw dates, constant columns, rule anomalies. |

---

## 4. Train / Test Approach
*   **Temporal Chronological Split**: Rather than random splitting, the clean datasets are sorted by `Submission_Date`, and the first **80%** is used for model training, with the remaining **20%** reserved for test evaluation. This mimics a real-world batch training and deployment scenario.
*   **Leakage Prevention**: Medians, category encoder vocabularies, and scaler parameters are fitted strictly on the 80% train split and reused during test prediction.

---

## 5. Model Parameters & Contamination Selection
*   **Configuration**: `random_state = 42`, `n_estimators = 300`.
*   **Contamination Rate**: Selected a contamination value of **`0.02` (2%)** for the final models.
*   **Significance**: The contamination parameter dictates the mathematical threshold below which records are classified as outliers. It is a modeling configuration and should **NOT** be interpreted as the actual fraud rate.

---

## 6. Output Schema and Severity Rules
The model generates:
1.  **Full predictions file (`*_ml_predictions.csv`)**: Record_ID, Record_Type, Isolation_Forest_Score, ML_Anomaly_Flag, Anomaly_Percentile.
2.  **Anomaly report (`*_ml_anomalies.csv`)**: Record_ID, Record_Type, Detection_Method, ML_Anomaly_Flag, Isolation_Forest_Score, Anomaly_Rank, Severity, Model_Name, Potential_Contributing_Factors.

### Severity Logic (based on Anomaly_Percentile)
*   **High Severity**: Claims in the top 1% of anomaly scores (Percentile >= 99.0).
*   **Medium Severity**: Claims in the 1% to 5% range (Percentile >= 95.0 and < 99.0).
*   **Low Severity**: Other flagged claims (Flag = 1, Percentile < 95.0).

---

## 7. Model Explanations (Contributing Factors)
Since tree partition path lengths are difficult for non-technical investigators to decipher, the pipeline calculates Z-score deviations for all numerical/ratio features using the preprocessor standard deviations. For any flagged anomaly, the top 3 features exhibiting the largest absolute Z-score deviation (Z-score > 2.0) are logged as **"Potential contributing factors"** to provide explainability.

---

## 8. Limitations & Anti-Fraud Disclaimer
> [!IMPORTANT]
> **ML Anomalies are NOT equivalent to Fraud.**
> An Isolation Forest anomaly simply flags a record that has an unusual combination of numeric behavioral features (e.g. extremely high retry counts combined with very high paid-to-billed ratios). These can represent legitimate edge cases, processing lags, or systemic errors. These scores should be combined with rules and statistical signals before triggering an audit investigation.
