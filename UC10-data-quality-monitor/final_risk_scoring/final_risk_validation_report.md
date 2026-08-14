# Final Risk Scoring Stage Validation Report

This report documents the PASS/FAIL results of validation checks for the final consolidation layer.

| Check Name | Status | Details |
| :--- | :--- | :--- |
| Record Counts Match | **PASS** | Total scored records is 10000 (Expected: 10000) |
| Record_ID Traceability | **PASS** | All Record_IDs are properly formatted and traceable (MC/PH/PA format) |
| No Duplicate Record IDs | **PASS** | Record_IDs are unique in final_risk_scores.csv |
| Detection Flag Presence | **PASS** | All flagged records contain at least one anomaly flag (ML, Rule, or Statistical) |
| Consensus Flag Alignment | **PASS** | Consensus_Level matches the sum of ML, Rule, and Statistical flags for all rows |
| Score Bounds [0, 100] | **PASS** | Risk scores are bound in [0, 100]. Min=0.0, Max=100.0 |
| No Missing Priorities | **PASS** | All flagged records contain non-null Risk_Priority values |
| No Missing Record Types | **PASS** | All records contain non-null Record_Type values |
| Provider Records Reconciliation | **PASS** | Sum of provider records count is 10000 (Expected: 10000) |
| Untouched Previous Pipelines | **PASS** | Verified that cleaned datasets exist and remain unmodified |