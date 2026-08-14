# UC10 Data Quality Monitor & SLA Risk Assessment

This project is the Role 2 module for the UC10 hackathon. It monitors healthcare data quality and assesses SLA risks without modifying the original data. It generates machine-readable JSON reports and provides a Streamlit dashboard.

## Setup Instructions

### 1. Install Packages
Ensure you have Python installed, then install the required dependencies:
```bash
pip install -r requirements.txt
```

### 2. Place the Dataset
Place the original dataset CSV file inside the `data/` directory and ensure it is named exactly:
`claims_pharmacy_auth_monitor_dataset_features.csv`

### 3. Run the Quality Engine
Run the main script to process the data, evaluate the rules, and generate the outputs:
```bash
python src/main.py
```
This will generate three JSON files in the `outputs/` directory:
- `data_profile.json`
- `quality_report.json`
- `batch_sla_risk.json`

### 4. Run the Dashboard
Start the Streamlit dashboard to visualize the results:
```bash
streamlit run src/app.py
```

## Project Structure
- `data/`: Directory for placing the input CSV file.
- `src/profiler.py`: Generates baseline statistics and data types.
- `src/rule_catalog.py`: Contains the configurable definitions for all 21 data quality rules.
- `src/quality_engine.py`: Executes the rules safely using Pandas.
- `src/scoring.py`: Calculates dimension scores, overall quality score, and SLA risk level.
- `src/main.py`: Orchestrates the execution of the profiler, engine, and scoring.
- `src/app.py`: Streamlit dashboard for visualizing the metrics and failed rules.
- `outputs/`: Contains the generated JSON reports.

## Architecture Handoff
This module **does not** contain an ML anomaly model. Instead, it computes and packages anomaly signals (e.g., volume versus trend ratio, SLA breach rate versus trend) into a structured section called `anomaly_signals` within the `quality_report.json`. This acts as a clean handoff to the teammate responsible for the anomaly-detection module.

## Assumptions & Data Dictionary Notes
- **Read-Only**: The original data is never modified, cleaned, or overwritten. 
- **Date Parsing**: Dates are parsed using `errors="coerce"`. Any unparseable strings will become `NaT` (Not a Time) and will be flagged as missing by completeness rules.
- **Rule R009 (Prior Auth Dates)**: It is assumed that only `APPROVED` and `DENIED` statuses require `Processed_Date` and `Decision_Date`. `PENDING` statuses are ignored for these fields.
- **Rule R018 (Referential Integrity)**: Checks if `Auth_Linked_ID` exists in the `Record_ID` column for `PRIOR_AUTH` records *within the same batch*. If historical authorizations are not in the same CSV batch, this rule might falsely fail. A comprehensive healthcare data dictionary and historical database access would be needed to treat this as a "hard failure" in production.
- **Source Systems**: It is assumed that `Source_System` precisely matches the strings `CARRIER_CLAIMS_SYS`, `PHARMACY_ADJ_SYS`, and `AUTH_MGMT_SYS`. Any slight variation (e.g., casing, trailing spaces) will trigger a failure.
