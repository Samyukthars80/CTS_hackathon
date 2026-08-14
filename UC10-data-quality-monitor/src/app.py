import streamlit as st
import json
import pandas as pd
import os

st.set_page_config(page_title="Data Quality Monitor", layout="wide")

st.title("Healthcare Data Quality Monitoring & SLA Risk Dashboard")

def load_json(filepath):
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            return json.load(f)
    return None

profile = load_json("outputs/data_profile.json")
quality_report = load_json("outputs/quality_report.json")
sla_risk = load_json("outputs/batch_sla_risk.json")

if not profile or not quality_report or not sla_risk:
    st.error("Could not find output JSON files. Please run the quality engine first.")
    st.stop()

# Top level metrics
st.header("Executive Summary")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Records", f"{profile.get('total_records', 0):,}")
with col2:
    st.metric("Overall Quality Score", f"{quality_report.get('overall_quality_score', 0):.2f}/100")
with col3:
    st.metric("Overall Risk Level", quality_report.get('overall_risk_level', 'UNKNOWN'))
with col4:
    st.metric("Critical Issues", quality_report.get('critical_issue_count', 0))

st.markdown("---")

# Dimension Scores
st.header("Dimension Scores")
dims = quality_report.get("dimension_scores", {})
dim_cols = st.columns(5)
for idx, (dim, score) in enumerate(dims.items()):
    with dim_cols[idx]:
        st.metric(dim, f"{score:.2f}%")
        
st.markdown("---")

st.header("SLA Risk Assessment")
col_s1, col_s2, col_s3, col_s4 = st.columns(4)
with col_s1:
    st.metric("Batch SLA Risk Level", sla_risk.get("batch_sla_risk_level", "UNKNOWN"))
with col_s2:
    st.metric("SLA Breach Rate", f"{sla_risk.get('sla_breach_rate', 0)*100:.2f}%")
with col_s3:
    st.metric("Breach Count", sla_risk.get('breach_count', 0))
with col_s4:
    st.metric("Retry Count", sla_risk.get('retry_count_summary', 0))

st.subheader("Anomaly Signals (Handoff)")
st.json(quality_report.get("anomaly_signals", {}))

st.markdown("---")

st.header("Rule Execution Details")

rules = quality_report.get("all_rule_results", [])
if rules:
    df_rules = pd.DataFrame(rules)
    
    # Severity count chart
    st.subheader("Failed Rules by Severity")
    failed_df = df_rules[df_rules["status"] == "FAILED"]
    
    if not failed_df.empty:
        severity_counts = failed_df["severity"].value_counts().reset_index()
        severity_counts.columns = ["Severity", "Count"]
        st.bar_chart(severity_counts.set_index("Severity"))
    else:
        st.success("No rules failed!")
        
    st.subheader("Top Issues")
    top_issues = quality_report.get("top_failed_rules", [])
    if top_issues:
        for issue in top_issues:
            with st.expander(f"{issue['rule_id']}: {issue['rule_name']} ({issue['severity']}) - {issue['failure_rate_pct']:.2f}% Failed"):
                st.write(f"**Description:** {issue['description']}")
                st.write(f"**Affected Records:** {issue['affected_records']}")
                st.write(f"**Recommended Fix:** {issue['recommended_fix']}")
                st.write(f"**Sample Record IDs:** {', '.join([str(x) for x in issue.get('sample_record_ids', [])])}")
    else:
        st.info("No top issues to display.")
        
    st.subheader("All Rules Table")
    display_df = df_rules[["rule_id", "rule_name", "dimension", "severity", "status", "failure_rate_pct", "affected_records"]]
    st.dataframe(display_df)
