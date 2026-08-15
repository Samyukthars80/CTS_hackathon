import streamlit as st
import json
import pandas as pd
import os

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Healthcare Claims Anomaly & Investigation Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Global */
    .main .block-container {
        padding-top: 1.2rem;
        max-width: 100% !important;
        padding-left: 2rem;
        padding-right: 2rem;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        border: 1px solid #475569;
        border-radius: 12px;
        padding: 14px 18px;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
    }
    div[data-testid="stMetric"] label {
        color: #94a3b8 !important;
        font-size: 0.82rem !important;
        font-weight: 500 !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #f1f5f9 !important;
        font-weight: 700 !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }
    section[data-testid="stSidebar"] .stRadio label {
        color: #e2e8f0 !important;
    }

    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, #3b82f6 0%, #8b5cf6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 1.8rem;
        font-weight: 800;
        margin-bottom: 0.3rem;
    }
    .subsection-header {
        color: #64748b;
        font-size: 0.9rem;
        font-weight: 500;
        margin-bottom: 1.2rem;
    }

    /* Method description cards */
    .method-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 8px;
    }
    .method-card h4 { color: #e2e8f0; margin: 0 0 6px 0; }
    .method-card p { color: #94a3b8; margin: 0; font-size: 0.88rem; }

    /* Status badges */
    .badge-pass { background: #22c55e; color: white; padding: 4px 14px; border-radius: 20px; font-weight: 700; font-size: 0.8rem; }
    .badge-fail { background: #dc2626; color: white; padding: 4px 14px; border-radius: 20px; font-weight: 700; font-size: 0.8rem; }

    /* Claim detail sections */
    .detail-section {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 18px;
        margin-bottom: 12px;
    }
    .detail-section h4 { color: #60a5fa; margin: 0 0 10px 0; }

    /* Dataframe — full column visibility */
    .stDataFrame { border-radius: 8px; }
    .stDataFrame [data-testid="stDataFrameResizable"] {
        width: 100% !important;
        overflow-x: auto !important;
    }
    .stDataFrame td, .stDataFrame th {
        white-space: nowrap !important;
        min-width: 90px !important;
    }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# BASE DIR
# ──────────────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def bp(rel):
    """Build absolute path from project-relative path."""
    return os.path.join(BASE, rel)


# ──────────────────────────────────────────────────────────────────────────────
# DATA LOADERS (cached)
# ──────────────────────────────────────────────────────────────────────────────
def load_json(filepath):
    fp = bp(filepath)
    if os.path.exists(fp):
        with open(fp, "r") as f:
            return json.load(f)
    return None


@st.cache_data
def load_csv(filepath):
    fp = bp(filepath)
    if os.path.exists(fp):
        df = pd.read_csv(fp)
        str_cols = df.select_dtypes(include=["object"]).columns
        df[str_cols] = df[str_cols].fillna("\u2014")
        return df
    return None


@st.cache_data
def load_text(filepath):
    fp = bp(filepath)
    if os.path.exists(fp):
        with open(fp, "r") as f:
            return f.read()
    return None


@st.cache_data
def load_all_cleaned():
    """Load and merge all three cleaned datasets."""
    dfs = []
    for f in ["outputs/medical_claim_cleaned.csv",
              "outputs/pharmacy_claim_cleaned.csv",
              "outputs/authorization_cleaned.csv"]:
        df = load_csv(f)
        if df is not None:
            dfs.append(df)
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return None


def data_unavailable(name):
    st.warning(f"Data unavailable: `{name}` file not found. Please ensure the pipeline has been run.")


# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────
st.sidebar.markdown("## 🏥 Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "🏠 Overview",
        "🚨 Investigation Queue",
        "🔎 Claims Explorer",
        "👤 Provider Risk",
        "📊 Statistical Analysis",
        "📋 Rule Violations",
        "🤖 ML Detection",
        "✅ Model Validation",
        "📁 Data Quality",
    ],
    label_visibility="collapsed",
)

# Sidebar search
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔍 Quick Search")
sidebar_search = st.sidebar.text_input("Record ID / Provider NPI / BENE ID", placeholder="e.g. MC103230", key="sidebar_search")
if sidebar_search:
    st.sidebar.info("Switch to **🔎 Claims Explorer** to see full results.")

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<div style='color:#64748b; font-size:0.78rem;'>"
    "Healthcare Claims Anomaly &amp; Investigation Dashboard<br>"
    "UC10 — CTS Hackathon"
    "</div>",
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: OVERVIEW (EXECUTIVE SUMMARY)
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown('<div class="section-header">Executive Summary</div>', unsafe_allow_html=True)
    st.markdown('<div class="subsection-header">Consolidated view across all pipeline stages — Healthcare Claims Anomaly Detection System</div>', unsafe_allow_html=True)

    df_overall = load_csv("final_risk_scoring/overall_risk_summary.csv")
    df_dm = load_csv("final_risk_scoring/detection_method_summary.csv")
    df_rt = load_csv("final_risk_scoring/record_type_risk_summary.csv")
    df_dist = load_csv("final_risk_scoring/risk_distribution.csv")
    df_consensus = load_csv("ml_anomaly_detection/model_validation/cross_method_consensus.csv")

    # ── Row 1: Key Metrics ──
    if df_overall is not None:
        r = df_overall.iloc[0]
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.metric("Total Records", f"{int(r['Total_Records']):,}")
        with c2:
            st.metric("Total Flagged", f"{int(r['Total_Flagged']):,}")
        with c3:
            st.metric("Flag Rate", f"{r['Flag_Rate_Percentage']:.1f}%")
        with c4:
            st.metric("🔴 Critical", f"{int(r['Critical_Count'])}")
        with c5:
            st.metric("🟠 High", f"{int(r['High_Count'])}")
    else:
        data_unavailable("final_risk_scoring/overall_risk_summary.csv")

    # ── Row 2: Record Type Counts ──
    if df_rt is not None:
        st.markdown("---")
        st.subheader("Records by Type")
        rc1, rc2, rc3 = st.columns(3)
        for i, (col_obj, label) in enumerate(zip([rc1, rc2, rc3], ["MEDICAL_CLAIM", "PHARMACY_CLAIM", "PRIOR_AUTH"])):
            row = df_rt[df_rt["Record_Type"] == label]
            if not row.empty:
                rr = row.iloc[0]
                with col_obj:
                    st.metric(label.replace("_", " ").title(), f"{int(rr['Total_Records']):,}")
                    st.caption(f"Flagged: {int(rr['Total_Flagged']):,} ({rr['Flag_Rate_Percentage']:.1f}%)")

    # ── Row 3: Detection Method Coverage ──
    if df_dm is not None:
        st.markdown("---")
        st.subheader("Detection Method Coverage")
        d1, d2, d3 = st.columns(3)
        with d1:
            st.markdown(
                '<div class="method-card"><h4>📋 Rule-Based</h4>'
                '<p>Identifies violations of known healthcare/business rules.</p></div>',
                unsafe_allow_html=True)
            st.metric("Rule Flags", f"{int(df_dm['Total_Rule_Flags'].iloc[0]):,}")
        with d2:
            st.markdown(
                '<div class="method-card"><h4>📊 Statistical</h4>'
                '<p>Identifies statistically unusual values or patterns.</p></div>',
                unsafe_allow_html=True)
            st.metric("Statistical Flags", f"{int(df_dm['Total_Statistical_Flags'].iloc[0]):,}")
        with d3:
            st.markdown(
                '<div class="method-card"><h4>🤖 Machine Learning</h4>'
                '<p>Identifies unusual feature combinations without predefined rules.</p></div>',
                unsafe_allow_html=True)
            st.metric("ML Flags", f"{int(df_dm['Total_ML_Flags'].iloc[0]):,}")

    # ── Row 4: Consensus Levels ──
    if df_consensus is not None:
        st.markdown("---")
        st.subheader("Cross-Method Consensus")
        cons_counts = df_consensus["Consensus_Level"].value_counts().sort_index()
        cc1, cc2, cc3, cc4 = st.columns(4)
        with cc1:
            st.metric("Level 0 (Unflagged)", f"{cons_counts.get(0, 0):,}")
        with cc2:
            st.metric("Level 1 (1 Method)", f"{cons_counts.get(1, 0):,}")
        with cc3:
            st.metric("Level 2 (2 Methods)", f"{cons_counts.get(2, 0):,}")
        with cc4:
            st.metric("Level 3 (All 3)", f"{cons_counts.get(3, 0):,}")

        chart_data = cons_counts.reset_index()
        chart_data.columns = ["Consensus Level", "Count"]
        chart_data["Consensus Level"] = chart_data["Consensus Level"].astype(str)
        st.bar_chart(chart_data.set_index("Consensus Level"), height=300)

    # ── Row 5: Multi-method consensus detail ──
    if df_dm is not None:
        st.markdown("---")
        st.subheader("Multi-Method Agreement")
        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            single = int(df_dm['ML_Only_Count'].iloc[0] + df_dm['Rule_Only_Count'].iloc[0] + df_dm['Statistical_Only_Count'].iloc[0])
            st.metric("Single Method Only", f"{single:,}")
        with mc2:
            st.metric("2-Method Consensus", f"{int(df_dm['Two_Method_Consensus_Count'].iloc[0]):,}")
        with mc3:
            st.metric("3-Method Consensus", f"{int(df_dm['Three_Method_Consensus_Count'].iloc[0]):,}")

    # ── Row 6: Risk Score Distribution ──
    if df_dist is not None:
        st.markdown("---")
        st.subheader("Risk Score Distribution")
        st.bar_chart(df_dist.set_index("Score_Bin"), height=350)

    # ── Row 7: Record Type Risk Summary Table ──
    if df_rt is not None:
        st.markdown("---")
        st.subheader("Risk by Record Type")
        st.dataframe(df_rt, width="stretch", hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: INVESTIGATION QUEUE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🚨 Investigation Queue":
    st.markdown('<div class="section-header">Investigation Queue</div>', unsafe_allow_html=True)
    st.markdown('<div class="subsection-header">Prioritized list of flagged records for auditor review — sorted by risk score and consensus</div>', unsafe_allow_html=True)

    df_queue = load_csv("final_risk_scoring/investigation_queue.csv")
    df_inv = load_csv("investigation_and_explanation/final_investigation_queue.csv")

    if df_queue is None:
        data_unavailable("final_risk_scoring/investigation_queue.csv")
        st.stop()

    # Enrich with explanations from investigation layer if available
    if df_inv is not None:
        extra_cols = ["Record_ID", "Explanation", "Primary_Investigation_Reason", "Supporting_Evidence", "recommended_priority"]
        extra_cols = [c for c in extra_cols if c in df_inv.columns]
        df_queue = df_queue.merge(df_inv[extra_cols], on="Record_ID", how="left", suffixes=("", "_inv"))

    # ── Top Metrics ──
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Total in Queue", f"{len(df_queue):,}")
    with c2:
        crit = len(df_queue[df_queue["Risk_Priority"] == "CRITICAL"])
        st.metric("🔴 Critical", f"{crit}")
    with c3:
        high = len(df_queue[df_queue["Risk_Priority"] == "HIGH"])
        st.metric("🟠 High", f"{high}")
    with c4:
        med = len(df_queue[df_queue["Risk_Priority"] == "MEDIUM"])
        st.metric("🟡 Medium", f"{med}")
    with c5:
        cons3 = len(df_queue[df_queue["Consensus_Level"] == 3])
        st.metric("3-Method Consensus", f"{cons3}")

    st.markdown("---")

    # ── Filters ──
    fc1, fc2, fc3, fc4 = st.columns(4)
    with fc1:
        rt_f = st.multiselect("Record Type", df_queue["Record_Type"].unique().tolist(),
                               default=df_queue["Record_Type"].unique().tolist(), key="iq_rt")
    with fc2:
        pri_f = st.multiselect("Investigation Priority", ["CRITICAL", "HIGH", "MEDIUM", "LOW"],
                                default=["CRITICAL", "HIGH"], key="iq_pri")
    with fc3:
        cons_options = sorted(df_queue["Consensus_Level"].unique().tolist(), reverse=True)
        cons_f = st.multiselect("Consensus Level", cons_options, default=cons_options, key="iq_cons")
    with fc4:
        sev_options = [s for s in df_queue["Rule_Severity"].unique().tolist() if s != "\u2014"]
        sev_f = st.multiselect("Rule Severity", sev_options, default=sev_options, key="iq_sev") if sev_options else []

    df_filt = df_queue[
        (df_queue["Record_Type"].isin(rt_f)) &
        (df_queue["Risk_Priority"].isin(pri_f)) &
        (df_queue["Consensus_Level"].isin(cons_f))
    ]
    if sev_f:
        df_filt = df_filt[(df_filt["Rule_Severity"].isin(sev_f)) | (df_filt["Rule_Severity"] == "\u2014")]

    st.markdown(f"**Showing {len(df_filt):,} records**")

    # ── Table ──
    display_cols = [c for c in ["Rank", "Record_ID", "Record_Type", "Final_Risk_Score", "Risk_Priority",
                                "Consensus_Level", "Detection_Methods", "Rule_Severity", "Rule_IDs",
                                "Anomaly_Categories", "Why_Flagged", "Explanation"] if c in df_filt.columns]
    st.dataframe(
        df_filt[display_cols].sort_values("Rank"),
        width="stretch",
        hide_index=True,
        height=600,
        column_config={
            "Rank": st.column_config.NumberColumn("Rank", width="small"),
            "Record_ID": st.column_config.TextColumn("Record ID", width="small"),
            "Record_Type": st.column_config.TextColumn("Type", width="small"),
            "Final_Risk_Score": st.column_config.NumberColumn("Score", format="%.2f", width="small"),
            "Risk_Priority": st.column_config.TextColumn("Priority", width="small"),
            "Consensus_Level": st.column_config.NumberColumn("Consensus", width="small"),
            "Detection_Methods": st.column_config.TextColumn("Methods", width="medium"),
            "Why_Flagged": st.column_config.TextColumn("Why Flagged", width="large"),
            "Explanation": st.column_config.TextColumn("Explanation", width="large"),
        },
    )

    st.markdown("---")

    # ── Record Drill-Down ──
    st.subheader("Record Drill-Down")
    record_id = st.text_input("Enter a Record ID to investigate", placeholder="e.g. MC103230", key="iq_drill")
    if record_id:
        match = df_queue[df_queue["Record_ID"] == record_id]
        if not match.empty:
            row = match.iloc[0]
            dc1, dc2, dc3 = st.columns(3)
            with dc1:
                st.metric("Risk Score", f"{row['Final_Risk_Score']:.2f}")
            with dc2:
                st.metric("Priority", row["Risk_Priority"])
            with dc3:
                st.metric("Consensus", f"{int(row['Consensus_Level'])} method(s)")
            st.markdown(f"**Detection Methods:** {row.get('Detection_Methods', '\u2014')}")
            st.markdown(f"**Rule IDs:** {row.get('Rule_IDs', '\u2014')}")
            st.markdown(f"**Why Flagged:** {row.get('Why_Flagged', '\u2014')}")
            if "Explanation" in row.index and row.get("Explanation", "\u2014") != "\u2014":
                st.markdown("---")
                st.markdown("#### Why was this flagged?")
                st.info(row["Explanation"])
        else:
            st.warning(f"Record ID `{record_id}` not found in the investigation queue.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: CLAIMS EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔎 Claims Explorer":
    st.markdown('<div class="section-header">Claims Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="subsection-header">Search and inspect individual claims — full investigation profile</div>', unsafe_allow_html=True)

    df_all_clean = load_all_cleaned()
    df_scores = load_csv("final_risk_scoring/final_risk_scores.csv")
    df_rules = load_csv("anomaly_detection/outputs/combined_rule_anomalies.csv")
    df_ml_expl = load_csv("ml_anomaly_detection/model_validation/ml_anomaly_explanations.csv")
    df_inv = load_csv("investigation_and_explanation/final_investigation_queue.csv")

    # Search
    search_val = sidebar_search if sidebar_search else ""
    search_input = st.text_input("Search by Record ID, Provider NPI, or BENE ID", value=search_val, placeholder="e.g. MC103230, 1215210273, BENE_1234", key="ce_search")

    if search_input and df_all_clean is not None:
        # Try to find matching records
        mask = (
            (df_all_clean["Record_ID"].astype(str) == search_input) |
            (df_all_clean["Provider_NPI"].astype(str) == search_input) |
            (df_all_clean["BENE_ID"].astype(str) == search_input)
        )
        matches = df_all_clean[mask]

        if matches.empty:
            st.warning(f"No records found for `{search_input}`.")
        else:
            st.success(f"Found {len(matches)} record(s)")

            for idx, claim in matches.iterrows():
                rec_id = claim["Record_ID"]
                with st.expander(f"📄 {rec_id} — {claim['Record_Type']}", expanded=(len(matches) == 1)):

                    # ── Claim Information ──
                    st.markdown("#### 📋 Claim Information")
                    info1, info2, info3 = st.columns(3)
                    with info1:
                        st.markdown(f"**Record ID:** {rec_id}")
                        st.markdown(f"**Record Type:** {claim['Record_Type']}")
                        st.markdown(f"**Provider NPI:** {claim.get('Provider_NPI', '\u2014')}")
                    with info2:
                        st.markdown(f"**BENE ID:** {claim.get('BENE_ID', '\u2014')}")
                        st.markdown(f"**Provider State:** {claim.get('Provider_State', '\u2014')}")
                        st.markdown(f"**Status:** {claim.get('Status', '\u2014')}")
                    with info3:
                        st.markdown(f"**Source System:** {claim.get('Source_System', '\u2014')}")
                        st.markdown(f"**Batch ID:** {claim.get('Batch_ID', '\u2014')}")
                        st.markdown(f"**Urgency Flag:** {claim.get('Urgency_Flag', '\u2014')}")

                    # ── Financial Information ──
                    st.markdown("#### 💰 Financial Information")
                    fin1, fin2, fin3, fin4 = st.columns(4)
                    with fin1:
                        st.metric("Billed Amount", f"${claim.get('Billed_Amount', 0):,.2f}")
                    with fin2:
                        st.metric("Allowed Amount", f"${claim.get('Allowed_Amount', 0):,.2f}")
                    with fin3:
                        st.metric("Paid Amount", f"${claim.get('Paid_Amount', 0):,.2f}")
                    with fin4:
                        st.metric("Patient Responsibility", f"${claim.get('Patient_Responsibility', 0):,.2f}")

                    # ── Timeline ──
                    st.markdown("#### 📅 Timeline")
                    t1, t2, t3, t4 = st.columns(4)
                    with t1:
                        st.markdown(f"**Service Date:** {claim.get('Service_Date', '\u2014')}")
                    with t2:
                        st.markdown(f"**Submission Date:** {claim.get('Submission_Date', '\u2014')}")
                    with t3:
                        st.markdown(f"**Processed Date:** {claim.get('Processed_Date', '\u2014')}")
                    with t4:
                        st.markdown(f"**Decision Date:** {claim.get('Decision_Date', '\u2014')}")

                    # ── Detection Results ──
                    st.markdown("#### 🔍 Detection Results")

                    # Get risk score row
                    score_row = None
                    if df_scores is not None:
                        score_match = df_scores[df_scores["Record_ID"] == rec_id]
                        if not score_match.empty:
                            score_row = score_match.iloc[0]

                    if score_row is not None:
                        det1, det2, det3 = st.columns(3)
                        with det1:
                            st.markdown("**ML Detection**")
                            ml_flag = int(score_row.get("ML_Flag", 0))
                            st.markdown(f"Flag: {'🔴 Yes' if ml_flag else '🟢 No'}")
                            st.markdown(f"Score: {score_row.get('ML_Anomaly_Score', '\u2014')}")
                            st.markdown(f"Percentile: {score_row.get('ML_Anomaly_Percentile', '\u2014')}")
                        with det2:
                            st.markdown("**Rule-Based Detection**")
                            rule_flag = int(score_row.get("Rule_Flag", 0))
                            st.markdown(f"Flag: {'🔴 Yes' if rule_flag else '🟢 No'}")
                            st.markdown(f"Severity: {score_row.get('Rule_Severity', '\u2014')}")
                            st.markdown(f"Rule IDs: {score_row.get('Rule_IDs', '\u2014')}")
                        with det3:
                            st.markdown("**Statistical Detection**")
                            stat_flag = int(score_row.get("Statistical_Flag", 0))
                            st.markdown(f"Flag: {'🔴 Yes' if stat_flag else '🟢 No'}")
                            st.markdown(f"Categories: {score_row.get('Statistical_Categories', '\u2014')}")

                        # Consensus & Risk
                        st.markdown("#### 📊 Risk Assessment")
                        r1, r2, r3, r4 = st.columns(4)
                        with r1:
                            st.metric("Risk Score", f"{score_row['Final_Risk_Score']:.2f}")
                        with r2:
                            st.metric("Priority", score_row["Risk_Priority"])
                        with r3:
                            st.metric("Consensus Level", f"{int(score_row['Consensus_Level'])}")
                        with r4:
                            st.metric("Detection Methods", score_row.get("Detection_Methods", "\u2014"))

                    # ── Rule Violations Detail ──
                    if df_rules is not None:
                        rec_rules = df_rules[df_rules["Record_ID"] == rec_id]
                        if not rec_rules.empty:
                            st.markdown("#### 📋 Rule Violations")
                            st.dataframe(
                                rec_rules[["Rule_ID", "Anomaly_Category", "Severity", "Affected_Columns",
                                           "Observed_Value", "Expected_Condition", "Explanation"]],
                                width="stretch", hide_index=True,
                                column_config={
                                    "Explanation": st.column_config.TextColumn("Explanation", width="large"),
                                },
                            )

                    # ── Why Was This Flagged? ──
                    st.markdown("#### ❓ Why Was This Flagged?")
                    explanation_shown = False

                    # From investigation layer
                    if df_inv is not None:
                        inv_match = df_inv[df_inv["Record_ID"] == rec_id]
                        if not inv_match.empty:
                            inv_row = inv_match.iloc[0]
                            expl = inv_row.get("Explanation", "\u2014")
                            if expl != "\u2014":
                                st.info(expl)
                                explanation_shown = True

                    # From ML explanations
                    if df_ml_expl is not None and not explanation_shown:
                        ml_match = df_ml_expl[df_ml_expl["Record_ID"] == rec_id]
                        if not ml_match.empty:
                            for _, ml_row in ml_match.iterrows():
                                st.info(ml_row.get("Plain_Language_Explanation", "\u2014"))
                                explanation_shown = True

                    # Fallback
                    if score_row is not None and not explanation_shown:
                        wf = score_row.get("Why_Flagged", "\u2014")
                        if wf != "\u2014":
                            st.info(wf)
                        else:
                            st.caption("No specific explanation available for this record.")

    elif not search_input:
        st.info("Enter a Record ID, Provider NPI, or BENE ID above to search for claim details.")
        # Show a sample of high-priority records for quick access
        if df_scores is not None:
            st.subheader("Top Critical Records (Quick Access)")
            top = df_scores[df_scores["Risk_Priority"] == "CRITICAL"].head(20)
            if not top.empty:
                st.dataframe(
                    top[["Record_ID", "Record_Type", "Final_Risk_Score", "Risk_Priority",
                         "Consensus_Level", "Detection_Methods", "Why_Flagged"]].sort_values("Final_Risk_Score", ascending=False),
                    width="stretch", hide_index=True,
                    column_config={
                        "Why_Flagged": st.column_config.TextColumn("Why Flagged", width="large"),
                    },
                )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4: PROVIDER RISK
# ══════════════════════════════════════════════════════════════════════════════
elif page == "👤 Provider Risk":
    st.markdown('<div class="section-header">Provider Risk Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="subsection-header">Provider NPI-level risk aggregations — elevated risk patterns requiring investigation</div>', unsafe_allow_html=True)

    df_prov = load_csv("final_risk_scoring/provider_risk_summary.csv")
    df_prov_inv = load_csv("investigation_and_explanation/provider_investigation_summary.csv")

    if df_prov is None:
        data_unavailable("final_risk_scoring/provider_risk_summary.csv")
        st.stop()

    # ── Top Metrics ──
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Total Providers", f"{len(df_prov):,}")
    with c2:
        crit_p = len(df_prov[df_prov["Provider_Risk_Level"] == "CRITICAL"])
        st.metric("🔴 Critical", f"{crit_p}")
    with c3:
        high_p = len(df_prov[df_prov["Provider_Risk_Level"] == "HIGH"])
        st.metric("🟠 High Risk", f"{high_p}")
    with c4:
        med_p = len(df_prov[df_prov["Provider_Risk_Level"] == "MEDIUM"])
        st.metric("🟡 Medium Risk", f"{med_p}")
    with c5:
        st.metric("Avg Risk Score", f"{df_prov['Average_Risk_Score'].mean():.2f}")

    st.markdown("---")

    # ── Risk Level Distribution ──
    st.subheader("Provider Risk Level Distribution")
    prl = df_prov["Provider_Risk_Level"].value_counts().reindex(["CRITICAL", "HIGH", "MEDIUM", "LOW"], fill_value=0).reset_index()
    prl.columns = ["Risk Level", "Count"]
    st.bar_chart(prl.set_index("Risk Level"), height=300)

    st.markdown("---")

    # ── Filters ──
    fc1, fc2 = st.columns(2)
    with fc1:
        risk_f = st.multiselect("Provider Risk Level", ["CRITICAL", "HIGH", "MEDIUM", "LOW"],
                                 default=["CRITICAL", "HIGH"], key="pr_rl")
    with fc2:
        min_claims = st.slider("Minimum Claim Count", 1, int(df_prov["Record_Count"].max()), 1, key="pr_mc")

    df_filt = df_prov[
        (df_prov["Provider_Risk_Level"].isin(risk_f)) &
        (df_prov["Record_Count"] >= min_claims)
    ].sort_values("Average_Risk_Score", ascending=False)

    st.markdown(f"**Showing {len(df_filt):,} providers**")

    # ── Top Suspicious Providers ──
    st.subheader("Top Providers by Average Risk Score")
    top_provs = df_filt.head(20)
    st.dataframe(
        top_provs[["Provider_NPI", "Record_Count", "Flagged_Record_Count", "Flagged_Record_Rate",
                    "Critical_Risk_Record_Count", "High_Risk_Record_Count", "Average_Risk_Score",
                    "Maximum_Risk_Score", "ML_Flag_Count", "Rule_Flag_Count", "Statistical_Flag_Count",
                    "Average_Consensus_Level", "Provider_Risk_Level"]],
        width="stretch", hide_index=True,
        column_config={
            "Provider_NPI": st.column_config.NumberColumn("Provider NPI", format="%d", width="medium"),
            "Flagged_Record_Rate": st.column_config.NumberColumn("Flag Rate %", format="%.1f"),
            "Average_Risk_Score": st.column_config.NumberColumn("Avg Score", format="%.2f"),
            "Maximum_Risk_Score": st.column_config.NumberColumn("Max Score", format="%.2f"),
            "Average_Consensus_Level": st.column_config.NumberColumn("Avg Consensus", format="%.2f"),
        },
    )

    st.markdown("---")

    # ── Full Provider Table ──
    st.subheader("All Matching Providers")
    st.dataframe(
        df_filt[["Provider_NPI", "Record_Count", "Flagged_Record_Count", "Flagged_Record_Rate",
                 "Critical_Risk_Record_Count", "Average_Risk_Score", "Provider_Risk_Level"]],
        width="stretch", hide_index=True, height=400,
        column_config={
            "Provider_NPI": st.column_config.NumberColumn("Provider NPI", format="%d"),
            "Flagged_Record_Rate": st.column_config.NumberColumn("Flag Rate %", format="%.1f"),
            "Average_Risk_Score": st.column_config.NumberColumn("Avg Score", format="%.2f"),
        },
    )

    st.markdown("---")

    # ── Provider Drill-Down ──
    st.subheader("Provider Drill-Down")
    npi_input = st.text_input("Enter Provider NPI", placeholder="e.g. 1215210273", key="pr_drill")
    if npi_input:
        try:
            npi_val = int(npi_input)
            match = df_prov[df_prov["Provider_NPI"] == npi_val]
        except ValueError:
            match = df_prov[df_prov["Provider_NPI"].astype(str) == npi_input]

        if not match.empty:
            row = match.iloc[0]
            st.markdown(f"### Provider NPI: {int(row['Provider_NPI'])}")
            pc1, pc2, pc3, pc4 = st.columns(4)
            with pc1:
                st.metric("Total Claims", f"{int(row['Record_Count'])}")
            with pc2:
                st.metric("Flagged Claims", f"{int(row['Flagged_Record_Count'])} ({row['Flagged_Record_Rate']:.1f}%)")
            with pc3:
                st.metric("Critical Claims", f"{int(row['Critical_Risk_Record_Count'])}")
            with pc4:
                st.metric("Avg Risk Score", f"{row['Average_Risk_Score']:.2f}")

            st.markdown(f"**Risk Level:** {row['Provider_Risk_Level']}")
            st.markdown(f"**Medical Claims:** {int(row['Medical_Claim_Count'])} | **Pharmacy Claims:** {int(row['Pharmacy_Claim_Count'])} | **Prior Auth:** {int(row['Prior_Auth_Count'])}")
            st.markdown(f"**ML Flags:** {int(row['ML_Flag_Count'])} | **Rule Flags:** {int(row['Rule_Flag_Count'])} | **Statistical Flags:** {int(row['Statistical_Flag_Count'])}")

            # Investigation explanation
            if df_prov_inv is not None:
                inv_match = df_prov_inv[df_prov_inv["Provider_NPI"] == npi_val]
                if not inv_match.empty:
                    inv_row = inv_match.iloc[0]
                    st.markdown("---")
                    st.markdown("#### Investigation Assessment")
                    st.markdown(f"**Status:** {inv_row.get('Provider_Status', '\u2014')}")
                    st.markdown(f"**Primary Risk Factors:** {inv_row.get('Primary_Risk_Factors', '\u2014')}")
                    if inv_row.get("Provider_Explanation", "\u2014") != "\u2014":
                        st.info(inv_row["Provider_Explanation"])
        else:
            st.warning(f"Provider NPI `{npi_input}` not found.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5: STATISTICAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Statistical Analysis":
    st.markdown('<div class="section-header">Statistical Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="subsection-header">Distribution summaries, outlier detection, and descriptive statistics per record type</div>', unsafe_allow_html=True)

    rt_map = {"Medical Claims": "medical", "Pharmacy Claims": "pharmacy", "Prior Authorization": "authorization"}
    selected_rt = st.selectbox("Select Record Type", list(rt_map.keys()), key="sa_rt")
    rt_key = rt_map[selected_rt]

    # ── Descriptive Statistics ──
    st.subheader(f"Descriptive Statistics — {selected_rt}")
    df_desc = load_csv(f"statistical_analysis/{rt_key}/descriptive_statistics.csv")
    if df_desc is not None:
        st.dataframe(
            df_desc,
            width="stretch", hide_index=True,
            column_config={
                "column": st.column_config.TextColumn("Variable", width="medium"),
                "mean": st.column_config.NumberColumn("Mean", format="%.2f"),
                "median": st.column_config.NumberColumn("Median", format="%.2f"),
                "std": st.column_config.NumberColumn("Std Dev", format="%.2f"),
                "min": st.column_config.NumberColumn("Min", format="%.2f"),
                "max": st.column_config.NumberColumn("Max", format="%.2f"),
                "skewness": st.column_config.NumberColumn("Skewness", format="%.3f"),
            },
        )
    else:
        data_unavailable(f"statistical_analysis/{rt_key}/descriptive_statistics.csv")

    st.markdown("---")

    # ── Outlier Statistics ──
    st.subheader(f"Outlier Statistics — {selected_rt}")
    df_outlier = load_csv(f"statistical_analysis/{rt_key}/outlier_statistics.csv")
    if df_outlier is not None:
        st.dataframe(
            df_outlier,
            width="stretch", hide_index=True,
            column_config={
                "column": st.column_config.TextColumn("Variable", width="medium"),
                "iqr_outlier_count": st.column_config.NumberColumn("IQR Outliers"),
                "iqr_outlier_percentage": st.column_config.NumberColumn("IQR Outlier %", format="%.2f%%"),
                "zscore_outlier_count_3sd": st.column_config.NumberColumn("Z-Score Outliers (3σ)"),
            },
        )
    else:
        data_unavailable(f"statistical_analysis/{rt_key}/outlier_statistics.csv")

    st.markdown("---")

    # ── Missingness ──
    st.subheader(f"Missing Value Analysis — {selected_rt}")
    df_miss = load_csv(f"statistical_analysis/{rt_key}/missingness_statistics.csv")
    if df_miss is not None:
        st.dataframe(df_miss, width="stretch", hide_index=True)
    else:
        data_unavailable(f"statistical_analysis/{rt_key}/missingness_statistics.csv")

    st.markdown("---")

    # ── Provider Statistics ──
    st.subheader(f"Provider Statistics — {selected_rt}")
    df_prov_stat = load_csv(f"statistical_analysis/{rt_key}/provider_statistics.csv")
    if df_prov_stat is not None:
        st.dataframe(df_prov_stat.head(30), width="stretch", hide_index=True, height=400)
    else:
        data_unavailable(f"statistical_analysis/{rt_key}/provider_statistics.csv")

    st.markdown("---")

    # ── Categorical Statistics ──
    st.subheader(f"Categorical Distributions — {selected_rt}")
    df_cat = load_csv(f"statistical_analysis/{rt_key}/categorical_statistics.csv")
    if df_cat is not None:
        cat_cols = df_cat["column"].unique().tolist()
        sel_cat = st.selectbox("Select Variable", cat_cols, key="sa_cat")
        cat_filt = df_cat[df_cat["column"] == sel_cat]
        st.dataframe(cat_filt, width="stretch", hide_index=True)
    else:
        data_unavailable(f"statistical_analysis/{rt_key}/categorical_statistics.csv")

    st.markdown("---")

    # ── Statistical Plots ──
    st.subheader(f"Distribution Plots — {selected_rt}")
    plot_dir = bp(f"statistical_analysis/{rt_key}/plots")
    if os.path.exists(plot_dir):
        plot_files = sorted([f for f in os.listdir(plot_dir) if f.endswith(".png")])
        if plot_files:
            plot_sel = st.selectbox("Select Plot", plot_files, key="sa_plot")
            st.image(os.path.join(plot_dir, plot_sel), use_container_width=True)
        else:
            st.caption("No plots available.")
    else:
        st.caption("Plot directory not found.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6: RULE VIOLATIONS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📋 Rule Violations":
    st.markdown('<div class="section-header">Rule-Based Anomaly Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="subsection-header">Domain-specific business rule violations — financial, chronological, technical, and categorical checks</div>', unsafe_allow_html=True)

    df_rules = load_csv("anomaly_detection/outputs/combined_rule_anomalies.csv")
    if df_rules is None:
        data_unavailable("anomaly_detection/outputs/combined_rule_anomalies.csv")
        st.stop()

    # ── Top Metrics ──
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Violations", f"{len(df_rules):,}")
    with c2:
        st.metric("Unique Records", f"{df_rules['Record_ID'].nunique():,}")
    with c3:
        st.metric("Unique Rules", f"{df_rules['Rule_ID'].nunique()}")
    with c4:
        high_count = len(df_rules[df_rules["Severity"] == "HIGH"])
        st.metric("High Severity", f"{high_count:,}")

    st.markdown("---")

    # ── Filters ──
    fc1, fc2, fc3, fc4 = st.columns(4)
    with fc1:
        rt_f = st.multiselect("Record Type", df_rules["Record_Type"].unique().tolist(),
                               default=df_rules["Record_Type"].unique().tolist(), key="rv_rt")
    with fc2:
        sev_f = st.multiselect("Severity", sorted(df_rules["Severity"].unique().tolist()),
                                default=df_rules["Severity"].unique().tolist(), key="rv_sev")
    with fc3:
        cat_options = sorted(df_rules["Anomaly_Category"].unique().tolist())
        cat_f = st.multiselect("Anomaly Category", cat_options, default=cat_options, key="rv_cat")
    with fc4:
        rule_options = sorted(df_rules["Rule_ID"].unique().tolist())
        rule_f = st.multiselect("Rule ID", rule_options, default=rule_options, key="rv_rule")

    df_filt = df_rules[
        (df_rules["Record_Type"].isin(rt_f)) &
        (df_rules["Severity"].isin(sev_f)) &
        (df_rules["Anomaly_Category"].isin(cat_f)) &
        (df_rules["Rule_ID"].isin(rule_f))
    ]

    st.markdown(f"**Showing {len(df_filt):,} violations**")

    # ── Severity Breakdown ──
    st.subheader("Violations by Severity")
    sev_counts = df_filt["Severity"].value_counts().reset_index()
    sev_counts.columns = ["Severity", "Count"]
    st.bar_chart(sev_counts.set_index("Severity"), height=300)

    # ── By Record Type ──
    st.subheader("Violations by Record Type")
    rt_counts = df_filt["Record_Type"].value_counts().reset_index()
    rt_counts.columns = ["Record Type", "Count"]
    st.bar_chart(rt_counts.set_index("Record Type"), height=300)

    # ── By Category ──
    st.subheader("Violations by Category")
    cat_counts = df_filt["Anomaly_Category"].value_counts().reset_index()
    cat_counts.columns = ["Category", "Count"]
    st.bar_chart(cat_counts.set_index("Category"), height=300)

    st.markdown("---")

    # ── Rule Summary Table ──
    st.subheader("Rule Summary")
    rule_summary = df_filt.groupby(["Rule_ID", "Anomaly_Category", "Severity"]).agg(
        Violations=("Record_ID", "count"),
        Unique_Records=("Record_ID", "nunique"),
        Example_Record=("Record_ID", "first"),
    ).reset_index().sort_values("Violations", ascending=False)
    st.dataframe(rule_summary, width="stretch", hide_index=True)

    st.markdown("---")

    # ── Full Violation Details ──
    st.subheader(f"Violation Details ({len(df_filt):,} rows)")
    st.dataframe(
        df_filt[["Record_ID", "Record_Type", "Rule_ID", "Severity", "Anomaly_Category",
                 "Affected_Columns", "Observed_Value", "Expected_Condition", "Explanation"]],
        width="stretch", hide_index=True, height=500,
        column_config={
            "Observed_Value": st.column_config.TextColumn("Observed", width="medium"),
            "Expected_Condition": st.column_config.TextColumn("Expected", width="medium"),
            "Explanation": st.column_config.TextColumn("Explanation", width="large"),
        },
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 7: ML DETECTION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 ML Detection":
    st.markdown('<div class="section-header">ML Anomaly Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="subsection-header">Isolation Forest models — one per record type — trained with n_estimators=300, contamination=0.02</div>', unsafe_allow_html=True)

    ml_configs = [
        ("MEDICAL_CLAIM", "medical"),
        ("PHARMACY_CLAIM", "pharmacy"),
        ("PRIOR_AUTH", "prior_auth"),
    ]

    # ── Per-Model Cards ──
    st.subheader("Model Summary")
    cols = st.columns(3)
    for i, (label, key) in enumerate(ml_configs):
        df_pred = load_csv(f"ml_anomaly_detection/outputs/{key}_ml_predictions.csv")
        df_anom = load_csv(f"ml_anomaly_detection/outputs/{key}_ml_anomalies.csv")
        with cols[i]:
            st.markdown(f"**{label.replace('_', ' ').title()}**")
            if df_pred is not None:
                total = len(df_pred)
                anom_count = int(df_pred["ML_Anomaly_Flag"].sum())
                rate = anom_count / total * 100 if total > 0 else 0
                st.metric("Training Records", f"{total:,}")
                st.metric("Anomalies Detected", f"{anom_count}")
                st.metric("Anomaly Rate", f"{rate:.2f}%")
            else:
                data_unavailable(f"ml_anomaly_detection/outputs/{key}_ml_predictions.csv")

    st.markdown("---")

    # ── Features Used ──
    st.subheader("Features Used per Model")
    df_feat = load_csv("ml_anomaly_detection/model_validation/model_feature_summary.csv")
    if df_feat is not None:
        sel_rt = st.selectbox("Select Record Type", [lbl for lbl, _ in ml_configs], key="ml_feat_rt")
        feat_filt = df_feat[df_feat["Record_Type"] == sel_rt]
        if not feat_filt.empty:
            used = feat_filt[feat_filt["Used_For_Model"] == True] if "Used_For_Model" in feat_filt.columns else feat_filt
            st.dataframe(used, width="stretch", hide_index=True)
            st.caption(f"Total features used: {len(used)}")
        else:
            st.caption("No feature data for this record type.")
    else:
        data_unavailable("model_feature_summary.csv")

    st.markdown("---")

    # ── Top ML Anomalies ──
    st.subheader("Top ML Anomalies")
    sel_rt2 = st.selectbox("Select Record Type", [lbl for lbl, _ in ml_configs], key="ml_anom_rt")
    key2 = dict(ml_configs)[sel_rt2]
    df_anom2 = load_csv(f"ml_anomaly_detection/outputs/{key2}_ml_anomalies.csv")
    if df_anom2 is not None:
        display_cols = [c for c in ["Record_ID", "Record_Type", "Isolation_Forest_Score", "Anomaly_Rank",
                                     "Severity", "Potential_Contributing_Factors"] if c in df_anom2.columns]
        st.dataframe(
            df_anom2[display_cols].sort_values("Anomaly_Rank") if "Anomaly_Rank" in df_anom2.columns else df_anom2[display_cols],
            width="stretch", hide_index=True, height=400,
            column_config={
                "Potential_Contributing_Factors": st.column_config.TextColumn("Contributing Factors", width="large"),
            },
        )
    else:
        data_unavailable(f"ml_anomaly_detection/outputs/{key2}_ml_anomalies.csv")

    st.markdown("---")

    # ── ML Explanations ──
    st.subheader("ML Anomaly Explanations")
    df_expl = load_csv("ml_anomaly_detection/model_validation/ml_anomaly_explanations.csv")
    if df_expl is not None:
        st.dataframe(
            df_expl,
            width="stretch", hide_index=True,
            column_config={
                "Plain_Language_Explanation": st.column_config.TextColumn("Explanation", width="large"),
                "Top_Contributing_Features": st.column_config.TextColumn("Top Features", width="medium"),
            },
        )
    else:
        data_unavailable("ml_anomaly_explanations.csv")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 8: MODEL VALIDATION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "✅ Model Validation":
    st.markdown('<div class="section-header">Model Validation & Reliability</div>', unsafe_allow_html=True)
    st.markdown('<div class="subsection-header">Comprehensive validation of Isolation Forest models — train/test generalization, stability, and data leakage checks</div>', unsafe_allow_html=True)

    # ── Overall Status ──
    df_val = load_csv("ml_anomaly_detection/model_validation/model_validation_summary.csv")
    if df_val is not None:
        all_pass = all(df_val["Validation_Status"] == "PASS") if "Validation_Status" in df_val.columns else False
        if all_pass:
            st.markdown('<span class="badge-pass">✅ Model Validation: PASSED</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge-fail">❌ Model Validation: ISSUES DETECTED</span>', unsafe_allow_html=True)
        st.markdown("")
    else:
        data_unavailable("model_validation_summary.csv")

    # ── Train/Test Generalization ──
    st.subheader("Train/Test Generalization")
    if df_val is not None:
        st.dataframe(
            df_val,
            width="stretch", hide_index=True,
            column_config={
                "Train_Anomaly_Rate": st.column_config.NumberColumn("Train Rate %", format="%.2f%%"),
                "Test_Anomaly_Rate": st.column_config.NumberColumn("Test Rate %", format="%.2f%%"),
                "Anomaly_Rate_Difference": st.column_config.NumberColumn("Difference", format="%.4f"),
            },
        )
        st.caption("Small difference between train and test anomaly rates indicates good generalization.")

    st.markdown("---")

    # ── Seed Stability ──
    st.subheader("Seed Stability Analysis")
    df_stab = load_csv("ml_anomaly_detection/model_validation/model_stability_report.csv")
    if df_stab is not None:
        st.dataframe(
            df_stab,
            width="stretch", hide_index=True,
            column_config={
                "Mean_Pairwise_Jaccard": st.column_config.NumberColumn("Avg Jaccard", format="%.4f"),
                "Mean_Pairwise_Agreement": st.column_config.NumberColumn("Avg Agreement", format="%.4f"),
            },
        )
        st.caption("High pairwise agreement (>99%) indicates the models are stable across different random seeds.")
    else:
        data_unavailable("model_stability_report.csv")

    st.markdown("---")

    # ── Contamination Sensitivity ──
    st.subheader("Contamination Sensitivity Sweep")
    df_contam = load_csv("ml_anomaly_detection/model_validation/contamination_sensitivity.csv")
    if df_contam is not None:
        st.dataframe(
            df_contam,
            width="stretch", hide_index=True,
            column_config={
                "Contamination": st.column_config.NumberColumn("Contamination", format="%.2f"),
                "Anomaly_Percentage": st.column_config.NumberColumn("Anomaly %", format="%.2f%%"),
                "Overlap_With_2_Percent_Pct": st.column_config.NumberColumn("Overlap w/ 2%", format="%.1f%%"),
                "Jaccard_Similarity": st.column_config.NumberColumn("Jaccard", format="%.4f"),
            },
        )
    else:
        data_unavailable("contamination_sensitivity.csv")

    st.markdown("---")

    # ── Data Leakage Check ──
    st.subheader("Data Leakage Audit")
    leakage_text = load_text("ml_anomaly_detection/model_validation/data_leakage_check.txt")
    if leakage_text:
        st.code(leakage_text, language="text")
    else:
        data_unavailable("data_leakage_check.txt")

    st.markdown("---")

    # ── Anomaly Score Statistics ──
    st.subheader("Anomaly Score Statistics")
    df_score_stats = load_csv("ml_anomaly_detection/model_validation/anomaly_score_statistics.csv")
    if df_score_stats is not None:
        st.dataframe(
            df_score_stats,
            width="stretch", hide_index=True,
            column_config={
                "Mean": st.column_config.NumberColumn("Mean", format="%.4f"),
                "Median": st.column_config.NumberColumn("Median", format="%.4f"),
                "Std": st.column_config.NumberColumn("Std Dev", format="%.4f"),
            },
        )
    else:
        data_unavailable("anomaly_score_statistics.csv")

    st.markdown("---")

    # ── Cross-Method Comparisons ──
    st.subheader("Cross-Method Agreement")
    df_rule_cmp = load_csv("ml_anomaly_detection/model_validation/ml_rule_comparison.csv")
    df_stat_cmp = load_csv("ml_anomaly_detection/model_validation/ml_statistical_comparison.csv")
    cmp1, cmp2 = st.columns(2)
    with cmp1:
        st.markdown("**ML vs Rule-Based**")
        if df_rule_cmp is not None:
            st.dataframe(df_rule_cmp, width="stretch", hide_index=True)
        else:
            data_unavailable("ml_rule_comparison.csv")
    with cmp2:
        st.markdown("**ML vs Statistical**")
        if df_stat_cmp is not None:
            st.dataframe(df_stat_cmp, width="stretch", hide_index=True)
        else:
            data_unavailable("ml_statistical_comparison.csv")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 9: DATA QUALITY (PRESERVED)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📁 Data Quality":
    st.markdown('<div class="section-header">Data Quality Monitoring</div>', unsafe_allow_html=True)
    st.markdown('<div class="subsection-header">Stage 1 — Data profiling, quality rules, and SLA risk assessment</div>', unsafe_allow_html=True)

    profile = load_json("outputs/data_profile.json")
    quality_report = load_json("outputs/quality_report.json")
    sla_risk = load_json("outputs/batch_sla_risk.json")

    if not profile or not quality_report or not sla_risk:
        st.error("Could not find output JSON files. Please run `python src/main.py` first.")
        st.stop()

    # Top metrics
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
    st.subheader("Dimension Scores")
    dims = quality_report.get("dimension_scores", {})
    if dims:
        dim_cols = st.columns(len(dims))
        for idx, (dim, score) in enumerate(dims.items()):
            with dim_cols[idx]:
                st.metric(dim, f"{score:.2f}%")

    st.markdown("---")

    # SLA Risk
    st.subheader("SLA Risk Assessment")
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

    # Rule Execution Details
    st.subheader("Rule Execution Details")
    rules = quality_report.get("all_rule_results", [])
    if rules:
        df_rules = pd.DataFrame(rules)

        st.markdown("**Failed Rules by Severity**")
        failed_df = df_rules[df_rules["status"] == "FAILED"]
        if not failed_df.empty:
            severity_counts = failed_df["severity"].value_counts().reset_index()
            severity_counts.columns = ["Severity", "Count"]
            st.bar_chart(severity_counts.set_index("Severity"), height=300)
        else:
            st.success("No rules failed!")

        st.markdown("**Top Issues**")
        top_issues = quality_report.get("top_failed_rules", [])
        if top_issues:
            for issue in top_issues:
                with st.expander(f"{issue['rule_id']}: {issue['rule_name']} ({issue['severity']}) \u2014 {issue['failure_rate_pct']:.2f}% Failed"):
                    st.write(f"**Description:** {issue['description']}")
                    st.write(f"**Affected Records:** {issue['affected_records']}")
                    st.write(f"**Recommended Fix:** {issue['recommended_fix']}")
                    st.write(f"**Sample Record IDs:** {', '.join([str(x) for x in issue.get('sample_record_ids', [])])}")
        else:
            st.info("No top issues to display.")

        st.markdown("**All Rules Table**")
        display_df = df_rules[["rule_id", "rule_name", "dimension", "severity", "status", "failure_rate_pct", "affected_records"]]
        st.dataframe(display_df, width="stretch", hide_index=True)
