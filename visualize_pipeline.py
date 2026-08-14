"""
visualize_pipeline.py
=====================
Before-vs-After visualization for the UC10 Data Cleaning Pipeline.

Produces a multi-page PDF report:  data_cleaning_visualization_report.pdf
And saves individual PNGs to:      visualizations/

Run:  python visualize_pipeline.py
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT      = os.path.dirname(os.path.abspath(__file__))
RAW_PATH  = os.path.join(ROOT, "Data_sets",
            "claims_pharmacy_auth_monitor_dataset_features (1).csv")
CLEAN_PATH = os.path.join(ROOT, "claims_cleaned_for_anomaly_detection.csv")
DQ_PATH    = os.path.join(ROOT, "data_quality_report.csv")
VIZ_DIR    = os.path.join(ROOT, "visualizations")
PDF_OUT    = os.path.join(ROOT, "data_cleaning_visualization_report.pdf")
os.makedirs(VIZ_DIR, exist_ok=True)

# ── Colour palette ─────────────────────────────────────────────────────────
C_RAW    = "#E74C3C"   # red   – raw / before
C_CLEAN  = "#2ECC71"   # green – clean / after
C_FLAG   = "#3498DB"   # blue  – anomaly flags
C_WARN   = "#F39C12"   # amber – warnings
C_DARK   = "#2C3E50"   # dark  – text / axes
C_LIGHT  = "#ECF0F1"   # light – backgrounds

sns.set_theme(style="whitegrid", font_scale=0.9)
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   C_LIGHT,
    "axes.edgecolor":   C_DARK,
    "axes.labelcolor":  C_DARK,
    "text.color":       C_DARK,
    "xtick.color":      C_DARK,
    "ytick.color":      C_DARK,
})

# ── Load data ───────────────────────────────────────────────────────────────
print("[load] Reading datasets …")
raw   = pd.read_csv(RAW_PATH,   low_memory=False)
clean = pd.read_csv(CLEAN_PATH, low_memory=False)
dq    = pd.read_csv(DQ_PATH)
print(f"       Raw: {raw.shape}  |  Clean: {clean.shape}")

# ── Helper utilities ────────────────────────────────────────────────────────

def save_fig(fig, name: str, pdf: PdfPages):
    """Save figure to both PNG and the shared PDF."""
    png_path = os.path.join(VIZ_DIR, f"{name}.png")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {name}.png")


def annotate_bar(ax, fmt="{:.0f}", color="white", fontsize=8):
    """Write value labels inside bars."""
    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(fmt.format(h),
                        (p.get_x() + p.get_width() / 2, h * 0.5),
                        ha="center", va="center",
                        color=color, fontsize=fontsize, fontweight="bold")


def section_title(pdf, title: str, subtitle: str = ""):
    """Insert a plain title-page section divider into the PDF."""
    fig, ax = plt.subplots(figsize=(16, 3))
    fig.patch.set_facecolor(C_DARK)
    ax.set_facecolor(C_DARK)
    ax.axis("off")
    ax.text(0.5, 0.65, title, transform=ax.transAxes,
            ha="center", va="center", color="white",
            fontsize=22, fontweight="bold")
    if subtitle:
        ax.text(0.5, 0.3, subtitle, transform=ax.transAxes,
                ha="center", va="center", color="#BDC3C7", fontsize=13)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
#  CHART 1 — Executive Overview Dashboard
# ══════════════════════════════════════════════════════════════════════════════

def chart_overview(pdf):
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle("UC10 – Data Cleaning Pipeline: Executive Overview",
                 fontsize=18, fontweight="bold", color=C_DARK, y=0.98)

    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.55, wspace=0.45)

    # ── KPI tiles (top row) ────────────────────────────────────────────────
    kpis = [
        ("Total Records",    f"{len(raw):,}",  f"{len(clean):,}",  "unchanged"),
        ("Columns",          str(raw.shape[1]), str(clean.shape[1]), f"+{clean.shape[1]-raw.shape[1]} new flag cols"),
        ("Missing Values",   f"{raw.isnull().sum().sum():,}",
                             f"{clean[raw.columns].isnull().sum().sum():,}", "identifiers preserved"),
        ("Rows Deleted",     "0",              "0",                "anomalies flagged, not removed"),
    ]
    colors_tile = [C_FLAG, C_CLEAN, C_WARN, C_CLEAN]

    for i, (label, before, after, note) in enumerate(kpis):
        ax = fig.add_subplot(gs[0, i])
        ax.set_facecolor(colors_tile[i])
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
        ax.text(0.5, 0.85, label,  ha="center", va="top",
                fontsize=11, color="white", fontweight="bold")
        ax.text(0.5, 0.58, f"Before: {before}", ha="center", va="center",
                fontsize=10, color="white")
        ax.text(0.5, 0.38, f"After:  {after}",  ha="center", va="center",
                fontsize=12, color="white", fontweight="bold")
        ax.text(0.5, 0.10, note, ha="center", va="bottom",
                fontsize=7.5, color="#ECF0F1", style="italic")

    # ── Record type distribution (bottom-left) ────────────────────────────
    ax2 = fig.add_subplot(gs[1, :2])
    rec_counts = raw["Record_Type"].value_counts()
    bars = ax2.bar(rec_counts.index, rec_counts.values,
                   color=[C_FLAG, C_WARN, C_RAW], edgecolor="white", width=0.5)
    for bar, val in zip(bars, rec_counts.values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                 f"{val:,}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax2.set_title("Record Type Distribution", fontweight="bold")
    ax2.set_ylabel("Count")
    ax2.set_ylim(0, rec_counts.max() * 1.2)

    # ── New columns added breakdown (bottom-right) ───────────────────────
    ax3 = fig.add_subplot(gs[1, 2:])
    new_cols = [c for c in clean.columns if c not in raw.columns]
    categories = {
        "Missingness\nIndicators": len([c for c in new_cols if c.startswith("Missing_")]),
        "Date\nAnomaly Flags":     len([c for c in new_cols if any(x in c for x in ["Date_Flag","Service_Flag","Submission_","Decision_Before"])]),
        "Financial\nAnomaly Flags":len([c for c in new_cols if any(x in c for x in ["Billed","Allowed","Paid","Zero"])]),
        "Outlier\nFlags":          len([c for c in new_cols if "Outlier" in c]),
        "Provider/Bene\nFlags":    len([c for c in new_cols if any(x in c for x in ["Provider","Beneficiary","Denial","Pipeline","Latency_Flag","Frequency"])]),
        "DQ Score\nColumns":       len([c for c in new_cols if "Quality" in c or "Duplicate" in c]),
    }
    pal = [C_FLAG, C_RAW, C_WARN, "#9B59B6", "#1ABC9C", C_CLEAN]
    bars2 = ax3.barh(list(categories.keys()), list(categories.values()),
                     color=pal, edgecolor="white")
    for bar, val in zip(bars2, categories.values()):
        ax3.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                 str(val), va="center", fontsize=10, fontweight="bold")
    ax3.set_title(f"New Columns Added by Category (Total: {len(new_cols)})", fontweight="bold")
    ax3.set_xlabel("Count")
    ax3.set_xlim(0, max(categories.values()) * 1.25)

    save_fig(fig, "01_overview_dashboard", pdf)


# ══════════════════════════════════════════════════════════════════════════════
#  CHART 2 — Missing Value Heatmap: Before vs After
# ══════════════════════════════════════════════════════════════════════════════

def chart_missing_values(pdf):
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    fig.suptitle("Missing Values — Before vs After Cleaning",
                 fontsize=16, fontweight="bold", y=1.01)

    for ax, df, label, cmap in zip(
        axes,
        [raw, clean[raw.columns]],
        ["BEFORE Cleaning (Raw)", "AFTER Cleaning (Original Columns Only)"],
        ["Reds", "Greens"]
    ):
        miss = df.isnull().mean().sort_values(ascending=False)
        miss_df = miss.reset_index()
        miss_df.columns = ["Column", "Missing_Pct"]
        miss_df["Missing_Pct_Val"] = miss_df["Missing_Pct"] * 100

        colors = []
        for v in miss_df["Missing_Pct_Val"]:
            if v == 0:      colors.append("#D5F5E3")
            elif v < 5:     colors.append("#F9E79F")
            elif v < 30:    colors.append("#F0B27A")
            else:           colors.append("#E74C3C")

        bars = ax.barh(miss_df["Column"], miss_df["Missing_Pct_Val"],
                       color=colors, edgecolor="white")
        for bar, val in zip(bars, miss_df["Missing_Pct_Val"]):
            if val > 0:
                ax.text(val + 0.3, bar.get_y() + bar.get_height()/2,
                        f"{val:.1f}%", va="center", fontsize=7, color=C_DARK)
        ax.set_xlim(0, 110)
        ax.set_xlabel("Missing %")
        ax.set_title(label, fontweight="bold", fontsize=12,
                     color=C_RAW if "BEFORE" in label else C_CLEAN)
        ax.invert_yaxis()

    patches = [
        mpatches.Patch(color="#D5F5E3", label="0% missing"),
        mpatches.Patch(color="#F9E79F", label="<5% missing"),
        mpatches.Patch(color="#F0B27A", label="5-30% missing"),
        mpatches.Patch(color="#E74C3C", label=">30% missing"),
    ]
    fig.legend(handles=patches, loc="lower center", ncol=4,
               frameon=True, fontsize=9, bbox_to_anchor=(0.5, -0.02))

    note = ("After cleaning: identifiers (BENE_ID, Provider_NPI) remain NaN — "
            "values are NOT fabricated. Missingness indicators created instead.")
    fig.text(0.5, -0.06, note, ha="center", fontsize=9,
             color=C_WARN, style="italic")

    plt.tight_layout()
    save_fig(fig, "02_missing_values_before_after", pdf)


# ══════════════════════════════════════════════════════════════════════════════
#  CHART 3 — Processing Latency: Before vs After (with negative latency)
# ══════════════════════════════════════════════════════════════════════════════

def chart_processing_latency(pdf):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle("Processing Latency Analysis — Before vs After",
                 fontsize=15, fontweight="bold")

    raw_lat   = raw["Processing_Latency_Days"].dropna()
    clean_lat = clean["Processing_Latency_Days"].dropna()

    # ── Panel 1: Overlapping distribution ─────────────────────────────────
    ax = axes[0]
    ax.hist(raw_lat.clip(-10, 45),   bins=40, alpha=0.6,
            color=C_RAW,   label=f"Before  (n={len(raw_lat):,})",   edgecolor="white")
    ax.hist(clean_lat.clip(-10, 45), bins=40, alpha=0.6,
            color=C_CLEAN, label=f"After   (n={len(clean_lat):,})", edgecolor="white")
    ax.axvline(0, color=C_DARK, linestyle="--", linewidth=1.5, label="Zero (normal boundary)")
    ax.set_xlabel("Processing Latency (days)")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution (clipped at -10 / +45)")
    ax.legend(fontsize=8)

    # ── Panel 2: Negative latency flagged ─────────────────────────────────
    ax2 = axes[1]
    neg_before = int((raw_lat < 0).sum())
    neg_after  = int((clean_lat < 0).sum())
    flag_count = int(clean["Negative_Processing_Latency_Flag"].sum())
    cats   = ["Negative\n(Before)", "Negative\n(After — Preserved)", "Flagged\n(New Flag Col)"]
    values = [neg_before, neg_after, flag_count]
    colors = [C_RAW, C_WARN, C_FLAG]
    bars   = ax2.bar(cats, values, color=colors, edgecolor="white", width=0.5)
    for bar, val in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                 f"{val:,}", ha="center", fontsize=11, fontweight="bold")
    ax2.set_title("Negative Latency: Preserved + Flagged\n(NOT deleted)")
    ax2.set_ylabel("Record Count")
    ax2.set_ylim(0, max(values) * 1.25)

    # ── Panel 3: Outlier flag summary ─────────────────────────────────────
    ax3 = axes[2]
    outlier_flag = clean["Processing_Latency_Outlier_Flag"]
    labels = ["Normal\nRecords", "IQR Outlier\n(Flagged)"]
    sizes  = [(outlier_flag == 0).sum(), (outlier_flag == 1).sum()]
    explode = (0, 0.08)
    wedges, texts, autotexts = ax3.pie(
        sizes, labels=labels, autopct="%1.1f%%",
        colors=[C_CLEAN, C_RAW], startangle=90, explode=explode,
        textprops={"fontsize": 10}, pctdistance=0.75,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5}
    )
    for at in autotexts:
        at.set_fontweight("bold")
    ax3.set_title("Processing Latency Outlier Flag\n(IQR method — values preserved)")

    note = "Key change: 2,785 negative-latency records PRESERVED with Negative_Processing_Latency_Flag=1 | 583 IQR outliers FLAGGED, not removed"
    fig.text(0.5, -0.04, note, ha="center", fontsize=9,
             color=C_RAW, style="italic", fontweight="bold")
    plt.tight_layout()
    save_fig(fig, "03_processing_latency", pdf)


# ══════════════════════════════════════════════════════════════════════════════
#  CHART 4 — Financial Amounts: Before vs After
# ══════════════════════════════════════════════════════════════════════════════

def chart_financials(pdf):
    fin_cols = ["Billed_Amount", "Allowed_Amount", "Paid_Amount", "Patient_Responsibility"]
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    fig.suptitle("Financial Amounts — Distribution Before vs After (Anomalies Preserved)",
                 fontsize=15, fontweight="bold")

    for col_idx, col in enumerate(fin_cols):
        raw_s   = raw[col].dropna()
        clean_s = clean[col].dropna()

        # Top row: distribution overlay (log scale)
        ax = axes[0, col_idx]
        all_vals = pd.concat([raw_s, clean_s])
        vmin = max(all_vals.min(), -15000)
        vmax = min(all_vals.quantile(0.99), 60000)
        bins = np.linspace(vmin, vmax, 50)

        ax.hist(raw_s.clip(vmin, vmax),   bins=bins, alpha=0.65,
                color=C_RAW,   label="Before", edgecolor="white", linewidth=0.3)
        ax.hist(clean_s.clip(vmin, vmax), bins=bins, alpha=0.65,
                color=C_CLEAN, label="After",  edgecolor="white", linewidth=0.3)
        if raw_s.min() < 0:
            ax.axvline(0, color=C_DARK, linestyle="--", linewidth=1.2)
        ax.set_title(col, fontweight="bold", fontsize=10)
        ax.set_xlabel("Amount ($)")
        ax.set_ylabel("Frequency")
        ax.legend(fontsize=7)

        # Bottom row: anomaly flag counts
        ax2 = axes[1, col_idx]
        neg_col  = f"Negative_{col.replace('_Amount','')}_Amount_Flag"
        out_col  = f"{col.replace('_Amount','')}_Amount_Outlier_Flag"

        counts = {}
        if neg_col in clean.columns:
            counts[f"Negative\n({neg_col[:12]}…)"] = int(clean[neg_col].sum())
        if out_col in clean.columns:
            counts[f"IQR Outlier\n({out_col[:12]}…)"] = int(clean[out_col].sum())
        if col == "Paid_Amount":
            counts["Paid >\nAllowed"] = int(clean.get("Paid_Exceeds_Allowed_Flag", pd.Series(0)).sum())
        if col == "Allowed_Amount":
            counts["Allowed >\nBilled"] = int(clean.get("Allowed_Exceeds_Billed_Flag", pd.Series(0)).sum())

        if counts:
            b = ax2.bar(list(counts.keys()), list(counts.values()),
                        color=[C_FLAG, C_WARN, C_RAW, "#9B59B6"][:len(counts)],
                        edgecolor="white")
            for bar, val in zip(b, counts.values()):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                         str(val), ha="center", fontsize=9, fontweight="bold")
            ax2.set_title(f"{col} — Anomaly Flags", fontsize=9, fontweight="bold")
            ax2.set_ylabel("Flagged Records")
            ax2.set_ylim(0, max(counts.values()) * 1.35 if counts else 10)
        else:
            ax2.text(0.5, 0.5, "No anomaly flags", ha="center", va="center",
                     transform=ax2.transAxes, fontsize=10, color="gray")
            ax2.axis("off")

    note = "Key: Negative amounts and outliers are PRESERVED with flag columns. No blanket deletion or clipping applied."
    fig.text(0.5, -0.01, note, ha="center", fontsize=9,
             color=C_RAW, style="italic", fontweight="bold")
    plt.tight_layout()
    save_fig(fig, "04_financial_amounts", pdf)


# ══════════════════════════════════════════════════════════════════════════════
#  CHART 5 — Date Anomaly Flags
# ══════════════════════════════════════════════════════════════════════════════

def chart_date_anomalies(pdf):
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.suptitle("Date Anomalies — Flagged After Cleaning (Not Removed)",
                 fontsize=15, fontweight="bold")

    # ── Panel 1: All date anomaly flag counts ─────────────────────────────
    date_flags = {
        "Invalid Service\nDate Range":          int(clean["Invalid_Service_Date_Flag"].sum()),
        "Submission\nBefore Service":           int(clean["Submission_Before_Service_Flag"].sum()),
        "Processing\nBefore Submission":        int(clean["Processing_Before_Submission_Flag"].sum()),
        "Decision\nBefore Submission":          int(clean["Decision_Before_Submission_Flag"].sum()),
        "Missing\nService Date":                int(clean["Missing_Service_Date_Flag"].sum()),
        "Missing\nDecision Date":               int(clean["Missing_Decision_Date_Flag"].sum()),
        "Missing\nProcessed Date":              int(clean["Missing_Processed_Date_Flag"].sum()),
    }
    colors = [C_RAW, C_WARN, C_FLAG, "#9B59B6", "#E67E22", "#1ABC9C", "#95A5A6"]
    ax = axes[0]
    bars = ax.bar(list(date_flags.keys()), list(date_flags.values()),
                  color=colors, edgecolor="white")
    for bar, val in zip(bars, date_flags.values()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                f"{val:,}", ha="center", fontsize=9, fontweight="bold")
    ax.set_title("Date Anomaly Flag Counts", fontweight="bold")
    ax.set_ylabel("Records Flagged")
    ax.set_ylim(0, max(date_flags.values()) * 1.25)
    ax.tick_params(axis="x", labelsize=7.5)

    # ── Panel 2: Processing_Before_Submission by Record_Type ─────────────
    ax2 = axes[1]
    grp = clean.groupby("Record_Type")["Processing_Before_Submission_Flag"].sum()
    bars2 = ax2.bar(grp.index, grp.values,
                    color=[C_FLAG, C_WARN, C_RAW], edgecolor="white", width=0.5)
    for bar, val in zip(bars2, grp.values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                 f"{int(val):,}", ha="center", fontsize=11, fontweight="bold")
    ax2.set_title("Processing Before Submission\nby Record Type (Preserved as Anomaly)",
                  fontweight="bold")
    ax2.set_ylabel("Records Flagged")
    ax2.set_ylim(0, grp.max() * 1.25)

    # ── Panel 3: Missing date flags stacked by record type ────────────────
    ax3 = axes[2]
    miss_date_cols = ["Missing_Service_Date_Flag", "Missing_Decision_Date_Flag",
                      "Missing_Processed_Date_Flag"]
    grp3 = clean.groupby("Record_Type")[miss_date_cols].sum()
    grp3.plot(kind="bar", ax=ax3, color=[C_RAW, C_WARN, C_FLAG],
              edgecolor="white", width=0.6)
    ax3.set_title("Missing Date Flags by Record Type\n(Context-aware — not deleted)",
                  fontweight="bold")
    ax3.set_ylabel("Records with Missing Date")
    ax3.set_xlabel("")
    ax3.tick_params(axis="x", rotation=15)
    ax3.legend(["Missing Service Date", "Missing Decision Date", "Missing Processed Date"],
               fontsize=8, loc="upper right")

    note = ("Processing_Before_Submission (2,745 records) is a genuine pipeline clock anomaly. "
            "All records retained with flag=1.")
    fig.text(0.5, -0.03, note, ha="center", fontsize=9,
             color=C_RAW, style="italic", fontweight="bold")
    plt.tight_layout()
    save_fig(fig, "05_date_anomalies", pdf)


# ══════════════════════════════════════════════════════════════════════════════
#  CHART 6 — Authorization Anomaly Analysis
# ══════════════════════════════════════════════════════════════════════════════

def chart_auth_anomalies(pdf):
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.suptitle("Authorization Anomalies — Before vs After",
                 fontsize=15, fontweight="bold")

    # ── Panel 1: Auth funnel ───────────────────────────────────────────────
    ax = axes[0]
    total      = len(raw)
    auth_y     = int((raw["Auth_Required_Flag"] == "Y").sum())
    auth_linked = int(raw["Auth_Linked_ID"].notna().sum())
    auth_y_linked = int(((raw["Auth_Required_Flag"] == "Y") & raw["Auth_Linked_ID"].notna()).sum())
    missing_link = auth_y - auth_y_linked

    stages = ["Total\nRecords", "Auth\nRequired (Y)", "Has Auth\nLinked ID",
              "Auth Required\n+ Has Link", "Auth Required\nBUT Missing Link\n(⚠ Anomaly)"]
    values = [total, auth_y, auth_linked, auth_y_linked, missing_link]
    colors_f = [C_FLAG, C_WARN, C_CLEAN, "#27AE60", C_RAW]
    bars = ax.barh(stages[::-1], values[::-1], color=colors_f[::-1], edgecolor="white")
    for bar, val in zip(bars, values[::-1]):
        ax.text(bar.get_width() + 50, bar.get_y() + bar.get_height()/2,
                f"{val:,}", va="center", fontsize=10, fontweight="bold")
    ax.set_title("Authorization Linkage Funnel\n(Raw Data)", fontweight="bold")
    ax.set_xlabel("Record Count")
    ax.set_xlim(0, total * 1.15)

    # ── Panel 2: New flag vs existing flag comparison ─────────────────────
    ax2 = axes[1]
    existing_flag = int(clean["Missing_Required_Auth_Link"].sum())
    new_flag      = int(clean["Missing_Required_Auth_Link_Flag"].sum())
    cats = ["Existing Feature\n(Missing_Required_Auth_Link)",
            "New Flag Column\n(Missing_Required_Auth_Link_Flag)"]
    vals = [existing_flag, new_flag]
    bars2 = ax2.bar(cats, vals, color=[C_WARN, C_FLAG], edgecolor="white", width=0.45)
    for bar, val in zip(bars2, vals):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                 f"{val:,}", ha="center", fontsize=12, fontweight="bold")
    ax2.set_title("Auth Missing Flags:\nExisting vs New (Both Preserved)", fontweight="bold")
    ax2.set_ylabel("Records Flagged")
    ax2.set_ylim(0, max(vals) * 1.25)

    note2 = f"Existing flag: {existing_flag:,} | New computed flag: {new_flag:,} | Difference: {abs(new_flag-existing_flag):,} (Auth=Y records not covered by existing feature)"
    ax2.text(0.5, -0.18, note2, transform=ax2.transAxes, ha="center",
             fontsize=8, color=C_DARK, style="italic")

    # ── Panel 3: Auth anomaly by Record_Type ──────────────────────────────
    ax3 = axes[2]
    grp = clean.groupby("Record_Type")["Missing_Required_Auth_Link_Flag"].agg(
        Total="count", Flagged="sum"
    ).reset_index()
    grp["Not_Flagged"] = grp["Total"] - grp["Flagged"]
    x = np.arange(len(grp))
    width = 0.35
    ax3.bar(x - width/2, grp["Not_Flagged"], width, label="Auth OK / N/A",
            color=C_CLEAN, edgecolor="white")
    ax3.bar(x + width/2, grp["Flagged"],     width, label="Auth Required + Missing Link",
            color=C_RAW,   edgecolor="white")
    for i, (nf, f) in enumerate(zip(grp["Not_Flagged"], grp["Flagged"])):
        ax3.text(i - width/2, nf + 20, f"{nf:,}", ha="center", fontsize=8)
        ax3.text(i + width/2, f  + 20, f"{f:,}",  ha="center", fontsize=8, color=C_RAW, fontweight="bold")
    ax3.set_xticks(x)
    ax3.set_xticklabels(grp["Record_Type"], rotation=10, fontsize=9)
    ax3.set_title("Auth Anomaly by Record Type\n(Records preserved, not deleted)",
                  fontweight="bold")
    ax3.set_ylabel("Record Count")
    ax3.legend(fontsize=8)

    plt.tight_layout()
    save_fig(fig, "06_auth_anomalies", pdf)


# ══════════════════════════════════════════════════════════════════════════════
#  CHART 7 — SLA & Pipeline Anomalies
# ══════════════════════════════════════════════════════════════════════════════

def chart_sla_pipeline(pdf):
    fig, axes = plt.subplots(2, 3, figsize=(21, 12))
    fig.suptitle("SLA & Pipeline Anomalies — Before vs After",
                 fontsize=15, fontweight="bold")

    # ── SLA Breach distribution before vs after ───────────────────────────
    ax = axes[0, 0]
    sla_raw   = raw["SLA_Breach_Flag"].value_counts()
    sla_clean = clean["SLA_Breach_Flag"].value_counts()
    x = np.arange(len(sla_raw))
    width = 0.35
    ax.bar(x - width/2, sla_raw.reindex(["N","Y","UNKNOWN_NO_DATE"], fill_value=0),
           width, label="Before", color=C_RAW, edgecolor="white")
    ax.bar(x + width/2, sla_clean.reindex(["N","Y","UNKNOWN_NO_DATE"], fill_value=0),
           width, label="After",  color=C_CLEAN, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(["No Breach (N)", "Breach (Y)", "Unknown\n(No Date)"])
    ax.set_title("SLA Breach Flag Distribution\n(Unchanged — preserved)")
    ax.set_ylabel("Count"); ax.legend()

    # ── Rolling 7D SLA breach rate distribution ───────────────────────────
    ax2 = axes[0, 1]
    ax2.hist(clean["Rolling_7D_Avg_SLA_Breach_Rate"].dropna(), bins=40,
             color=C_FLAG, edgecolor="white", alpha=0.85)
    ax2.axvline(clean["Rolling_7D_Avg_SLA_Breach_Rate"].median(),
                color=C_RAW, linestyle="--", linewidth=1.5,
                label=f"Median: {clean['Rolling_7D_Avg_SLA_Breach_Rate'].median():.3f}")
    ax2.set_title("Rolling 7D Avg SLA Breach Rate\n(Existing Feature — Preserved)")
    ax2.set_xlabel("Rate"); ax2.set_ylabel("Frequency"); ax2.legend(fontsize=8)

    # ── SLA Breach Vs DOW Norm ────────────────────────────────────────────
    ax3 = axes[0, 2]
    ax3.hist(clean["SLA_Breach_Vs_DOW_Norm"].dropna(), bins=40,
             color=C_WARN, edgecolor="white", alpha=0.85)
    ax3.axvline(0, color=C_RAW, linestyle="--", linewidth=1.5, label="Zero baseline")
    ax3.set_title("SLA Breach Vs DOW Norm\n(Existing Feature — Preserved)")
    ax3.set_xlabel("Normalised Score"); ax3.set_ylabel("Frequency"); ax3.legend(fontsize=8)

    # ── Pipeline gap flags ────────────────────────────────────────────────
    ax4 = axes[1, 0]
    gap_existing  = int(clean["Pipeline_Gap_Flag"].sum())
    gap_extended  = int(clean["Extended_Pipeline_Gap_Flag"].sum())
    gap_normal    = len(clean) - max(gap_existing, gap_extended)
    cats = ["No Gap\n(Normal)", "Pipeline Gap\n(Existing Flag)", "Extended Gap\n(>2 days, New Flag)"]
    vals = [gap_normal, gap_existing, gap_extended]
    bars = ax4.bar(cats, vals, color=[C_CLEAN, C_WARN, C_RAW], edgecolor="white", width=0.5)
    for bar, val in zip(bars, vals):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                 f"{val:,}", ha="center", fontsize=11, fontweight="bold")
    ax4.set_title("Pipeline Gap Flags\n(Existing + New supplemental flag)")
    ax4.set_ylabel("Records")
    ax4.set_ylim(0, max(vals) * 1.2)

    # ── Days_Since_Prev_Batch distribution ────────────────────────────────
    ax5 = axes[1, 1]
    dsb_raw   = raw["Days_Since_Prev_Batch"].dropna()
    dsb_clean = clean["Days_Since_Prev_Batch"].dropna()
    ax5.hist(dsb_raw.clip(0, 6),   bins=10, alpha=0.6,
             color=C_RAW,   label=f"Before (NaN={raw['Days_Since_Prev_Batch'].isna().sum()})")
    ax5.hist(dsb_clean.clip(0, 6), bins=10, alpha=0.6,
             color=C_CLEAN, label=f"After  (NaN=0 — filled with 1.0)")
    ax5.set_title("Days Since Prev Batch\n(18 missing → imputed with 1.0)")
    ax5.set_xlabel("Days"); ax5.set_ylabel("Frequency"); ax5.legend(fontsize=8)

    # ── Batch Volume Vs Trend ─────────────────────────────────────────────
    ax6 = axes[1, 2]
    ax6.scatter(clean["Rolling_7D_Avg_Volume"],
                clean["Volume_Vs_Trend_Ratio"],
                c=clean["Batch_Volume_Outlier_Flag"].map({0: C_CLEAN, 1: C_RAW}),
                alpha=0.4, s=15)
    ax6.axhline(1.0, color=C_DARK, linestyle="--", linewidth=1,
                label="Trend ratio = 1.0 (expected)")
    norm_patch = mpatches.Patch(color=C_CLEAN, label="Normal Volume")
    out_patch  = mpatches.Patch(color=C_RAW,   label="Outlier Flagged")
    ax6.legend(handles=[norm_patch, out_patch], fontsize=8)
    ax6.set_title("Batch Volume vs Rolling Avg\n(Outliers flagged, not removed)")
    ax6.set_xlabel("Rolling 7D Avg Volume"); ax6.set_ylabel("Volume Vs Trend Ratio")

    plt.tight_layout()
    save_fig(fig, "07_sla_pipeline_anomalies", pdf)


# ══════════════════════════════════════════════════════════════════════════════
#  CHART 8 — Provider & Beneficiary Anomalies
# ══════════════════════════════════════════════════════════════════════════════

def chart_provider_bene(pdf):
    fig, axes = plt.subplots(2, 3, figsize=(21, 12))
    fig.suptitle("Provider & Beneficiary Anomalies — Before vs After",
                 fontsize=15, fontweight="bold")

    # ── Provider denial rate distribution ────────────────────────────────
    ax = axes[0, 0]
    ax.hist(raw["Provider_Denial_Rate"].dropna(), bins=40, alpha=0.6,
            color=C_RAW,   label="Before", edgecolor="white")
    ax.hist(clean["Provider_Denial_Rate"].dropna(), bins=40, alpha=0.6,
            color=C_CLEAN, label="After",  edgecolor="white")
    ax.axvline(0.5, color=C_DARK, linestyle="--", linewidth=1.5,
               label="High denial threshold (0.5)")
    ax.set_title("Provider Denial Rate Distribution\n(Preserved; high-denial providers flagged)")
    ax.set_xlabel("Denial Rate"); ax.set_ylabel("Frequency"); ax.legend(fontsize=8)

    # ── High denial rate flag ─────────────────────────────────────────────
    ax2 = axes[0, 1]
    hdr_count = int(clean["High_Denial_Rate_Flag"].sum())
    normal    = len(clean) - hdr_count
    wedges, texts, at = ax2.pie(
        [normal, hdr_count], labels=["Normal\nDenial Rate", "High Denial Rate\n(>50% — Flagged)"],
        autopct="%1.1f%%", colors=[C_CLEAN, C_RAW], startangle=90,
        explode=(0, 0.08), pctdistance=0.75,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5}
    )
    for a in at: a.set_fontweight("bold")
    ax2.set_title(f"High Denial Rate Flag\n({hdr_count:,} records flagged — NOT removed)")

    # ── Provider total records (log scale) ────────────────────────────────
    ax3 = axes[0, 2]
    ptr_raw   = raw["Provider_Total_Records"].dropna()
    ptr_clean = clean["Provider_Total_Records"].dropna()
    ax3.hist(np.log1p(ptr_raw),   bins=30, alpha=0.6,
             color=C_RAW,   label="Before", edgecolor="white")
    ax3.hist(np.log1p(ptr_clean), bins=30, alpha=0.6,
             color=C_CLEAN, label="After",  edgecolor="white")
    ax3.set_title("Provider Total Records (log1p scale)\n(IQR outliers flagged, not removed)")
    ax3.set_xlabel("log(1 + Provider_Total_Records)")
    ax3.set_ylabel("Frequency"); ax3.legend(fontsize=8)

    # ── Beneficiary record count ──────────────────────────────────────────
    ax4 = axes[1, 0]
    brc_raw   = raw["Beneficiary_Record_Count"].dropna()
    brc_clean = clean["Beneficiary_Record_Count"].dropna()
    ax4.hist(brc_raw.clip(0, 20),   bins=20, alpha=0.6,
             color=C_RAW,   label=f"Before (NaN={raw['Beneficiary_Record_Count'].isna().sum()})")
    ax4.hist(brc_clean.clip(0, 20), bins=20, alpha=0.6,
             color=C_CLEAN, label=f"After  (NaN=0 — median imputed)")
    ax4.set_title("Beneficiary Record Count\n(150 NaN imputed with median — per Record_Type)")
    ax4.set_xlabel("Record Count"); ax4.set_ylabel("Frequency"); ax4.legend(fontsize=8)

    # ── High frequency beneficiary ────────────────────────────────────────
    ax5 = axes[1, 1]
    hfb_existing = int(clean["High_Frequency_Beneficiary_Flag"].sum())
    hfb_extreme  = int(clean["Extreme_Beneficiary_Frequency_Flag"].sum())
    x = [0, 1]
    bars = ax5.bar(
        ["Existing:\nHigh_Frequency_Beneficiary_Flag",
         "New:\nExtreme_Beneficiary_Frequency_Flag\n(top 5%)"],
        [hfb_existing, hfb_extreme],
        color=[C_WARN, C_RAW], edgecolor="white", width=0.4
    )
    for bar, val in zip(bars, [hfb_existing, hfb_extreme]):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                 f"{val:,}", ha="center", fontsize=12, fontweight="bold")
    ax5.set_title("Beneficiary Frequency Flags\n(Existing preserved + New added)")
    ax5.set_ylabel("Records Flagged")
    ax5.set_ylim(0, max(hfb_existing, hfb_extreme) * 1.3)

    # ── Provider state distribution ───────────────────────────────────────
    ax6 = axes[1, 2]
    state_raw   = raw["Provider_State"].value_counts().head(12)
    state_clean = clean["Provider_State"].value_counts().head(12)
    unknown_added = int((clean["Provider_State"] == "UNKNOWN").sum())
    x = np.arange(len(state_raw))
    ax6.bar(x - 0.2, state_raw.values,   0.4, label="Before", color=C_RAW,   edgecolor="white")
    ax6.bar(x + 0.2, state_clean.reindex(state_raw.index, fill_value=0).values,
            0.4, label="After",  color=C_CLEAN, edgecolor="white")
    ax6.set_xticks(x)
    ax6.set_xticklabels(state_raw.index, rotation=45, ha="right", fontsize=8)
    ax6.set_title(f"Top 12 Provider States\n({unknown_added} NaN → 'UNKNOWN' after cleaning)")
    ax6.set_ylabel("Count"); ax6.legend(fontsize=8)

    plt.tight_layout()
    save_fig(fig, "08_provider_bene_anomalies", pdf)


# ══════════════════════════════════════════════════════════════════════════════
#  CHART 9 — Record-Type-Aware Missingness
# ══════════════════════════════════════════════════════════════════════════════

def chart_record_type_missingness(pdf):
    """Show that field missingness is EXPECTED for certain record types."""
    fig, axes = plt.subplots(1, 3, figsize=(21, 8))
    fig.suptitle("Record-Type-Aware Missingness — Why Not All NaN Is a Problem",
                 fontsize=14, fontweight="bold")

    rtype_cols = {
        "MEDICAL_CLAIM":  ["Diagnosis_Code", "Procedure_Code", "NDC_Code",
                           "Drug_Name", "Days_Supply", "Quantity_Dispensed"],
        "PHARMACY_CLAIM": ["Diagnosis_Code", "Procedure_Code", "NDC_Code",
                           "Drug_Name", "Days_Supply", "Quantity_Dispensed"],
        "PRIOR_AUTH":     ["Diagnosis_Code", "Procedure_Code", "NDC_Code",
                           "Drug_Name", "Days_Supply", "Quantity_Dispensed"],
    }
    colors_rt = {"MEDICAL_CLAIM": C_FLAG, "PHARMACY_CLAIM": C_WARN, "PRIOR_AUTH": C_RAW}

    for ax_idx, (rtype, cols) in enumerate(rtype_cols.items()):
        ax = axes[ax_idx]
        subset = raw[raw["Record_Type"] == rtype]
        miss_pct = (subset[cols].isnull().mean() * 100).round(1)

        bar_colors = []
        for col, pct in miss_pct.items():
            # Expected missing = natural absence for this record type
            if rtype == "MEDICAL_CLAIM" and col in ["NDC_Code","Drug_Name","Days_Supply","Quantity_Dispensed"]:
                bar_colors.append("#95A5A6")   # grey = expected
            elif rtype == "PHARMACY_CLAIM" and col in ["Diagnosis_Code","Procedure_Code"]:
                bar_colors.append("#95A5A6")
            elif rtype == "PRIOR_AUTH" and col == "Diagnosis_Code":
                bar_colors.append("#95A5A6")
            elif pct > 50:
                bar_colors.append(C_WARN)
            elif pct > 0:
                bar_colors.append(C_RAW)
            else:
                bar_colors.append(C_CLEAN)

        bars = ax.bar(cols, miss_pct.values, color=bar_colors, edgecolor="white")
        for bar, val in zip(bars, miss_pct.values):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 1, f"{val:.0f}%",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.set_ylim(0, 120)
        ax.set_title(f"{rtype}\n(n={len(subset):,})",
                     fontweight="bold", color=colors_rt[rtype])
        ax.set_ylabel("Missing %")
        ax.set_xticklabels(cols, rotation=30, ha="right", fontsize=8)

    legend_patches = [
        mpatches.Patch(color="#95A5A6", label="Expected missing (field not applicable to this type)"),
        mpatches.Patch(color=C_CLEAN,   label="Complete (0% missing)"),
        mpatches.Patch(color=C_WARN,    label="High missing but contextual"),
        mpatches.Patch(color=C_RAW,     label="Unexpected missing — flagged"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=4,
               fontsize=9, frameon=True, bbox_to_anchor=(0.5, -0.04))

    note = ("Key insight: NDC_Code/Drug_Name are 100% missing for MEDICAL_CLAIM — that is CORRECT.\n"
            "Diagnosis_Code is 100% missing for PHARMACY_CLAIM — also CORRECT. These are NOT anomalies.")
    fig.text(0.5, -0.11, note, ha="center", fontsize=9,
             color=C_FLAG, style="italic")
    plt.tight_layout()
    save_fig(fig, "09_record_type_missingness", pdf)


# ══════════════════════════════════════════════════════════════════════════════
#  CHART 10 — Data Quality Score Distribution
# ══════════════════════════════════════════════════════════════════════════════

def chart_dq_score(pdf):
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    fig.suptitle("Data Quality Risk Score — New Feature Added by Pipeline",
                 fontsize=15, fontweight="bold")

    dq_score = clean["Data_Quality_Risk_Score"]
    dq_count = clean["Data_Quality_Issue_Count"]

    # ── Score distribution ────────────────────────────────────────────────
    ax = axes[0]
    n_bins = 20
    ax.hist(dq_score, bins=n_bins, color=C_FLAG, edgecolor="white", alpha=0.85)
    ax.axvline(dq_score.mean(), color=C_RAW, linestyle="--", linewidth=2,
               label=f"Mean: {dq_score.mean():.3f}")
    ax.axvline(dq_score.median(), color=C_WARN, linestyle="--", linewidth=2,
               label=f"Median: {dq_score.median():.3f}")
    ax.set_title("Data_Quality_Risk_Score Distribution\n(0 = clean, 1 = worst)")
    ax.set_xlabel("DQ Risk Score"); ax.set_ylabel("Frequency")
    ax.legend(fontsize=9)

    # ── Issue count breakdown ─────────────────────────────────────────────
    ax2 = axes[1]
    count_vc = dq_count.value_counts().sort_index()
    pal = [C_CLEAN if i == 0 else
           (C_WARN if i <= 2 else C_RAW)
           for i in count_vc.index]
    bars = ax2.bar(count_vc.index.astype(str), count_vc.values,
                   color=pal, edgecolor="white")
    for bar, val in zip(bars, count_vc.values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
                 f"{val:,}", ha="center", fontsize=10, fontweight="bold")
    ax2.set_title("Data_Quality_Issue_Count Distribution\n(Number of flags triggered per record)")
    ax2.set_xlabel("Issue Count"); ax2.set_ylabel("Records")

    # ── DQ score by Record_Type ───────────────────────────────────────────
    ax3 = axes[2]
    for rtype, color in [("MEDICAL_CLAIM", C_FLAG),
                          ("PHARMACY_CLAIM", C_WARN),
                          ("PRIOR_AUTH",    C_RAW)]:
        subset = clean[clean["Record_Type"] == rtype]["Data_Quality_Risk_Score"]
        ax3.hist(subset, bins=15, alpha=0.6, color=color,
                 label=f"{rtype} (μ={subset.mean():.3f})", edgecolor="white")
    ax3.set_title("DQ Risk Score by Record Type\n(Higher = more anomalous patterns)")
    ax3.set_xlabel("DQ Risk Score"); ax3.set_ylabel("Frequency")
    ax3.legend(fontsize=8)

    # Summary stats table
    stats_text = (
        f"Records with DQ issues: {int((dq_count>0).sum()):,} / {len(clean):,} ({(dq_count>0).mean()*100:.1f}%)\n"
        f"Avg issues per record: {dq_count.mean():.2f}  |  Max: {int(dq_count.max())} flags on a single record\n"
        f"Records with score > 0.5 (High Risk): {int((dq_score>0.5).sum()):,}"
    )
    fig.text(0.5, -0.04, stats_text, ha="center", fontsize=10,
             color=C_DARK, style="italic",
             bbox=dict(boxstyle="round,pad=0.4", facecolor=C_LIGHT, edgecolor=C_DARK))

    plt.tight_layout()
    save_fig(fig, "10_dq_score", pdf)


# ══════════════════════════════════════════════════════════════════════════════
#  CHART 11 — Full Anomaly Flag Heatmap (All 36 New Flags)
# ══════════════════════════════════════════════════════════════════════════════

def chart_anomaly_heatmap(pdf):
    new_cols = [c for c in clean.columns if c not in raw.columns
                and c not in ["Data_Quality_Issue_Count", "Data_Quality_Risk_Score"]]
    flag_counts = clean[new_cols].sum().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(14, 12))
    fig.suptitle("All New Anomaly / DQ Flag Columns — Record Count per Flag\n(Added by Cleaning Pipeline)",
                 fontsize=14, fontweight="bold")

    colors_bar = []
    for val in flag_counts.values:
        if val == 0:            colors_bar.append("#BDC3C7")
        elif val < 100:         colors_bar.append(C_CLEAN)
        elif val < 500:         colors_bar.append(C_WARN)
        elif val < 2000:        colors_bar.append(C_FLAG)
        else:                   colors_bar.append(C_RAW)

    bars = ax.barh(flag_counts.index[::-1], flag_counts.values[::-1],
                   color=colors_bar[::-1], edgecolor="white", height=0.7)
    for bar, val in zip(bars, flag_counts.values[::-1]):
        ax.text(bar.get_width() + 10, bar.get_y() + bar.get_height()/2,
                f"{int(val):,}", va="center", fontsize=8.5, fontweight="bold")

    ax.set_xlabel("Records Flagged", fontsize=11)
    ax.set_title("")
    ax.set_xlim(0, flag_counts.max() * 1.15)

    legend_patches = [
        mpatches.Patch(color="#BDC3C7", label="0 records"),
        mpatches.Patch(color=C_CLEAN,   label="< 100 records"),
        mpatches.Patch(color=C_WARN,    label="100 – 499"),
        mpatches.Patch(color=C_FLAG,    label="500 – 1,999"),
        mpatches.Patch(color=C_RAW,     label="≥ 2,000 records"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=9, frameon=True)

    plt.tight_layout()
    save_fig(fig, "11_anomaly_flag_heatmap", pdf)


# ══════════════════════════════════════════════════════════════════════════════
#  CHART 12 — Cleaning Rules Applied: Change Summary Table
# ══════════════════════════════════════════════════════════════════════════════

def chart_cleaning_summary_table(pdf):
    """Visual table showing every cleaning rule applied and its impact."""
    fig, ax = plt.subplots(figsize=(18, 11))
    fig.suptitle("Cleaning Rules Applied — Before vs After Change Summary",
                 fontsize=15, fontweight="bold")
    ax.axis("off")

    rows = [
        # [Rule, Scope, Before, After, Impact]
        ["Duplicate Detection",  "All records",
         "0 full duplicates\n0 dup Record_IDs",
         "0 removed",
         "No change — dataset clean"],
        ["Date Parsing",         "7 date/datetime cols",
         "Stored as raw strings",
         "Converted to datetime\nerrors='coerce'",
         "Enables date arithmetic\nfor anomaly flags"],
        ["Missing Dates Flagged","Service/Decision/\nProcessed Date",
         "1,971 – 8,227\nmissing per col",
         "7 missingness flags\ncreated",
         "Records kept; ML can\nlearn from absence"],
        ["Date Order Anomalies", "Service/Submission/\nProcessed/Decision",
         "101 invalid ranges\n2,745 backward process",
         "4 binary flags\ncreated",
         "2,846 anomaly records\npreserved with flags"],
        ["Neg. Latency Preserved","Processing_Latency_Days",
         "2,785 negative values\n(pipeline clock issue)",
         "Negative_Processing\n_Latency_Flag = 1",
         "All 2,785 preserved\nNOT deleted"],
        ["Financial Anomalies",  "Billed/Allowed/Paid\nAmounts",
         "23 neg Billed\n29 neg Allowed\n110 Paid > Allowed",
         "5 financial flags\ncreated",
         "All 184 records\npreserved"],
        ["Auth Link Flagging",   "Auth_Required_Flag=Y\n+ missing Auth_Linked_ID",
         "3,492 auth required\nbut no link",
         "Missing_Required_Auth\n_Link_Flag = 1",
         "Critical anomaly\npreserved for ML"],
        ["ID Preservation",      "BENE_ID, Provider_NPI\nAuth_Linked_ID",
         "150/60/9,897 NaN",
         "Missingness flags\nNO imputation",
         "IDs never fabricated\nFlags created instead"],
        ["Categorical Imputation","Status, Source_System\nProvider_State",
         "Some NaN values",
         "Filled with 'UNKNOWN'",
         "No data leakage\nfrom fabricated IDs"],
        ["Numerical Imputation", "Processing_Latency\nProvider_Total_Records\nDays_Since_Prev_Batch",
         "286 / 60 / 18 NaN",
         "Median-imputed\nper Record_Type",
         "Only non-identifier\nnumerics imputed"],
        ["IQR Outlier Flagging", "Latency, Billed Amt\nBatch Volume, Provider Vol",
         "Values present but\nno flags",
         "5 outlier flags\ncreated",
         "Original values kept\noutliers preserved"],
        ["Provider Flags",       "Provider_Denial_Rate\nProvider_Total_Records",
         "No flags",
         "High_Denial_Rate_Flag\nHigh_Volume_Provider_Flag",
         "980 high-denial\nproviders flagged"],
        ["Beneficiary Flags",    "Beneficiary_Record_Count",
         "High_Frequency flag\nexisted",
         "Extreme_Beneficiary\n_Frequency_Flag added",
         "Both flags preserved\n87/468 records flagged"],
        ["DQ Score",             "All 24 flag columns",
         "No composite score",
         "Data_Quality_Issue_Count\nData_Quality_Risk_Score",
         "6,626 records have\n≥1 DQ issue (66.3%)"],
    ]

    col_labels = ["Rule Applied", "Scope", "Before", "After", "Impact / Records Preserved"]
    col_widths  = [0.18, 0.14, 0.18, 0.22, 0.28]

    y_start = 0.97
    row_h   = 0.061
    x_starts = [0.0]
    for w in col_widths[:-1]:
        x_starts.append(x_starts[-1] + w)

    # Header
    for j, (label, xp) in enumerate(zip(col_labels, x_starts)):
        ax.text(xp + col_widths[j]/2, y_start + 0.005, label,
                ha="center", va="center", fontsize=9.5, fontweight="bold",
                color="white",
                transform=ax.transAxes)
    ax.add_patch(mpatches.FancyBboxPatch(
        (0, y_start - 0.005), 1.0, row_h * 0.9,
        boxstyle="round,pad=0.002", facecolor=C_DARK, edgecolor="none",
        transform=ax.transAxes, clip_on=False))

    # Data rows
    for i, row in enumerate(rows):
        y = y_start - (i + 1) * row_h
        bg = C_LIGHT if i % 2 == 0 else "white"
        ax.add_patch(mpatches.FancyBboxPatch(
            (0, y - 0.005), 1.0, row_h * 0.95,
            boxstyle="round,pad=0.001", facecolor=bg, edgecolor="#D5D8DC",
            transform=ax.transAxes, clip_on=False))
        for j, (cell, xp) in enumerate(zip(row, x_starts)):
            color = C_RAW if j == 2 else (C_CLEAN if j == 3 else C_DARK)
            if j == 4:  color = "#1A5276"
            ax.text(xp + col_widths[j]/2 + 0.005, y + row_h * 0.42,
                    cell, ha="center", va="center", fontsize=7.5,
                    color=color, transform=ax.transAxes,
                    multialignment="center")

    plt.tight_layout()
    save_fig(fig, "12_cleaning_rules_table", pdf)


# ══════════════════════════════════════════════════════════════════════════════
#  CHART 13 — Correlation Heatmap of New DQ Flags
# ══════════════════════════════════════════════════════════════════════════════

def chart_flag_correlation(pdf):
    """Show how the new DQ flags co-occur — useful for root-cause clustering."""
    flag_cols = [
        "Negative_Processing_Latency_Flag", "Processing_Before_Submission_Flag",
        "Invalid_Service_Date_Flag",         "Submission_Before_Service_Flag",
        "Missing_Required_Auth_Link_Flag",   "Negative_Billed_Amount_Flag",
        "Paid_Exceeds_Allowed_Flag",         "Allowed_Exceeds_Billed_Flag",
        "High_Denial_Rate_Flag",             "Extreme_Beneficiary_Frequency_Flag",
        "Processing_Latency_Outlier_Flag",   "Billed_Amount_Outlier_Flag",
        "Pipeline_Gap_Flag",                 "Extended_Pipeline_Gap_Flag",
        "Missing_BENE_ID_Flag",              "Missing_Service_Date_Flag",
    ]
    present = [c for c in flag_cols if c in clean.columns]
    corr = clean[present].corr()

    short_names = [c.replace("_Flag","").replace("_"," ") for c in present]
    corr.index   = short_names
    corr.columns = short_names

    fig, ax = plt.subplots(figsize=(16, 13))
    fig.suptitle("Anomaly Flag Co-occurrence Correlation\n(New DQ Flags Added by Pipeline)",
                 fontsize=14, fontweight="bold")

    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, ax=ax, annot=True, fmt=".2f", cmap="RdYlGn",
                center=0, vmin=-1, vmax=1, linewidths=0.5,
                linecolor="#BDC3C7", square=True,
                annot_kws={"size": 7}, mask=False,
                cbar_kws={"shrink": 0.8, "label": "Pearson Correlation"})

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0,  fontsize=8)

    note = ("High positive correlation → flags often fire together (same root cause).\n"
            "Negative_Processing_Latency_Flag ≈ Processing_Before_Submission_Flag (same pipeline clock anomaly).")
    fig.text(0.5, -0.01, note, ha="center", fontsize=9,
             color=C_DARK, style="italic")
    plt.tight_layout()
    save_fig(fig, "13_flag_correlation_heatmap", pdf)


# ══════════════════════════════════════════════════════════════════════════════
#  CHART 14 — Before vs After: Column Count & Schema Change
# ══════════════════════════════════════════════════════════════════════════════

def chart_schema_change(pdf):
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle("Schema Change: Before vs After Cleaning Pipeline",
                 fontsize=15, fontweight="bold")

    # ── Donut: before column composition ─────────────────────────────────
    ax = axes[0]
    b_cats = {
        "Identifiers (5)":          5,
        "Date/Time (7)":            7,
        "Categorical (9)":          9,
        "Numerical — Raw (14)":    14,
        "Existing Engineered (15)":15,
    }
    colors_b = [C_RAW, C_WARN, "#9B59B6", C_FLAG, C_CLEAN]
    wedges, texts, at = ax.pie(
        list(b_cats.values()), labels=list(b_cats.keys()),
        autopct="%1.0f%%", colors=colors_b, startangle=90,
        pctdistance=0.78, wedgeprops={"edgecolor": "white", "linewidth": 2},
        textprops={"fontsize": 9}
    )
    for a in at: a.set_fontweight("bold")
    ax.set_title(f"BEFORE Cleaning\n{raw.shape[1]} columns", fontweight="bold",
                 color=C_RAW, fontsize=13)

    # ── Donut: after column composition ──────────────────────────────────
    ax2 = axes[1]
    new_cols = [c for c in clean.columns if c not in raw.columns]
    a_cats = {
        "Identifiers (5)":               5,
        "Date/Time (7)":                 7,
        "Categorical (9)":               9,
        "Numerical — Raw (14)":         14,
        "Existing Engineered (15)":     15,
        "Missingness Flags (12)":       12,
        "DQ / Anomaly Flags (22)":      22,
        "DQ Score Columns (2)":          2,
    }
    colors_a = [C_RAW, C_WARN, "#9B59B6", C_FLAG, C_CLEAN,
                "#1ABC9C", "#E74C3C", "#2C3E50"]
    wedges2, texts2, at2 = ax2.pie(
        list(a_cats.values()), labels=list(a_cats.keys()),
        autopct="%1.0f%%", colors=colors_a, startangle=90,
        pctdistance=0.78, wedgeprops={"edgecolor": "white", "linewidth": 2},
        textprops={"fontsize": 9}
    )
    for a in at2: a.set_fontweight("bold")
    ax2.set_title(f"AFTER Cleaning\n{clean.shape[1]} columns  (+{clean.shape[1]-raw.shape[1]} new)",
                  fontweight="bold", color=C_CLEAN, fontsize=13)

    note = (f"Total columns: {raw.shape[1]} → {clean.shape[1]}  (+{clean.shape[1]-raw.shape[1]} columns added, 0 removed).\n"
            f"All original 50 columns preserved. 36 new flag/score columns added.")
    fig.text(0.5, -0.02, note, ha="center", fontsize=10,
             color=C_DARK, style="italic", fontweight="bold")
    plt.tight_layout()
    save_fig(fig, "14_schema_change", pdf)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN — Run all charts and build PDF
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print(f"\n{'='*60}")
    print("  UC10 — Data Cleaning Visualization Report")
    print(f"{'='*60}\n")

    with PdfPages(PDF_OUT) as pdf:

        # Cover page
        fig_cover, ax_cover = plt.subplots(figsize=(18, 10))
        fig_cover.patch.set_facecolor(C_DARK)
        ax_cover.set_facecolor(C_DARK)
        ax_cover.axis("off")
        ax_cover.text(0.5, 0.72,
            "UC10 — Claims & Authorization",
            ha="center", va="center", color="white",
            fontsize=26, fontweight="bold", transform=ax_cover.transAxes)
        ax_cover.text(0.5, 0.60,
            "Data-Quality Anomaly Monitor",
            ha="center", va="center", color="white",
            fontsize=22, fontweight="bold", transform=ax_cover.transAxes)
        ax_cover.text(0.5, 0.48,
            "Data Cleaning Pipeline — Before vs After Visualization Report",
            ha="center", va="center", color="#BDC3C7",
            fontsize=16, transform=ax_cover.transAxes)
        stats_str = (f"Raw: {raw.shape[0]:,} rows × {raw.shape[1]} cols   →   "
                     f"Cleaned: {clean.shape[0]:,} rows × {clean.shape[1]} cols\n"
                     f"0 records deleted  |  36 new DQ/anomaly flag columns added  |  "
                     f"6,626 records flagged with ≥1 DQ issue")
        ax_cover.text(0.5, 0.33, stats_str,
            ha="center", va="center", color="#F0B27A",
            fontsize=13, transform=ax_cover.transAxes)
        ax_cover.text(0.5, 0.18,
            "14 Visualization Sections:\n"
            "Overview  |  Missing Values  |  Latency  |  Financials  |  Dates  "
            "|  Auth  |  SLA/Pipeline\n"
            "Provider/Bene  |  Record-Type Missingness  |  DQ Score  |  "
            "All Flags  |  Cleaning Table  |  Correlation  |  Schema",
            ha="center", va="center", color="#AED6F1",
            fontsize=11, transform=ax_cover.transAxes)
        pdf.savefig(fig_cover, bbox_inches="tight")
        plt.close(fig_cover)

        # All sections
        print("[1/14]  Overview dashboard …")
        chart_overview(pdf)

        section_title(pdf, "SECTION 1 — Missing Value Analysis",
                      "Before vs After | Identifiers never imputed | Missingness indicators created")
        print("[2/14]  Missing values …")
        chart_missing_values(pdf)

        section_title(pdf, "SECTION 2 — Processing Latency",
                      "2,785 negative-latency records PRESERVED with flag | IQR outliers flagged, not removed")
        print("[3/14]  Processing latency …")
        chart_processing_latency(pdf)

        section_title(pdf, "SECTION 3 — Financial Amounts",
                      "Negative amounts preserved | Paid>Allowed flagged | IQR outliers flagged")
        print("[4/14]  Financial amounts …")
        chart_financials(pdf)

        section_title(pdf, "SECTION 4 — Date Anomalies",
                      "4 date-order flags | 2,845+ date anomaly records preserved, not deleted")
        print("[5/14]  Date anomalies …")
        chart_date_anomalies(pdf)

        section_title(pdf, "SECTION 5 — Authorization Anomalies",
                      "3,492 Auth-required but missing link records PRESERVED with flag")
        print("[6/14]  Auth anomalies …")
        chart_auth_anomalies(pdf)

        section_title(pdf, "SECTION 6 — SLA & Pipeline Anomalies",
                      "Existing SLA features preserved | Pipeline gap supplement added")
        print("[7/14]  SLA & pipeline …")
        chart_sla_pipeline(pdf)

        section_title(pdf, "SECTION 7 — Provider & Beneficiary Anomalies",
                      "High-denial & high-frequency records flagged, not removed")
        print("[8/14]  Provider & beneficiary …")
        chart_provider_bene(pdf)

        section_title(pdf, "SECTION 8 — Record-Type-Aware Missingness",
                      "Not all NaN is wrong — context-aware handling per MEDICAL/PHARMACY/PRIOR_AUTH")
        print("[9/14]  Record-type missingness …")
        chart_record_type_missingness(pdf)

        section_title(pdf, "SECTION 9 — Data Quality Score",
                      "New composite DQ score — separate from ML anomaly score")
        print("[10/14] DQ score …")
        chart_dq_score(pdf)

        section_title(pdf, "SECTION 10 — All Anomaly Flags",
                      "36 new flag columns added — count of records flagged per rule")
        print("[11/14] Anomaly flag heatmap …")
        chart_anomaly_heatmap(pdf)

        section_title(pdf, "SECTION 11 — Cleaning Rules Table",
                      "Every rule applied, scope, before state, after state, impact")
        print("[12/14] Cleaning rules table …")
        chart_cleaning_summary_table(pdf)

        section_title(pdf, "SECTION 12 — Flag Correlation Heatmap",
                      "How DQ flags co-occur — useful for root-cause analysis")
        print("[13/14] Flag correlation …")
        chart_flag_correlation(pdf)

        section_title(pdf, "SECTION 13 — Schema Change Summary",
                      "50 → 86 columns | 0 deleted | 36 new DQ/flag columns")
        print("[14/14] Schema change …")
        chart_schema_change(pdf)

    print(f"\n{'='*60}")
    print(f"  Report saved:")
    print(f"  PDF → {PDF_OUT}")
    print(f"  PNGs→ {VIZ_DIR}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
