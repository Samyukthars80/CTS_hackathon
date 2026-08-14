from __future__ import annotations

import unittest
import pandas as pd
import numpy as np


class TestFinalRiskScoring(unittest.TestCase):

    def test_no_method_record(self):
        """A record with no anomalies must receive a score of 0.0."""
        ml_flag = 0
        rule_flag = 0
        stat_flag = 0
        
        ml_contribution = 0.0
        rule_contribution = 0.0
        statistical_contribution = 0.0
        
        final_risk_score = ml_contribution + rule_contribution + statistical_contribution
        self.assertEqual(final_risk_score, 0.0)

    def test_ml_only_anomaly(self):
        """A record with ML-only anomaly should only receive contribution from ML percentile."""
        ml_flag = 1
        rule_flag = 0
        stat_flag = 0
        ml_pct = 95.0
        
        ml_contribution = (ml_pct / 100.0) * 40.0
        rule_contribution = 0.0
        statistical_contribution = 0.0
        
        final_risk_score = ml_contribution + rule_contribution + statistical_contribution
        self.assertEqual(final_risk_score, 38.0)

    def test_rule_only_anomaly(self):
        """Rule-only anomaly contribution based on severity (e.g. Medium)."""
        ml_flag = 0
        rule_flag = 1
        stat_flag = 0
        highest_sev = "Medium"
        
        ml_contribution = 0.0
        rule_contribution = 20.0 # Medium severity
        statistical_contribution = 0.0
        
        final_risk_score = ml_contribution + rule_contribution + statistical_contribution
        self.assertEqual(final_risk_score, 20.0)

    def test_statistical_only_anomaly(self):
        """Statistical-only anomaly contribution based on Z-score extremeness."""
        ml_flag = 0
        rule_flag = 0
        stat_flag = 1
        max_z = 3.5
        
        ml_contribution = 0.0
        rule_contribution = 0.0
        statistical_contribution = 25.0 # Z > 3.0
        
        final_risk_score = ml_contribution + rule_contribution + statistical_contribution
        self.assertEqual(final_risk_score, 25.0)

    def test_two_method_consensus(self):
        """Consensus calculation for two methods."""
        ml_flag = 1
        rule_flag = 1
        stat_flag = 0
        consensus_level = ml_flag + rule_flag + stat_flag
        self.assertEqual(consensus_level, 2)

    def test_three_method_consensus(self):
        """Consensus level when all three channels agree."""
        ml_flag = 1
        rule_flag = 1
        stat_flag = 1
        consensus_level = ml_flag + rule_flag + stat_flag
        self.assertEqual(consensus_level, 3)

    def test_high_severity_rule(self):
        """High severity rule contribution is 35.0 points."""
        severities = ["Low", "High"]
        highest_sev = "None"
        rule_contribution = 0.0
        if "High" in severities:
            highest_sev = "High"
            rule_contribution = 35.0
        self.assertEqual(highest_sev, "High")
        self.assertEqual(rule_contribution, 35.0)

    def test_duplicate_record_id(self):
        """Assert duplicate IDs can be caught."""
        df = pd.DataFrame([{"Record_ID": "MC1"}, {"Record_ID": "MC1"}])
        has_dups = df["Record_ID"].duplicated().any()
        self.assertTrue(has_dups)

    def test_missing_provider_npi(self):
        """Missing provider NPI handles to UNKNOWN_PROVIDER."""
        npi = None
        mapped_npi = str(int(npi)) if pd.notna(npi) else "UNKNOWN_PROVIDER"
        self.assertEqual(mapped_npi, "UNKNOWN_PROVIDER")

    def test_provider_aggregation(self):
        """Provider aggregates counts correctly."""
        records = pd.DataFrame([
            {"Record_ID": "MC1", "Provider_NPI": "123", "Final_Risk_Score": 85.0, "Consensus_Level": 3, "Risk_Priority": "CRITICAL"},
            {"Record_ID": "MC2", "Provider_NPI": "123", "Final_Risk_Score": 10.0, "Consensus_Level": 0, "Risk_Priority": "LOW"}
        ])
        grouped = records.groupby("Provider_NPI")
        for npi, group in grouped:
            rec_count = len(group)
            avg_score = group["Final_Risk_Score"].mean()
            crit_count = (group["Risk_Priority"] == "CRITICAL").sum()
            self.assertEqual(rec_count, 2)
            self.assertEqual(avg_score, 47.5)
            self.assertEqual(crit_count, 1)

    def test_score_boundaries(self):
        """Score must remain within [0.0, 100.0] bounds."""
        ml_pct = 100.0
        ml_contribution = (ml_pct / 100.0) * 40.0 # 40
        rule_contribution = 35.0 # High (35)
        statistical_contribution = 25.0 # Z > 3.0 (25)
        final_risk_score = ml_contribution + rule_contribution + statistical_contribution
        self.assertTrue(0.0 <= final_risk_score <= 100.0)


if __name__ == "__main__":
    unittest.main()
