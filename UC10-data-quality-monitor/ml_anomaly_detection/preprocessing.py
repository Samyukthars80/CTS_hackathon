from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


class ClaimsPreprocessor(BaseEstimator, TransformerMixin):
    """
    Robust preprocessing pipeline for healthcare claim records.
    Fits imputers, encoders, and scalers on training data and applies them to test/inference data.
    """
    def __init__(self, numerical_features: list[str], categorical_features: list[str], ratio_features: list[str]):
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.ratio_features = ratio_features
        
        # Fitted states
        self.medians_ = {}
        self.missing_indicator_cols_ = []
        self.constant_cols_to_drop_ = []
        self.encoder_ = None
        self.scaler_ = StandardScaler()
        self.final_feature_names_ = []
        
    def _compute_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create derived financial and operational ratios safely."""
        df_out = df.copy()
        
        # Safe division helper
        def safe_div(num_col, den_col):
            # Return NaN where denominator is zero, negative, or NaN
            mask = (df[den_col] > 0) & df[den_col].notna() & df[num_col].notna()
            res = pd.Series(np.nan, index=df.index)
            res[mask] = df.loc[mask, num_col] / df.loc[mask, den_col]
            return res

        if "Paid_to_Allowed_Ratio" in self.ratio_features:
            df_out["Paid_to_Allowed_Ratio"] = safe_div("Paid_Amount", "Allowed_Amount")
        if "Paid_to_Billed_Ratio" in self.ratio_features:
            df_out["Paid_to_Billed_Ratio"] = safe_div("Paid_Amount", "Billed_Amount")
        if "Allowed_to_Billed_Ratio" in self.ratio_features:
            df_out["Allowed_to_Billed_Ratio"] = safe_div("Allowed_Amount", "Billed_Amount")
        if "Quantity_to_Days_Ratio" in self.ratio_features:
            df_out["Quantity_to_Days_Ratio"] = safe_div("Quantity_Dispensed", "Days_Supply")
            
        return df_out

    def fit(self, X: pd.DataFrame, y=None) -> ClaimsPreprocessor:
        # 1. Compute Ratios
        X_ratios = self._compute_ratios(X)
        
        # Combine base numericals and generated ratio features
        all_numeric = self.numerical_features + self.ratio_features
        
        # 2. Find missing values and record training medians
        for col in all_numeric:
            non_nulls = X_ratios[col].dropna()
            # If all are null, use 0.0 as default, otherwise median
            median_val = non_nulls.median() if len(non_nulls) > 0 else 0.0
            self.medians_[col] = median_val
            
            # Determine if we should create a missingness indicator (at least 1 missing in training)
            if X_ratios[col].isna().sum() > 0:
                self.missing_indicator_cols_.append(col)
                
        # 3. Handle Categorical Columns - fit OrdinalEncoder
        cat_cols_present = [col for col in self.categorical_features if col in X.columns]
        if cat_cols_present:
            self.encoder_ = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1
            )
            # Fill NA with 'MISSING' before encoding
            X_cat_filled = X_ratios[cat_cols_present].fillna("MISSING").astype(str)
            self.encoder_.fit(X_cat_filled)
            
        # 4. Remove constant features (std == 0 in training)
        X_numeric_filled = X_ratios[all_numeric].copy()
        for col in all_numeric:
            X_numeric_filled[col] = X_numeric_filled[col].fillna(self.medians_[col])
            
        for col in all_numeric:
            if X_numeric_filled[col].std() == 0:
                self.constant_cols_to_drop_.append(col)
                
        # 5. Determine Final Features Structure
        active_numeric = [col for col in all_numeric if col not in self.constant_cols_to_drop_]
        missing_indicators = [f"{col}_is_missing" for col in self.missing_indicator_cols_ if col not in self.constant_cols_to_drop_]
        
        self.final_feature_names_ = active_numeric + missing_indicators + cat_cols_present
        
        # 6. Fit StandardScaler on active numerical features (including ratio columns)
        if active_numeric:
            self.scaler_.fit(X_numeric_filled[active_numeric])
            
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # 1. Compute Ratios
        X_ratios = self._compute_ratios(X)
        all_numeric = self.numerical_features + self.ratio_features
        
        # Create output DataFrame
        X_out = pd.DataFrame(index=X.index)
        
        # 2. Add missingness indicators
        for col in self.missing_indicator_cols_:
            if col not in self.constant_cols_to_drop_:
                X_out[f"{col}_is_missing"] = X_ratios[col].isna().astype(float)
                
        # 3. Impute and Scale active numerical columns
        X_num_imputed = X_ratios[all_numeric].copy()
        for col in all_numeric:
            X_num_imputed[col] = X_num_imputed[col].fillna(self.medians_[col])
            
        active_numeric = [col for col in all_numeric if col not in self.constant_cols_to_drop_]
        if active_numeric:
            scaled_vals = self.scaler_.transform(X_num_imputed[active_numeric])
            scaled_df = pd.DataFrame(scaled_vals, columns=active_numeric, index=X.index)
            # Merge with output
            X_out = pd.concat([X_out, scaled_df], axis=1)
            
        # 4. Encode Categorical
        cat_cols_present = [col for col in self.categorical_features if col in X.columns]
        if cat_cols_present and self.encoder_ is not None:
            X_cat_filled = X_ratios[cat_cols_present].fillna("MISSING").astype(str)
            encoded_vals = self.encoder_.transform(X_cat_filled)
            encoded_df = pd.DataFrame(encoded_vals, columns=cat_cols_present, index=X.index)
            X_out = pd.concat([X_out, encoded_df], axis=1)
            
        # Ensure column ordering aligns exactly with final_feature_names_
        # Filter final_feature_names_ to only keep those present in X_out
        cols_to_keep = [col for col in self.final_feature_names_ if col in X_out.columns]
        return X_out[cols_to_keep]
