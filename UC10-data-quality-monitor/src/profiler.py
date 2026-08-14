import pandas as pd
import json
import os
import numpy as np

def generate_profile(df, output_path="outputs/data_profile.json"):
    profile = {}
    
    # total records and columns
    profile["total_records"] = len(df)
    profile["total_columns"] = len(df.columns)
    
    # all column names and data types
    profile["columns"] = {}
    
    # exact duplicate rows
    profile["exact_duplicate_rows"] = int(df.duplicated().sum())
    
    # duplicate Record_ID values
    if "Record_ID" in df.columns:
        profile["duplicate_record_ids"] = int(df["Record_ID"].duplicated().sum())
    else:
        profile["duplicate_record_ids"] = 0
        
    date_columns = ["Service_Date", "Service_End_Date", "Processed_Date", "Decision_Date", "Submission_Date"]
    
    for col in df.columns:
        col_data = df[col]
        col_profile = {}
        
        col_profile["data_type"] = str(col_data.dtype)
        
        # missing count and missing percentage by column
        missing_count = int(col_data.isnull().sum())
        col_profile["missing_count"] = missing_count
        col_profile["missing_percentage"] = float(missing_count / len(df) * 100) if len(df) > 0 else 0.0
        
        # unique count by column
        col_profile["unique_count"] = int(col_data.nunique(dropna=True))
        
        if pd.api.types.is_numeric_dtype(col_data):
            col_profile["min"] = float(col_data.min()) if pd.notnull(col_data.min()) else None
            col_profile["max"] = float(col_data.max()) if pd.notnull(col_data.max()) else None
            col_profile["mean"] = float(col_data.mean()) if pd.notnull(col_data.mean()) else None
            
        elif pd.api.types.is_object_dtype(col_data) or pd.api.types.is_categorical_dtype(col_data):
            # categorical value counts (top 10 for brevity)
            col_profile["value_counts"] = col_data.value_counts(dropna=True).head(10).to_dict()
            
        if col in date_columns:
            # Check date parsing failures (values that are NaT after pd.to_datetime with coerce but were not initially null)
            # This requires knowing the original string format, but since we assume data is already loaded with coerce,
            # we can only do our best. A better way is checking if there were values originally but are now NaT.
            # Assuming df[col] is datetime already if it was parsed.
            pass
            
        profile["columns"][col] = col_profile

    # We will compute date parsing failures separately in the load function or engine if needed, 
    # but let's add a placeholder or evaluate strings.
    # Actually, the requirement says "date parsing failures". If the dataframe is already parsed with `errors='coerce'`, 
    # we'd need the original strings to know failures. We'll handle this during data ingestion.
    profile["date_parsing_failures"] = {}

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(profile, f, indent=4)
        
    return profile
