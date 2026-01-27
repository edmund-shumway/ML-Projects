# -*- coding: utf-8 -*-
"""
Created on Fri Jan  9 18:16:10 2026

@author: EdmundShumway
"""

import pandas as pd

# Load the 80% SOH EOL file
df = pd.read_csv("eol_df_80SOH_20240227.csv")

# Select the 6 protocol input features and the EFC target
feature_cols = [
    "charge_constant_current_1",
    "charge_constant_current_2",
    "discharge_constant_current",
    "charge_cutoff_voltage",
    "discharge_cutoff_voltage",
    "charge_constant_voltage_time"
]

target_col = "equivalent_full_cycles"

# Create the modeling DataFrame
df_model = df[feature_cols + [target_col]].dropna()

# Optional: rename for clarity
df_model.columns = ['CC1', 'CC2', 'CC_discharge', 'V_charge', 'V_discharge', 't_CV', 'EFC']

# Preview
print(df_model)

# Save to Excel
df_model.to_excel("efc_training_data.xlsx", index=False)

print("Saved to efc_training_data.xlsx")
