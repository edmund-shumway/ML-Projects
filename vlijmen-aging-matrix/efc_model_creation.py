import pandas as pd
import xgboost as xgb
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv("eol_df_80SOH_20240227.csv")

# Select relevant columns
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

df_model.columns = ['CC1', 'CC2', 'CC_discharge', 'V_charge', 'V_discharge', 't_CV', 'EFC']

# Split into features and target
X = df_model.drop(columns=["EFC"])
y = df_model["EFC"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define and train XGBoost model
model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    objective="reg:squarederror",
    random_state=42
)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae:.2f}")
print(f"R² Score: {r2:.3f}")

# Plot XGBoost built-in feature importance
xgb.plot_importance(model)
plt.title("XGBoost Feature Importance")
plt.tight_layout()
plt.show()

# ----------------------------
# SHAP EXPLAINABILITY SECTION
# ----------------------------
print("\nRunning SHAP explainability analysis...")


# Create a masker using your input data
masker = shap.maskers.Independent(X_train)

# Use the model’s predict function
explainer = shap.Explainer(model.predict, masker)

# Get SHAP values on test set
shap_values = explainer(X_test)

# Plot summary
shap.summary_plot(shap_values, X_test)

# Summary plot (global feature importance with SHAP values)
shap.plots.beeswarm(shap_values)

# Optional: force plot for a single prediction
# shap.plots.force(shap_values[0])

# Optional: dependence plot for a single feature
# shap.plots.scatter(shap_values[:, "CC1"], color=shap_values)
