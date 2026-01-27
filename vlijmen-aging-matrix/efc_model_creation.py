import pandas as pd
import numpy as np
import xgboost as xgb
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import LinearSegmentedColormap

# ----------------------------
# Forest-green colormap (light gray -> forest green) so LOW values are visible
# ----------------------------
FOREST_GREEN = "#228B22"
LOW_COLOR = "#E8F5E9"  # light gray; change to "#E8F5E9" for a light green tint instead
forest_cmap = LinearSegmentedColormap.from_list("forest_gray_to_green", [LOW_COLOR, FOREST_GREEN])

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
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

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

# ----------------------------
# SHAP EXPLAINABILITY
# ----------------------------
print("\nRunning SHAP explainability analysis...")

masker = shap.maskers.Independent(X_train)
explainer = shap.Explainer(model.predict, masker)
shap_values = explainer(X_test)

# ----------------------------
# ORDERING
# ----------------------------
bee_paper_order = ["CC1", "CC2", "V_discharge", "V_charge", "CC_discharge", "t_CV"]
bee_plot_order = bee_paper_order[::-1]  # reverse only for beeswarm rendering

heat_order = ["CC1", "CC2", "V_charge", "CC_discharge", "V_discharge", "t_CV"]

# ----------------------------
# PREP DATA FOR BEESWARM
# ----------------------------
X_bee = X_test[bee_plot_order]
bee_idx = [X_test.columns.get_loc(c) for c in bee_plot_order]
shap_bee = shap_values.values[:, bee_idx]

# ----------------------------
# PREP DATA FOR HEAT ROW
# ----------------------------
heat_idx = [X_test.columns.get_loc(c) for c in heat_order]
shap_heat = shap_values.values[:, heat_idx]
mean_abs_shap_heat = np.abs(shap_heat).mean(axis=0)

# RAE (%)
y_true = y_test.to_numpy()
den = np.sum(np.abs(y_true - y_true.mean()))
rae = 100.0 * np.sum(np.abs(y_true - y_pred)) / den if den != 0 else np.nan

# ----------------------------
# FIGURE LAYOUT
# ----------------------------
fig = plt.figure(figsize=(9, 6))
gs = gridspec.GridSpec(2, 1, height_ratios=[4.2, 0.9], hspace=0.35)

# ---- Top: SHAP beeswarm (forest green theme) ----
ax_top = fig.add_subplot(gs[0])
plt.sca(ax_top)

# NOTE: If your SHAP version errors on cmap=..., delete that single argument.
shap.summary_plot(
    shap_bee,
    X_bee,
    show=False,
    plot_size=None,
    cmap=forest_cmap,
    color_bar_label="Feature Value"
)

ax_top.set_xlabel("SHAP Value (Impact on EFC)")

# n annotation
n = X_bee.shape[0]
ax_top.text(0.5, 1.02, f"n = {n}", transform=ax_top.transAxes,
            ha="center", va="bottom", fontsize=10)

# Labels for arrows
ax_top.annotate("Increasing\nnegative impact",
                xy=(0.03, 1.08), xycoords="axes fraction",
                ha="left", va="bottom", fontsize=9)
ax_top.annotate("Increasing\npositive impact",
                xy=(0.97, 1.08), xycoords="axes fraction",
                ha="right", va="bottom", fontsize=9)

# OUTWARD arrows (BLACK)
ax_top.annotate("", xy=(0.05, 1.06), xytext=(0.48, 1.06),
                xycoords="axes fraction",
                arrowprops=dict(arrowstyle="->", lw=1, color="black"))
ax_top.annotate("", xy=(0.95, 1.06), xytext=(0.52, 1.06),
                xycoords="axes fraction",
                arrowprops=dict(arrowstyle="->", lw=1, color="black"))

# ---- Bottom: mean |SHAP| heat row (forest green), RAE cell gray ----
ax_bot = fig.add_subplot(gs[1])
ax_bot.set_yticks([])
for spine in ax_bot.spines.values():
    spine.set_visible(False)

# Dynamic scaling across the features
shap_norm = plt.Normalize(vmin=0, vmax=float(mean_abs_shap_heat.max()))
err_norm = plt.Normalize(vmin=0, vmax=100)

# Feature cells (forest green)
im_shap = ax_bot.imshow(
    mean_abs_shap_heat[np.newaxis, :],
    aspect="auto",
    cmap=forest_cmap,
    norm=shap_norm,
    extent=(0, len(heat_order), 0, 1)
)

# RAE cell (gray)
im_err = ax_bot.imshow(
    np.array([[rae]]),
    aspect="auto",
    cmap=plt.cm.Greys,
    norm=err_norm,
    extent=(len(heat_order), len(heat_order) + 1, 0, 1)
)

ax_bot.set_xticks(np.arange(len(heat_order) + 1) + 0.5)
ax_bot.set_xticklabels(heat_order + ["RAE"], rotation=90)
ax_bot.set_xlim(0, len(heat_order) + 1)
ax_bot.set_ylim(0, 1)

ax_bot.text(-0.2, 0.5, "EFC", va="center", ha="right", fontsize=10)
ax_bot.set_title("f", pad=6)

# Colorbars: RAE gray, SHAP forest green
cax_err = inset_axes(ax_bot, width="2.5%", height="100%", loc="upper left",
                     bbox_to_anchor=(1.02, 0., 1, 1),
                     bbox_transform=ax_bot.transAxes, borderpad=0)
cb_err = fig.colorbar(im_err, cax=cax_err)
cb_err.set_label("Error (RAE)")
cb_err.set_ticks([0, 100])

cax_shap = inset_axes(ax_bot, width="2.5%", height="100%", loc="upper left",
                      bbox_to_anchor=(1.12, 0., 1, 1),
                      bbox_transform=ax_bot.transAxes, borderpad=0)
cb_shap = fig.colorbar(im_shap, cax=cax_shap)
cb_shap.set_label("|SHAP mean value|")

plt.show()
