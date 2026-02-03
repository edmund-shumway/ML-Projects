"""Recreate Fig. 4-style protocol-model SHAP matrix (feature importance + RAE).

- Rows: mechanistic SOH metrics at EOL
- Columns: cycling parameters (protocol inputs) + RAE

Outputs:
  - fig4_protocol_matrix.png
  - fig4_protocol_matrix_values.csv
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# --- numpy compatibility patch for shap<=0.39 on numpy>=1.24 ---
if not hasattr(np, "bool"):
    np.bool = np.bool_  # type: ignore

# --- Patch for environments where `coverage` lacks some `coverage.types.*` that numba expects ---
try:
    import coverage, types as _types  # type: ignore
    if not hasattr(coverage, "types"):
        coverage.types = _types.SimpleNamespace()
    needed_attrs = [
        "Tracer",
        "TTraceData",
        "TShouldTraceFn",
        "TFileDisposition",
        "TWarnFn",
        "TTraceFn",
        "TShouldStartContextFn",
        "TShouldStopContextFn",
        "TStartContextFn",
        "TStopContextFn",
    ]
    for a in needed_attrs:
        if not hasattr(coverage.types, a):
            setattr(coverage.types, a, object)
except Exception:
    pass

import shap

CSV_PATH = r"eol_df_80SOH_20240227.csv"
OUT_FIG = r"fig4_protocol_matrix.png"
OUT_TABLE = r"fig4_protocol_matrix_values.csv"

# Protocol-model inputs (cycling parameters)
feature_cols_raw = [
    "charge_constant_current_1",
    "charge_constant_current_2",
    "charge_cutoff_voltage",
    "discharge_constant_current",
    "charge_constant_voltage_time",
    "discharge_cutoff_voltage",
]
feature_labels = ["CC1", "CC2", "Vcharge", "CCdischarge", "Vdischarge", "tCV"]

df = pd.read_csv(CSV_PATH)

# Derived resistances at ~50% SOC from HPPC resistance features
SOC_IDX_50 = 4
df["Rohm_50"] = 0.5 * (df[f"r_c_0s_4"] + df[f"r_d_0s_4"])
df["Rct_50"]  = 0.5 * ((df[f"r_c_3s_4"] - df[f"r_c_0s_4"]) +
                       (df[f"r_d_3s_4"] - df[f"r_d_0s_4"]))
df["Rp_50"]   = 0.5 * ((df[f"r_c_end_4"] - df[f"r_c_3s_4"]) +
                       (df[f"r_d_end_4"] - df[f"r_d_3s_4"]))

# NP ratio
df["NP_ratio"] = df["Q_ne_opt_interp"] / df["Q_pe_opt_interp"]

# Targets: (label_on_plot, column_in_csv)
targets = [
    ("EFC", "equivalent_full_cycles"),
    ("QRPT,1.0C", "rpt_1C_discharge_capacity"),
    ("QRPT,2.0C", "rpt_2C_discharge_capacity"),
    ("Rohm", "Rohm_50"),
    ("Rct", "Rct_50"),
    ("Rp", "Rp_50"),
    ("QPE", "Q_pe_opt_interp"),
    ("QNE", "Q_ne_opt_interp"),
    ("QLi", "Q_li_inv"),
    ("SOCPE,2.7V", "pe_soc_FC2p7V"),
    ("SOCNE,2.7V", "ne_soc_FC2p7V"),
    ("SOCPE,4.0V", "pe_soc_FC4p0V"),
    ("SOCNE,4.0V", "ne_soc_FC4p0V"),
    # Replace proxies below if you have the true columns:
    ("Knee (proxy)", "area_total_weighted"),
    ("R'' (proxy)", "IR_offset_matching"),
    ("NP Ratio", "NP_ratio"),
]

# Use the same rows for all models
all_needed = feature_cols_raw + [col for _, col in targets]
df_model = df[all_needed].dropna().copy()

X = df_model[feature_cols_raw].copy()
X.columns = feature_labels

X_train, X_test, idx_train, idx_test = train_test_split(
    X, df_model.index.values, test_size=0.2, random_state=42
)

def rae_percent(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    den = np.sum(np.abs(y_true - y_true.mean()))
    return float(100.0 * np.sum(np.abs(y_true - y_pred)) / den) if den != 0 else float("nan")

mean_abs_shap_norm_matrix = []
rae_list = []

for metric_label, target_col in targets:
    y_train = df_model.loc[idx_train, target_col].to_numpy()
    y_test = df_model.loc[idx_test, target_col].to_numpy()

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=1,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rae_list.append(rae_percent(y_test, y_pred))

    explainer = shap.TreeExplainer(model)
    shap_vals = np.asarray(explainer.shap_values(X_test))  # (n_samples, n_features)
    mean_abs = np.abs(shap_vals).mean(axis=0)

    # Row-normalize so each row sums to 1 (paper-style relative importance)
    s = float(mean_abs.sum())
    mean_abs_shap_norm_matrix.append(mean_abs / s if s > 0 else mean_abs)

mean_abs_shap_norm_matrix = np.vstack(mean_abs_shap_norm_matrix)
rae_arr = np.array(rae_list)

# Save numeric matrix
table = pd.DataFrame(mean_abs_shap_norm_matrix, columns=feature_labels, index=[t[0] for t in targets])
table["RAE"] = rae_arr
table.to_csv(OUT_TABLE)

# Plot matrix + RAE column
n_metrics, n_features = mean_abs_shap_norm_matrix.shape
fig = plt.figure(figsize=(9.5, 7.0))
ax = fig.add_subplot(111)

shap_norm = plt.Normalize(vmin=0.0, vmax=float(mean_abs_shap_norm_matrix.max()))
err_norm = plt.Normalize(vmin=0.0, vmax=100.0)

im_shap = ax.imshow(
    mean_abs_shap_norm_matrix,
    aspect="auto",
    cmap=plt.cm.Blues,
    norm=shap_norm,
    extent=(0, n_features, n_metrics, 0),
)

im_err = ax.imshow(
    rae_arr[:, None],
    aspect="auto",
    cmap=plt.cm.Greys,
    norm=err_norm,
    extent=(n_features, n_features + 1, n_metrics, 0),
)

ax.set_xticks(np.arange(n_features + 1) + 0.5)
ax.set_xticklabels(feature_labels + ["RAE"], rotation=90)
ax.set_yticks(np.arange(n_metrics) + 0.5)
ax.set_yticklabels([t[0] for t in targets])
ax.set_xlim(0, n_features + 1)
ax.set_ylim(n_metrics, 0)

# Optional group separators (performance / capacities / SOCs / trajectory)
for y in [6, 9, 13]:
    ax.hlines(y, xmin=0, xmax=n_features + 1, colors="black", linewidth=0.6, alpha=0.6)

# light cell grid
ax.set_xticks(np.arange(n_features + 2), minor=True)
ax.set_yticks(np.arange(n_metrics + 1), minor=True)
ax.grid(which="minor", linestyle="-", linewidth=0.3, alpha=0.35)
ax.tick_params(which="minor", bottom=False, left=False)

ax.set_title(f"Protocol-model aging matrix (row-normalized mean |SHAP|), n_test = 48", pad=10)

cax_err = inset_axes(ax, width="2.8%", height="46%", loc="upper left",
                     bbox_to_anchor=(1.02, 0.02, 1, 1),
                     bbox_transform=ax.transAxes, borderpad=0)
cb_err = fig.colorbar(im_err, cax=cax_err)
cb_err.set_label("Error (RAE)")
cb_err.set_ticks([0, 100])

cax_shap = inset_axes(ax, width="2.8%", height="46%", loc="upper left",
                      bbox_to_anchor=(1.12, 0.52, 1, 1),
                      bbox_transform=ax.transAxes, borderpad=0)
cb_shap = fig.colorbar(im_shap, cax=cax_shap)
cb_shap.set_label("mean |SHAP| (row-normalized)")

fig.tight_layout()
fig.savefig(OUT_FIG, dpi=200, bbox_inches="tight")
plt.show()
