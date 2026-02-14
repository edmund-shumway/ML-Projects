\
"""
Recreate Fig. 4-style protocol-model aging matrix with:

- Restored rows (no global dropna across all targets).
- Protocol-repeat split (global), then intersect per metric.
- Toggle between:
    * paper view: raw mean(|SHAP|), globally scaled to 0..0.5 for display
    * row_norm view: row-normalized mean(|SHAP|), scaled 0..1 (diagnostic)

Outputs:
  /mnt/data/fig4_protocol_matrix_v6_paper.png
  /mnt/data/fig4_protocol_matrix_v6_row_norm.png
  /mnt/data/fig4_protocol_matrix_values_v6_raw.csv
  /mnt/data/fig4_protocol_matrix_values_v6_row_norm.csv
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from collections import defaultdict

# --- numpy compatibility for shap<=0.39 on numpy>=1.24 ---
if not hasattr(np, "bool"):
    np.bool = np.bool_  # type: ignore

# --- Patch for environments where numba expects coverage.types.Tracer etc. ---
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

# ----------------------------
# Toggles
# ----------------------------
VIEW_MODE = "paper"   # "paper" or "row_norm"
MAKE_BOTH = True      # if True, saves both figures regardless of VIEW_MODE


# ----------------------------
# Paths
# ----------------------------


CSV_PATH = "eol_df_80SOH_20240227.csv"

# If not found, try extracting the provided zip and search inside for the dataset
if CSV_PATH is None and os.path.exists(ZIP_PATH):
    os.makedirs(EXTRACT_DIR, exist_ok=True)
    try:
        import zipfile, glob
        with zipfile.ZipFile(ZIP_PATH, "r") as z:
            z.extractall(EXTRACT_DIR)
        hits = glob.glob(os.path.join(EXTRACT_DIR, "**", "eol_df_80SOH_20240227.csv"), recursive=True)
        if not hits:
            hits = glob.glob(os.path.join(EXTRACT_DIR, "**", "*.csv"), recursive=True)
        CSV_PATH = hits[0] if hits else None
    except Exception:
        CSV_PATH = None

if CSV_PATH is None:
    raise FileNotFoundError("Could not find eol_df_80SOH_20240227.csv (or any csv) in expected locations.")

OUT_FIG_PAPER = "fig4_protocol_matrix_v6_paper.png"
OUT_FIG_ROW   = "fig4_protocol_matrix_v6_row_norm.png"
OUT_RAW_CSV   = "fig4_protocol_matrix_values_v6_raw.csv"
OUT_ROW_CSV   = "fig4_protocol_matrix_values_v6_row_norm.csv"

# ----------------------------
# Protocol features: mapping + order
# ----------------------------
feature_map = {
    "charge_constant_current_1": "CC1",
    "charge_constant_current_2": "CC2",
    "charge_cutoff_voltage": "Vcharge",
    "discharge_constant_current": "CCdischarge",
    "discharge_cutoff_voltage": "Vdischarge",
    "charge_constant_voltage_time": "tCV",
}
FEATURE_ORDER = ["CC1","CC2","Vcharge","CCdischarge","Vdischarge","tCV"]
raw_feature_cols = list(feature_map.keys())

# ----------------------------
# Load + derive metrics
# ----------------------------
df = pd.read_csv(CSV_PATH)

SOC_IDX_50 = 4
needed_r_cols = [
    f"r_c_0s_{SOC_IDX_50}", f"r_d_0s_{SOC_IDX_50}",
    f"r_c_3s_{SOC_IDX_50}", f"r_d_3s_{SOC_IDX_50}",
    f"r_c_end_{SOC_IDX_50}", f"r_d_end_{SOC_IDX_50}",
]
if all(c in df.columns for c in needed_r_cols):
    df["Rohm_50"] = 0.5 * (df[f"r_c_0s_{SOC_IDX_50}"] + df[f"r_d_0s_{SOC_IDX_50}"])
    df["Rct_50"]  = 0.5 * ((df[f"r_c_3s_{SOC_IDX_50}"] - df[f"r_c_0s_{SOC_IDX_50}"]) +
                           (df[f"r_d_3s_{SOC_IDX_50}"] - df[f"r_d_0s_{SOC_IDX_50}"]))
    df["Rp_50"]   = 0.5 * ((df[f"r_c_end_{SOC_IDX_50}"] - df[f"r_c_3s_{SOC_IDX_50}"]) +
                           (df[f"r_d_end_{SOC_IDX_50}"] - df[f"r_d_3s_{SOC_IDX_50}"]))
else:
    df["Rohm_50"] = np.nan
    df["Rct_50"]  = np.nan
    df["Rp_50"]   = np.nan

if "Q_ne_opt_interp" in df.columns and "Q_pe_opt_interp" in df.columns:
    df["NP_ratio"] = df["Q_ne_opt_interp"] / df["Q_pe_opt_interp"]
else:
    df["NP_ratio"] = np.nan

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
    ("Knee (proxy)", "area_total_weighted"),
    ("R'' (proxy)", "IR_offset_matching"),
    ("NP Ratio", "NP_ratio"),
]

row_labels = [t[0] for t in targets]
feature_labels = FEATURE_ORDER

# ----------------------------
# Protocol cohort (ONLY protocol-feature completeness)
# ----------------------------
df_protocol = df[raw_feature_cols].dropna().copy()
X_all = df_protocol.rename(columns=feature_map)[FEATURE_ORDER]
protocol_index = X_all.index.values

# ----------------------------
# Protocol-repeat split (global)
# ----------------------------
X_tuple = [tuple(row) for row in X_all.to_numpy()]
groups = defaultdict(list)
for idx, t in zip(protocol_index, X_tuple):
    groups[t].append(idx)

train_idx, test_idx = [], []
rng = np.random.RandomState(42)
for t, idxs in groups.items():
    idxs = list(idxs)
    rng.shuffle(idxs)
    if len(idxs) > 2:
        train_idx.extend(idxs[:2])
        test_idx.extend(idxs[2:])
    else:
        train_idx.extend(idxs)

train_idx = np.array(sorted(train_idx))
test_idx  = np.array(sorted(test_idx))
n_train_global, n_test_global = len(train_idx), len(test_idx)

def rae_percent(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    den = np.sum(np.abs(y_true - y_true.mean()))
    return float(100.0 * np.sum(np.abs(y_true - y_pred)) / den) if den != 0 else float("nan")

# ----------------------------
# Fit per metric on available subset
# ----------------------------
n_features = X_all.shape[1]
raw_matrix = np.full((len(targets), n_features), np.nan, dtype=float)
row_matrix = np.full((len(targets), n_features), np.nan, dtype=float)
rae_arr    = np.full((len(targets),), np.nan, dtype=float)
n_train_by_row = np.zeros((len(targets),), dtype=int)
n_test_by_row  = np.zeros((len(targets),), dtype=int)

for i, (metric_label, target_col) in enumerate(targets):
    if target_col not in df.columns:
        continue

    y = df[target_col]
    valid_idx = y.index.intersection(protocol_index)
    valid_idx = valid_idx[y.loc[valid_idx].notna().values]

    train_i = np.intersect1d(train_idx, valid_idx)
    test_i  = np.intersect1d(test_idx, valid_idx)

    if len(train_i) < 5:
        continue

    n_train_by_row[i] = len(train_i)
    n_test_by_row[i]  = len(test_i)

    X_train = X_all.loc[train_i]
    y_train = y.loc[train_i].to_numpy()

    model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    if len(test_i) >= 3:
        X_test = X_all.loc[test_i]
        y_test = y.loc[test_i].to_numpy()
        y_pred = model.predict(X_test)
        rae_arr[i] = rae_percent(y_test, y_pred)

    explainer = shap.TreeExplainer(model)
    shap_vals = np.asarray(explainer.shap_values(X_train))
    mean_abs = np.abs(shap_vals).mean(axis=0)

    raw_matrix[i, :] = mean_abs
    s = float(np.nansum(mean_abs))
    row_matrix[i, :] = mean_abs / s if s > 0 else mean_abs

# Save tables
raw_df = pd.DataFrame(raw_matrix, index=row_labels, columns=feature_labels)
raw_df["RAE"] = rae_arr
raw_df["n_train"] = n_train_by_row
raw_df["n_test"] = n_test_by_row
raw_df.to_csv(OUT_RAW_CSV)

row_df = pd.DataFrame(row_matrix, index=row_labels, columns=feature_labels)
row_df["RAE"] = rae_arr
row_df["n_train"] = n_train_by_row
row_df["n_test"] = n_test_by_row
row_df.to_csv(OUT_ROW_CSV)

# ----------------------------
# Plot helper
# ----------------------------
def plot_matrix(shap_mat: np.ndarray, mode: str, out_path: str):
    n_metrics = shap_mat.shape[0]

    if mode == "paper":
        gmax = float(np.nanmax(shap_mat)) if np.isfinite(np.nanmax(shap_mat)) else 0.0
        disp = 0.5 * (shap_mat / gmax) if gmax > 0 else shap_mat.copy()
        vmin, vmax = 0.0, 0.5
        cbar_label = "mean |SHAP| value"
        title = f"Protocol-model aging matrix (mean |SHAP|; global scale 0–0.5), n_train = {n_train_global}, n_test = {n_test_global}"
    else:
        disp = shap_mat.copy()
        vmin, vmax = 0.0, 1.0
        cbar_label = "mean |SHAP| (row-normalized)"
        title = f"Protocol-model aging matrix (row-normalized mean |SHAP|), n_train = {n_train_global}, n_test = {n_test_global}"

    fig = plt.figure(figsize=(9.5, 7.0))
    ax = fig.add_subplot(111)

    shap_norm = plt.Normalize(vmin=vmin, vmax=vmax)
    err_norm = plt.Normalize(vmin=0.0, vmax=100.0)

    im_shap = ax.imshow(
        disp,
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
    ax.set_yticklabels(row_labels)
    ax.set_xlim(0, n_features + 1)
    ax.set_ylim(n_metrics, 0)

    for y in [3, 6, 9, 13]:
        ax.hlines(y, xmin=0, xmax=n_features + 1, colors="black", linewidth=0.6, alpha=0.55)

    ax.set_xticks(np.arange(n_features + 2), minor=True)
    ax.set_yticks(np.arange(n_metrics + 1), minor=True)
    ax.grid(which="minor", linestyle="-", linewidth=0.3, alpha=0.35)
    ax.tick_params(which="minor", bottom=False, left=False)

    ax.set_title(title, pad=10)

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
    cb_shap.set_label(cbar_label)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()

if MAKE_BOTH:
    plot_matrix(raw_matrix, "paper", OUT_FIG_PAPER)
    plot_matrix(row_matrix, "row_norm", OUT_FIG_ROW)
else:
    if VIEW_MODE == "paper":
        plot_matrix(raw_matrix, "paper", OUT_FIG_PAPER)
    else:
        plot_matrix(row_matrix, "row_norm", OUT_FIG_ROW)
