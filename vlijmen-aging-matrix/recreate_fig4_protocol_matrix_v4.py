
"""
Recreate Fig. 4b-style protocol-model aging matrix (row-normalized mean |SHAP| + RAE).

Key fixes vs earlier recreations:
1) "Inside-of-domain" (protocol-repeat) split to match the paper’s effective cohort sizes:
   - For each unique protocol (CC1, CC2, Vcharge, CCdischarge, Vdischarge, tCV):
       * If the protocol appears > 2 times, pick 2 cells for TRAIN and put the remaining repeats into TEST.
       * If the protocol appears <= 2 times, keep all of them in TRAIN.
   This typically yields ~160 train / ~80 test (instead of a naive 80/20 split → n_test≈48).

2) SHAP is computed on the TRAIN cohort (so the “n” in the title reflects the cohort used to
   estimate feature importance), while RAE is computed on the TEST cohort.

3) Safe, explicit feature mapping: values cannot become "agnostic" to labels when reordering.

Outputs:
  - fig4_protocol_matrix_v4.png
  - fig4_protocol_matrix_values_v4.csv
  - fig3_efc_bar_from_fig4_cohort_v4.png  (EFC row extracted directly from the matrix)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# --- numpy compatibility patch for shap<=0.39 on numpy>=1.24 ---
if not hasattr(np, "bool"):
    np.bool = bool  # type: ignore


# --- compatibility patch: numba<->coverage issues when importing SHAP ---
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

# ---------------------------------------------------------------------
# Inputs / outputs
# ---------------------------------------------------------------------
CSV_PATH = "eol_df_80SOH_20240227.csv"

OUT_FIG = "fig4_protocol_matrix_v4.png"
OUT_VALUES = "fig4_protocol_matrix_values_v4.csv"
OUT_EFC_BAR = "fig3_efc_bar_from_fig4_cohort_v4.png"

# ---------------------------------------------------------------------
# Feature definition (raw cols -> display labels), and display order
# ---------------------------------------------------------------------
feature_map = {
    "charge_constant_current_1": "CC1",
    "charge_constant_current_2": "CC2",
    "charge_cutoff_voltage": "Vcharge",
    "discharge_constant_current": "CCdischarge",
    "discharge_cutoff_voltage": "Vdischarge",
    "charge_constant_voltage_time": "tCV",
}
feature_order = ["CC1", "CC2", "Vcharge", "CCdischarge", "Vdischarge", "tCV"]

protocol_raw_cols = list(feature_map.keys())
seq_col = "seq_num"

# ---------------------------------------------------------------------
# Targets (row label, raw column)
# NOTE: Replace the proxy columns if/when you have the exact ones from the paper.
# ---------------------------------------------------------------------
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
    ("NP Ratio", "NP_ratio_at_eol"),
]

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def protocol_repeat_split(df: pd.DataFrame, protocol_cols_labeled: list[str], n_train_per_protocol: int = 2, seed: int = 44):
    """
    Implements the paper-style split:
      - If protocol repeats > n_train_per_protocol: put n_train_per_protocol into TRAIN and the rest into TEST.
      - Else: all into TRAIN.
    Returns: train_seq_nums (set), test_seq_nums (set)
    """
    rng = np.random.default_rng(seed)

    # group key is the protocol tuple
    prot_key = list(map(tuple, df[protocol_cols_labeled].to_numpy()))
    df_keyed = df[[seq_col] + protocol_cols_labeled].copy()
    df_keyed["_prot_key"] = prot_key

    train_seq = set()
    test_seq = set()

    for _, grp in df_keyed.groupby("_prot_key"):
        seqs = grp[seq_col].tolist()
        if len(seqs) > n_train_per_protocol:
            rng.shuffle(seqs)
            train_seq.update(seqs[:n_train_per_protocol])
            test_seq.update(seqs[n_train_per_protocol:])
        else:
            train_seq.update(seqs)

    # sanity: disjoint
    test_seq = test_seq.difference(train_seq)
    return train_seq, test_seq


def rae_percent(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    den = np.sum(np.abs(y_true - y_true.mean()))
    if den == 0:
        return float("nan")
    return 100.0 * np.sum(np.abs(y_true - y_pred)) / den


# ---------------------------------------------------------------------
# Load + clean (protocol cols + seq_num must exist)
# ---------------------------------------------------------------------
df_raw = pd.read_csv(CSV_PATH)

# Many of these dataframes ship with to_include flags; use if present.
if "to_include" in df_raw.columns:
    df_raw = df_raw.loc[df_raw["to_include"] == 1.0].copy()

need_cols = [seq_col] + protocol_raw_cols
missing = [c for c in need_cols if c not in df_raw.columns]
if missing:
    raise ValueError(f"Missing required columns in {CSV_PATH}: {missing}")

# Labeled protocol dataframe for splitting/modeling
df_proto = df_raw[[seq_col] + protocol_raw_cols].dropna().copy()
df_proto = df_proto.rename(columns=feature_map)
df_proto = df_proto[[seq_col] + feature_order].copy()

# Build the paper-style split once, using the protocol cohort
train_seq, test_seq = protocol_repeat_split(df_proto, feature_order, n_train_per_protocol=2, seed=44)

# ---------------------------------------------------------------------
# Loop targets: train model, compute mean |SHAP| on TRAIN, RAE on TEST
# ---------------------------------------------------------------------
mean_abs_shap_matrix = []
rae_list = []
n_train_list = []
n_test_list = []

for label, ycol in targets:
    if ycol not in df_raw.columns:
        # preserve row order, but mark as missing
        mean_abs_shap_matrix.append([np.nan] * len(feature_order))
        rae_list.append(np.nan)
        n_train_list.append(0)
        n_test_list.append(0)
        continue

    df_y = df_raw[[seq_col] + protocol_raw_cols + [ycol]].dropna().copy()
    df_y = df_y.rename(columns=feature_map)
    df_y = df_y[[seq_col] + feature_order + [ycol]].copy()

    df_train = df_y[df_y[seq_col].isin(train_seq)].copy()
    df_test = df_y[df_y[seq_col].isin(test_seq)].copy()

    # Some metrics may be missing for parts of the split → adjust counts.
    n_train_list.append(len(df_train))
    n_test_list.append(len(df_test))

    if len(df_train) < 10 or len(df_test) < 5:
        mean_abs_shap_matrix.append([np.nan] * len(feature_order))
        rae_list.append(np.nan)
        continue

    X_train = df_train[feature_order]
    y_train = df_train[ycol].to_numpy()
    X_test = df_test[feature_order]
    y_test = df_test[ycol].to_numpy()

    model = RandomForestRegressor(
        n_estimators=500,
        random_state=44,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # Error on TEST
    y_pred = model.predict(X_test)
    rae_list.append(rae_percent(y_test, y_pred))

    # SHAP on TRAIN cohort (to match paper-style "n" for feature importance)
    explainer = shap.TreeExplainer(model)
    shap_values_train = explainer.shap_values(X_train)
    mean_abs = np.mean(np.abs(shap_values_train), axis=0)
    mean_abs_shap_matrix.append(mean_abs.tolist())

mean_abs_shap_matrix = np.array(mean_abs_shap_matrix, dtype=float)
rae_arr = np.array(rae_list, dtype=float)

# Row-normalize mean |SHAP|
row_max = np.nanmax(mean_abs_shap_matrix, axis=1)
mean_abs_shap_norm_matrix = mean_abs_shap_matrix / row_max[:, None]

# ---------------------------------------------------------------------
# Save numeric matrix for debugging / auditing
# ---------------------------------------------------------------------
row_labels = [t[0] for t in targets]
values_df = pd.DataFrame(mean_abs_shap_norm_matrix, index=row_labels, columns=feature_order)
values_df["RAE"] = rae_arr
values_df["n_train"] = n_train_list
values_df["n_test"] = n_test_list
values_df.to_csv(OUT_VALUES, index=True)

# ---------------------------------------------------------------------
# Plot matrix + RAE column
# ---------------------------------------------------------------------
n_metrics, n_features = mean_abs_shap_norm_matrix.shape
fig = plt.figure(figsize=(9.8, 7.2))
ax = fig.add_subplot(111)

shap_norm = plt.Normalize(vmin=0.0, vmax=float(np.nanmax(mean_abs_shap_norm_matrix)))
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

# axis labels
ax.set_xticks(np.arange(n_features + 1) + 0.5)
ax.set_xticklabels(feature_order + ["RAE"], rotation=90)
ax.set_yticks(np.arange(n_metrics) + 0.5)
ax.set_yticklabels(row_labels)

ax.set_xlim(0, n_features + 1)
ax.set_ylim(n_metrics, 0)

# grid lines between sections (match the paper’s layout if desired)
for y in [1, 6, 13]:
    ax.axhline(y=y, color="k", lw=0.8, alpha=0.6)

# Title: report both cohorts to avoid ambiguity
n_train_eff = max(n_train_list) if len(n_train_list) else 0
n_test_eff = max(n_test_list) if len(n_test_list) else 0
ax.set_title(f"Protocol-model aging matrix (row-normalized mean |SHAP|), n_train = {n_train_eff}, n_test = {n_test_eff}")

# Colorbars
cax_err = inset_axes(ax, width="2.5%", height="100%", loc="upper left",
                     bbox_to_anchor=(1.02, 0., 1, 1),
                     bbox_transform=ax.transAxes, borderpad=0)
cb_err = fig.colorbar(im_err, cax=cax_err)
cb_err.set_label("Error (RAE)")
cb_err.set_ticks([0, 100])

cax_shap = inset_axes(ax, width="2.5%", height="100%", loc="upper left",
                      bbox_to_anchor=(1.12, 0., 1, 1),
                      bbox_transform=ax.transAxes, borderpad=0)
cb_shap = fig.colorbar(im_shap, cax=cax_shap)
cb_shap.set_label("mean |SHAP| (row-normalized)")

fig.tight_layout()
fig.savefig(OUT_FIG, dpi=200, bbox_inches="tight")
plt.show()

# ---------------------------------------------------------------------
# Single-row EFC bar extracted from the matrix (matches EFC row exactly)
# ---------------------------------------------------------------------
efc_row_i = row_labels.index("EFC") if "EFC" in row_labels else 0
efc_vals = mean_abs_shap_norm_matrix[efc_row_i, :]
efc_rae = rae_arr[efc_row_i]

fig2 = plt.figure(figsize=(8.5, 1.6))
ax2 = fig2.add_subplot(111)

ax2.imshow(efc_vals[None, :], aspect="auto", cmap=plt.cm.Blues, vmin=0.0, vmax=float(np.nanmax(mean_abs_shap_norm_matrix)))
ax2.set_yticks([])
ax2.set_xticks(np.arange(n_features) + 0.5)
ax2.set_xticklabels(feature_order, rotation=90)
ax2.set_title(f"EFC row extracted from matrix (RAE={efc_rae:.1f}%)")

fig2.tight_layout()
fig2.savefig(OUT_EFC_BAR, dpi=200, bbox_inches="tight")
plt.show()