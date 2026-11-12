import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from voxcell import RegionMap

from simulator.bold_analysis import BoldTimeSeries, compute_fc, compute_sfc, compute_sfc_network
from simulator.mouse_conn import MouseConnectivity

BASE              = os.path.dirname(os.path.abspath(__file__))
BOLD_DIR          = os.path.join(BASE, "../emp_data/Data_to_share/BOLD_ts")
ATLAS_XLSX        = os.path.join(BASE, "../emp_data/macro_atlas_n_172_rois_excel_final.xlsx")
REGION_MAP_JSON   = os.path.join(BASE, "../conn/conn_data/1.json")
CONNECTIVITY_FILE = os.path.join(BASE, "../conn/conn_data/Conn_Allen/ConnAllen.h5")
OUTPUT_PATH       = os.path.join(BASE, "../emp_data/features")
FIGS_PATH         = os.path.join(BASE, "../figs_bold")
VOXEL_MM          = 0.025

TR_MS = 1000.0  # repetition time of the empirical recordings (Pagani et al. 2021)
MOUSE_ID_RE = re.compile(r"ag(\w+?)_")

EXCLUDE_LABELS = {
    "Right Hippocampo-amygdalar transition area",
    "Left Hippocampo-amygdalar transition area",
}

THRESHOLD_PCTS = [0, 25, 50, 75, 90]  # 0 = log-scaled SC, no thresholding


if __name__ == "__main__":

    print("Loading structural connectivity…")
    region_map = RegionMap.load_json(REGION_MAP_JSON)
    mc = MouseConnectivity(
        CONNECTIVITY_FILE,
        voxel_size_mm=VOXEL_MM,
        region_map=region_map,
    )
    sc_weights = MouseConnectivity.log_scale(mc.weights)
    sc_labels = mc.region_labels
    print(f"  SC: {sc_weights.shape[0]} regions")

    print("Loading functional connectivity…")
    mat_files = sorted(glob.glob(os.path.join(BOLD_DIR, "*.mat")))
    print(f"  Found {len(mat_files)} empirical BOLD recordings")

    fc_by_mouse = {}
    fc_labels = None
    fc_macros = None
    for path in mat_files:
        fname = os.path.basename(path)
        m = MOUSE_ID_RE.search(fname)
        mouse_id = m.group(1) if m else fname

        bold = BoldTimeSeries(
            path,
            tr_ms=TR_MS,
            atlas_xlsx=ATLAS_XLSX,
            region_map_json=REGION_MAP_JSON,
        )
        fc_by_mouse[mouse_id] = compute_fc(bold.ts)
        fc_labels = fc_labels or bold.region_labels
        fc_macros = fc_macros or bold.macros
        print(f"  Mouse {mouse_id}: FC {fc_by_mouse[mouse_id].shape}")

    print("\nComparing SC and FC region sets…")
    sc_label_set = set(sc_labels.tolist())
    fc_label_set = set(fc_labels)
    only_in_sc = sorted(sc_label_set - fc_label_set)
    only_in_fc = sorted(fc_label_set - sc_label_set)
    macro_by_sc_label = dict(zip(sc_labels, mc.macros))
    macro_by_fc_label = dict(zip(fc_labels, fc_macros))
    print(f"  SC has {len(sc_label_set)} regions, FC has {len(fc_label_set)} regions")
    print(f"  Only in SC ({len(only_in_sc)}):")
    for lab in only_in_sc:
        print(f"    {lab} ({macro_by_sc_label[lab]})")
    print(f"  Only in FC ({len(only_in_fc)}):")
    for lab in only_in_fc:
        print(f"    {lab} ({macro_by_fc_label[lab]})")

    common_labels = [lab for lab in fc_labels if lab in sc_label_set]
    sc_label_to_idx = {lab: i for i, lab in enumerate(sc_labels)}
    fc_label_to_idx = {lab: i for i, lab in enumerate(fc_labels)}
    sc_idx = [sc_label_to_idx[lab] for lab in common_labels]
    fc_idx = [fc_label_to_idx[lab] for lab in common_labels]
    sc_common = sc_weights[np.ix_(sc_idx, sc_idx)]
    print(f"\n{len(common_labels)} common regions")

    print("\nComputing whole-network SFC_N per mouse, at various SC thresholds (log-scaled SC)…")
    rows = []
    for pct in THRESHOLD_PCTS:
        sc_thr = sc_common if pct == 0 else MouseConnectivity.threshold(sc_common, pct)
        n_edges = int((sc_thr > 0).sum())
        for mouse_id, fc in fc_by_mouse.items():
            fc_common = fc[np.ix_(fc_idx, fc_idx)]
            sfc_i = compute_sfc(sc_thr, fc_common)
            sfc_n = compute_sfc_network(sfc_i)
            rows.append({
                "mouse_id": mouse_id,
                "threshold_pct": pct,
                "n_sc_edges": n_edges,
                "sfc_n": sfc_n,
            })
            print(f"  Mouse {mouse_id}, threshold {pct}%: SFC_N = {sfc_n:.3f}")

    sfc_n_df = pd.DataFrame(rows)
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    out_csv = os.path.join(OUTPUT_PATH, "sfc_network.csv")
    sfc_n_df.to_csv(out_csv, index=False)
    print(f"\nSaved {out_csv} ({len(sfc_n_df)} rows)")

    pivot = sfc_n_df.pivot(index="mouse_id", columns="threshold_pct", values="sfc_n")

    fig, ax = plt.subplots(figsize=(8, 5.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    box_data = [pivot[pct].values for pct in THRESHOLD_PCTS]
    bp = ax.boxplot(box_data, positions=THRESHOLD_PCTS, widths=4, patch_artist=True,
                     manage_ticks=False, zorder=2)
    for patch in bp["boxes"]:
        patch.set_facecolor("#0057FF")
        patch.set_alpha(0.35)
    for median in bp["medians"]:
        median.set_color("#C0392B")
        median.set_linewidth(1.8)

    ax.set_xticks(THRESHOLD_PCTS)
    ax.set_xlabel("SC threshold percentile", fontsize=14, labelpad=8)
    ax.set_ylabel("SFC ", fontsize=14, labelpad=8)
    ax.set_title(
        f"Whole-network SFC vs SC threshold ({len(pivot)} mice)",
        fontsize=16, pad=12,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    os.makedirs(FIGS_PATH, exist_ok=True)
    out_fig = os.path.join(FIGS_PATH, "sfc_network_by_threshold.png")
    fig.savefig(out_fig, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out_fig}")