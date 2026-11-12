import glob
import os
import re
from simulator.bold_analysis import BoldTimeSeries, extract_fmri_features_df
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
BOLD_DIR = os.path.join(BASE, "../emp_data/Data_to_share/BOLD_ts")
OUTPUT_PATH=os.path.join(BASE, "../emp_data/features")
ATLAS_XLSX = os.path.join(BASE, "../emp_data/macro_atlas_n_172_rois_excel_final.xlsx")
REGION_MAP_JSON = os.path.join(BASE, "../conn/conn_data/1.json")
TR_MS = 1000.0  # repetition time of the empirical recordings (Pagani et al. 2021)
MOUSE_ID_RE = re.compile(r"ag(\w+?)_")
COMPUTE = False


def compute_metrics_df(mat_files, tags_to_keep=['fmri'], save=False):
    bold_list = []
    mouse_ids = []
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
        bold_list.append(bold)
        mouse_ids.append(mouse_id)
    # compute metrics
    print("Extracting fMRI features: ")
    features_df = extract_fmri_features_df(bold_list, mouse_ids=mouse_ids, tags_to_keep=tags_to_keep)

    if save:
        os.makedirs(OUTPUT_PATH, exist_ok=True)
        out_csv = os.path.join(OUTPUT_PATH, "fmri_features.csv")
        features_df.to_csv(out_csv)
        print(f"Saved {out_csv} ({features_df.shape[0]} mice x {features_df.shape[1]} features)")

    return features_df

if __name__ == "__main__":
    if COMPUTE:
        mat_files = sorted(glob.glob(os.path.join(BOLD_DIR, "*.mat")))
        feature_df = compute_metrics_df(mat_files, save=True)
    else:
        feature_df = pd.read_csv(os.path.join(OUTPUT_PATH, "fmri_features.csv"))









