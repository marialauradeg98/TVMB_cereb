import glob
import os
import re
from simulator.bold_analysis import (
    BoldTimeSeries,
    PlotBold,
    compute_fc,
    compute_fcd,
)

BASE             = os.path.dirname(os.path.abspath(__file__))
BOLD_DIR         = os.path.join(BASE, "../emp_data/Data_to_share/BOLD_ts")
ATLAS_XLSX       = os.path.join(BASE, "../emp_data/macro_atlas_n_172_rois_excel_final.xlsx")
REGION_MAP_JSON  = os.path.join(BASE, "../conn_data/1.json")
FIGS_PATH        = os.path.join(BASE, "../figs_bold")
TS_FIGS_PATH     = os.path.join(FIGS_PATH, "ts")
FC_FIGS_PATH     = os.path.join(FIGS_PATH, "fc_mat")
FCD_FIGS_PATH    = os.path.join(FIGS_PATH, "fcd_mat")
os.makedirs(TS_FIGS_PATH, exist_ok=True)
os.makedirs(FC_FIGS_PATH, exist_ok=True)
os.makedirs(FCD_FIGS_PATH, exist_ok=True)

TR_MS   = 1000.0   # repetition time of the empirical recordings (Pagani et al. 2021)
WIN_LEN = 30    # [s]
WIN_SP  = 1     # [s]

MOUSE_ID_RE = re.compile(r"ag(\w+?)_")


if __name__ == "__main__":

    plot_bold = PlotBold()

    mat_files = sorted(glob.glob(os.path.join(BOLD_DIR, "*.mat")))
    print(f"Found {len(mat_files)} empirical BOLD recordings")

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
        print(
            f"Mouse {mouse_id}: {bold.n_timepoints} timepoints, "
            f"{bold.n_regions} regions, duration {bold.duration_s:.1f} s"
        )

        fc = compute_fc(bold.ts)
        FCD, fcs, speed_ts = compute_fcd(bold.ts, win_len=WIN_LEN, win_sp=WIN_SP)

        plot_bold.plot_timeseries(
            bold,
            out_path=os.path.join(TS_FIGS_PATH, f"ts_{mouse_id}.png"),
            title=f"Mouse {mouse_id} — BOLD time series",
            save=True,
        )
        plot_bold.plot_fc(
            fc, out_path=os.path.join(FC_FIGS_PATH, f"fc_{mouse_id}.png"),
            title=f"Mouse {mouse_id} — static FC",
            save=True,
        )
        plot_bold.plot_fcd(
            FCD, out_path=os.path.join(FCD_FIGS_PATH, f"fcd_{mouse_id}.png"),
            title=f"Mouse {mouse_id} — FCD",
            save=True,
        )


