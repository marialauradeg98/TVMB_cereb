"""
explore_conn.py
===============

Characterize the **cerebellar connectivity** in the Knox mouse connectome
(``ConnKnox_182.h5`` by default, 182 regions) with an anatomical focus, to sanity-check
that the structural matrix used by the NEST+TVB co-simulation reproduces the
input/output pathways expected from the literature.

Two analyses
------------
A. **Afferents INTO the cerebellum** (cerebellar cortex + deep nuclei): rank
   the source regions and highlight the anatomically-known mossy- and
   climbing-fiber sources.
B. **Efferents OUT of the deep cerebellar nuclei (DCN)**: rank the target
   regions and highlight the cerebello-thalamic (dentato-thalamo-cortical)
   projection.

Anatomical background (literature)
----------------------------------
Mossy fibers (the main excitatory afferent to cerebellar cortex) arise mostly
from the **pontine nuclei** (cortico-ponto-cerebellar relay of cerebral cortex)
and, to a lesser extent, the **lateral reticular nucleus** (spino-reticulo-
cerebellar) plus vestibular / dorsal-column / trigeminal / external-cuneate
nuclei. Climbing fibers arise **solely from the inferior olive**. The dominant
DCN output is the **dentato-thalamo-cortical** pathway (dentate -> contralateral
ventral-lateral / VPLo / MD thalamus -> motor, premotor and parietal cortex);
the dentate and interposed also project to the **red nucleus**.

  - Pontocerebellar / mossy sources: ScienceDirect "Pontocerebellar Fibers",
    "Cerebellar Mossy Fiber"; Wikipedia "Mossy fiber (cerebellum)".
  - Dentato-thalamo-cortical output: J Neurophysiol 2002 "Unfolded map of the
    cerebellar dentate nucleus"; ScienceDirect "Deep Cerebellar Nuclei".

Regions present in THIS atlas that match the known pathways
-----------------------------------------------------------
Afferent SOURCES to cerebellum (present):
  - Pontine nuclei : "Pons, sensory/motor/behavioral related"  (idx 94/95/96 R, 192/193/194 L)
  - Lateral reticular nucleus                                  (idx 97 R, 195 L)
  - Inferior olivary complex  (climbing fibers)                (idx 88 R, 186 L)
  (Red nucleus, vestibular, dorsal-column, trigeminal, spinal cord, external
   cuneate are NOT separate ROIs here; red nucleus folds into "Midbrain, motor
   related" 77/175.)

DCN efferent TARGETS (present):
  - Thalamus, sensory-motor cortex related     (idx 70 R, 168 L)
  - Thalamus, polymodal association related     (idx 71 R, 169 L)
  - Red nucleus -> "Midbrain, motor related"    (idx 77 R, 175 L)

Each case is analyzed on the **TVB-ready matrix** produced by
``tvb_cereb_preprocessing`` (mirror of simulator/mouse_conn.tvb_cereb_preprocessing):
cerebellum-internal blocks zeroed (DCN->cortex, cortex<->cortex, DCN<->DCN;
cortex->DCN kept) and TRANSPOSED to rows=TARGET, with an optional log transform.

Finding: in RAW weights the thalamus is actually the #1 DCN efferent target; the
"thalamus ranks low" behaviour only appears under log_scale (the log compression
reorders sums — see the note printed by analyze_efferents).

Usage
-----
    python explore_conn.py                 # both cases (raw + log), prints + saves
    python explore_conn.py --mode raw      # only raw
    python explore_conn.py --mode log_scale
    python explore_conn.py --h5 /path/to/ConnKnox_196.h5 --outdir ./conn_report

Outputs (in <outdir>_raw/ and <outdir>_log/):
    afferents_to_cortex.png / afferents_to_dcn.png / afferents_to_cerebellum.csv
    efferents_from_dcn.csv / .png
    cortex_nodes_afferents.csv / .png
    weight_distribution.png / tract_length_distribution.png

Global overview (analysis D, in <outdir>_overview/, skip with --no-overview):
    fig_weights_raw[_hist].png / fig_weights_log[_hist].png
    fig_tract_lengths[_hist].png / fig_delays[_hist].png
    fig_network_3d.png / fig_regions_per_macroarea.png / fig_w_threshold_<pct>.png
    + top-10 longest tracts printed to stdout
"""

import os
import argparse
import numpy as np
import h5py

# Headless-safe plotting (no display needed on the cluster / venv).
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Single source of truth for region classification + matrix preprocessing +
# normalization lives in simulator.mouse_conn (imported TVB-free thanks to the
# lazy import).
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from simulator.mouse_conn import MouseConnectivity


# ─────────────────────────────────────────────────────────────────────────────
# Region groups — reuse the canonical keyword lists from mouse_conn so there is
# ONE definition shared by all scripts (matching here is case-insensitive).
# ─────────────────────────────────────────────────────────────────────────────
CEREB_CORTEX_KEYS = list(MouseConnectivity.CEREB_CORTEX_KW)
DCN_KEYS = list(MouseConnectivity.DCN_KW)

# Inferior olive: source of climbing fibers (an AFFERENT source, not a DCN).
OLIVE_KEY = "Inferior olivary complex"

# Known afferent SOURCES to the cerebellum, grouped by pathway.
AFFERENT_SOURCES = {
    "Pontine nuclei (cortico-ponto-cereb, mossy)": [
        "Pons, sensory related", "Pons, motor related",
        "Pons, behavioral state related",
    ],
    "Lateral reticular nucleus (spino-reticulo-cereb, mossy)": [
        "Lateral reticular nucleus",
    ],
    "Inferior olive (climbing fibers)": [
        "Inferior olivary complex",
    ],
}

# Known DCN efferent TARGETS of interest.
EFFERENT_TARGETS = {
    "Thalamus (dentato-thalamo-cortical)": [
        "Thalamus, sensory-motor cortex related",
        "Thalamus, polymodal association cortex related",
    ],
    "Midbrain motor (incl. red nucleus)": [
        "Midbrain, motor related",
    ],
}

# Macroarea classification (MACRO_RULES) now lives in mouse_conn — classify_macro
# below delegates to MouseConnectivity.classify_macro (single source of truth).


# ─────────────────────────────────────────────────────────────────────────────
# Loading / helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_conn(h5_path):
    """Load the RAW weights, tract lengths and labels from the HDF5 file.

    Convention as stored (verified against conn_data/knox_voxel_conn.py, which
    builds this matrix): **rows = source, columns = target**, i.e. ``W[i, j]`` is
    the weight of the connection FROM region i TO region j. The matrix is directed
    (asymmetric). The diagonal is zeroed. No transpose / zeroing / log is applied
    here — call tvb_cereb_preprocessing() for the TVB-ready matrix. Returns
    ``(W, tract_lengths, labels)``; tract_lengths is None if absent."""
    with h5py.File(h5_path, "r") as f:
        labels = np.array(f["region_labels"][()]).astype("<U128")
        W = np.array(f["weights"][()], dtype=np.float64)
        tl = np.array(f["tract_lengths"][()], dtype=np.float64) \
            if "tract_lengths" in f else None
    np.fill_diagonal(W, 0.0)
    return W, tl, labels


def tvb_cereb_preprocessing(W, labels, do_log=False, zero_internal=True, verbose=True):
    """Thin wrapper over mouse_conn.MouseConnectivity.preprocess_matrix (single
    source of truth): zero diagonal + zero cerebellum-internal blocks (keep
    cortex->DCN) + transpose to rows=TARGET + optional log_scale."""
    return MouseConnectivity.preprocess_matrix(
        W, labels, zero_internal=zero_internal, transpose=True,
        do_log=do_log, verbose=verbose)


def _log_scale(w_in):
    """Log-transform + normalize to [0, 1] — delegates to mouse_conn."""
    return MouseConnectivity.log_scale(w_in)


def idx_of(labels, key):
    """All region indices whose label contains ``key`` (case-insensitive, both
    hemispheres) — case-insensitive so the lowercase canonical keyword lists
    from mouse_conn match the capitalised Allen labels."""
    k = key.lower()
    return [i for i, lb in enumerate(labels) if k in lb.lower()]


def idx_of_any(labels, keys):
    out = []
    for k in keys:
        out.extend(idx_of(labels, k))
    return sorted(set(out))


def short_label(lb):
    """Drop the hemisphere prefix but keep an L/R tag."""
    if lb.startswith("Left "):
        return "L " + lb[5:]
    if lb.startswith("Right "):
        return "R " + lb[6:]
    return lb


def hemi(lb):
    return "L" if lb.startswith("Left") else ("R" if lb.startswith("Right") else "?")


def classify_macro(lb):
    """Delegate to the single source of truth in mouse_conn."""
    return MouseConnectivity.classify_macro(lb)


# ─────────────────────────────────────────────────────────────────────────────
# Analysis A — afferents INTO the cerebellum
# ─────────────────────────────────────────────────────────────────────────────

def analyze_afferents(W, labels, outdir):
    """
    W is the TVB-ready matrix: **rows = TARGET, cols = SOURCE** (W[t, s] = weight
    of the connection FROM source s TO target t). Afferents into target t = row
    W[t, :].
    """
    cortex = idx_of_any(labels, CEREB_CORTEX_KEYS)
    dcn = idx_of_any(labels, DCN_KEYS)

    cset, dset = set(cortex), set(dcn)
    # Afferent weight per SOURCE (all regions) into each target compartment.
    # rows=target => sum the TARGET rows over every SOURCE column.
    aff_cortex = W[cortex, :].sum(axis=0)   # per source, into cerebellar cortex
    aff_dcn = W[dcn, :].sum(axis=0)         # per source, into DCN

    def _recs(exclude):
        return [(s, labels[s], classify_macro(labels[s]),
                 float(aff_cortex[s]), float(aff_dcn[s]))
                for s in range(len(labels)) if s not in exclude]
    # Sources = all regions EXCEPT the target group itself (its self-block is
    # zeroed anyway). So the cerebellar cortex IS kept as a source for the DCN
    # plot (cortex->DCN = Purkinje->nucleus, the dominant DCN input); likewise
    # the DCN stays a potential source for the cortex plot.
    rows_cortex = sorted(_recs(cset), key=lambda r: r[3], reverse=True)  # by ->cortex
    rows_dcn = sorted(_recs(dset), key=lambda r: r[4], reverse=True)     # by ->DCN

    print("\n" + "=" * 78)
    print("A.  AFFERENTS  INTO  THE  CEREBELLUM   (who drives the cerebellum)")
    print("=" * 78)
    print(f"Targets: cerebellar cortex ({len(cortex)}) and DCN ({len(dcn)});  "
          f"sources = every region except the target group itself.")
    ext = [i for i in range(len(labels)) if i not in cset | dset]
    print(f"  cortex->DCN (Purkinje->nuclei, KEPT) total = "
          f"{W[np.ix_(dcn, cortex)].sum():.4f}  vs  all external->DCN total = "
          f"{W[np.ix_(dcn, ext)].sum():.4f}  (cortex dominates the DCN input).")

    def _tbl(title, ranked, col):
        print(f"\nTop sources afferent to {title}:")
        print(f"{'rank':>4}  {'idx':>3}  {'region':<42} {'macro':<20} {'weight':>10}")
        for r, rec in enumerate(ranked[:15], 1):
            print(f"{r:>4}  {rec[0]:>3}  {short_label(rec[1]):<42} "
                  f"{rec[2]:<20} {rec[col]:>10.4f}")

    _tbl("CEREBELLAR CORTEX", rows_cortex, 3)
    _tbl("DEEP CEREBELLAR NUCLEI (DCN)", rows_dcn, 4)

    # Known mossy/climbing sources, with their rank in BOTH target rankings.
    rc = {r[0]: i for i, r in enumerate(rows_cortex, 1)}
    rd = {r[0]: i for i, r in enumerate(rows_dcn, 1)}
    print("\nKnown anatomical afferent sources (literature):")
    for pathway, keys in AFFERENT_SOURCES.items():
        print(f"  • {pathway}")
        for src in idx_of_any(labels, keys):
            if src in rc:
                print(f"      idx {src:>3} {short_label(labels[src]):<40} "
                      f"->cortex {W[cortex, src].sum():.4f} (rank {rc[src]:>3})   "
                      f"->DCN {W[dcn, src].sum():.4f} (rank {rd[src]:>3})")

    # Contralateral dominance check for pontine input into the right Declive.
    declive_r = idx_of(labels, "Declive")
    pons = idx_of_any(labels, AFFERENT_SOURCES[
        "Pontine nuclei (cortico-ponto-cereb, mossy)"])
    if declive_r and pons:
        print("\nPontine input laterality into Declive (VI) "
              "[expected: contralateral-dominant]:")
        for t in declive_r:
            ipsi = [s for s in pons if hemi(labels[s]) == hemi(labels[t])]
            contra = [s for s in pons if hemi(labels[s]) != hemi(labels[t])]
            print(f"  {short_label(labels[t]):<16}  ipsi={W[t, ipsi].sum():.4f}"
                  f"   contra={W[t, contra].sum():.4f}")

    # CSV (both weights, sorted by ->cortex)
    csv = os.path.join(outdir, "afferents_to_cerebellum.csv")
    with open(csv, "w") as fh:
        fh.write("idx,region,macro,w_to_cortex,w_to_DCN\n")
        for src, lb, mac, wc, wd in rows_cortex:
            fh.write(f"{src},\"{lb}\",{mac},{wc:.6f},{wd:.6f}\n")
    print(f"\nSaved {csv}")

    # Two separate figures.
    _plot_afferents_into(rows_cortex[:15], 3, "cerebellar cortex", "#00A0C4",
                         os.path.join(outdir, "afferents_to_cortex.png"))
    _plot_afferents_into(rows_dcn[:15], 4, "deep cerebellar nuclei (DCN)", "#8B4513",
                         os.path.join(outdir, "afferents_to_dcn.png"))
    return rows_cortex, rows_dcn


def _plot_afferents_into(top_rows, col, target_name, color, out_path):
    labels = [short_label(r[1]) for r in top_rows]
    w = [r[col] for r in top_rows]
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(y, w, color=color)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Summed structural weight")
    ax.set_title(f"Afferents into {target_name} (top 15 sources)",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Analysis B — efferents OUT of the DCN
# ─────────────────────────────────────────────────────────────────────────────

def analyze_efferents(W, labels, outdir):
    dcn = idx_of_any(labels, DCN_KEYS)
    cereb_cortex = idx_of_any(labels, CEREB_CORTEX_KEYS)
    olive = idx_of_any(labels, [OLIVE_KEY])
    cereb_all = set(dcn + cereb_cortex + olive)

    print("\n" + "=" * 78)
    print("B.  EFFERENTS  OUT  OF  THE  DEEP  CEREBELLAR  NUCLEI")
    print("=" * 78)

    # rows=target, cols=source: efferents FROM the DCN sources = columns W[:, dcn],
    # summed over the DCN source columns -> per-target total. Report EXTERNAL targets.
    eff = W[:, dcn].sum(axis=1)                 # summed over all DCN sources
    rows = []
    for t in range(len(labels)):
        if t in cereb_all:
            continue
        rows.append((t, labels[t], classify_macro(labels[t]), eff[t]))
    rows.sort(key=lambda r: r[3], reverse=True)

    print("\nTop EXTERNAL targets of the DCN (sum over all deep nuclei):")
    print(f"{'rank':>4}  {'idx':>3}  {'region':<42} {'macro':<20} {'weight':>10}")
    for r, (t, lb, mac, w) in enumerate(rows[:15], 1):
        print(f"{r:>4}  {t:>3}  {short_label(lb):<42} {mac:<20} {w:>10.4f}")

    # Per-nucleus breakdown for the key targets, esp. thalamus.
    rank_by_t = {t: r for r, (t, *_rest) in enumerate(rows, 1)}
    print("\nKnown DCN efferent targets (literature) and their rank:")
    for pathway, keys in EFFERENT_TARGETS.items():
        print(f"  • {pathway}")
        for t in idx_of_any(labels, keys):
            if t in rank_by_t:
                print(f"      idx {t:>3} {short_label(labels[t]):<40} "
                      f"rank {rank_by_t[t]:>3}/{len(rows)}   total {eff[t]:.4f}")

    # Focused per-DCN -> thalamus matrix (the connection of interest).
    thal = idx_of_any(labels, EFFERENT_TARGETS["Thalamus (dentato-thalamo-cortical)"])
    print("\nDCN → Thalamus weights per nucleus "
          "[dentate is the thalamus-projecting nucleus; expect it strongest]:")
    print(f"{'DCN source':<34} " + " ".join(f"{short_label(labels[t]):>22}" for t in thal))
    for s in dcn:
        vals = " ".join(f"{W[t, s]:>22.5f}" for t in thal)   # rows=target: s->t = W[t,s]
        print(f"{short_label(labels[s]):<34} {vals}")

    # Where does thalamus rank vs the top target, to expose the under-weighting?
    if thal:
        best_thal_rank = min(rank_by_t[t] for t in thal if t in rank_by_t)
        print(f"\n→ Best thalamic rank among DCN efferents: "
              f"{best_thal_rank}/{len(rows)}  "
              f"(top target = {short_label(rows[0][1])}, w={rows[0][3]:.4f}).")
        print("  If thalamus ranks low this confirms the Knox mouse connectome "
              "under-weights the cerebello-thalamic tract (see co-sim notes).")

    csv = os.path.join(outdir, "efferents_from_dcn.csv")
    with open(csv, "w") as fh:
        fh.write("idx,region,macro,weight_from_DCN\n")
        for t, lb, mac, w in rows:
            fh.write(f"{t},\"{lb}\",{mac},{w:.6f}\n")
    print(f"\nSaved {csv}")

    _plot_efferents(rows[:15], thal, outdir)
    return rows


def _plot_efferents(top_rows, thal_idx, outdir):
    labels = [short_label(r[1]) for r in top_rows]
    w = [r[3] for r in top_rows]
    colors = ["#FF1A1A" if r[0] in thal_idx else "#4C72B0" for r in top_rows]
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(y, w, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Summed structural weight from DCN")
    ax.set_title("DCN efferents (top 15 targets) — thalamus in red",
                 fontweight="bold")
    fig.tight_layout()
    out = os.path.join(outdir, "efferents_from_dcn.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Analysis C — per-NODE afferents of the cerebellar cortex (single lobules)
# ─────────────────────────────────────────────────────────────────────────────

def analyze_cortex_nodes(W, labels, outdir):
    """Per individual cerebellar-cortex lobule (not aggregated), break down the
    EXTERNAL afferent drive. rows=target => afferents into node c = row W[c, :]."""
    cortex = idx_of_any(labels, CEREB_CORTEX_KEYS)
    dcn = idx_of_any(labels, DCN_KEYS)
    olive = idx_of_any(labels, [OLIVE_KEY])
    cereb_all = set(cortex + dcn + olive)
    external = [i for i in range(len(labels)) if i not in cereb_all]

    # Known afferent pathways (+ olive) as index groups, for the per-node columns.
    pons = idx_of_any(labels, AFFERENT_SOURCES[
        "Pontine nuclei (cortico-ponto-cereb, mossy)"])
    lrn = idx_of_any(labels, AFFERENT_SOURCES[
        "Lateral reticular nucleus (spino-reticulo-cereb, mossy)"])
    pathways = [("Pons", pons), ("Lat.ret.", lrn), ("Inf.olive", olive)]

    print("\n" + "=" * 78)
    print("C.  PER-NODE AFFERENTS OF THE CEREBELLAR CORTEX   (single lobules)")
    print("=" * 78)

    # Build a per-node record: total external drive, each pathway, top-3 sources.
    recs = []
    for c in cortex:
        row = W[c, :]
        tot_ext = row[external].sum()
        pw = {name: row[idx].sum() for name, idx in pathways}
        order = sorted(external, key=lambda s: row[s], reverse=True)
        top3 = [(short_label(labels[s]), row[s]) for s in order[:3] if row[s] > 0]
        recs.append((c, labels[c], tot_ext, pw, top3))
    recs.sort(key=lambda r: r[2], reverse=True)

    hdr = f"{'idx':>3}  {'lobule':<22} {'totExt':>8} " + \
        " ".join(f"{nm:>9}" for nm, _ in pathways) + "   top-3 external sources"
    print(hdr)
    for c, lb, tot, pw, top3 in recs:
        cols = " ".join(f"{pw[nm]:>9.4f}" for nm, _ in pathways)
        tops = ", ".join(f"{nm}({w:.3f})" for nm, w in top3)
        print(f"{c:>3}  {short_label(lb):<22} {tot:>8.4f} {cols}   {tops}")

    # CSV
    csv = os.path.join(outdir, "cortex_nodes_afferents.csv")
    with open(csv, "w") as fh:
        fh.write("idx,lobule,total_external," +
                 ",".join(nm for nm, _ in pathways) + ",top_source,top_weight\n")
        for c, lb, tot, pw, top3 in recs:
            ts, tw = (top3[0] if top3 else ("", 0.0))
            fh.write(f'{c},"{lb}",{tot:.6f},' +
                     ",".join(f"{pw[nm]:.6f}" for nm, _ in pathways) +
                     f',"{ts}",{tw:.6f}\n')
    print(f"\nSaved {csv}")

    _plot_cortex_nodes(recs, [nm for nm, _ in pathways], outdir)
    return recs


def _plot_cortex_nodes(recs, pw_names, outdir):
    """Heatmap: cortical lobules (rows) x known afferent pathways (cols)."""
    node_labels = [short_label(r[1]) for r in recs]
    M = np.array([[r[3][nm] for nm in pw_names] for r in recs])
    fig, ax = plt.subplots(figsize=(7, max(4, 0.42 * len(recs))))
    im = ax.imshow(M, cmap="magma", aspect="auto")
    ax.set_xticks(range(len(pw_names)))
    ax.set_xticklabels(pw_names, fontsize=10)
    ax.set_yticks(range(len(node_labels)))
    ax.set_yticklabels(node_labels, fontsize=8)
    ax.set_title("Cerebellar cortex: afferent drive per lobule",
                 fontweight="bold", fontsize=12)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center",
                    fontsize=7, color="w" if M[i, j] < M.max() * 0.6 else "k")
    fig.colorbar(im, ax=ax, shrink=0.8, label="afferent weight")
    fig.tight_layout()
    out = os.path.join(outdir, "cortex_nodes_afferents.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Distributions (weights / tract lengths)
# ─────────────────────────────────────────────────────────────────────────────

def _hist(data, title, xlabel, out_path, color):
    """Histogram of the non-zero values of `data`, with mean/median vlines."""
    nz = data[data > 0].ravel()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(nz, bins=80, color=color, edgecolor="none", alpha=0.85)
    mean, med = float(nz.mean()), float(np.median(nz))
    ax.axvline(mean, color="#C0392B", ls="--", lw=1.6, label=f"mean = {mean:.4g}")
    ax.axvline(med, color="#E67E22", ls=":", lw=1.6, label=f"median = {med:.4g}")
    ax.set_xlabel(xlabel); ax.set_ylabel("count")
    ax.set_title(title, fontweight="bold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def _summary(nz, name):
    print(f"  {name}: n_nonzero={nz.size}  min={nz.min():.4g}  max={nz.max():.4g}  "
          f"mean={nz.mean():.4g}  median={np.median(nz):.4g}  "
          f"p90={np.percentile(nz, 90):.4g}")


def plot_weight_distribution(W, outdir, tag):
    print(f"\nWeight distribution [{tag}]:")
    _summary(W[W > 0], "weights")
    _hist(W, f"Weight distribution ({tag})", "connection weight",
          os.path.join(outdir, "weight_distribution.png"), "#4C72B0")


def plot_tract_lengths(tl, outdir):
    if tl is None:
        print("\nNo tract_lengths dataset in the HDF5 file.")
        return
    t = tl.copy(); np.fill_diagonal(t, 0.0)
    print("\nTract-length distribution:")
    _summary(t[t > 0], "tract_lengths")
    _hist(t, "Tract-length distribution", "tract length (mm)",
          os.path.join(outdir, "tract_length_distribution.png"), "#55A868")


# ─────────────────────────────────────────────────────────────────────────────
# Analysis D — global overview of the whole connectome (ex scripts/conn_exploration.py)
# ─────────────────────────────────────────────────────────────────────────────

def longest_tracts(tl, labels, n_top=10):
    """Print the n_top longest (undirected) tracts. Returns [(i, j, length)]."""
    if tl is None:
        print("\nNo tract_lengths dataset in the HDF5 file.")
        return []
    upper = np.triu(tl, k=1)  # upper triangle: no diagonal, no symmetric duplicates
    flat = np.argsort(upper, axis=None)[::-1][:n_top]
    pairs = [np.unravel_index(k, upper.shape) for k in flat]
    print(f"\nTop {n_top} longest tracts:")
    for r, (a, b) in enumerate(pairs, 1):
        print(f"{r:>3}. {short_label(labels[a])} <-> {short_label(labels[b])}: "
              f"{upper[a, b]:.2f} mm")
    return [(int(a), int(b), float(upper[a, b])) for a, b in pairs]


def plot_overview(h5_path, rvm_path, json_path, outdir, threshold_pct=75,
                  speed=3.0, voxel_mm=None, rvm_voxel_mm=0.1):
    """Whole-connectome overview with simulator.mouse_conn.PlotConn, on the
    STORED matrix (rows=source, cols=target; no cerebellar preprocessing):
    raw / log-scaled / thresholded weight matrices + histograms, tract lengths,
    delays (tract_length / speed), 3D network, ROIs per macroarea and the
    longest tracts.

    ``voxel_mm`` converts centres/tract lengths from voxel units (None for the
    Knox files, already in mm); ``rvm_voxel_mm`` is the RegionVolumeMapping
    voxel size used for the brain-surface point cloud (0.1 mm for Knox)."""
    import warnings
    from voxcell import RegionMap
    from simulator.mouse_conn import PlotConn
    warnings.filterwarnings("ignore", message=".*non-interactive.*")  # plt.show() on Agg

    os.makedirs(outdir, exist_ok=True)
    print("\n" + "=" * 78)
    print("D.  GLOBAL OVERVIEW OF THE CONNECTOME")
    print("=" * 78)

    region_map = RegionMap.load_json(json_path) if json_path else None
    mc = MouseConnectivity(h5_path, speed=speed, voxel_size_mm=voxel_mm,
                           region_map=region_map)
    labels = mc.region_labels
    macros = mc.macros if mc.macros is not None else MouseConnectivity.macros_of(labels)
    mc.tract_lengths = mc.preprocess_tract_lengths(mc.tract_lengths)
    mc.centres = mc.preprocess_centres(mc.centres)
    mc.set_delays()
    print(f"  {len(labels)} regions, speed = {speed} m/s  ->  {outdir}")

    pc = PlotConn()
    out = lambda name: os.path.join(outdir, name)
    weights_raw = mc.weights.copy()
    weights_raw_norm = weights_raw / weights_raw.max()
    weights_log = MouseConnectivity.log_scale(weights_raw)

    pc.plot_matrix(weights_raw_norm, "Weights matrix (raw)", pc.CMAP_W,
                   "weight [0, 1]", vmin=0, vmax=1,
                   out_path=out("fig_weights_raw.png"), save=True)
    pc.plot_histogram(weights_raw_norm, "Weights distribution (raw, non-zero)",
                      "Raw weight", color=pc.C_W, fmt=".4f",
                      out_path=out("fig_weights_raw_hist.png"), save=True)

    pc.plot_matrix(weights_log, "Weights matrix (log-scaled)", pc.CMAP_W,
                   "weight [0, 1]", vmin=0, vmax=1,
                   out_path=out("fig_weights_log.png"), save=True)
    pc.plot_histogram(weights_log, "Weights distribution (log-scaled, non-zero)",
                      "Log-scaled weight", color=pc.C_W, fmt=".2f", loc="upper left",
                      out_path=out("fig_weights_log_hist.png"), save=True)

    pc.plot_matrix(mc.tract_lengths, "Tract lengths matrix (mm)", pc.CMAP_TL, "mm",
                   out_path=out("fig_tract_lengths.png"), save=True)
    pc.plot_histogram(mc.tract_lengths, "Tract lengths distribution (non-zero, mm)",
                      "Tract length (mm)", color=pc.C_TL, fmt=".2f",
                      out_path=out("fig_tract_lengths_hist.png"), save=True)

    pc.plot_matrix(mc.delays, "Delays matrix", "Reds", "[ms]",
                   out_path=out("fig_delays.png"), save=True)
    pc.plot_histogram(mc.delays, "Delays distribution", "[ms]", color="pink",
                      fmt=".2f", out_path=out("fig_delays_hist.png"), save=True)

    if rvm_path and os.path.exists(rvm_path):
        pc.plot_network_3d(
            weights_raw, mc.centres, macros,
            surf_pts=PlotConn.brain_surface_points(rvm_path, voxel_mm=rvm_voxel_mm),
            out_path=out("fig_network_3d.png"), save=True)
    else:
        print(f"  RegionVolumeMapping not found ({rvm_path}): skipping 3D network.")
    pc.plot_macroarea_distribution(labels, macros,
                                   out_path=out("fig_regions_per_macroarea.png"),
                                   save=True)

    mc.set_weights(weights_log)
    mc.threshold_weights(threshold_pct)
    n_edges = int((mc.weights > 0).sum())
    pc.plot_matrix(mc.weights,
                   f"Log-scaled weights — threshold percentile {threshold_pct}% "
                   f"({n_edges} edges)",
                   pc.CMAP_W, "weight [0, 1]", vmin=0, vmax=1,
                   out_path=out(f"fig_w_threshold_{threshold_pct}.png"), save=True)

    longest_tracts(mc.tract_lengths, labels)
    return mc


def run_one(W_raw, tl, labels, do_log, outdir, normalize="none"):
    """Preprocess (zero internal + transpose to rows=target + optional log) and
    run all analyses + distributions into ``outdir``. ``normalize`` applies one
    of mouse_conn's weight normalizations (e.g. 'floor_log') AFTER preprocessing."""
    os.makedirs(outdir, exist_ok=True)
    tag = "log_scale" if do_log else "raw"
    if normalize != "none":
        tag = normalize
    print("\n" + "#" * 78)
    print(f"#  CASE: {tag.upper()}   (TVB-ready matrix: internal-zeroed, rows=TARGET)")
    print("#" * 78)
    W = tvb_cereb_preprocessing(W_raw, labels, do_log=do_log, verbose=True)
    if normalize != "none":
        W = MouseConnectivity.normalize(W, method=normalize)
        print(f"  applied normalize='{normalize}'  (rho={MouseConnectivity.spectral_radius(W):.3f})")
    print(f"  W range: [{W.min():.4g}, {W.max():.4g}]  ->  {outdir}")
    analyze_afferents(W, labels, outdir)
    analyze_efferents(W, labels, outdir)
    analyze_cortex_nodes(W, labels, outdir)
    plot_weight_distribution(W, outdir, tag)   # weights depend on the case
    plot_tract_lengths(tl, outdir)             # geometry: same for both cases


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5", default=os.path.join(here, "./conn_data/Knox_conn_182/ConnKnox_182.h5"),
                    help="Path to the connectivity HDF5 file.")
    ap.add_argument("--mode", default="both", choices=["raw", "log_scale", "both"],
                    help="Which case(s) to analyze (default both). raw = true "
                         "anatomy; log_scale = what the co-sim log variant integrates.")
    ap.add_argument("--outdir", default=os.path.join(here, "conn_182_report"),
                    help="Base directory for CSV/PNG outputs (a _raw / _log "
                         "suffix is added per case).")
    ap.add_argument("--normalize", default="none",
                    choices=["none", "floor_log", "log", "spectral", "row"],
                    help="Weight normalization applied AFTER preprocessing "
                         "(floor_log = threshold noise floor + log -> ~Gaussian).")
    ap.add_argument("--rvm", default=os.path.join(here, "./conn_data/Knox_conn_182/RegionVolumeMapping_182.h5"),
                    help="RegionVolumeMapping HDF5 (brain surface for the 3D network plot).")
    ap.add_argument("--json", default=os.path.join(here, "./conn_data/1.json"),
                    help="Allen ontology JSON (voxcell RegionMap) for the macroarea classification.")
    ap.add_argument("--threshold-pct", type=float, default=75,
                    help="Percentile threshold for the thresholded log-weights matrix (overview).")
    ap.add_argument("--speed", type=float, default=3.0,
                    help="Conduction speed in m/s for the delays (overview).")
    ap.add_argument("--no-overview", action="store_true",
                    help="Skip the global-overview plots (analysis D).")
    args = ap.parse_args()

    W_raw, tl, labels = load_conn(args.h5)
    print(f"Loaded {args.h5}")
    print(f"  regions: {len(labels)}   raw W range: [{W_raw.min():.4g}, {W_raw.max():.4g}]"
          f"   (stored rows=source; preprocessing transposes to rows=target)")

    cases = {"raw": False, "log_scale": True} if args.mode == "both" \
        else {args.mode: (args.mode == "log_scale")}
    for tag, do_log in cases.items():
        suffix = args.normalize if args.normalize != "none" else ("log" if do_log else "raw")
        run_one(W_raw, tl, labels, do_log, f"{args.outdir}_{suffix}",
                normalize=args.normalize)

    if not args.no_overview:
        plot_overview(args.h5, args.rvm, args.json, f"{args.outdir}_overview",
                      threshold_pct=args.threshold_pct, speed=args.speed)

    print("\nDone.")


if __name__ == "__main__":
    main()
