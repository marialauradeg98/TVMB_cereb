"""
Regionalize the Knox et al. 2019 voxel-scale connectivity model onto our TVB ROIs.
"""
import os
import sys
import numpy as np
import pandas as pd
import h5py

from mcmodels.core import VoxelModelCache
from mcmodels.models.voxel import RegionalizedModel
from voxcell import RegionMap

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))   # TVMB_cereb/ (conn/conn_data -> repo root)
sys.path.insert(0, _HERE)                          # functions.py lives next to this script
from functions import rotate_reference, construct_tract_lengths  # noqa: E402

MANIFEST = os.path.join(_HERE, "voxel_model_cache", "voxel_model_manifest.json")
RESOLUTION = 100  # the Knox voxel model is only available at 100 um


def build_roi_ids(cache):
    """Resolve the exact same ROI set used by AllenConnectomeBuilder.py to Allen ids,
    preserving order. Returns (roi_ids, roi_acronyms, roi_names)."""
    structure_tree = cache.get_structure_tree()

    fmri = pd.read_excel(os.path.join(_REPO, "emp_data",
                                      "macro_atlas_n_172_rois_excel_final.xlsx"))
    fmri = fmri.drop(fmri[fmri["ACRO"] == "HATA"].index)  # no Allen exps for HATA
    rois = [acro for acro in fmri["ACRO"]]

    region_map = RegionMap.load_json(os.path.join(_HERE, "1.json"))
    with open(os.path.join(_HERE, "rois_to_add_min.txt")) as f:
        rois_to_add = [line.strip().replace("_", " ") for line in f if line.strip()]
    for name in rois_to_add:
        region = next(iter(region_map.find(name, "name")), None)
        if region is None:
            print(f"Region not found, skipped: {name}")
            continue
        acro = region_map.get(int(region), "acronym")
        if acro in rois:
            print(f"Skipping duplicate ROI already in the list: {name} ({acro})")
        else:
            rois.append(acro)

    structures = pd.DataFrame(structure_tree.get_structures_by_acronym(rois))
    roi_ids = np.asarray(structures["id"], dtype=int)
    roi_names = np.asarray(structures["name"])  # full region names (Allen ontology)
    print(f"Resolved {len(roi_ids)} ROIs (x2 hemispheres = {len(roi_ids) * 2}).")

    # sanity: warn on ancestor/descendant pairs (would make the voxel->ROI key ambiguous)
    id_set = set(int(i) for i in roi_ids)
    for s in structure_tree.get_structures_by_id([int(i) for i in roi_ids]):
        anc = set(s["structure_id_path"][:-1])
        if anc & id_set:
            clash = [structure_tree.get_structures_by_id([a])[0]["acronym"]
                     for a in (anc & id_set)]
            print(f"  WARNING nested ROI: {s['acronym']} is descendant of {clash}")
    return roi_ids, np.asarray(rois), roi_names


def regionalized_metrics(cache, roi_ids):
    """Return dict metric-name -> (M_ipsi, M_contra) as DataFrames indexed/ordered by
    roi_ids. ipsi = right-source -> right-target, contra = right-source -> left-target."""
    voxel_array, source_mask, target_mask = cache.get_voxel_connectivity_array()

    source_key = source_mask.get_key(structure_ids=roi_ids)
    ipsi_key = target_mask.get_key(structure_ids=roi_ids, hemisphere_id=2)
    contra_key = target_mask.get_key(structure_ids=roi_ids, hemisphere_id=1)

    ipsi = RegionalizedModel.from_voxel_array(
        voxel_array, source_key, ipsi_key, ordering=roi_ids, dataframe=True)
    contra = RegionalizedModel.from_voxel_array(
        voxel_array, source_key, contra_key, ordering=roi_ids, dataframe=True)

    metrics = {}
    for name in RegionalizedModel.VALID_REGION_METRICS:
        Mi = getattr(ipsi, name).reindex(index=roi_ids, columns=roi_ids)
        Mc = getattr(contra, name).reindex(index=roi_ids, columns=roi_ids)
        metrics[name] = (Mi, Mc)
    return metrics


def build_centres(cache, roi_ids):
    """Centroid of each ROI's structure mask, in the TVB reference frame, expressed in
    millimetres (resolution-independent). Same logic as functions.construct_centres, but
    kept in roi_ids order so it stays aligned with the weights matrix (construct_centres
    would re-sort by graph_order). Returns centres (2N x 3): [Right R1..RN, Left R1..RN]."""
    n = len(roi_ids)
    centres = np.zeros((2 * n, 3), dtype=float)
    for row, nid in enumerate(roi_ids):
        mask, _ = cache.get_structure_mask(int(nid))
        mask = rotate_reference(mask)
        mask_r = mask[:int(mask.shape[0] / 2), :, :]  # right hemisphere
        xyz = np.where(mask_r)
        coord = [0.0, 0.0, 0.0]
        if xyz[0].shape[0] > 0:  # structure present in the annotation volume
            coord = [float(np.mean(xyz[0])), float(np.mean(xyz[1])), float(np.mean(xyz[2]))]
        centres[row, :] = coord
        # left = bilateral mirror along the first (L-R) axis
        centres[row + n, :] = [mask.shape[0] - coord[0], coord[1], coord[2]]
    return centres * (RESOLUTION / 1000.0)  # voxel indices -> mm


def build_region_volume_mapping(cache, roi_ids):
    """Per-voxel ROI index in the TVB reference frame, aligned with the weights
    matrix order: a voxel in the RIGHT hemisphere of roi_ids[i] gets index i, in
    the LEFT hemisphere index i+n, background gets -1.

    Uses the SAME mask fetch + rotate_reference + first-axis hemisphere split as
    build_centres, so the mapping stays consistent with the centres and the
    ConnKnox weights order (no AllenConnectomeBuilder + remap needed)."""
    n = len(roi_ids)
    rvm = None
    for i, nid in enumerate(roi_ids):
        mask, _ = cache.get_structure_mask(int(nid))
        mask = rotate_reference(mask) > 0.5          # reorient to TVB frame -> bool
        if rvm is None:
            rvm = np.full(mask.shape, -1, dtype=np.int64)
        half = int(mask.shape[0] / 2)
        right = np.zeros_like(mask); right[:half] = mask[:half]
        left = np.zeros_like(mask);  left[half:] = mask[half:]
        rvm[right] = i                                # right hemisphere -> i
        rvm[left] = i + n                             # left hemisphere  -> i+n
    return rvm


def assemble_tvb_matrix(M_ipsi, M_contra):
    """Assemble the full 2N x 2N TVB weights matrix from the right-source ipsi/contra
    blocks, mirroring bilaterally (same convention as construct_structural_conn)."""
    Mi = np.nan_to_num(np.asarray(M_ipsi, dtype=float))
    Mc = np.nan_to_num(np.asarray(M_contra, dtype=float))
    n = Mi.shape[0]
    W = np.zeros((2 * n, 2 * n), dtype=float)
    # right-source rows [0:n]:  right-targets = ipsi, left-targets = contra
    W[:n, :n] = Mi
    W[:n, n:] = Mc
    # left-source rows [n:2n] (bilateral mirror): left-targets = ipsi, right-targets = contra
    W[n:, n:] = Mi
    W[n:, :n] = Mc
    return W


def main():
    os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)
    cache = VoxelModelCache(manifest_file=MANIFEST)

    roi_ids, roi_acros, roi_names = build_roi_ids(cache)
    metrics = regionalized_metrics(cache, roi_ids)

    labels = np.array(["Right " + n for n in roi_names] +
                      ["Left " + n for n in roi_names])

    # geometry: region centres in MILLIMETRES (build_centres already applies
    # RESOLUTION/1000), so the Euclidean tract lengths come out in mm too (NOT
    # voxel units) — no further scaling needed.
    centres = build_centres(cache, roi_ids)                # mm
    tract_lengths = construct_tract_lengths(centres)       # mm (Euclidean of mm centres)
    print(f"Centres {centres.shape} (mm), tract_lengths {tract_lengths.shape} (mm), "
          f"range [{tract_lengths[tract_lengths>0].min():.3f}, {tract_lengths.max():.3f}] mm")

    out_dir = os.path.join(_HERE, f"Knox_conn_{len(labels)}")
    os.makedirs(out_dir, exist_ok=True)

    npz = {"region_labels": labels, "roi_ids": roi_ids,
           "centres": centres, "tract_lengths": tract_lengths}
    for name, (Mi, Mc) in metrics.items():
        W = assemble_tvb_matrix(Mi, Mc)
        empties = int(np.sum(W.sum(axis=1) == 0))
        print(f"[{name}] matrix {W.shape}, empty rows: {empties}, "
              f"max={W.max():.3e}")
        npz[name] = W
        # per-metric ipsi/contra CSV (right-source blocks) for inspection
        Mi.to_csv(os.path.join(out_dir, f"{name}_ipsi.csv"))
        Mc.to_csv(os.path.join(out_dir, f"{name}_contra.csv"))

    np.savez_compressed(os.path.join(out_dir, "knox_regionalized.npz"), **npz)

    # primary TVB matrix = normalized_connection_density, normalized to max
    W = assemble_tvb_matrix(*metrics["normalized_connection_density"])
    W = W / W.max() if W.max() > 0 else W
    with h5py.File(os.path.join(out_dir, f"ConnKnox_{len(labels)}.h5"), "w") as f:
        f.create_dataset("region_labels", data=np.array(labels, dtype="S128"))
        f.create_dataset("weights", data=W)
        f.create_dataset("centres", data=centres)
        f.create_dataset("tract_lengths", data=tract_lengths)
        f.create_dataset("roi_ids", data=roi_ids)
    print(f"Saved ConnKnox to {out_dir}")

    # --- Volume geometry in the SAME Knox ROI order (100 um, TVB frame) ----------
    # RegionVolumeMapping: per-voxel ROI index; Template: anatomical MRI; Volume:
    # origin + voxel_size. All from the same cache, so RVM is natively aligned to
    # ConnKnox (no AllenConnectomeBuilder + make_knox_volume_mapping remap needed).
    print("Building RegionVolumeMapping / Template / Volume (Knox order)...")
    rvm = build_region_volume_mapping(cache, roi_ids)
    template = rotate_reference(cache.get_template_volume()[0])
    origin = np.array([[0.0, 0.0, 0.0]])
    voxel_size = np.array([RESOLUTION, RESOLUTION, RESOLUTION], dtype=np.float64)
    print(f"  RVM {rvm.shape} values [{int(rvm.min())},{int(rvm.max())}]  "
          f"template {template.shape}  voxel_size {RESOLUTION} um")

    with h5py.File(os.path.join(out_dir, f"RegionVolumeMapping_{len(labels)}.h5"), "w") as f:
        f.create_dataset("array_data", data=rvm)
    with h5py.File(os.path.join(out_dir, f"TemplateAllen_{len(labels)}.h5"), "w") as f:
        f.create_dataset("array_data", data=np.asarray(template, dtype=np.float64))
    with h5py.File(os.path.join(out_dir, f"VolumeAllen_{len(labels)}.h5"), "w") as f:
        f.create_dataset("origin", data=origin)
        f.create_dataset("voxel_size", data=voxel_size)
    print(f"Saved RegionVolumeMapping / Template / Volume to {out_dir}")


if __name__ == "__main__":
    main()
