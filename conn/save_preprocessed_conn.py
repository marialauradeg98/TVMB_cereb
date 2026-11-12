"""
save_preprocessed_conn.py
=========================

Build a **preprocessed cerebellar connectivity** and save it to HDF5, so a
simulation can load a ready-to-use matrix instead of re-running the
preprocessing every time.

What it does (single source of truth: simulator/mouse_conn.tvb_cereb_preprocessing):

  1. load the raw Knox connectivity (ConnKnox_196.h5);
  2. ZERO the cerebellum-internal blocks — DCN->cortex, cortex<->cortex,
     DCN<->DCN — while KEEPING cortex->DCN (Purkinje->nucleus);
  3. TRANSPOSE source->target so the matrix follows TVB's weights[target, source]
     convention (node i input = sum_j weights[i,j]*state_j);
  4. (only if --log) apply the log_scale weight transform, AFTER the zeroing
     (so the removed internal links don't pollute the log normalization).

The output HDF5 mirrors the input datasets (region_labels, weights,
tract_lengths, centres, roi_ids) and stores the preprocessing choices as root
attributes for provenance.

Usage
-----
    python save_preprocessed_conn.py                      # zeroed + transposed, RAW weights
    python save_preprocessed_conn.py --log                # + log transform
    python save_preprocessed_conn.py --no-zero            # transpose only (no zeroing)
    python save_preprocessed_conn.py --in ConnKnox_196.h5 --out my_conn.h5 --log
"""

import os
import argparse

import numpy as np
import h5py

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from simulator.mouse_conn import MouseConnectivity, tvb_cereb_preprocessing


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp",
                    default=os.path.join(here, "./conn_data/Knox_conn_216/ConnKnox_216.h5"),
                    help="Input raw connectivity HDF5 (rows=source).")
    ap.add_argument("--out", default=None,
                    help="Output HDF5 path (default: auto-named next to --in).")
    ap.add_argument("--log", action="store_true",
                    help="Apply the log_scale weight transform after zeroing.")
    ap.add_argument("--normalize", default="none",
                    choices=["none", "floor_log", "log", "spectral", "row"],
                    help="Weight normalization applied after zeroing/transpose "
                         "(floor_log = threshold noise floor + log -> ~Gaussian). "
                         "Overrides --log when not 'none'.")
    ap.add_argument("--no-zero", action="store_true",
                    help="Skip the cerebellum-internal zeroing (transpose only).")
    args = ap.parse_args()

    # 1. load raw via MouseConnectivity (zeroes diagonal, rows=source)
    mc = MouseConnectivity(path=args.inp, weight_mode="raw")

    # 2-3. zero internal blocks (unless --no-zero) + transpose source->target
    tvb_cereb_preprocessing(mc, zero_internal=not args.no_zero, verbose=True)
    W = mc.weights                                    # now rows = TARGET

    # 4. optional weight transform / normalization (AFTER zeroing)
    if args.normalize != "none":
        W = MouseConnectivity.normalize(W, method=args.normalize)
        print(f"[save_preprocessed_conn] applied normalize='{args.normalize}' -> "
              f"weights in [{W.min():.3f}, {W.max():.3f}]  "
              f"(rho={MouseConnectivity.spectral_radius(W):.3f})")
    elif args.log:
        W = MouseConnectivity.log_scale(W)
        print(f"[save_preprocessed_conn] applied log_scale -> "
              f"weights in [{W.min():.3f}, {W.max():.3f}]")

    # tract_lengths: transpose too so delays align with weights (Knox tract
    # lengths are Euclidean/symmetric, so this is a no-op, but keep it consistent).
    tl = mc.tract_lengths.T

    # roi_ids is not loaded by MouseConnectivity; copy it from the input if present.
    with h5py.File(args.inp, "r") as f:
        roi_ids = np.array(f["roi_ids"][()]) if "roi_ids" in f else None

    # output name
    if args.out is None:
        base = os.path.splitext(os.path.basename(args.inp))[0]
        norm_tag = args.normalize if args.normalize != "none" \
            else ("log" if args.log else "raw")
        suffix = ("_preproc"
                  + ("" if args.no_zero else "_zeroed")
                  + "_" + norm_tag)
        args.out = os.path.join(here, base + suffix + ".h5")

    with h5py.File(args.out, "w") as f:
        f.create_dataset("region_labels",
                         data=np.array(mc.region_labels, dtype="S128"))
        f.create_dataset("weights", data=np.asarray(W, dtype=np.float64))
        f.create_dataset("tract_lengths", data=np.asarray(tl, dtype=np.float64))
        f.create_dataset("centres", data=np.asarray(mc.centres, dtype=np.float64))
        if roi_ids is not None:
            f.create_dataset("roi_ids", data=roi_ids)
        # provenance
        f.attrs["source_file"] = os.path.abspath(args.inp)
        f.attrs["weight_convention"] = "rows=target, cols=source (TVB)"
        f.attrs["transposed"] = True
        f.attrs["zeroed_internal"] = not args.no_zero
        f.attrs["zeroed_blocks"] = ("DCN->cortex; cortex<->cortex; DCN<->DCN "
                                    "(cortex->DCN kept)") if not args.no_zero else "none"
        f.attrs["log_scale"] = bool(args.log) and args.normalize == "none"
        f.attrs["normalize"] = args.normalize

    print(f"[save_preprocessed_conn] regions={len(mc.region_labels)}  "
          f"zeroed_internal={not args.no_zero}  log={args.log}")
    print(f"[save_preprocessed_conn] saved -> {args.out}")


if __name__ == "__main__":
    main()
