import os
import shutil
import argparse
from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache

# Default location: a dedicated cache folder inside conn_data (next to this script),
# so the whole cache lives in the repo and can be safely wiped on each rebuild.
_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MANIFEST = os.path.join(_HERE, "allen_cache", "mouse_connectivity_manifest.json")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", default=DEFAULT_MANIFEST,
                   help="Path of the manifest.json to create/use "
                        "(default: %(default)s)")
    p.add_argument("--resolution", type=int, default=25, choices=[10, 25, 50, 100],
                   help="Allen atlas resolution in micron (default: 25)")
    p.add_argument("--no-clean", action="store_true",
                   help="Do NOT delete the previous cache before downloading "
                        "(default: the existing cache folder is wiped first).")
    p.add_argument("--volumes", action="store_true",
                   help="Also download the annotation and template volumes "
                        "(large, only needed for the volumetric parcellation).")
    return p.parse_args()


def main():
    args = parse_args()
    manifest = os.path.abspath(args.manifest)
    cache_dir = os.path.dirname(manifest)

    print(f"Manifest        : {manifest}")
    print(f"Cache folder    : {cache_dir}")
    print(f"Resolution      : {args.resolution} um")
    print(f"Clean previous  : {not args.no_clean}")
    print(f"Download volumes: {args.volumes}")
    print("-" * 60)

    # Delete the previous cache so every run starts from a fresh download.
    if not args.no_clean and os.path.isdir(cache_dir):
        print(f"Removing previous cache folder: {cache_dir}")
        shutil.rmtree(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)

    # Instantiating the cache with a non-existing manifest_file writes the manifest.json.
    mcc = MouseConnectivityCache(resolution=args.resolution, manifest_file=manifest)

    # Structure ontology (needed by AllenConnectomeBuilder.py).
    print("Fetching structure tree ...")
    mcc.get_structure_tree()

    # Experiments list. cre=None downloads/caches the full table; AllenConnectomeBuilder.py
    # then filters it with cre=False. Caching the full table keeps every later call offline.
    print("Fetching experiments list (full table) ...")
    experiments = mcc.get_experiments(dataframe=True, cre=None)

    print(f"Experiments cached: {len(experiments)} "
          f"(wild-type only, cre=False: "
          f"{len(mcc.get_experiments(dataframe=True, cre=False))})")

    if args.volumes:
        print("Fetching annotation volume ...")
        mcc.get_annotation_volume()
        print("Fetching template volume ...")
        mcc.get_template_volume()

    print("-" * 60)
    print("Done. Use this manifest in AllenConnectomeBuilder.py:")
    print(f"  MouseConnectivityCache(resolution={args.resolution}, "
          f"manifest_file='{manifest}')")


if __name__ == "__main__":
    main()
