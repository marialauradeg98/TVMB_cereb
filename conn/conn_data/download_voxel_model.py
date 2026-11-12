"""
Download / bootstrap the Knox et al. 2019 voxel model cache used by knox_voxel_conn.py.

mcmodels.core.VoxelModelCache creates `voxel_model_manifest.json` on first instantiation
and lazily downloads each asset the first time it is requested:
    - the voxel connectivity model (weights + nodes arrays, several GB)
    - the reference space (annotation volume) and the structure tree
    - the per-experiment data referenced by the manifest
"""
import os

from mcmodels.core import VoxelModelCache

_HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(_HERE, "voxel_model_cache")
MANIFEST = os.path.join(CACHE_DIR, "voxel_model_manifest.json")


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Instantiating with the manifest path creates voxel_model_manifest.json if absent.
    cache = VoxelModelCache(manifest_file=MANIFEST)
    print(f"Manifest: {MANIFEST}")

    # The heavy download: voxel connectivity array + source/target masks.
    print("Fetching voxel connectivity model...")
    voxel_array, source_mask, target_mask = cache.get_voxel_connectivity_array()
    print(f"  voxel_array: {voxel_array.shape[0]} source x {voxel_array.shape[1]} target voxels")
    print(f"  source voxels: {len(source_mask.coordinates)}  "
          f"target voxels: {len(target_mask.coordinates)}")

    # Reference space assets that knox_voxel_conn.build_centres relies on.
    print("Fetching structure tree and annotation volume...")
    cache.get_structure_tree()
    annot, _ = cache.get_annotation_volume()
    print(f"  annotation volume: {annot.shape} at {cache.resolution} um")

    size = sum(os.path.getsize(os.path.join(dp, f))
               for dp, _, fs in os.walk(CACHE_DIR) for f in fs)
    print(f"\nDone. Cache at {CACHE_DIR} ({size / 1e9:.2f} GB).")


if __name__ == "__main__":
    main()