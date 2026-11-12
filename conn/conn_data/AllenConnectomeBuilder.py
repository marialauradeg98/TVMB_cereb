import pandas as pd
from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache
import numpy as np
import functions as fun
import matplotlib.pyplot as plt
import h5py
import os
from tvb.datatypes.connectivity import Connectivity
from tvb.datatypes.region_mapping import RegionVolumeMapping
from tvb.datatypes.volumes import Volume
from tvb.datatypes.structural import StructuralMRI
from voxcell import RegionMap

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Load regions from empirical parcellation
fmri_data = pd.read_excel("../emp_data/macro_atlas_n_172_rois_excel_final.xlsx")
fmri_data = fmri_data.drop(fmri_data[fmri_data["ACRO"] == "HATA"].index)    # no exps found in Allen for HATA
rois = [acro for acro in fmri_data["ACRO"]]

# Add other ROIs from file
with open('rois_to_add.txt', 'r') as f:
    rois_to_add = [line.strip().replace("_", " ") for line in f if line.strip()]

# Map in the ontology
region_map = RegionMap.load_json("./1.json")
acro_to_add = []
for name in rois_to_add:
    region = next(iter(region_map.find(name, 'name')), None)
    if region is not None:
        acro = region_map.get(int(region), 'acronym')
        if acro in rois or acro in acro_to_add:
            print(f"Skipping duplicate ROI already in the list: {name} ({acro})")
        else:
            acro_to_add.append(acro)
    else:
        print(f"Region not found, skipped: {name}")

rois.extend(acro_to_add)
print('Conn rois number : ', len(rois)*2)

# Load Mouse Connectivity Cache (Allen Experimental dataset)
mcc = MouseConnectivityCache(resolution=25, manifest_file='./allen_cache/mouse_connectivity_manifest.json')
structure_tree = mcc.get_structure_tree()

structures = pd.DataFrame(structure_tree.get_structures_by_acronym(rois))
roi_ids = structures['id']

all_experiments = mcc.get_experiments(dataframe=True, cre=False)
print(all_experiments.keys())

# Injection-fraction-weighted pooling
# For every ROI we pool ALL experiments whose injection has volume inside it (structure mask, descendant-inclusive) and
# weight each by its injection fraction in the ROI, so focal injections dominate and grazing (contaminated) ones are
# down-weighted instead of being dropped.

INJ_FRACTION_THRESH = 0.0  # keep any experiment with positive injection fraction in the ROI
experiments_dict, inj_weights = fun.injection_fraction_weights(
    mcc, roi_ids, all_experiments['id'], thresh=INJ_FRACTION_THRESH)

for roi, id in zip(rois, roi_ids):
    exps = experiments_dict[int(id)]
    fmax = max(inj_weights[int(id)].values()) if exps else 0.0
    print(f'For {roi}: n = {len(exps)} experiments (max inj_fraction = {fmax:.3f}).')

still_missing = [roi for roi, id in zip(rois, roi_ids) if len(experiments_dict[int(id)]) == 0]
print(f'ROIs still without experiments: n = {len(still_missing)} -> {still_missing}')

# Per-ROI reliability report
fun.injection_fraction_report(structure_tree, roi_ids, rois, inj_weights,
                              all_experiments, 'conn_quality_report.csv')

# control: keep ROI order and drop any ROI that ended up without experiments
experiments_dict = {int(your_key): experiments_dict[int(your_key)]
                    for your_key in roi_ids if experiments_dict[int(your_key)]}

# Create projection matrix
projmaps = fun.download_an_construct_matrix(mcc,1, experiments_dict, False)

# Processing and cleaning of the projmaps
# the function 'pms_cleaner' cleans the file projmaps in 2 steps
# 1) Set all the target sites to be the same for all the injection sites
# 2) Remove Nan
projmaps = fun.pms_cleaner(projmaps)
# the function 'areas_volume_threshold' remove  areas smaller than a threshold resolution
# Select only the areas that have a volume greater than (micron^3): default: '1000000000'}]
resolution=25.
vol_thresh=0.
projmaps=fun.areas_volume_threshold(mcc, projmaps, vol_thresh, resolution)

# Organize labels and create parcellation of the Allen Atlas according to the ROIs

# the function 'create_file_order' creates file order and keyword that will be the link between the SC order and the
# id key in the Allen database
[order, key_ord] = fun.create_file_order(projmaps, structure_tree)

# order has graph_order as keys and (target_id ,acronyms, region labels) as values
labels=np.asarray(list(order.values()))[:,1]
regions=np.asarray(list(order.values()))[:,2]

# Construct the Structural Connectivity and the tract lengths

# 'construct_structural_conn' builds the Structural Connectivity (SC) matrix
structural_conn = fun.construct_structural_conn(projmaps, order, key_ord, inj_weights)

# 'construct_centres' returns the coordinate of the centres and the name of the brain areas in the selected parcellation
[centres, names, acronyms] = fun.construct_centres(mcc, order, key_ord)

# construct_centres returns centroids in voxel indices at the cache resolution (25 um).
# Convert to millimetres so the geometry matches the Knox pipeline (knox_voxel_conn.py,
# build_centres) and the TVB delay convention: the two Connectivity objects are then
# delay-interchangeable. tract_lengths are computed from the mm centres, so they are in mm too.
RESOLUTION_UM = 25
centres = centres * (RESOLUTION_UM / 1000.0)

# 'construct_tract_lengths' returns the tract lengths between the brain areas in the selected parcellation
tract_lengths = fun.construct_tract_lengths(centres)



plt.figure(figsize=(5,5))
plt.title('Log Allen Structural Connectivity',fontsize=14)
plt.imshow(np.log10(structural_conn),interpolation='nearest',cmap='Reds')
plt.xlabel('ROIs',fontsize=14);plt.ylabel('ROIs',fontsize=14)
cbar=plt.colorbar(shrink=0.75)
cbar.set_label(label='Normalized Weights',size=14)
plt.tight_layout()
plt.show()

print(f'Structural conn dims: {structural_conn.shape}')

plt.figure(figsize=(5,5))
plt.title('Tract Lengths',fontsize=14)
plt.imshow(tract_lengths,interpolation='nearest',cmap='viridis')
plt.xlabel('ROIs',fontsize=14);plt.ylabel('ROIs',fontsize=14)
plt.colorbar(shrink=0.75)
plt.tight_layout()
plt.show()

print(f'Tract lengths conn dims: {tract_lengths.shape}')

# Connectivity
result_connectivity = Connectivity()
result_connectivity.centres = centres
result_connectivity.region_labels = np.array(names)
result_connectivity.weights = structural_conn
result_connectivity.tract_lengths = tract_lengths
result_connectivity.configure()
n_rois = len(result_connectivity.region_labels)

os.makedirs(f"My_conn_{n_rois}", exist_ok=True)
with h5py.File(f"./My_conn_{n_rois}/ConnAllen_{n_rois}.h5", "w") as f:
    f.create_dataset("region_labels", data=np.array(result_connectivity.region_labels, dtype="S128"))
    f.create_dataset("centres", data=result_connectivity.centres)
    f.create_dataset("weights", data=result_connectivity.weights)
    f.create_dataset("tract_lengths", data=result_connectivity.tract_lengths)


# download from the AllenSDK the annotation volume, the template volume
vol, annot_info = mcc.get_annotation_volume()
template, template_info = mcc.get_template_volume()

# rotate template in the TVB 3D reference:
template = fun.rotate_reference(template)

# the method associated the parent and the grandparents to the child in the selected parcellation with
# the biggest volume
[unique_parents, unique_grandparents] = fun.parents_and_grandparents_finder(mcc, order, key_ord, structure_tree)

# the method returns a volume indexed between 0 and N-1, with N=tot brain areas in the parcellation.
# -1=background and areas that are not in the parcellation
vol_parcel = fun.mouse_brain_visualizer(vol, order, key_ord, unique_parents, unique_grandparents,
                                        structure_tree, projmaps)

# Volume
result_volume = Volume()
result_volume.origin = np.array([[0.0, 0.0, 0.0]])

resolution = 25
result_volume.voxel_size = np.array([resolution, resolution, resolution], dtype=np.float64)
# result_volume.voxel_unit= micron
# Region Volume Mapping
result_rvm = RegionVolumeMapping()
result_rvm.volume = result_volume
result_rvm.array_data = vol_parcel.astype(np.int64)
result_rvm.connectivity = result_connectivity
result_rvm.title = "Volume mouse brain "
result_rvm.dimensions_labels = ["X", "Y", "Z"]
# Volume template
result_template = StructuralMRI()
result_template.array_data = template
result_template.weighting = 'T1'
result_template.volume = result_volume

# Region Volume Mapping
result_rvm = RegionVolumeMapping()
result_rvm.volume = result_volume
result_rvm.array_data =  vol_parcel.astype(np.int64)
result_rvm.connectivity = result_connectivity
result_rvm.title = "Volume mouse brain "
result_rvm.dimensions_labels = ["X", "Y", "Z"]

with h5py.File(f"./My_conn_{n_rois}/RegionVolumeMapping_{n_rois}.h5", "w") as f:
    f.create_dataset("array_data", data=result_rvm.array_data)

with h5py.File(f"./My_conn_{n_rois}/VolumeAllen_{n_rois}.h5", "w") as f:
    f.create_dataset("origin", data=result_volume.origin)
    f.create_dataset("voxel_size", data=result_volume.voxel_size)

with h5py.File(f"./My_conn_{n_rois}/TemplateAllen_{n_rois}.h5", "w") as f:
    f.create_dataset("array_data", data=result_template.array_data)