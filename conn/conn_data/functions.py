from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache
import numpy as np
import pandas as pd
import copy

def mcc(resolution=100):
    """
    The function imports the mouse connectivity cache
    """
    return MouseConnectivityCache(resolution=resolution)


def dictionary_builder(mcc, structure_tree, ROIs, transgenic_line):
    """
    The function creates a dictionary with information about which experiments need to be downloaded
    """
    #define map
    ia_map = structure_tree.get_id_acronym_map()
    # open up a list of all of the experiments
    all_experiments = mcc.get_experiments(dataframe=True, cre=transgenic_line)
    # build dict of injection structure id to experiment list
    ist2e = {}
    for eid in all_experiments.index:
        sab = all_experiments.loc[eid]['structure_abbrev']
        if sab in ROIs:
            isti = ia_map[sab]
            if isti not in ist2e:
                ist2e[isti] = []
            ist2e[isti].append(eid)
        else:
            ist = all_experiments.loc[eid]['injection_structures']
            for isti in ist:
                if isti not in ist2e:
                    ist2e[isti] = []
                ist2e[isti].append(eid)
    return ist2e


def download_an_construct_matrix(mcc, weighting, ist2e, transgenic_line):
    """
    The function downloads experiments necessary to build the connectivity and returns the projection maps
    """
    projmaps = {}
    if weighting == 3:  # download projection energy
        for isti, elist in ist2e.items():
            projmaps[isti] = mcc.get_projection_matrix(
                experiment_ids=elist,
                projection_structure_ids=list(ist2e),  # summary_structure_ids,
                parameter='projection_energy')
            print('injection site id', isti, ' has ', len(elist), ' experiments with pm shape ',
                  projmaps[isti]['matrix'].shape)
    else:  # download projection density:
        eli=[]
        for isti, elist in ist2e.items():
            projmaps[isti] = mcc.get_projection_matrix(
                experiment_ids=elist,
                projection_structure_ids=list(ist2e),  # summary_structure_ids,
                parameter='projection_density')
            eli=list(set(eli + elist))
            print('injection site id', isti, ' has ', len(elist), ' experiments with pm shape ',
                  projmaps[isti]['matrix'].shape)
        if weighting == 1:  # download injection density
            injdensity = {}
            #all_experiments = mcc.get_experiments(dataframe=True, cre=transgenic_line)
            for exp_id in eli:#all_experiments['id']:
                inj_d = mcc.get_injection_density(exp_id, file_name=None)
                # all the experiments have only an injection sites (only 3 coordinates),
                # thus it is possible to sum the injection matrix
                injdensity[exp_id] = (np.sum(inj_d[0]) / np.count_nonzero(inj_d[0]))
                print('Experiment id', exp_id, ', the total injection density is ', injdensity[exp_id])
            # in this case projmaps will contain PD/ID
            for inj_id in range(len(list(projmaps.values()))):
                index = 0
                for exp_id in list(projmaps.values())[inj_id]['rows']:
                    list(projmaps.values())[inj_id]['matrix'][index] = list(projmaps.values())[inj_id]['matrix'][index] / \
                                                                       injdensity[exp_id]
                    index += 1
    return projmaps


def injection_fraction_weights(mcc, roi_ids, experiment_ids, thresh=0.0):
    """
    For each ROI, pool every experiment whose injection has volume inside the ROI
    (structure mask, descendant-inclusive) and weight it by the fraction of its total
    injection volume that fell within the ROI:
        inj_fraction(exp, ROI) = injection_volume_in_ROI / total_injection_volume(exp)
    Returns (experiments_dict, weights):
        experiments_dict[roi_id] = [experiment ids with inj_fraction > thresh]
        weights[roi_id][exp_id]  = inj_fraction(exp, roi)
    """
    exp_ids = [int(e) for e in experiment_ids]
    # injection structure-unionizes (descendant-inclusive), fetched/cached in chunks
    frames = []
    for i in range(0, len(exp_ids), 100):
        frames.append(mcc.get_structure_unionizes(exp_ids[i:i + 100], is_injection=True))
    U = pd.concat(frames, ignore_index=True)
    U = U[U['hemisphere_id'] == 3]  # both hemispheres -> total injection per structure
    inj_vol = {(int(r.experiment_id), int(r.structure_id)): float(r.projection_volume)
               for r in U.itertuples()}
    # total injection volume of an experiment = its root structure value (i.e. the maximum)
    total = U.groupby('experiment_id')['projection_volume'].max().to_dict()

    experiments_dict = {int(rid): [] for rid in roi_ids}
    weights = {int(rid): {} for rid in roi_ids}
    for rid in [int(r) for r in roi_ids]:
        for e in exp_ids:
            tot = total.get(e, 0.0)
            if tot <= 0:
                continue
            frac = inj_vol.get((e, rid), 0.0) / tot
            if frac > thresh:
                experiments_dict[rid].append(e)
                weights[rid][e] = frac
    return experiments_dict, weights


def injection_fraction_report(structure_tree, roi_ids, roi_acronyms, weights,
                              all_experiments, out_csv):
    """
    Per-ROI reliability report for the injection-fraction pooling, written to `out_csv`.
    Unlike the TVB/Rabuffo and rising-net Allen Connectome Builders (which emit no
    reliability information), for every ROI we record how many of its contributing
    experiments are:
        n_primary       - injection centred exactly on the ROI (structure_id == roi_id),
        n_primary_desc  - injection centred on a descendant of the ROI (aggregation
                          source for coarse regions),
        n_overlap       - ROI only grazed as a secondary injection structure,
    plus the injection-fraction and dilution statistics. `inj_fraction_max` is the single
    most informative number: ~1 = the ROI has a dedicated injection (reliable), <<1 = the
    ROI is only ever grazed by injections centred elsewhere (low confidence). Returns the
    DataFrame.
    """
    exp_df = all_experiments.set_index('id')
    rows = []
    for rid, acro in zip([int(r) for r in roi_ids], roi_acronyms):
        w = weights.get(rid, {})
        exps = list(w.keys())
        desc = set(structure_tree.descendant_ids([rid])[0])  # includes rid itself
        n_primary = n_primary_desc = n_overlap = 0
        dil = []
        for e in exps:
            prim = int(exp_df.loc[e, 'structure_id'])
            if prim == rid:
                n_primary += 1
            elif prim in desc:
                n_primary_desc += 1
            else:
                n_overlap += 1
            inj_s = exp_df.loc[e, 'injection_structures']
            dil.append(len(inj_s) if hasattr(inj_s, '__len__') else 1)
        fracs = list(w.values())
        rows.append({
            'roi': acro, 'roi_id': rid,
            'n_primary': n_primary, 'n_primary_desc': n_primary_desc,
            'n_overlap': n_overlap, 'n_effective': len(exps),
            'inj_fraction_max': max(fracs) if fracs else 0.0,
            'inj_fraction_mean': float(np.mean(fracs)) if fracs else 0.0,
            'dilution_min': min(dil) if dil else 0,
            'dilution_mean': float(np.mean(dil)) if dil else 0.0,
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f'Wrote per-ROI quality report ({len(df)} ROIs) to {out_csv}')
    return df


# the function 'pms_cleaner'
def pms_cleaner(input_projmaps):
    """
    the method cleans the file projmaps in 2 steps:
    1) Set all the target sites to be the same for all the injection sites
    2) Remove Nan
    """
    projmaps_clean1={}
    # 1) Set all the target sites to be the same for all the injection sites
    for inj_id in input_projmaps:
        c=input_projmaps[inj_id]['columns']
        m=input_projmaps[inj_id]['matrix']
        excl=[]
        for ic in range(len(c)):
            if c[ic]['structure_id'] not in input_projmaps.keys():
                excl.append(ic)
        for icc in excl[::-1]:
            c.pop(icc)
        m=np.delete(m,excl,axis=1)
        projmaps_clean1[inj_id]={'columns':c,'matrix':m,'rows':input_projmaps[inj_id]['rows']}
    print('In total %d injection sites have been removed after imposing that targets and injection sites are the same. New number of keys in projmaps is N=%d'%(len(input_projmaps.keys())-len(projmaps_clean1.keys()),len(projmaps_clean1.keys())))

    projmaps={}
    # 2) Remove Nan
    for inj_id in projmaps_clean1:
        c=projmaps_clean1[inj_id]['columns']
        m=projmaps_clean1[inj_id]['matrix']
        excl=[]
        for ic in range(len(c)):
            if 1 in np.isnan(m[:,ic]):
                excl.append(ic)
        for icc in excl[::-1]:
            c.pop(icc)
        m=np.delete(m,excl,axis=1)
        projmaps[inj_id]={'columns':c,'matrix':m,'rows':projmaps_clean1[inj_id]['rows']}
    print('In total %d injection sites have been removed after imposing that there is no Nan. New number of keys in projmaps is N=%d'%(len(projmaps_clean1.keys())-len(projmaps.keys()),len(projmaps.keys())))
    return projmaps

def areas_volume_threshold(mcc, projmaps_old, vol_thresh, resolution):
    """
    the method includes in the parcellation only brain regions whose volume is greater than vol_thresh
    """
    threshold = vol_thresh / (resolution ** 3)
    id_all=list(projmaps_old.keys())
    id_ok = []

    projmaps_rmvol1=copy.deepcopy(projmaps_old)
    for ID in projmaps_old:
        mask, _ = mcc.get_structure_mask(ID)
        tot_voxels = int((np.count_nonzero(mask)) / 2)  # mask contains both left and right hemisphere
        if tot_voxels > threshold:
            id_ok.append(ID)
    print('Regions removed because too small')
    print(list(set(id_all)-set(id_ok)))

    print('Big regions left')
    print(id_ok)

    # Remove keys of all areas that are not in id_ok from the injection list
    for checkid in id_all:
        if checkid not in id_ok:
            projmaps_rmvol1.pop(checkid, None)

    # Remove areas that are not in id_ok from target list (columns+matrix)
    projmaps_rmvol={}
    for inj_id in projmaps_rmvol1:
        c=projmaps_rmvol1[inj_id]['columns']
        m=projmaps_rmvol1[inj_id]['matrix']
        excl=[]
        for ic in range(len(c)):
            if c[ic]['structure_id'] not in id_ok:
                excl.append(ic)
        for icc in excl[::-1]:
            c.pop(icc)
        m=np.delete(m,excl,axis=1)
        projmaps_rmvol[inj_id]={'columns':c,'matrix':m,'rows':projmaps_rmvol1[inj_id]['rows']}
    return projmaps_rmvol

def create_file_order(projmaps, structure_tree):
    """
    the method creates file order and keyord that will be the link between the structural conn
    order and the id key in the Allen database
    """
    order = {}
    for target_id in projmaps.keys():
        order[structure_tree.get_structures_by_id([target_id])[0]['graph_order']] = [target_id]
        order[structure_tree.get_structures_by_id([target_id])[0]['graph_order']].append(
            structure_tree.get_structures_by_id([target_id])[0]['acronym'])
        order[structure_tree.get_structures_by_id([target_id])[0]['graph_order']].append(
            structure_tree.get_structures_by_id([target_id])[0]['name'])#name
    key_ord = list(order)
    key_ord.sort()
    return order, key_ord

def mouse_brain_visualizer(vol, order, key_ord, unique_parents, unique_grandparents, structure_tree, projmaps):
    """
    the method returns a volume indexed between 0 and N-1, with N=tot brain areas in the parcellation.
    -1=background and areas that are not in the parcellation
    """
    tot_areas = len(key_ord) * 2
    indexed_vec = np.arange(tot_areas).reshape(tot_areas, )
    # vec indexed between 0 and (N-1), with N=total number of area in the parcellation
    indexed_vec = indexed_vec + 1  # vec indexed between 1 and N
    indexed_vec = indexed_vec * (10 ** (-(1 + int(np.log10(tot_areas)))))
    # vec indexed between 0 and 0,N (now all the entries of vec_indexed are < 1 in order to not create confusion
    # with the entry of Vol (always greater than 1)
    vol_r = vol[:, :, :(vol.shape[2] // 2)]
    vol_r = vol_r.astype(np.float64)
    vol_l = vol[:, :, (vol.shape[2] // 2):]
    vol_l = vol_l.astype(np.float64)
    index_vec = 0  # this is the index of the vector
    left = len(indexed_vec) // 2
    for graph_ord_inj in key_ord:
        node_id = order[graph_ord_inj][0]
        if node_id in vol_r:  # check if the area is in the annotation volume
            vol_r[vol_r == node_id] = indexed_vec[index_vec]
            vol_l[vol_l == node_id] = indexed_vec[index_vec + left]
        child = []
        for ii in range(len(structure_tree.children([node_id])[0])):
            child.append(structure_tree.children([node_id])[0][ii]['id'])
        while len(child) != 0:
            if (child[0] in vol_r) and (child[0] not in list(projmaps)):
                vol_r[vol_r == child[0]] = indexed_vec[index_vec]
                vol_l[vol_l == child[0]] = indexed_vec[index_vec + left]
            child.remove(child[0])
        index_vec += 1  # index of vector
    vol_parcel = np.concatenate((vol_r, vol_l), axis=2)
    # Since the parcellation is reduced some areas are in the annotation volume but not in the parcellation,
    # so it is possible to plot also those areas with trick explained in ParentsAndGrandPArentsFinder
    # Parents:
    bool_idx = (vol_parcel > np.amax(indexed_vec))
    # Find the elements of vol_parcel that are yet not associated to a value of the indexed_vec in the parcellation
    not_assigned = np.unique(vol_parcel[bool_idx])
    vol_r = vol_parcel[:, :, :(vol.shape[2] // 2)]
    vol_r = vol_r.astype(np.float64)
    vol_l = vol_parcel[:, :, (vol.shape[2] // 2):]
    vol_l = vol_l.astype(np.float64)

    for node_id in not_assigned:
        node_id = int(node_id)
        if structure_tree.get_structures_by_id([node_id])[0] is not None:
            ancestor = structure_tree.get_structures_by_id([node_id])[0]['structure_id_path']
        else:
            ancestor = []
        while len(ancestor) > 0:
            pp = ancestor[-1]
            if pp in list(unique_parents):
                vol_r[vol_r == node_id] = indexed_vec[unique_parents[pp]]
                vol_l[vol_l == node_id] = indexed_vec[unique_parents[pp] + left]
                ancestor = []
            else:
                ancestor.remove(pp)
    vol_parcel = np.concatenate((vol_r, vol_l), axis=2)
    # Grand parents:
    bool_idx = (vol_parcel > np.amax(indexed_vec))
    # Find the elements of vol_parcel that are yet not associated to a value of the indexed_vec in the parcellation
    not_assigned = np.unique(vol_parcel[bool_idx])
    vol_r = vol_parcel[:, :, :(vol.shape[2] // 2)]
    vol_r = vol_r.astype(np.float64)
    vol_l = vol_parcel[:, :, (vol.shape[2] // 2):]
    vol_l = vol_l.astype(np.float64)
    for node_id in not_assigned:
        node_id = int(node_id)
        if structure_tree.get_structures_by_id([node_id])[0] is not None:
            ancestor = structure_tree.get_structures_by_id([node_id])[0]['structure_id_path']
        else:
            ancestor = []
        while len(ancestor) > 0:
            pp = ancestor[-1]
            if pp in list(unique_grandparents):
                vol_r[vol_r == node_id] = indexed_vec[unique_grandparents[pp]]
                vol_l[vol_l == node_id] = indexed_vec[unique_grandparents[pp] + left]
                ancestor = []
            else:
                ancestor.remove(pp)
    vol_parcel = np.concatenate((vol_r, vol_l), axis=2)
    vol_parcel[vol_parcel >= 1] = 0  # set all the areas not in the parcellation to 0 since the background is zero
    vol_parcel = vol_parcel * (10 ** (1 + int(np.log10(tot_areas))))  # return to indexed between
    # 1 and N (with N=tot number of areas in the parcellation)
    vol_parcel = vol_parcel - 1  # with this operation background and areas not in parcellation will be -1
    # and all the others with the indexed between 0 and N-1
    vol_parcel = np.round(vol_parcel)
    vol_parcel = rotate_reference(vol_parcel)
    return vol_parcel

def rotate_reference(allen):
    """
    'rotate_reference' rotate the Allen 3D (x1,y1,z1) reference
     in the TVB (https://www.thevirtualbrain.org/tvb/zwei) 3D reference (x2,y2,z2).
    The relation between the different reference system is: x1=z2, y1=x2, z1=y2
    """
    # first rotation in order to obtain: x1=x2, y1=z2, z1=y2
    vol_trans = np.zeros((allen.shape[0], allen.shape[2], allen.shape[1]), dtype=float)
    for x in range(allen.shape[0]):
        vol_trans[x, :, :] = (allen[x, :, :][::-1]).transpose()

    # second rotation in order to obtain: x1=z2, y1=x1, z1=y2
    allen_rotate = np.zeros((allen.shape[2], allen.shape[0], allen.shape[1]), dtype=float)
    for y in range(allen.shape[1]):
        allen_rotate[:, :, y] = (vol_trans[:, :, y]).transpose()
    return allen_rotate


def construct_structural_conn(projmaps, order, key_ord, weights=None):
    """
    the method builds the Structural Connectivity (SC) matrix.
    If 'weights' (roi_id -> {exp_id: inj_fraction}) is given, experiments are combined
    with an injection-fraction-weighted average (Oh 2014 style), so focal injections
    dominate and grazing/contaminated ones are down-weighted; otherwise a plain mean.
    """
    len_right = len(list(projmaps))
    structural_conn = np.zeros((len_right, 2 * len_right), dtype=float)
    row = -1
    for graph_ord_inj in key_ord:
        row += 1
        inj_id = order[graph_ord_inj][0]
        columns = projmaps[inj_id]['columns']
        matrix = projmaps[inj_id]['matrix']
        # average on the experiments (NB: if there are NaN values not average!)
        if np.isnan(np.sum(matrix)):
            print('There is a Nan in matrix of injection structure %d'%inj_id)
        else:
            if weights is not None:
                w = np.array([weights.get(int(inj_id), {}).get(int(e), 0.0)
                              for e in projmaps[inj_id]['rows']], dtype=float)
                matrix = np.average(matrix, axis=0, weights=w) if w.sum() > 0 \
                    else np.mean(matrix, axis=0)
            else:
                matrix = np.mean(matrix,axis=0) #(np.array([sum(matrix[:, i]) for i in range(matrix.shape[1])]) / (matrix.shape[0]))
        # order the target
        col = -1
        for graph_ord_targ in key_ord:
            col += 1
            targ_id = order[graph_ord_targ][0]
            for index in range(len(columns)):
                if columns[index]['structure_id'] == targ_id:
                    if columns[index]['hemisphere_id'] == 2:
                        structural_conn[row, col] = matrix[index]
                    if columns[index]['hemisphere_id'] == 1:
                        structural_conn[row, col + len_right] = matrix[index]
    # save the complete matrix (both left and right inj):
    first_quarter = structural_conn[:, :int(structural_conn.shape[1] / 2)]
    second_quarter = structural_conn[:, int(structural_conn.shape[1] / 2):]
    sc_down = np.concatenate((second_quarter, first_quarter), axis=1)
    structural_conn = np.concatenate((structural_conn, sc_down), axis=0)
    structural_conn = structural_conn / (np.amax(structural_conn))  # normalize the matrix
    return structural_conn


def construct_centres(mcc, order, key_ord):
    """
    the method returns the centres of the brain areas in the selected parcellation
    """
    centres = np.zeros((len(key_ord) * 2, 3), dtype=float)
    names = []
    acro = []
    row = -1
    for graph_ord_inj in key_ord:
        node_id = order[graph_ord_inj][0]
        coord = [0, 0, 0]
        mask, _ = mcc.get_structure_mask(node_id)
        mask = rotate_reference(mask)
        mask_r = mask[:int(mask.shape[0] / 2), :, :]
        xyz = np.where(mask_r)
        if xyz[0].shape[0] > 0:  # Check if the area is in the annotation volume
            coord[0] = np.mean(xyz[0])
            coord[1] = np.mean(xyz[1])
            coord[2] = np.mean(xyz[2])
        row += 1
        centres[row, :] = coord
        coord[0] = (mask.shape[0]) - coord[0]
        centres[row + len(key_ord), :] = coord
        n = order[graph_ord_inj][2]
        N = order[graph_ord_inj][1]
        right = 'Right '
        names.append(str(right+n))
        acro.append(str(right+N))
    for graph_ord_inj in key_ord:
        n = order[graph_ord_inj][2]
        N = order[graph_ord_inj][1]
        left = 'Left '
        names.append(str(left+n))
        acro.append(str(left+N))
    return centres, names, acro



def construct_tract_lengths(centres):
    """
    the method returns the tract lengths between the brain areas in the selected parcellation
    """
    len_right = int(len(centres) / 2)
    tracts = np.zeros((len_right, len(centres)), dtype=float)
    for inj in range(len_right):
        center_inj = centres[inj]
        for targ in range(len_right):
            targ_r = centres[targ]
            targ_l = centres[targ + len_right]
            tracts[inj, targ] = np.sqrt(
                (center_inj[0] - targ_r[0]) ** 2 + (center_inj[1] - targ_r[1]) ** 2 + (center_inj[2] - targ_r[2]) ** 2)
            tracts[inj, targ + len_right] = np.sqrt(
                (center_inj[0] - targ_l[0]) ** 2 + (center_inj[1] - targ_l[1]) ** 2 + (center_inj[2] - targ_l[2]) ** 2)
    # Save the complete matrix (both left and right inj):
    first_quarter = tracts[:, :int(tracts.shape[1] / 2)]
    second_quarter = tracts[:, int(tracts.shape[1] / 2):]
    tracts_down = np.concatenate((second_quarter, first_quarter), axis=1)
    tracts = np.concatenate((tracts, tracts_down), axis=0)
    return tracts

# the method associated the parent and the grandparents to the child in the selected parcellation with the biggest vol
# Since the parcellation is reduced some areas are in the annotation volume but not in the parcellation,
# so it is possible to plot also those areas with following trick:
# If an area that is not in the parcellation is brother of an area that is in the parcellation (same parent),
# the areas not in the parcellation will be plotted in the vol with the
# same vec_indexed of the area in the parcellation.
# In order to have an univocal relation, since some areas in the parcellation have some parent
# for each parent it will be link the child with the biggest volume in the parcellation
# the same is done for the grandparents
def parents_and_grandparents_finder(tvb_mcc, order, key_ord, structure_tree):
    parents = []  # Here it will be the id of the parents of the areas in the parcellation
    grandparents = []  # Here it will be the id of the grandparents of the areas in the parcellation
    vol_areas = []  # Here it will be the volume of the areas in the parcellation
    vec_index = []  # Here it will be the index of the vector of the areas in the parcellation
    index = 0
    for graph_ord_inj in key_ord:
        node_id = order[graph_ord_inj][0]
        parents.append(structure_tree.get_structures_by_id([node_id])[0]['structure_id_path'][-2])
        grandparents.append(structure_tree.get_structures_by_id([node_id])[0]['structure_id_path'][-3])
        vec_index.append(index)
        index += 1
        mask, _ = tvb_mcc.get_structure_mask(node_id)
        tot_voxels = np.count_nonzero(mask)
        vol_areas.append(tot_voxels)
    # I will order parents, grandparents, vec_index according to the volume of the areas
    parents = [parents for (vv, parents) in sorted(zip(vol_areas, parents))]
    grandparents = [grandparents for (vv, grandparents) in sorted(zip(vol_areas, grandparents))]
    vec_index = [iid for (vv, iid) in sorted(zip(vol_areas, vec_index))]
    k = len(parents)
    unique_parents = {}  # Unique parents will be a dictionary with keys the parent id and as value the index vec
    # of the region in parcellation which has that parent id
    for p in reversed(parents):
        k -= 1
        if p not in list(unique_parents):
            unique_parents[p] = vec_index[k]
    k = len(grandparents)
    unique_gradparents = {}  # Unique parents will be a dictionary with keys the parent id and as value the index vec
    # of the region in my parcellation that has that parent id
    for p in reversed(grandparents):
        k -= 1
        if np.isnan(p) == 0:
            if p not in list(unique_gradparents):
                unique_gradparents[p] = vec_index[k]
    return unique_parents, unique_gradparents