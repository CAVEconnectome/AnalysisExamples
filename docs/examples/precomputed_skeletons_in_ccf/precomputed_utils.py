import pandas as pd
import numpy as np
import pcg_skel
import cloudvolume
import caveclient

from tqdm.notebook import tqdm, trange
tqdm.pandas()

# ccf transform
m = [-0.20923994, -0.01442708, -1.15535762, 9296.273617, 0.48709059, 0.9850834, -0.16600506, 230.5823285366, 0.72729732, -0.66388746, -0.22121276, 8605.414249918707]

def get_meshwork_from_segment(segment_id, nucleus_position, root_point_resolution, client, cv):
    # create whole neuron with radius info
    # takes segment information; outputs meshwork
    synapse_table = client.info.get_datastack_info()['synapse_table']
    resample_spacing = 1510
    collapse_soma = True
    collapse_radius = 10_000
    mw =  pcg_skel.coord_space_meshwork(segment_id,
                                        client=client,
                                        root_point=nucleus_position,
                                        root_point_resolution=root_point_resolution,
                                        collapse_soma=collapse_soma,
                                        collapse_radius=collapse_radius,
                                        synapses='all',
                                        synapse_table=synapse_table,
                                        cv = cv)
    
    # add radius properties df to annotations 
    pcg_skel.features.add_volumetric_properties(mw, client)
    pcg_skel.features.add_segment_properties(mw)
    pcg_skel.features.add_is_axon_annotation(mw,'pre_syn', 'post_syn')

    return mw

def get_skeleton_features_from_meshwork(mw, properties_dict):
    # make skeleton node labels
    # takes meshwork and propteries dictionary; outputs updated properties dictionary
    ## adapted from skeletonization_tools.skeletonize_clean_metadata.make_node_labels()

    # extract vertices, edges, radius, 
    vertices = mw.skeleton.vertices
    edges = mw.skeleton.edges

    # transform vertices into ccf directly
    vertices_transform = np.apply_along_axis(ccf_vertex_transform, 1, vertices)
    
    # create compartment label, default to dendrite
    compartment = np.array([3]*len(mw.skeleton.vertices))
    
    # add soma label
    compartment[int(mw.skeleton.root)]=1
    
    # add axon label
    axon_nodes_mesh = mw.anno['is_axon']['mesh_index_filt']
    axon_nodes = mw.MeshIndex(axon_nodes_mesh).to_skel_index_padded
    compartment[axon_nodes]=2
    
    # pulls the segment properties from meshwork anno and translates into skel index
    r_df = mw.anno.segment_properties.df[['r_eff', 'mesh_ind_filt']].set_index('mesh_ind_filt')
    radius = r_df.loc[mw.skeleton_indices.to_mesh_region_point].r_eff.values/1000

    # update properties
    properties_dict['compartment'] = compartment
    properties_dict['vertices'] = vertices
    properties_dict['vertices_transform'] = vertices_transform
    properties_dict['edges'] = edges
    properties_dict['radius'] = radius

    return properties_dict

def get_postsynapse_features_from_meshwork(mw, properties_dict):
    # gets synapse properties for each vertex
    # takes meshwork and propteries dictionary; outputs updated properties dictionary

    # generate index lookup
    mw_index_lookup = get_meshwork_index_lookup(mw)
    
    # Quantify synapse counts by node
    # postsynaptic counts
    postsyn_counts_df = pd.DataFrame(mw.anno.post_syn.df['post_pt_mesh_ind_filt'].value_counts())
    postsyn_counts_df.index.names = ['mesh_ind_filt']
    
    postsyn_mesh = mw_index_lookup.copy()
    postsyn_mesh['counts'] = 0
    postsyn_mesh.loc[postsyn_counts_df.index, 'counts'] = postsyn_counts_df['count']
    postsyn_skel_counts = postsyn_mesh.groupby('skel_ind')['counts'].sum()

    # Quantify synapses size by node
    # postsynaptic size
    postsyn_size_df = pd.DataFrame(mw.anno.post_syn.df.groupby('post_pt_mesh_ind_filt')['size'].transform('sum').values, 
                                   columns=['size'],
                                   index = mw.anno.post_syn.df['post_pt_mesh_ind_filt'].values)
    postsyn_size_df.index.names = ['mesh_ind_filt']
    
    postsyn_mesh = mw_index_lookup.copy()
    postsyn_mesh['size'] = 0
    postsyn_mesh.loc[postsyn_size_df.index, 'size'] = postsyn_size_df['size']
    postsyn_skel_size = postsyn_mesh.groupby('skel_ind')['size'].sum()

    # update properties
    properties_dict['postsyn_counts'] = postsyn_skel_counts.values
    properties_dict['postsyn_size'] = postsyn_skel_size.values

    return properties_dict


def get_presynapse_features_from_meshwork(mw, properties_dict):
    # gets synapse properties for each vertex
    # takes meshwork and propteries dictionary; outputs updated properties dictionary

    # generate index lookup
    mw_index_lookup = get_meshwork_index_lookup(mw)

    # Quantify synapse counts by node
    # presynaptic counts
    presyn_counts_df = pd.DataFrame(mw.anno.pre_syn.df['pre_pt_mesh_ind_filt'].value_counts())
    presyn_counts_df.index.names = ['mesh_ind_filt']
    
    presyn_mesh = mw_index_lookup.copy()
    presyn_mesh['counts'] = 0
    presyn_mesh.loc[presyn_counts_df.index, 'counts'] = presyn_counts_df['count']
    presyn_skel_counts = presyn_mesh.groupby('skel_ind')['counts'].sum()

    # Quantify synapses size by node
    # presynaptic size
    presyn_size_df = pd.DataFrame(mw.anno.pre_syn.df.groupby('pre_pt_mesh_ind_filt')['size'].transform('sum').values, 
                                   columns=['size'],
                                   index = mw.anno.pre_syn.df['pre_pt_mesh_ind_filt'].values)
    presyn_size_df.index.names = ['mesh_ind_filt']
    
    presyn_mesh = mw_index_lookup.copy()
    presyn_mesh['size'] = 0
    presyn_mesh.loc[presyn_size_df.index, 'size'] = presyn_size_df['size']
    presyn_skel_size = presyn_mesh.groupby('skel_ind')['size'].sum()

    # update properties
    properties_dict['presyn_counts'] = presyn_skel_counts.values
    properties_dict['presyn_size'] = presyn_skel_size.values

    return properties_dict

def get_myelin_at_vertex(mw, myelin_df, properties_dict, client, cv):
    # given a df of all myelinated points on the axon, return the corresponding skeleton labels to the properties dictionary

    # get level2 nodes at myelinated positions
    myelin_lvl2 = myelin_df.progress_apply(pd_get_level2_point, axis=1, client=client, cv=cv)

    # generate index lookup
    mw_index_lookup = get_meshwork_index_lookup(mw)

    # merge myelin_lvl2 to index lookup
    myelin_merge = pd.merge(myelin_lvl2, mw_index_lookup, on='lvl2_id', how='inner')
    
    # empty myelin labels
    try:
        vertices = properties_dict['vertices']
    except:
        print('no vertex properties found; call get_skeleton_features_from_meshwork() first') 
        
    myelin = np.zeros(len(vertices))
    
    # where myelin is present, set to 1
    myelin[myelin_merge.skel_ind.values] = 1

    properties_dict['myelin'] = myelin

    return properties_dict

def get_meshwork_index_lookup(mw):
    # generate index lookup with lvl2, mesh, and skeleton indices
    mw_index_lookup = mw.anno.lvl2_ids.df
    mw_index_lookup['skel_ind'] = mw.mesh_indices.to_skel_index_padded
    mw_index_lookup.set_index('mesh_ind_filt', inplace=True)

    return mw_index_lookup

def pd_get_level2_point(row, client, cv, voxel_resolution=[4,4,40]):
    point, root_id = row[['pt_position','valid_id']]

    try:
        lvl2_id = pcg_skel.chunk_tools.get_closest_lvl2_chunk(point,
                                                              root_id,
                                                              client,
                                                              voxel_resolution=voxel_resolution,
                                                              radius=200)
        row['lvl2_id'] = lvl2_id
    except:
        row['lvl2_id'] = np.nan
        
    return row

## Transform into nm
def ccf_vertex_transform(a):
    
    out = np.zeros(len(a))
    x = a[0]
    y = a[1]
    z = a[2]
    
    out[0] = (m[0] * x + m[1] * y + m[2] * z) + m[3]*1000;
    out[1] = (m[4] * x + m[5] * y + m[6] * z) + m[7]*1000;
    out[2] = (m[8] * x + m[9] * y + m[10] * z) + m[11]*1000;
    
    return out

## Transform into nm
def ccf_vertex_translate(a):
    
    out = np.zeros(len(a))
    x = a[0]
    y = a[1]
    z = a[2]
    
    out[0] = x + m[3]*1000;
    out[1] = y + m[7]*1000;
    out[2] = z + m[11]*1000;
    
    return out