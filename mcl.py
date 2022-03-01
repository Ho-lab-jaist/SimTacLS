"""
Author: Quan Khanh Luu (JAIST)
Contact: quan-luu@jaist.ac.jp
Descriptions: 
- Multi-contact localization module for ViTac devices (e.g., TacLink)
- based on the proposed contact region labeling (CRL) method.
It is important that the soft artificial skin is represeted by a mesh (graph).
"""

import numpy as np
from utils import simtacls_utils


skin_mesh_vtk = './resources/skin.vtk'
num_of_nodes = 707

tactile_skin = simtacls_utils.TactileSkin(skin_path=skin_mesh_vtk)
# extract list of connected cells of the sensor skin
skin_cells = tactile_skin.get_skin_cells()
# extract non-deformed positions of skin nodes in Cartesian space
init_positions = tactile_skin.get_init_nodal_positions()
# extract fixed inward radial vectors
radial_vectors = tactile_skin.get_radial_vectors()
# generate a graph that represents the sensor skin mesh
skin_graph = simtacls_utils.graph_generation(skin_cells, num_of_nodes)


def dfs(current_node_id, current_region_id, labelled_graph, binary_graph):
    """ Depth first search to traverse across skin nodes
    - Parameters
        current_node_id: type int
        current_region_id: type int
    - Returns
    """
    if not binary_graph[current_node_id] or labelled_graph[current_node_id]:
        return labelled_graph
    labelled_graph[current_node_id] = current_region_id
    # Iterate over the connected nodes of the current node id
    for node_id in skin_graph[str(current_node_id)]:
        dfs(int(node_id), current_region_id, labelled_graph, binary_graph)
    return labelled_graph

def crl(binary_graph):
    """ 
    Contact region labeling fucntion
    Based on the concept of connected components labelling
    in order to extract contact regions
    """
    # initialize a set of contact region labels \mathbf{y}
    # by a graph whose vertices/nodes contain region labels
    labelled_graph = np.zeros_like(binary_graph, dtype=np.int8)
    no_of_touch = 0
    region_id = 1
    for node_id in range(num_of_nodes):
        if binary_graph[node_id] and not labelled_graph[node_id]:
            no_of_touch = no_of_touch + 1
            labelled_graph = dfs(node_id, region_id, labelled_graph, binary_graph)
            region_id = region_id + 1

    return no_of_touch, labelled_graph


def mcl(nodal_displacements, threshold):
    """
    Multi-contact localization for ViTac devices.
    Parameters:
        - nodal_displacements (N,3): 3D predicted nodal displacement vectors for N nodes
        - threshold: deformed region threshold
    Returns:
        - no_of_contacts: the number of contacts detected
        - contact_positions: 3-D contact positions
    """
    # compute directional similarity at every nodes on the skin surface
    dir_sim = simtacls_utils.compute_directional_similarity(nodal_displacements, radial_vectors)
    # compute deviation intensity at every nodes on the skin surface
    norm_deviations = np.linalg.norm(nodal_displacements, axis=1)
    # directional signals
    dir_signals = dir_sim > 0.
    # contact threshold signals
    contact_signals = norm_deviations > threshold
    # obtain nodal contact signals \mathbf{s}
    nodal_contact_signals = np.array(contact_signals) & np.array(dir_signals)
    # create a graph whose nodes contain binary contact signals
    binary_graph = np.array(nodal_contact_signals)
    # revoke contact region labeling (CRL)
    # which returns (number of possible separated contacts) and a graph contains region labels
    no_of_contacts, labelled_graph = crl(binary_graph)
    # extract multiple "apparent" contact regions, each region represented by node indexes
    contact_regions = [np.where(labelled_graph==patch_id)[0] for patch_id in range(1, no_of_contacts+1)]
    contact_positions = list()
    for contact_region in contact_regions:
        # extract the node index which maximizes the magnitude of nodal displacement
        contact_depth = np.max(norm_deviations[contact_region])
        contact_location_id = np.where(norm_deviations==contact_depth)[0].squeeze()
        # 3D-coordinated contact position of given contact region
        contact_position = init_positions[contact_location_id]
        contact_positions.append(contact_position)
    return no_of_contacts, contact_positions

if __name__ == '__main__':
    """
    Test 'mcl' module
    """
    import os
    import pandas as pd
    root_dir = 'Z:/'
    exp_name = 'r2sTN-cGAN-SSIM-L1'
    image_type = 'translated' # virtual | translated | real
    # load predicted displacements
    saved_path = os.path.join(root_dir, 'iotouch_env/results/{0}'.format(exp_name))
    saved_file_name = 'predicted_displacements_{0}_data.csv'.format(image_type)
    pred_displacements_df = pd.read_csv(os.path.join(saved_path, saved_file_name))
    true_displacements_df = pd.read_csv(os.path.join(root_dir, 'iotouch_env/results', 'groundtruth_displacements.csv'))
    pred_displacements = np.array(pred_displacements_df.iloc[:, 1:], dtype=float)
    true_displacements = np.array(true_displacements_df.iloc[:, 1:], dtype=float)
    assert pred_displacements.shape == true_displacements.shape, 'prediction/groundtruth mismatched'

    prediction = pred_displacements[10].reshape(-1,3)
    prediction = tactile_skin.update_nodal_displacements(prediction)
    threshold = 2
    no_of_contacts, contact_positions = mcl(prediction, threshold)
    print('The number of contacts: {0}'.format(no_of_contacts))
    print('The number of contacts: {0}'.format(contact_positions))