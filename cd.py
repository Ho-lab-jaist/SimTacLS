"""
Author: Quan Khanh Luu (JAIST)
Contact: quan-luu@jaist.ac.jp
Descriptions: 
- Contact detection for ViTac devices (e.g., TacLink)
It is important that the soft artificial skin is represeted by a mesh (graph).
"""

import numpy as np
from utils import simtacls_utils

skin_mesh_vtk = './resources/skin.vtk'
tactile_skin = simtacls_utils.TactileSkin(skin_path=skin_mesh_vtk)
# extract fixed inward radial vectors
radial_vectors = tactile_skin.get_radial_vectors()

def contact_detection(nodal_displacements, threshold):
    """
    Contact detection for ViTac devices based on nodal displacements
    Parameters:
        - nodal_displacements (N,3): 3D predicted nodal displacement vectors for N nodes
        - threshold: deformed region threshold
    Returns:
        - no_of_contacts: the number of contacts detected
        - contact_positions: 3-D contact positions
    """
    # compute directional similarity at every nodes on the skin surface
    dir_sim = simtacls_utils.compute_directional_similarity(nodal_displacements, radial_vectors)
    nodal_displacement_magnitude = np.linalg.norm(nodal_displacements, axis=1)
    # directional signals
    dir_signals = dir_sim > 0.
    # assign True to the deformed nodes, whose intensities > threshold
    contact_signals = nodal_displacement_magnitude > threshold
    # obtain nodal contact signals
    nodal_contact_signals = contact_signals & dir_signals
    # extract the number of deformed nodes
    num_of_deformed_nodes = len(nodal_displacement_magnitude[nodal_contact_signals])
    return True if num_of_deformed_nodes > 0 else False