import numpy as np
import pyvista as pv
import pyvistaqt as pvqt
import matplotlib.pyplot as plt
from collections import OrderedDict
import pandas as pd

def create_nodal_radial_vectors(X0):
    """
    Create a list of fixed nodal inward radial vectors names (unit vectors)
    Parameters:
        - X0: undeformed nodal positions (N, 3)
    Returns:
        - N: nodal radial vectors (N, 3)
    """
    N = list()
    for x0 in X0:
        n = np.array([0, 0, x0[2]]) - x0
        n_unit = n/np.linalg.norm(n)  # normalize vector
        N.append(n_unit)
    return np.array(N)

def compute_directional_similarity(D, N):
    """
    Compute the directional similarity between two vectors
    which is measured by cos(\phi); \phi is angle between the two vectors
    cos(\phi) = D.N/||D||.||N||
    Parameters:
        - D: estimated nodal displacement vectors (N, 3)
    Returns:
        - N: nodal radial vectors (N, 3)
    """
    assert len(D)==len(N), 'The two vectors should be same size'
    Phi_ls = list()
    if len(D) > 1:
        for d, n in zip(D, N):
            Phi_ls.append(np.dot(d, n))    
    else:
        return np.dot(D, N)
    return np.array(Phi_ls)

def modify_displacement_direction(Dm, dir_signals):
    """
    Assign sign for for the displacement magnitude
    Parameters:
        - Dm: estimated displacement magnitudes (N, 1)
    Returns:
        - dir_signals: binary directonal signal (N, 1)
    """
    assert len(Dm)==len(dir_signals), 'The two vectors should be same size'
    signed_Dm = list()
    for dm, dir_sig in zip(Dm, dir_signals):
        if not dir_sig:
            signed_Dm.append(-dm)
        else:
            signed_Dm.append(dm)
    return np.array(signed_Dm)
    
def extract_skin_cells(vtk_file='./resources/skin.vtk'):
    """
    Returns:
        - list of connected nodes : [[id1, id2, id3], [], ..., []]
    """

    cells_ls = []
    with open(vtk_file, 'r') as rf:
        for idx, line in enumerate(rf):
            if idx < 715:
                continue
            elif 715 <= idx <= 2088:
                cells = line.split()[1:]
                cells_ls.append(cells)
            elif idx > 2088:
                break

    return cells_ls

def extract_skin_points(vtk_file):
    points_ls = []
    with open(vtk_file, 'r') as rf:
        for idx, line in enumerate(rf):
            if 6 <= idx <= 712:
                points = [float(x) for x in line.split()]
                points_ls.append(points)
            elif idx > 712:
                break

    return np.array(points_ls, dtype=float)

def graph_generation(skin_cells, num_of_nodes):

    # a graph can be stored with a dictionary data structure
    graph = OrderedDict()
    for node_idx in range(num_of_nodes):
        node_id = str(node_idx)
        # find the id of nodes connecting to the given node
        # by searching the list of cells
        connected_nodes = []
        for cell in skin_cells:
            if node_id in cell:
                connected_nodes.extend(cell)
        # remove the id of given node and duplicale ones
        connected_nodes = list(set(connected_nodes))
        connected_nodes.remove(node_id)
        graph[node_id] = connected_nodes
    return graph

class TactileSkin():
    def __init__(self, skin_path='./resources/skin.vtk', node_idx_path='./resources/node_idx.csv', label_idx_path='./resources/label_idx.csv'):
        self._skin_path = skin_path
        self.file_idx = get_file_idx(node_idx_path, label_idx_path)

        # extract fixed initial nodal positions
        self.init_positions = extract_skin_points(self._skin_path)
        # extract fixed skin cells
        self.skin_cells = extract_skin_cells(self._skin_path)
        # extract fixed inward radial vectors
        self.radial_vectors = create_nodal_radial_vectors(self.init_positions)

        self.positions = np.array(self.init_positions) # full running positions
        self.nodal_displacements = np.zeros_like(self.init_positions) # full running nodal displacements

    def get_init_nodal_positions(self):
        return self.init_positions
    
    def get_skin_cells(self):
        return self.skin_cells

    def get_radial_vectors(self):
        return self.radial_vectors

    def update_nodal_positions(self, pred_displacements):
        """
        @state: the state of current nodes/points nd.array nx3
        n is the number of target nodes, used for network outputs
        @init_points: the state of initial nodes/points nd.array mx3
        (m >= n)
        """
        self.positions = self.init_positions + self.update_nodal_displacements(pred_displacements)
        return self.positions

    def update_nodal_displacements(self, pred_displacements):
        """
        @state: the state of current nodes/points nd.array nx3
        n is the number of target nodes, used for network outputs
        @init_points: the state of initial nodes/points nd.array mx3
        (m >= n)
        """
        self.nodal_displacements[self.file_idx, :] = pred_displacements
        return self.nodal_displacements

def get_file_idx(node_idx_path, label_idx_path):
    df_node_idx = pd.read_csv(node_idx_path)
    df_label_idx = pd.read_csv(label_idx_path)

    node_idx = np.array(df_node_idx.iloc[:,0], dtype=int) # (full skin) face node indices in vtk file exported from SOFA 
    node_idx = list(set(node_idx)) # eleminate duplicate elements (indices)
    node_idx = sorted(node_idx) # sorted the list of indices

    label_idx = list(df_label_idx.iloc[:,0]) #(not full skin) at nodes used for training - labels
    file_idx = [node_idx.index(idx) for idx in label_idx]

    return file_idx