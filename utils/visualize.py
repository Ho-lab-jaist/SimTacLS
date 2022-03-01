import numpy as np
import pyvista as pv
import pyvistaqt as pvqt
import matplotlib.pyplot as plt
import pandas as pd
from threading import Thread
# import util functions
from utils import simtacls_utils

tactile_skin = simtacls_utils.TactileSkin(skin_path='./resources/skin.vtk')
# extract fixed inward radial vectors
radial_vectors = tactile_skin.get_radial_vectors()

class TactileSkinVisualization():
    def __init__(self, skin_path='./resources/skin.vtk',  
                        plot_init = True,
                        color_map = "bwr"):
        self.skin_path = skin_path
        self.color_map = plt.cm.get_cmap(color_map)  
        # update only once as initialization
        if plot_init:
            self.plot_initialize()

    def plot_initialize(self):
        pv.global_theme.font.color = 'black' 
        pv.global_theme.font.title_size = 16 
        pv.global_theme.font.label_size = 16  
        
        self.plotter = pvqt.BackgroundPlotter()
        self.plotter.subplot(0, 0)
        self.plotter.set_background("white", top="white")
        self.plotter.show_axes()      
        self.skin = pv.read(self.skin_path) # for pyvista visualization
        self.skin['contact depth (unit:mm)'] = np.zeros(self.skin.n_points)
        color_map = plt.cm.get_cmap("bwr")  
        self.plotter.add_mesh(self.skin, cmap=color_map, clim=[-10, 15], show_scalar_bar=False)

    def update_visualization(self, nodal_displacements, nodal_positions, bandwidth, cd_text, num_contact_text):
        norm_deviations = np.linalg.norm(nodal_displacements, axis=1)
        dir_sim = simtacls_utils.compute_directional_similarity(nodal_displacements, radial_vectors)
        dir_signals = dir_sim > 0
        signed_norm_deviations = simtacls_utils.modify_displacement_direction(norm_deviations, dir_signals)
        self.skin['contact depth (unit:mm)'] = signed_norm_deviations # for contact depth visualization
        self.skin.points = nodal_positions
        self.plotter.add_text('{0} Hz'.format(round(bandwidth)), name='bw_text', position='upper_right')
        self.plotter.add_text('Contact: {0}'.format(cd_text), name='cd_text', position=(5, 1255))
        self.plotter.add_text('#Contact: {0}'.format(num_contact_text), name='mcl_text', position=(5, 1150))