"""
Author: Quan Khanh Luu (JAIST)
Contact: quan-luu@jaist.ac.jp
Descriptions: 
- SimTacLS demonstration on barrel-shaped TacLink (Supplementary for T-RO22)
- Real-time evaluate Unet-based TacNet on REAL images of TacLink sensor
- And realtime perception (contact detection and muti-contac localization).
It is important that the soft artificial skin is represeted by a mesh (grap).
"""

import cv2
import torch
from utils.utils import apply_transform, tensor2img
from utils.visualize import TactileSkinVisualization
import time
import os
from threading import Thread

# import TacNet
from networks.unet_model import TacNetUNet2, print_networks

# import modules related to GAN model
from gan.options.test_options import TestOptions
from gan.models import create_model

# import perception modules
from cd import contact_detection
from mcl import mcl

from utils import simtacls_utils

opt = TestOptions().parse()  # get test options
model = create_model(opt)      # create a model given opt.model and other options
model.setup(opt)               # regular setup: load and print networks; create schedulers

# Load TacNet
root_dir = 'Z:/'
model_name = 'TacNetUnet2_2022_01_25_single_double_touch_dataset.pt' # TacNet_2022_01_06_single_double_touch_dataset
model_name_path = os.path.join('iotouch_env/train_model', model_name)
model_dir = os.path.join(root_dir, model_name_path)
tacnet = TacNetUNet2()
print('model [TacNet] was created')
print_networks(tacnet, False)
print('loading the model from {0}'.format(model_dir))
tacnet.load_state_dict(torch.load(model_dir))
print('---------- Tactile Networks initialized -------------')
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(dev)
tacnet.to(dev)
tacnet.eval()

"""
For sucessfully read two video camera streams simultaneously,
we need to use two seperated USB bus for cameras
e.g, USB ports in front and back of the CPU
"""

cam_bot = cv2.VideoCapture(0)
cam_top = cv2.VideoCapture(1)

tactile_skin = simtacls_utils.TactileSkin()
skin_visualize = TactileSkinVisualization()
cd_threshold = 1.0
mcl_threshold = 5.0
def update():
    while cam_bot.isOpened():
        start = time.time_ns()
        frame_top_o = cv2.cvtColor(cam_top.read()[1], cv2.COLOR_BGR2RGB)
        frame_bot_o = cv2.cvtColor(cam_bot.read()[1], cv2.COLOR_BGR2RGB)        
        frame_top = apply_transform(frame_top_o)
        frame_bot = apply_transform(frame_bot_o)

        # generate virtual image for top video stream
        model.set_input(frame_top)  # unpack data from data loader
        with torch.no_grad():
            frame_top_tf = model.forward()
            generated_top = (frame_top_tf + 1) / 2.0
        
        # generate virtual image for bottom video stream
        model.set_input(frame_bot)  # unpack data from data loader
        with torch.no_grad():
            frame_bot_tf = model.forward()
            generated_bot = (frame_bot_tf + 1) / 2.0

        # concatenate two video stream
        tac_img = torch.cat((generated_top, generated_bot), dim=1)
        # forward pass to TacNet
        with torch.no_grad():
            prediction = tacnet(tac_img).cpu().numpy().reshape(-1,3)
        
        nodal_displacements = tactile_skin.update_nodal_displacements(prediction)
        nodal_positions = tactile_skin.update_nodal_positions(prediction)
        if contact_detection(nodal_displacements, cd_threshold):
            cd_text = 'Detected'
            no_of_contacts, contact_positions = mcl(nodal_displacements, mcl_threshold)
            num_contact_text = str(no_of_contacts)
        else:
            cd_text = 'No'
            num_contact_text = str(0)
        end = time.time_ns()
        bandwidth = 1/((end-start)*10**-9 + 10**-5)
        # skin_visualize.plotter.add_text('{0} Hz'.format(round(bandwidth)), name='bw_text', position='upper_right')
        # skin_visualize.plotter.add_text('Contact: {0}'.format(cd_text), name='cd_text', position=(5, 1255))
        # skin_visualize.plotter.add_text('#Contact: {0}'.format(num_contact_text), name='mcl_text', position=(5, 1150))
        skin_visualize.update_visualization(nodal_displacements, nodal_positions, bandwidth, cd_text, num_contact_text)
        # skin_visualize.plotter.remove_actor(text)
        # Display the resulting frame
        cv2.imshow("Real (top)", tensor2img(frame_top))
        cv2.imshow("Real (bottom)", tensor2img(frame_bot))
        cv2.imshow("Transformed Real (top)", tensor2img(frame_top_tf))
        cv2.imshow("Transformed Real (bottom)", tensor2img(frame_bot_tf))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # When everything done, release the capture
    cam_top.release()
    cam_bot.release()
    cv2.destroyAllWindows()

thread = Thread(target=update, args=())
thread.start()
