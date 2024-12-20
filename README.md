# SimTacLS: Toward a Platform for Simulation and Learning of Vision-based Tactile Sensing at Large Scale
Open-source simulation tool and sim2real method for large-scale vision-based tactile (ViTac) sensing devices, based on SOFA-GAZEBO-GAN pipeline. 

A video demonstration is at:

[![IMAGE ALT TEXT](https://img.youtube.com/vi/bFY4CV5Wk9g/0.jpg)](https://youtu.be/bFY4CV5Wk9g "SimTacLS: Toward a Platform for Simulation & Learning of Vision-based Tactile Sensing at Large Scale")

The overview of the pipeline is at:
![Overview](https://github.com/Ho-lab-jaist/TacLink-Sim2Real/blob/main/figures/Fig_simtacls_overview.png)

## Documentation

### Training R2S-TN (Gan) network
Create a `data` directory with `train` and `test` subfolders to organize the respective datasets. Each dataset contains pairs of `sim` and `real` images.



### Real-time perception
Run: 
```
$ python simtacls_run_realtime.py
```
Please note that the following dependecies required:
- pytorch
- opencv-python
- pyvista
- pyvistaqt

For the pre-trained R2S-TN and TacNet models, please contact [quan-luu@jaist.ac.jp](mailto:quan-luu@jaist.ac.jp) for more information.

The program was tested on a couple of fisheye-lense cameras (ELP, USB2.0 Webcam Board 2 Mega Pixels 1080P OV2710 CMOS Camera Module) which optionally provide up to 120fps. And the processing speed heavily depends on computing resources, e.g., GPU (NVIDIA, RTX8000, RTX3090, GTX 1070Ti were proved to be compatible)
.
### Soft skin modelling, using SOFA:

### Image acquisition module for obtaining tactile image dataset, using ROS/GAZEBO:

1. Move to Catkin workspace (ROS) working directory.

```
$ cd [home/username]/catkin_ws/src
```
Then make the catkin environment
```
$ catkin_make
```
Next, move to the home directory of the module to avoid unfound reference path
```
$ cd [home/username]/catkin_ws/src/sofa_gazebo_interface
```

2. Start Gazebo simulation environment for TacLink.

```
$ roslaunch vitaclink_gazebo vitaclink_world.launch
```

3. Add PATH for Gazebo models (model uri), if necessary. This would be helful if the skin/marker states (.STL file) were not in the default Gazebo model directory.

```
$ nano ~/.bashrc
```
Then, write the following command to the end of *.basrc* file, given the skin/marker states are in the *skin_state* directory.

```
source /usr/share/share/gazebo/setup.sh
```
```
export GAZEBO_MODEL_PATH=$[path/to/data]/skin_state:$GAZEBO_MODEL_PATH
```

4. Run program file (.py) for acquisition of tactile image dataset
   1. Single contact dataset: Run the following python program for starting the acquisition process (single-touch data)
      ```
      $ rosrun vitaclink_gazebo single_point_tactile_image_acquisition.py
      ```
    2. Multiple contact dataset: 
        
        The file name of skin and marker states (.STL) exported from SOFA should be saved as follows: `skin{group:04}_{depth:02}.stl` and `marker{group:04}_{depth:02}.stl`; e.g., `skin0010_15.stl` is the skin state of `data group: 10` and `contact depth: 15`.

       The output tactile images will be formatted as `{group}_{depth}.jpg`. (For our demostration, we have 500 contact groups, each consists of 20 incremental contact depths).

       Run the following python program for starting the acquisition process (multi-touch data)
      ```
      $ rosrun vitaclink_gazebo multiple_point_tactile_image_acquisition.py
      ```
      
### Learning and Sim2Real, based on TacNet and GAN networks:
For TacNet training, run:
```
$ python tacnet_train_mutitouch_dataset.py
```
Note that the full simulation dataset is required for the program executaion, for the full dataset please contact correspondances for more information.

## License
Distributed under the MIT License. See `LICENSE` for more information.
## Contact
- Project Manager
	- Ho Anh Van - [van-ho[at]jaist.ac.jp](mailto:van-ho[at]jaist.ac.jp)
- Developers
	- Nguyen Huu Nhan - [nhnha[at]jaist.ac.jp](nhnhan[at]jaist.ac.jp)
	- Luu Khanh Quan - [quan-luu[at]jaist.ac.jp](mailto:quan-luu[at]jaist.ac.jp)
## Acknowledgements
This work was supported in part by JSPS KAKENHI under Grant 18H01406 and JST Precursory Research for Embryonic Science and Technology PRESTO under Grant JPMJPR2038.

## Disclaimer
Since we are an acedamic lab, the software provided here is not as computationally robust or efficient as possible and is dedicated for research purpose only. There is no bug-free guarantee and we also do not have personnel devoted for regular maintanence. 
