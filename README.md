# TacLink-Sim2Real
Open-source simulation tool and sim2real method for Large-scale Tactile Sensing Devices, based on SOFA-GAZEBO-GAN framework. The overview of the simulation framework could be seen below.

![Overview](https://github.com/Ho-lab-jaist/TacLink-Sim2Real/blob/main/figures/Fig_overview.PNG)

## For Developers

## TO-DO
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
      
## License
Distributed under the MIT License. See `LICENSE` for more information.
## Contact
- Project Manager
	- Ho Anh Van - [van-ho@jaist.ac.jp](mailto:van-ho@jaist.ac.jp)
- Developers
	- Nguyen Huu Nhan - [nhnhan@jaist.ac.jp](nhnhan@jaist.ac.jp)
	- Luu Khanh Quan - [quan-luu@jaist.ac.jp](mailto:quan-luu@jaist.ac.jp)
## Acknowledgements
This work was supported in part by JSPS KAKENHI under Grant 18H01406 and JST Precursory Research for Embryonic Science and Technology PRESTO under Grant JPMJPR2038.
