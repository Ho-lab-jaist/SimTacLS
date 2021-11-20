# TacLink-Sim2Real
Open-source simulation tool and sim2real method for Large-scale Tactile Sensing Devices, based on SOFA-GAZEBO-GAN framework.

## TO-DO
Run image acquisition module for obtaining tactile image dataset, using ROS/GAZEBO.

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

export GAZEBO_MODEL_PATH=$[path/to/data]/skin_state:$MY_MODEL_PATH:$GAZEBO_MODEL_PATH

```
