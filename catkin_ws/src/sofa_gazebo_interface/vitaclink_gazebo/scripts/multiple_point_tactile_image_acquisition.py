#!/usr/bin/env python

from numpy.lib.function_base import _unwrap_dispatcher
import rospy, tf
from rospy.core import loginfo
import numpy as np
# ROS-Gazebo server messages
from gazebo_msgs.srv import DeleteModel, SpawnModel
from std_srvs.srv import Empty
# ROS Geometry messages
from geometry_msgs.msg import Pose, Point, Quaternion
# ROS Image message
from sensor_msgs.msg import Image
# ROS Image message --> OpenCV2 image converter
from cv_bridge import CvBridge, CvBridgeError
# OpenCV2 for saving an image
import cv2
import time
import os
from parse_csv import CSV_Parse, CSV_FILE_NAME


class Image_Processing:
	def __init__(self, image_topic = ["/vitaclink/camera_up/image_raw", "/vitaclink/camera_bot/image_raw"]):
	  rospy.Subscriber(image_topic[0], Image, self.callback_up)
	  rospy.Subscriber(image_topic[1], Image, self.callback_bot)
	  self.cv2_image_up = None
	  self.cv2_image_bot = None
	  self.bridge = CvBridge()

	def callback_up(self, msg):
	  try:
		  ros2cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
	  except CvBridgeError as e:
		  print(e)
	  else:
		  self.cv2_image_up = ros2cv_img
	
	def callback_bot(self, msg):
	  try:
		  ros2cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")		
	  except CvBridgeError as e:
		  print(e)
	  else:
		  self.cv2_image_bot = ros2cv_img

	def get_cvimage_up(self):
		return self.cv2_image_up
	
	def get_cvimage_bot(self):
		return self.cv2_image_bot


out_dir = "/media/holab/SSD-PGU3/iotouch/data" # output directory
up_path = os.path.join(out_dir, "up")
bot_path = os.path.join(out_dir, "bot")
if not os.path.exists(up_path):
	os.makedirs(up_path)
if not os.path.exists(bot_path):
	os.makedirs(bot_path)

if __name__ == '__main__':
	# Initialize ROS node
	rospy.init_node("multiple_point_tactile_image_acquisition")
	
	# Wait for ROS services
	rospy.wait_for_service("gazebo/delete_model")
	rospy.wait_for_service("gazebo/spawn_sdf_model")
	rospy.wait_for_service("gazebo/reset_simulation")
	loginfo("All ros services are ready!")

	# Initialize ROS services to communicate with GAZEBO
	delete_model = rospy.ServiceProxy("gazebo/delete_model", DeleteModel)
	spawn_model = rospy.ServiceProxy("gazebo/spawn_sdf_model", SpawnModel)
	reset_sim = rospy.ServiceProxy("gazebo/reset_simulation", Empty)

	# Instantiate image acquisition object
	img_processing = Image_Processing()
	csv_parse = CSV_Parse()

	with open("./vitaclink_gazebo/models/skin.sdf", "r") as f:
        	skin_xml = f.read()	
	with open("./vitaclink_gazebo/models/marker.sdf", "r") as f:
        	marker_xml = f.read()	

	orient = Quaternion(*tf.transformations.quaternion_from_euler(0,0,0))
	skin_pose   = Pose(Point(0, 0, 0.807), orient)
	marker_pose = Pose(Point(0, 0, 0.807), orient)

	num_group = 500 # modify the code for user-options later
	num_depth = 20 # modify the code for user-options later 
	groups = range(1, num_group+1)
	depths = range(1, num_depth+1)

	count = 0
	start_idx = 0
	for group in groups[start_idx:]:
		index_current_group = groups.index(group)
		print("[{0}/{1}] - The current data group : {2}".format(len(groups[:index_current_group+1]), num_group, group))
	
		if count == 10:
			count = 0
			reset_sim()
			loginfo("Simulation time reset!")		

		count += 1

		for depth in depths:
			state_num =  '{0:04}_{1:02}'.format(group, depth) #e.g., 0025_02 (group: 25, depth 2)
			skin_name = "skin{0}.stl".format(state_num)
			marker_name = "marker{0}.stl".format(state_num)
			skin_xml = skin_xml[:238] + skin_name + skin_xml[253:]
			marker_xml = marker_xml[:238] + marker_name + marker_xml[255:]

			# Spawn skin and marker model
			spawn_model(skin_name, skin_xml, "", skin_pose, "world")
			time.sleep(2.0) # 2	
			spawn_model(marker_name, marker_xml, "", marker_pose, "world")
			time.sleep(8.0) # 8
			
			# Save your OpenCV2 image as a jpeg
			image_name = "{0}.jpg".format(state_num)
			cv2.imwrite(os.path.join(up_path, image_name), img_processing.get_cvimage_up())
			cv2.imwrite(os.path.join(bot_path, image_name), img_processing.get_cvimage_bot())
			print("Saved images : {0}".format(image_name))

			# Delete skin and marker model
			delete_model(skin_name)
			time.sleep(1) # 1
			delete_model(marker_name)
			time.sleep(2) # 2
