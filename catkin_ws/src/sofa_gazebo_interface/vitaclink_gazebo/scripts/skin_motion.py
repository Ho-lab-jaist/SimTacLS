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


pwd = "/media/holab/SSD-PGU3/iotouch/data"
up_path = os.path.join(pwd, "up")
bot_path = os.path.join(pwd, "bot")


if __name__ == '__main__':
	rospy.init_node("update_skin_meshes")
	rospy.wait_for_service("gazebo/delete_model")
	rospy.wait_for_service("gazebo/spawn_sdf_model")
	rospy.wait_for_service("gazebo/reset_simulation")
	loginfo("All ros services are ready!")

	delete_model = rospy.ServiceProxy("gazebo/delete_model", DeleteModel)
	spawn_model = rospy.ServiceProxy("gazebo/spawn_sdf_model", SpawnModel)
	reset_sim = rospy.ServiceProxy("gazebo/reset_simulation", Empty)

	img_processing = Image_Processing()
	csv_parse = CSV_Parse()
	node_id = csv_parse.get_node_id(CSV_FILE_NAME)
	num_of_depth = csv_parse.get_num_of_depth(CSV_FILE_NAME)

	with open("../models/skin.sdf", "r") as f:
        	skin_xml = f.read()	
	with open("../models/marker.sdf", "r") as f:
        	marker_xml = f.read()	

	orient = Quaternion(*tf.transformations.quaternion_from_euler(0,0,0))
	skin_pose   = Pose(Point(0, 0, 0.807), orient)
	marker_pose = Pose(Point(0, 0, 0.807), orient)

	# start = time.time()
	count = 0
	start_idx = 47
	for idx, num in zip(node_id[start_idx:], num_of_depth[start_idx:]):
		index_current_node = node_id.index(idx)
		print("[{0}/{1}] - The current skin node : {2}".format(len(node_id[:index_current_node]), len(node_id), idx))
	
		if count == 10:
			count = 0
			reset_sim()
			loginfo("Simulation time reset!")		

		count += 1

		for i in range(1, num+1):
			state_num =  str(idx) + '_' + str(i).zfill(2) #e.g., 7351_02
			skin_name = "skin{0}.stl".format(state_num)
			marker_name = "marker{0}.stl".format(state_num)
			skin_xml = skin_xml[:238] + skin_name + skin_xml[253:]
			marker_xml = marker_xml[:238] + marker_name + marker_xml[255:]

			# Spawn skin and marker model
			spawn_model(skin_name, skin_xml, "", skin_pose, "world")
			time.sleep(2.0) # 2	
			spawn_model(marker_name, marker_xml, "", marker_pose, "world")
			time.sleep(8.0) # 8
			print("[{1}/{2}] - Contact Depth : {0}".format(state_num, i, num))

			# Save your OpenCV2 image as a jpeg
			image_name = "{0}.jpg".format(state_num)
			cv2.imwrite(os.path.join(up_path, image_name), img_processing.get_cvimage_up())
			cv2.imwrite(os.path.join(bot_path, image_name), img_processing.get_cvimage_bot())

			# Delete skin and marker model
			delete_model(skin_name)
			time.sleep(1) # 1
			delete_model(marker_name)
			time.sleep(2) # 2

	# finish = time.time()
	# runtime = finish-start
	# fps = num/runtime
	# duration = 1/fps
	# print("Runtime: {}".format(runtime))
	# print("Frame per second (fps): {}".format(fps))
	# print("Duration: {}".format(duration))
