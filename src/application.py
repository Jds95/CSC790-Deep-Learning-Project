import collections
import os
import io
import sys
import tarfile
import tempfile
import urllib
import imutils

from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import cv2
from lib.data.panorama import Stitcher

import tensorflow as tf
import get_dataset_colormap

## Load model in TensorFlow
TAR_NAME = "ADA_MODEL_1.tar.gz"
GRAPH_NAME = 'ADA_FINF_GRAPH'
GRAPH_PATH = "lib/graphs/"
VIDEO_PATH = "lib/data/testing/"
LEFT_VIDEO = "l.avi"
RIGHT_VIDEO = "r.avi"


class DeepLabModel(object):
	"""Class to load deeplab model and run inference."""
	
	INPUT_TENSOR_NAME = 'ImageTensor:0'
	OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
	INPUT_SIZE = 513

	def __init__(self, tarball_path):
		"""Creates and loads pretrained deeplab model."""
		self.graph = tf.Graph()
		
		graph_def = None
		# Extract frozen graph from tar archive.
		tar_file = tarfile.open(tarball_path)
		for tar_info in tar_file.getmembers():
			if GRAPH_NAME in os.path.basename(tar_info.name):
				file_handle = tar_file.extractfile(tar_info)
				graph_def = tf.GraphDef.FromString(file_handle.read())
				break

		tar_file.close()
		
		if graph_def is None:
			raise RuntimeError('Cannot find inference graph in tar archive.')

		with self.graph.as_default():      
			tf.import_graph_def(graph_def, name='')
		
		self.sess = tf.Session(graph=self.graph)
			
	def run(self, image):
		"""Runs inference on a single image.
		
		Args:
			image: A PIL.Image object, raw input image.
			
		Returns:
			resized_image: RGB image resized from original input image.
			seg_map: Segmentation map of `resized_image`.
		"""
		width, height = image.size
		resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
		target_size = (int(resize_ratio * width), int(resize_ratio * height))
		resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
		batch_seg_map = self.sess.run(
			self.OUTPUT_TENSOR_NAME,
			feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
		seg_map = batch_seg_map[0]
		return resized_image, seg_map

model = DeepLabModel(GRAPH_PATH+TAR_NAME)


if(__name__=="__main__"):
	#=============================================================
	#CHANGE THIS PART TO SWITCH TO LIVE VIDEO FEEDS FROM ROBOT
	lReader = cv2.VideoCapture(VIDEO_PATH+LEFT_VIDEO)
	rReader = cv2.VideoCapture(VIDEO_PATH+RIGHT_VIDEO)
	stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
	#=============================================================

	ret1, ret2 = True
	while(ret1 and ret2):
		ret1, lFrame = lReader.read()
		ret2, rFrame = rReader.read()
		
		if(ret1 and ret2):
			#get pan image and depth
			pan_frame = lFrame[:]#stitcher.stitch([lFrame, rFrame])
			disparity = stereo.compute(lFrame,rFrame)
			
			# From cv2 to PIL
			cv2_im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			pil_im = Image.fromarray(cv2_im)
			
			# Run model
			resized_im, seg_map = model.run(pil_im)
			
			# Adjust color of mask
			seg_image = get_dataset_colormap.label_to_color_image(
				seg_map, get_dataset_colormap.get_pascal_name()).astype(np.uint8)
			
			# Convert PIL image back to cv2 and resize
			frame = np.array(pil_im)
			r = seg_image.shape[1] / frame.shape[1]
			dim = (int(frame.shape[0] * r), seg_image.shape[1])[::-1]
			resized = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
			resized = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
			
			# Stack horizontally color frame and mask
			color_and_mask = np.hstack((resized, seg_image))

			cv2.imshow('frame', color_and_mask)
			cv2.imshow('depth mask', disparity)
			if cv2.waitKey(25) & 0xFF == ord('q'):
				cap.release()
				cv2.destroyAllWindows()
				break