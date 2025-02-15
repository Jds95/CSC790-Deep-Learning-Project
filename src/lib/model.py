import collections
import os
import io
import sys
import tarfile
import tempfile
import urllib
import imutils

import numpy as np
from PIL import Image
import cv2

class DeepLabModel(object):
	"""Class to load deeplab model and run inference."""
	
	INPUT_TENSOR_NAME = 'ImageTensor:0'
	OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
	INPUT_SIZE = 513

	def __init__(self, tarball_path):
		"""Creates and loads pretrained deeplab model."""
		self.graph = tf.Graph()
		
		graph_def = None
		
		print("extracting graph")
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
		# From cv2 to PIL
		cv2_im = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		image = Image.fromarray(cv2_im)
			
		width, height = image.size
		resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
		target_size = (int(resize_ratio * width), int(resize_ratio * height))
		resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
		batch_seg_map = self.sess.run(
			self.OUTPUT_TENSOR_NAME,
			feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
		seg_map = batch_seg_map[0]
		
		return seg_map