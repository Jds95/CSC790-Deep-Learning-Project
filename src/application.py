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
from lib.synth import Stitcher
from lib.dsp import StereoDisparity
from lib.model import DeepLabModel
from lib.seg import StereoZip

import tensorflow as tf

## Load model in TensorFlow
TAR_NAME = "ADA_MODEL_1.tar.gz"
GRAPH_NAME = 'frozen_inference_graph'
GRAPH_PATH = "lib/graphs/"
VIDEO_PATH = "lib/data/testing/"
LEFT_VIDEO = "l.avi"
RIGHT_VIDEO = "r.avi"

print("imports done setting up model..")


model = DeepLabModel(GRAPH_PATH+TAR_NAME)
synth = ViewSynthisizer()
dProcessor = StereoDisparity()
classes = {
			0 : (0,0,0),
			1 : (0,0,255),
			2 : (0,255,255),
			3 : (0,255,0)
}
seg = StereoZip(classes)
print("model built. starting main run cycle")

if(__name__=="__main__"):
	#=============================================================
	#CHANGE THIS PART TO SWITCH TO LIVE VIDEO FEEDS FROM ROBOT
	lReader = cv2.VideoCapture(VIDEO_PATH+LEFT_VIDEO)
	rReader = cv2.VideoCapture(VIDEO_PATH+RIGHT_VIDEO)
	#=============================================================

	ret1 = True
	ret2 = True
	while(ret1 and ret2):
		#=============================================================
		#CHANGE THIS PART TO SWITCH TO LIVE VIDEO FEEDS FROM ROBOT
		ret1, lFrame = lReader.read()
		ret2, rFrame = rReader.read()
		#=============================================================
		
		if(ret1 and ret2):
			#STEP-01: Get pan_image
			pan_frame = lFrame[:]#stitcher.stitch([lFrame, rFrame])
			
			#STEP-02: Get depthmap
			d_map = dProcessor.calculate_disparity(lFrame, rFrame)
			
			#STEP-03: Run model to get segmentation map
			seg_map = model.run(pan_frame)
			
			#STEP-04: zip seg_map and d_map togeather
			seg_d_map = seg.zip(seg_map, d_map)
			
			#STEP-05: OPTIONAL - Use seg map to apply color segmentation to d_map
			3DSeg_map = seg.apply_colormap(seg_map, d_map)
			
			#==============DEMO PURPOSES ONLY==================
			#display output
			og_frames = np.hstack([lFrame, rFrame])
			color_map = seg.get_colormap(seg_map)
			cv2.imshow('Original Frames', og_frames)
			cv2.imshow('Depth Map', d_map)
			cv2.imshow('Color Map', color_map)
			cv2.imshow('3D Segmentation', 3DSeg_map)
			key = cv2.waitKey(10)
			if(key == ord('q')):
				break