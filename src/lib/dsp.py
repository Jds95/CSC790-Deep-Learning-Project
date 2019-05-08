import numpy as np
import cv2

class StereoDepth():

	def __init__(self, focal_length = 100, cam_dist = 10):
		#Depth alg variables
		lmbda = 80000
		sigma = 1.2
		visual_multiplier = 1.0
		window_size = 3
		self.focal_length = focal_length
		self.cam_dist = cam_dist
		 
		self.lMatcher = cv2.StereoSGBM_create(
			minDisparity = 0,
			numDisparities = 160,
			blockSize = 5,
			P1 = 8 * 3 * window_size ** 2,
			P2 = 32 * 3 * window_size ** 2,
			disp12MaxDiff = 1,
			uniquenessRatio = 15,
			speckleWindowSize = 0,
			speckleRange = 2,
			preFilterCap = 63,
			mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
		)

		self.rMatcher = cv2.ximgproc.createRightMatcher(self.lMatcher)

		self.wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=self.lMatcher)
		self.wls_filter.setLambda(lmbda)
		self.wls_filter.setSigmaColor(sigma)
		
	def calculate_disparity(self, lFrame, rFrame):
		lGray = cv2.cvtColor(lFrame, cv2.COLOR_BGR2GRAY)
		rGray = cv2.cvtColor(rFrame, cv2.COLOR_BGR2GRAY)
		dispL = self.lMatcher.compute(lGray, rGray)
		dispR = self.rMatcher.compute(rGray, lGray)
		dispL = np.int16(dispL)
		dispR = np.int16(dispR)
		d_map = self.wls_filter.filter(dispL, lFrame, None, dispR)
		
		#reformat d_map for opencv
		d_map = cv2.normalize(src=d_map, dst=d_map, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
		d_map = np.uint8(d_map[:, 160:])
		d_map = cv2.resize(d_map, (400, 400), interpolation = cv2.INTER_AREA)
		return d_map
	
	def apply_color_map(self, d_map, color_map):
		pass

if(__name__=="__main__"):
	dProcessor = StereoDepth()
	lReader = cv2.VideoCapture('l.avi')
	rReader = cv2.VideoCapture('r.avi')
	ret1 = True
	ret2 = True
	while(ret1 and ret2):
		ret1, lFrame = lReader.read()
		ret2, rFrame = rReader.read()
		
		if(ret1 and ret2):
			#calculate disparity
			d_map = dProcessor.calculate_disparity(lFrame, rFrame)
			
			#Show the frame for demo purposes
			raw_frame = np.hstack([lFrame, rFrame])
			cv2.imshow('input', raw_frame)
			cv2.imshow('depth map', d_map)
			key = cv2.waitKey(40)
			if(key == ord('q')):
				break

	lReader.release()
	rReader.release()
	cv2.destroyAllWindows()