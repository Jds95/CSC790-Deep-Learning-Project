import numpy as np
import cv2

# import the necessary packages
import numpy as np
import imutils
import cv2

class ViewSynthisizer:
	def __init__(self):
		# determine if we are using OpenCV v3.X and initialize the
		# cached homography matrix
		self.isv3 = imutils.is_cv3(or_better=True)
		self.cachedH = None

	def stitch(self, images, ratio=0.75, reprojThresh=4.0):
		# unpack the images
		(imageB, imageA) = images

		# if the cached homography matrix is None, then we need to
		# apply keypoint matching to construct it
		if self.cachedH is None:
			# detect keypoints and extract
			(kpsA, featuresA) = self.detectAndDescribe(imageA)
			(kpsB, featuresB) = self.detectAndDescribe(imageB)

			# match features between the two images
			M = self.matchKeypoints(kpsA, kpsB,
				featuresA, featuresB, ratio, reprojThresh)

			# if the match is None, then there aren't enough matched
			# keypoints to create a panorama
			if M is None:
				return None

			# cache the homography matrix
			self.cachedH = M[1]

		# apply a perspective transform to stitch the images together
		# using the cached homography matrix
		result = cv2.warpPerspective(imageA, self.cachedH,
			(imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
		result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB

		# return the stitched image
		return result

	def detectAndDescribe(self, image):
		# convert the image to grayscale
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		# detect and extract features from the image
		descriptor = cv2.ORB_create()
		(kps, features) = descriptor.detectAndCompute(image, None)

		# convert the keypoints from KeyPoint objects to NumPy
		# arrays
		kps = np.float32([kp.pt for kp in kps])

		# return a tuple of keypoints and features
		return (kps, features)

	def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB,
		ratio, reprojThresh):
		# compute the raw matches and initialize the list of actual
		# matches
		matcher = cv2.DescriptorMatcher_create("BruteForce")
		rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
		matches = []

		# loop over the raw matches
		for m in rawMatches:
			# ensure the distance is within a certain ratio of each
			# other (i.e. Lowe's ratio test)
			if len(m) == 2 and m[0].distance < m[1].distance * ratio:
				matches.append((m[0].trainIdx, m[0].queryIdx))

		# computing a homography requires at least 4 matches
		if len(matches) > 4:
			# construct the two sets of points
			ptsA = np.float32([kpsA[i] for (_, i) in matches])
			ptsB = np.float32([kpsB[i] for (i, _) in matches])

			# compute the homography between the two sets of points
			(H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
				reprojThresh)

			# return the matches along with the homograpy matrix
			# and status of each matched point
			return (matches, H, status)

		# otherwise, no homograpy could be computed
		return None


if(__name__=="__main__"):
	synth = ViewSynthisizer()
	lReader = cv2.VideoCapture('l.avi')
	rReader = cv2.VideoCapture('r.avi')
	ret1 = True
	ret2 = True
	while(ret1 and ret2):
		ret1, lFrame = lReader.read()
		ret2, rFrame = rReader.read()
		
		if(ret1 and ret2):
			#build the stitched image
			pan_frame = synth.stitch([lFrame, rFrame])
			
			#Show the frame for demo purposes
			og = np.hstack([lFrame, rFrame])
			cv2.imshow('original stereo input', og)
			cv2.imshow('synth output', pan_frame)
			key = cv2.waitKey(40)
			if(key == ord('q')):
				break

	lReader.release()
	rReader.release()
	cv2.destroyAllWindows()
