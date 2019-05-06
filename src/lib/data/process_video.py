"""
This file builds the dataset for the deeplab semantic segmentation nerual network.
Programmed by: Jared Hall

===IMPORTANT====
Only run this file after doing the following steps:
1. Run the "build_file_structure.bat file.
2. Place your test videos in the testing folder (the plane videos).
2. Place your color coded videos in the training folder.
================= 
"""
import cv2
import csv
from tqdm import tqdm
from PIL import Image
import numpy as np
from panorama import Stitcher

#Dataset options
PATH_TO_TRANING = "training/" #path to your training folder
PATH_TO_TESTING = "testing/"  #path to your testing folder
NUM_FRAMES = 1000             #Number of frames to use for training and testing
LEFT_VIDEO = "l.avi"          #Filename for the sterio left video
RIGHT_VIDEO = "r.avi"         #filename to use for the sterio right video
HEADER = "ADA"                #header for the output images i.e. <header>_1.jpg


def get_segment_frame(img, pallette, boundries):
	
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	
	#Get the green segment
	gMask =  cv2.inRange(hsv, boundries["green"][0], boundries["green"][1])
	gMask = cv2.morphologyEx(gMask, cv2.MORPH_OPEN, (3,3), iterations=3)
	gMask = cv2.morphologyEx(gMask, cv2.MORPH_CLOSE, (3,3), iterations=3)
	blur = cv2.GaussianBlur(gMask,(3,3),0)

	#Get the pink segment
	pkMask = cv2.inRange(hsv, boundries["pink"][0], boundries["pink"][1])
	pkMask = cv2.morphologyEx(pkMask, cv2.MORPH_OPEN, (3,3), iterations=3)
	pkMask = cv2.morphologyEx(pkMask, cv2.MORPH_CLOSE, (3,3), iterations=3)
	blur = cv2.GaussianBlur(pkMask,(3,3),0)
	
	#get the purple segment
	prMask = cv2.inRange(hsv, boundries["purple"][0], boundries["purple"][1])
	prMask = cv2.morphologyEx(prMask, cv2.MORPH_OPEN, (3,3), iterations=3)
	prMask = cv2.morphologyEx(prMask, cv2.MORPH_CLOSE, (3,3), iterations=3)
	blur = cv2.GaussianBlur(prMask,(3,3),0)

	kernel = (5, 5)
	bkg = gMask+pkMask+prMask
	blur = cv2.GaussianBlur(bkg, kernel,0)
	ret3,bkg = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	cv2.waitKey(10)

	img[bkg==0] = pallette[0]
	img[prMask>0] = pallette[3]
	img[gMask>0] = pallette[1]
	img[pkMask>0] = pallette[2]
	
	return img

def get_color_palette():
	#TODO: Generalize this function to base all outputs from the csv file cLabels
	infile = open('training/cLabels.csv')
	reader = csv.reader(infile)
	next(reader)
	low = next(reader)
	mid = next(reader)
	high = next(reader)
	infile.close()

	classes = { 
				(0,   0, 255) : 1, #low mobility
				(0, 255, 255) : 2, #mid mobility
				(0,   255, 0) : 3  #High Mobility
			  }

	palette = [
				(0,   0,   0), #background color
				(0,   0, 255), #low mobility
				(0, 255, 255), #mid mobility
				(0, 255,   0)  #High Mobility
			  ]

	#HSV boundries for gazebo segment 
	bound = {
				 "green":  (np.array([25,  75, 44], dtype="uint8"), np.array([ 55, 255, 255], dtype="uint8")), #BGR
				 "pink":   (np.array([145, 190, 88], dtype="uint8"), np.array([255, 255, 255], dtype="uint8")),
				 "purple": (np.array([130, 50, 80], dtype="uint8"), np.array([155, 125, 255], dtype="uint8")),
				}
	return classes, palette, bound
	
def get_ground_truth_annotation(frame, classes):
	ground_truth = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)

	for color, label in classes.items():
		mask = cv2.inRange(frame, (color[0]-10, color[1]-10, color[2]-10), (color[0]+10, color[1]+10, color[2]+10))
		ground_truth[mask>0] = label
	return ground_truth
	
#build our dataset from a video file
def build_train_dataset(dir, header):
	lReader = cv2.VideoCapture(dir+LEFT_VIDEO)
	rReader = cv2.VideoCapture(dir+RIGHT_VIDEO)
	stitcher = Stitcher()
	classes, pal, bound = get_color_palette()

	print("Processing the first ", NUM_FRAMES, " frames from the given video which has ", int(lReader.get(cv2.CAP_PROP_FRAME_COUNT)), " frames.")
	for i in tqdm(range(NUM_FRAMES)):
		condition,  lFrame = lReader.read()
		condition2, rFrame = rReader.read()
		if(condition and condition2):
			#STEP-01: Get panoramic image
			pan_frame = stitcher.stitch([lFrame, rFrame])
			
			#STEP-02: Build and write segmented image
			seg_frame = get_segment_frame(pan_frame, pal, bound)
			label = "ADA/dataset/SegmentationClass/"+header+"_"+str(i)+".jpg"
			cv2.imwrite(label, seg_frame)
			
			#STEP-03: Build and write annotated ground truth
			gt_frame = get_ground_truth_annotation(seg_frame, classes)
			label = "ADA/dataset/SegmentationClassRaw/"+header+"_"+str(i)+'.png'
			Image.fromarray(gt_frame).save(label, 'PNG')

	lReader.release()
	rReader.release()
	print()
	
def build_test_dataset(dir, header, labels):
	lReader = cv2.VideoCapture(dir+LEFT_VIDEO)
	rReader = cv2.VideoCapture(dir+RIGHT_VIDEO)
	stitcher = Stitcher()
	print("Processing the first ", NUM_FRAMES, " frames from the given video which has ", int(lReader.get(cv2.CAP_PROP_FRAME_COUNT)), " frames.")

	for i in tqdm(range(NUM_FRAMES)):
		condition,  lFrame = lReader.read()
		condition2, rFrame = rReader.read()
		if(condition and condition2):
			#STEP-01: Get panoramic image
			pan_frame = stitcher.stitch([lFrame, rFrame])
			
			#STEP-02: Write image to file
			label = header+"_"+str(i)
			labels.append(label)
			cv2.imwrite("ADA/dataset/JPEGImages/"+label+".jpg", pan_frame)
	lReader.release()
	rReader.release()
	print()
	return labels
	
def build_image_lists(labels):
	print("Building image lists...")
	trFile = open("ADA/dataset/ImageSets/train.txt", 'w')
	vlFile = open("ADA/dataset/ImageSets/val.txt", 'w')
	trvlFile = open("ADA/dataset/ImageSets/trainval.txt", 'w')
	for i in tqdm(range(len(labels))):
		label = labels[i]+"\n"
		if(i%5==0):
			vlFile.write(label)
			trvlFile.write(label)
		else:
			trFile.write(label)
			trvlFile.write(label)
	print()
	

if(__name__=="__main__"):
	labels = []
	#build test dataset
	print("build the testing dataset...")
	labels = build_test_dataset(PATH_TO_TESTING, 'ADA', labels)
	
	#Build the training dataset
	print("Translating training videos...")
	build_train_dataset(PATH_TO_TRANING, "ADA")
	
	#build the image lists (train, val, train+val
	build_image_lists(labels)