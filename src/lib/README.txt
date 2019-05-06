##### INSTALLATION #####
1. Pip install all of the required packages in the src/requirements.txt file.
2. clone the tensorflow/models library.
	i.   cd src/lib 
	ii.  git clone https://github.com/tensorflow/models.git (recomend just downloading the zip)
	iii. rename the repo as models. 
	iv.  Move the models folder to the lib folder.
3. append the models to your python path.
	WINDOWS:
		i.   search: "Environmental Variables" click on the option to edit your environmental variables for the system.
		ii.  Add or edit the PYTHONPATH variable.
		iii. Add the following folders:
			a. C:\<your_path>\Python37;
			b. C:\<your_path>>\Python37\DLLs;
			c. C:\<your_path>\src\lib\models\research\slim;
			d. C:\<your_path>\src\lib\models\research\deeplab;

##### TEST YOUR DEEPLAB INSTALLATION #####

1. Test DeepLab v3+
	i.   open a terminal or cmd.
	ii.  cd models\research
	iii. execute: python deeplab\model_test.py (it should give quite a few warnings but the end should output OK.)


##### BUILD THE DATASET FOR TRAINING #####
1. place the sterio left and right videos into the folders for testing and training. (data\testing, data\training)
	i. The files must be labeled 'l.avi' or 'r.avi'.
2. place the class labels csv file into the training folder.
3. Edit the file process_video.py to include the number of frames for training.
	i. Edit line 21 with the number of frames to use for training. ###IMPORTANT### The video for testing and training need to be the exact same video frame for frame. 
4. cd out to the lib folder.
5. build the dataset.
	i.  open a command prompt.
	ii. execute: build_dataset.bat. (it should run with no problems if you followed the previous steps.)
6. In lib/data edit the file data_generator.py to include the new dataset.
	i. Edit lines 101-103 to have the number of files in the train, val, and train+val sets.
7. copy the data_generator.py to lib/models/research/deeplab/datasets
8. Setup the model.
	Either:
	i.  Download a model from: https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/model_zoo.md and place the contents into the ADA folder under a new folder name 'models'.
	ii. Move the files provided model folder in lib/data to lib/data/ADA/models and to models/research/deeplab/datasets/ADA/models

##### TRAINING #####
1. cd to lib
2. set the number of iterations in train.sh to greater than 5000
3. execute: train.bat (it should run with alot of output but no errors)
