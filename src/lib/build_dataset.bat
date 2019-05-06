echo off
echo Building ADA file structure..
rem build the file structure
set ROOT=%cd%
cd data
md ADA
cd ADA
md dataset
md tfrecord
md models
cd dataset
md ImageSets
md JPEGImages
md SegmentationClass
md SegmentationClassRaw
cd ..
cd ..
echo File structure built.
pause

rem process the videos and build dataset
echo Processing videos...
python process_video.py

rem setup the folders
set ADA_ROOT=%cd%\ADA
set DATASET_FOLDER=%ADA_ROOT%\dataset
set SEG_FOLDER=%DATASET_FOLDER%\SegmentationClass
set SEMANTIC_SEG_FOLDER=%DATASET_FOLDER%\SegmentationClassRaw
set IMAGE_FOLDER=%DATASET_FOLDER%\JPEGImages
set LIST_FOLDER=%DATASET_FOLDER%\ImageSets
set OUTPUT_DIR=%ADA_ROOT%\tfrecord

rem Build TFRecords of the dataset.
python build_tf_record.py ^
--image_folder "%IMAGE_FOLDER%" ^
--semantic_segmentation_folder "%SEMANTIC_SEG_FOLDER%" ^
--list_folder "%LIST_FOLDER%" ^
--image_format "jpg" ^
--output_dir "%OUTPUT_DIR%"
echo Videos Processed.
pause

rem copy dataset to deeplab
echo Importing dataset to deeplab...
cd ROOT
cd models\research\deeplab\datasets
md ADA
cd ROOT
robocopy %ROOT%\data\ADA %ROOT%\models\research\deeplab\datasets/ADA /COPYALL /E