echo off
cd models\research\deeplab
cd ..
rem Set up the working environment.
set CURRENT_DIR=%cd%
set WORK_DIR=%CURRENT_DIR%\deeplab

rem setup the paths
set CHECKPOINT_NAME=model.ckpt-951

set CHECKPOINT_PATH=%WORK_DIR%\datasets\ADA\exp\train_on_trainval_set\train


python %WORK_DIR%\export_model.py ^
--checkpoint_path=%CHECKPOINT_PATH%\%CHECKPOINT_NAME% ^
--export_path=graphs\
--num_classes=4 ^
--atrous_rates=6 ^
--atrous_rates=12 ^
--atrous_rates=18 ^
--output_stride=16 ^