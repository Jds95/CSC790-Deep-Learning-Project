echo off
cd models\research\deeplab
cd ..
rem Set up the working environment.
set CURRENT_DIR=%cd%
set WORK_DIR=%CURRENT_DIR%\deeplab
set DATASET_DIR=datasets

rem Set up the working directories.
set ADA_FOLDER=ADA
set EXP_FOLDER=exp\train_on_trainval_set
set INIT_FOLDER=%WORK_DIR%\%DATASET_DIR%\%ADA_FOLDER%\%EXP_FOLDER%\init_models
set TRAIN_LOGDIR=%WORK_DIR%\%DATASET_DIR%\%ADA_FOLDER%\%EXP_FOLDER%\train
set DATASET=%WORK_DIR%\%DATASET_DIR%\%ADA_FOLDER%\tfrecord"

md %WORK_DIR%\%DATASET_DIR%\%ADA_FOLDER%\exp
md %TRAIN_LOGDIR%

set NUM_ITERATIONS=1000
python %WORK_DIR%\train.py ^
--logtostderr ^
--train_split=train ^
--model_variant=xception_65 ^
--atrous_rates=6 ^
--atrous_rates=12 ^
--atrous_rates=18 ^
--output_stride=16 ^
--decoder_output_stride=4 ^
--train_crop_size=513 ^
--train_crop_size=513 ^
--train_batch_size=4 ^
--training_number_of_steps=%NUM_ITERATIONS% ^
--fine_tune_batch_norm=True ^
--dataset="ADA" ^
--initialize_last_layer=False ^
--last_layers_contain_logits_only=True ^
--tf_initial_checkpoint=%WORK_DIR%\%DATASET_DIR%\%ADA_FOLDER%\models\model.ckpt ^
--train_logdir=%TRAIN_LOGDIR% ^
--dataset_dir=%DATASET%