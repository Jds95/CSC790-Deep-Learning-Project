rem setup the paths
set CHECKPOINT_NAME=model.ckpt-942
set CHECKPOINT_PATH=%cd%\models\research\deeplab\datasets\ADA\exp\train_on_trainval_set\train\

python export_model.py ^
--checkpoint_path=%CHECKPOINT_PATH%\%CHECKPOINT_NAME% ^
--export_path=graphs\
--num_classes=4 ^
--atrous_rates=6 ^
--atrous_rates=12 ^
--atrous_rates=18 ^
--output_stride=16 ^