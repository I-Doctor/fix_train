#!/usr/bin/env sh

#--------------------set global param------------#
data_root="/home/eva_share/datasets/"
current_path=$(pwd)
checkpoint_path="../checkpoint/fix_experiment/cifar"


#--------------------get date--------------------#
time=`date +"%Y%m%d_%H-%M-%S"`
echo "Processing at time: ${time}"

#--------------------mkdir-----------------------#
output_dir_name="log_quantize_${time}"
output_path="${checkpoint_path}/${output_dir_name}"
echo "Creating output dir: ${output_path}"
mkdir -p ${output_path}

#--------------------run python------------------#
<<<<<<< HEAD:code/qrun.sh
cfg_file="70-4bit-linear-glevel4-block"
=======
cfg_file="00-4bit-linear-glevel4-block"
>>>>>>> 3289f02983dbce3279d1cf3f2453dfc7575b6f75:code/cifar_run.sh
cfg_path="../config/cifar10/fix_cfg/"
python -u main.py				        \
	${data_root}					\
	${cfg_path}${cfg_file}.yaml		\
	${output_path}					\
<<<<<<< HEAD:code/qrun.sh
    --gpu 7                         \
=======
    --gpu 0                         \
>>>>>>> 3289f02983dbce3279d1cf3f2453dfc7575b6f75:code/cifar_run.sh
	2>&1 | tee ${output_path}/${cfg_file}.log

