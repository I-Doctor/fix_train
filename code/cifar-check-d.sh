#!/usr/bin/env sh

#--------------------set global param------------#
data_root="/home/eva_share/datasets/"
current_path=$(pwd)
checkpoint_path="../checkpoint/fix_experiment/cifar"


#--------------------get date--------------------#
time=`date +"%Y%m%d_%H-%M-%S"`
echo "Processing at time: ${time}"

#--------------------mkdir-----------------------#
output_dir_name="log_quantize_check-d"
output_path="${checkpoint_path}/${output_dir_name}"
echo "Creating output dir: ${output_path}"
mkdir -p ${output_path}

#--------------------run python------------------#
cfg_file="d-cifar-check"
cfg_path="../config/cifar10/fix_cfg/"
python -u main.py			        \
	${data_root}					\
	${cfg_path}${cfg_file}.yaml		\
	${output_path}					\
	2>&1 | tee ${output_path}/${cfg_file}.log

