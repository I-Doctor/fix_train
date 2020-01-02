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
#cfg_file="0-4bit-linear-glevel4"
#cfg_file="00-4bit-linear-glevel4-block"
#cfg_file="10-4bit-linear-glevel4-block"
#cfg_file="20-4bit-linear-glevel4-block"
#cfg_file="30-4bit-linear-glevel4-block"
#cfg_file="00-4bit-linear-glevel4-block2"
#cfg_file="10-4bit-linear-glevel4-block2"
#cfg_file="20-4bit-linear-glevel4-block2"
cfg_file="30-4bit-linear-glevel4-block2"
#cfg_file="0-4bit-linear-glevel4-resume"
#cfg_file="0-4bit-linear-glevel4-evaluate"
#cfg_file="0-8bit-linear-glevel2-resume"
#cfg_file="0-8bit-linear-glevel2-evaluate"
cfg_path="../config/cifar10/fix_cfg/"
python main.py				        \
	${data_root}					\
	${cfg_path}${cfg_file}.yaml		\
	${output_path}					\
    --gpu 6                         \
	2>&1 | tee ${output_path}/${cfg_file}.log

