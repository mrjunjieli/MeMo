#!/bin/sh

export PATH=$PWD:$PATH
export PYTHONPATH=$PWD:$(dirname $(dirname $PWD)):$(dirname $PWD):$(dirname $PWD)/Model:$PYTHONPATH
# source /home/junjie/.bashrc
# /home/anaconda3/bin/conda init bash 
# conda activate /home/junjie/.conda/envs/py2.1

#model_name:
#TDSE : baseline 
#TDSE_SelfMem: contextual bank
#TDSE_SpkEmbMem: speaker bank
#TDSE_SelfMem_SpkEmbMem contextual bank+speaker bank 

#USEV 
#USEV_SelfMem

#SEANET
#SEANET_SelfMem

model_name='SEANET_SelfMem'
gpu_id='0'
continue_from=
if [ -z ${continue_from} ]; then
	log_name='logs/'${model_name}/
	if [ -d $log_name ]; then
		echo "$log_name exists! "
		exit 1 
	else 
		mkdir -p $log_name
		continue_from=FALSE
	fi 
else
	log_name=${continue_from}
fi

num_gpus=$(echo $gpu_id | awk -F ',' '{print NF}')
CUDA_VISIBLE_DEVICES="$gpu_id" \
python -W ignore \
-m torch.distributed.launch \
--nproc_per_node=$num_gpus \
--master_port=3130 \
main.py \
\
--log_name $log_name \
\
--visual_direc '/home/data1/voxceleb2_video/voxceleb2/mp4/' \
--mix_lst_path '../../data_preparation/mixture_data_list_2mix_with_occludded.csv' \
--mixture_direc '/home/data1/voxceleb2_video/voxceleb2//mixture/' \
--obj_dir '../../data_preparation/Asset/object_image_sr' \
--obj_mask_dir '../../data_preparation/Asset/object_mask_x4' \
--C 2 \
--epochs 100 \
--max_length 6 \
--accu_grad 0 \
--batch_size 8 \
--num_workers 16 \
--use_tensorboard 1 \
--model_name $model_name \
--lr 1e-3 \
--shift_range '0,1' \
--Self_enroll_amplitude_scaling '0.1,1' \
--teacher_point 50 \
--num_slots '1,5' \
--loss_weight '28' \
--continue_from ${continue_from} \
>>$log_name/train.log 


