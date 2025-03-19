#!/bin/bash 

export PATH=$PWD:$PATH
export PYTHONPATH=$PWD:$(dirname $(dirname $PWD)):$(dirname $PWD):$(dirname $PWD)/Model:$PYTHONPATH

# source /home/lijunjie/.bashrc
# conda activate /mntnfs/lee_data1/lijunjie/anaconda3/envs/py2.1


stage=1
stop_stage=1


#model_name:
#TDSE : baseline 
#TDSE_SelfMem: context bank 
#TDSE_SpkEmbMem: spk bank
#TDSE_SelfMem_SpkEmbMem : context bank + spk bank 

#USEV_SelfMem


model_name=TDSE_SelfMem
video_path='/home/junjie/MeMo/src/UttLevel_src/output.mp4'
checkpoint='/home/junjie/MeMo/src/UttLevel_src/logs/TDSE_SelfMem/model_dict_best.pt'
sampling_rate=16000
max_audio_length=20 # the max audio length to avoid OOM 

gpu_id=1

#offline eval 
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    CUDA_VISIBLE_DEVICES="$gpu_id" \
    python evaluate_realdemo.py \
    --video_path $video_path \
    --checkpoint $checkpoint \
    --sampling_rate $sampling_rate \
    --model_name $model_name \
    --max_audio_length 20 \

fi 


#only be used to compute online audio_scores
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    

#online_evaluation param 
chunk_time=0.2
recep_field=2
initilization=2       # duration of initilization
clean_visual_init=1   # whether use clean visual as initilization 0:false 1:true
mem_bank_slot=1       # the number of slots in memory bank(minimum value is 1 )
mem_bank_update=0 # 0: first in first out  1: pop out one with lower attention weights
clean_visual=1 # whether the visual cues are avaialibe 1: clean visual 0: impaired visuals


    python evaluate_online.py --continue_from $log_name \
                        --visual_direc '/data/junjie/data/vox2/mp4/' \
                        --mix_lst_path '../../data_preparation/mixture_data_list_2mix_with_occludded.csv' \
                        --mixture_direc '/data/junjie/data/vox2/mixture/' \
                        --obj_dir '../../data_preparation/Asset/object_image_sr_test' \
                        --obj_mask_dir '../../data_preparation/Asset/object_mask_x4_test' \
                        --save_audio 0 \
                        --save_dir $log_name'/save_audio_online/'${clean_visual}/ \
                        --model_name $model_name \
                        --clean_visual_init $clean_visual_init \
                        --chunk_time $chunk_time \
                        --recep_field $recep_field \
                        --initilization $initilization \
                        --mem_bank_slot $mem_bank_slot \
                        --sample_rate 16000 \
                        --mem_bank_update $mem_bank_update \
                        --num_workers 6 \
                        --clean_visual $clean_visual \
    >> $log_name/online_sisnr_${chunk_time}_${recep_field}_${initilization}_${clean_visual_init}_${mem_bank_slot}_${mem_bank_update}_${clean_visual}.log


fi 

