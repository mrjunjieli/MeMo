#!/bin/bash 

export PATH=$PWD:$PATH
export PYTHONPATH=$PWD:$(dirname $(dirname $PWD)):$(dirname $PWD):$(dirname $PWD)/Model:$PYTHONPATH

# source /home/lijunjie/.bashrc
# conda activate /mntnfs/lee_data1/lijunjie/anaconda3/envs/py2.1


stage=1
stop_stage=1

#model_name:
#TDSE : baseline 
#TDSE_SelfMem: contextual bank
#TDSE_SpkEmbMem: speaker bank
#TDSE_SelfMem_SpkEmbMem contextual bank+speaker bank 

#USEV 
#USEV_SelfMem

#SEANET
#SEANET_SelfMem

gpu_id=1

model_name=MoMuSE
# log_name=logs/$model_name
log_name=/home/junjie/MeMo/src/Model/MoMuSE

#online_evaluation param 
chunk_time=0.2
recep_field=2
initilization=2       # duration of initilization
clean_visual_init=1   # whether use clean visual as initilization 0:false 1:true
mem_bank_slot=1       # the number of slots in memory bank(minimum value is 1 )
mem_bank_update=0 # 0: first in first out  1: pop out one with lower attention weights
clean_visual=0 # whether the visual cues are avaialibe in the whole utterace 1: clean visual 0: impaired visuals


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # CUDA_VISIBLE_DEVICES="$gpu_id" \
# python evaluate_offline.py --continue_from $log_name \
    #                     --visual_direc '/data/junjie/data/vox2/mp4/' \
    #                     --mix_lst_path '../../data_preparation/mixture_data_list_2mix_with_occludded.csv' \
    #                     --mixture_direc '/data/junjie/data/vox2/mixture/' \
    #                     --obj_dir '../../data_preparation/Asset/object_image_sr_test' \
    #                     --obj_mask_dir '../../data_preparation/Asset/object_mask_x4_test' \
    #                     --save_audio 0 \
    #                     --save_dir ${log_name}'/save_audio_offline/' \
    #                     --sample_rate 16000 \
    #                     --model_name $model_name \
    #                     --num_workers 6 \
    #                     --clean_visual $clean_visual \
    # >> $log_name/offline_sisnr_${clean_visual}.log

    
    CUDA_VISIBLE_DEVICES="$gpu_id"  python evaluate_online.py --continue_from $log_name \
                        --visual_direc '/home/data1/voxceleb2_video/voxceleb2/mp4/' \
                        --mix_lst_path '../../data_preparation/mixture_data_list_2mix_with_occludded.csv' \
                        --mixture_direc '/home/data1/junjie/memo_test/' \
                        --obj_dir '../../data_preparation/Asset/object_image_sr_test' \
                        --obj_mask_dir '../../data_preparation/Asset/object_mask_x4_test' \
                        --save_audio 1 \
                        --save_dir $log_name'/save_audio_online/'${clean_visual}/ \
                        --model_name $model_name \
                        --clean_visual_init $clean_visual_init \
                        --chunk_time $chunk_time \
                        --recep_field $recep_field \
                        --initilization $initilization \
                        --mem_bank_slot $mem_bank_slot \
                        --sample_rate 16000 \
                        --mem_bank_update $mem_bank_update \
                        --num_workers 8 \
                        --clean_visual $clean_visual \
    >> $log_name/online_sisnr_${chunk_time}_${recep_field}_${initilization}_${clean_visual_init}_${mem_bank_slot}_${mem_bank_update}_${clean_visual}.log

fi 


#only be used to compute online audio_scores
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    use_pesq=true
    use_sdr=true 
    use_stoi=true 
    use_dnsmos=true
    if [ $use_pesq = true ] || [ $use_sdr = true ] || [ $use_stoi = true ] || [ $use_dnsmos = true ]; then 
        python3 tools/compute_pesq_online.py --sisnr_log $log_name/online_sisnr_${chunk_time}_${recep_field}_${initilization}_${clean_visual_init}_${mem_bank_slot}_${mem_bank_update}_${clean_visual}.log \
                                --audio_dir $log_name'/save_audio_online/'${clean_visual}/  \
                                --use_pesq $use_pesq \
                                --use_stoi $use_stoi \
                                --use_sdr $use_sdr \
                                --use_dnsmos $use_dnsmos \
                                --save_path $log_name/pesq_sdr_stoi_dns_${chunk_time}_${recep_field}_${initilization}_${clean_visual_init}_${mem_bank_slot}_${mem_bank_update}_${clean_visual}.log
    fi

fi 


# compute performance per mask_ratio; per mask type 
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    mkdir -p  $log_name/score/
    # python3 tools/score_offline.py --sisnr_log "$log_name/offline_sisnr_${clean_visual}.log" \
    #                             --sisnr_score_out_path $log_name/score/offline_sisnr_${clean_visual}.score 

    python3 tools/score_online.py --sisnr_log $log_name/online_sisnr_${chunk_time}_${recep_field}_${initilization}_${clean_visual_init}_${mem_bank_slot}_${mem_bank_update}_${clean_visual}.log \
                                --pesq_log $log_name/pesq_sdr_stoi_dns_${chunk_time}_${recep_field}_${initilization}_${clean_visual_init}_${mem_bank_slot}_${mem_bank_update}_${clean_visual}.log \
                                --sisnr_score_out_path $log_name/score/online_sisnr_${chunk_time}_${recep_field}_${initilization}_${clean_visual_init}_${mem_bank_slot}_${mem_bank_update}_${clean_visual}.score \
                                --pesq_score_out_path $log_name/score/pesq_${chunk_time}_${recep_field}_${initilization}_${clean_visual_init}_${mem_bank_slot}_${mem_bank_update}_${clean_visual}.score \

fi 



# compute performance per mask_ratio; per mask type 
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    python3 tools/cal_flops_rtf.py --model_name $model_name \
                                    --chunk_time $chunk_time \
                                    --recep_field $recep_field \
                                    --sample_rate 16000 \
                                    --initilization 0 \

fi 
