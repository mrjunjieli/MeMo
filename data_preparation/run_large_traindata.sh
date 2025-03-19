#!/bin/bash 


#path 
direc=/home/data1/voxceleb2_video/voxceleb2/ # Your folder of VoxCeleb2 
data_direc_mp4=${direc}/mp4/  #video folder of VoxCeleb2
audio_data_direc=${direc}/wav/ # Audio folder of VoxCeleb2 
mixture_audio_direc=$direc/mixture_large/ # Audio mixture saved directory


train_samples=80000 # no. of train mixture samples simulated
C=2 # only 2 here 
mix_db=10 # random db ratio from -10 to 10db
sampling_rate=16000 # audio sampling rate
min_length=4 # minimum length of audio. Audios will be removed if less than min_length
max_length=6 # maxmun length of mixture. Audios will be cutted if longer than max_length.  Only used for train and dev


mixture_data_list=./mixture_data_list_${C}mix_large_data.csv #mixture datalist
mixture_data_list_with_occludded=./mixture_data_list_${C}mix_with_occludded_large_data.csv

mixture_dev_test_list=./mixture_data_list_${C}mix.csv #mixture datalist
mixture_dev_test_list_with_occludded=./mixture_data_list_${C}mix_with_occludded.csv

# # stage 1: Remove repeated datas in pretrain and train set, extract audio from mp4, create mixture list
if [ ! -f $mixture_data_list ];then
    echo 'stage 1: create mixture list'
    python 1_create_mixture_list_large.py \
    --data_direc_mp4 $data_direc_mp4 \
    --C $C \
    --mix_db $mix_db \
    --train_samples $train_samples \
    --audio_data_direc $audio_data_direc \
    --min_length $min_length \
    --sampling_rate $sampling_rate \
    --mixture_data_list $mixture_data_list \
    --max_length $max_length
    else
    echo $mixture_data_list' exist!'
fi


# stage 2: create audio mixture from list
echo 'stage 2: create mixture audios'
python 2_create_mixture.py \
--C $C \
--audio_data_direc $audio_data_direc \
--mixture_audio_direc $mixture_audio_direc \
--mixture_data_list $mixture_data_list \
--sampling_rate $sampling_rate \
--max_length $max_length # only used for train and dev 



# stage 3: create low_quality visual list 
echo 'stage 3: create low_quality visual list '
if [ ! -f $mixture_data_list_with_occludded ];then
python 3_create_LowQuality_visual_list.py \
--video_data_direc $data_direc_mp4 \
--mix_data_list $mixture_data_list \
--mixture_data_list_with_occludded $mixture_data_list_with_occludded

# Extract lines starting with 'val' and append them to mix_largetrain.csv
grep '^val' "$mixture_dev_test_list" >> "$mixture_data_list"
grep '^test' "$mixture_dev_test_list" >> "$mixture_data_list"


grep '^val' "$mixture_dev_test_list_with_occludded" >> "$mixture_data_list_with_occludded"
grep '^test' "$mixture_dev_test_list_with_occludded" >> "$mixture_data_list_with_occludded"
else 
    echo $mixture_data_list_with_occludded' exist!'
fi 


