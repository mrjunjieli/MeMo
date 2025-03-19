#!/bin/bash

direc=/home/data1/voxceleb2_video/voxceleb2/ # Your folder of VoxCeleb2 
data_direc_mp4=${direc}/mp4/  #video folder of VoxCeleb2
audio_data_direc=${direc}/wav/ # Audio folder of VoxCeleb2 
sampling_rate=16000
vox2_lst=./vox2_wav.lst


# echo 'stage 1: check whether there are broken videos'
# python "5_check_is_video_avaliable.py" \
# --data_direc_mp4 $data_direc_mp4 \
# --num_threads 10 \
# --output_file './broken_file.txt'

# #broken_file.txt is used to save broken_video path.
# #you need to remove these videos, or the training may be broken

# echo 'stage 2: extract audio from video' 
# python 'extract_wav_from_mp4.py' \
# --data_direc_mp4 $data_direc_mp4 \
# --audio_data_direc $audio_data_direc \
# --sampling_rate $sampling_rate


echo 'stage 3: generate dynamic_training data list'
> "$vox2_lst"
find "$audio_data_direc"/train/ -type f -name "*.wav" -print | while read filepath; do
    echo "$filepath" >> "$vox2_lst"
done

