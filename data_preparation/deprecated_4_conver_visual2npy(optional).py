# Copyright (c) 2021 Mobvoi Inc. (authors: Binbin Zhang
#               2023    SRIBD              Shuai Wang )
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import io
import logging
import multiprocessing
import os
import random
import tarfile
import time
import sys
import numpy as np 
import cv2 as cv 
import pdb 


def write_tar_file(data_list, output_dir, index=0, total=1,output_folder_2=None):
    for item in data_list:
        visual_paths = item 
        try:
            for visual_path in visual_paths:
                visual_file_name = visual_path[:-3].split('/')[-4:]
                visual_file_name = '/'.join(visual_file_name)
                visual_file_name = visual_file_name+'npy'
                visual_file_name = output_dir+'/'+visual_file_name
                if os.path.exists(visual_file_name):
                    continue
                else: 
                    visual_file_name = visual_path[:-3].split('/')[-4:]
                    visual_file_name = '/'.join(visual_file_name)
                    visual_file_name = visual_file_name+'npy'
                    visual_file_name = output_folder_2+'/'+visual_file_name
                    if os.path.exists(visual_file_name):
                        continue
                os.makedirs(os.path.dirname(visual_file_name),exist_ok=True)
                captureObj = cv.VideoCapture(visual_path)
                roiSequence=[]
                roiSize=112
                while captureObj.isOpened():
                    ret, frame = captureObj.read()
                    if ret == True:
                        grayed = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                        grayed = grayed / 255
                        grayed = cv.resize(grayed, (roiSize * 2, roiSize * 2))
                        roi = grayed[
                            int(roiSize - (roiSize / 2)) : int(
                                roiSize + (roiSize / 2)
                            ),
                            int(roiSize - (roiSize / 2)) : int(
                                roiSize + (roiSize / 2)
                            ),]
                        roiSequence.append(roi)
                    else:
                        break
                visual_data = np.asarray(roiSequence)
                np.save(visual_file_name,visual_data)
        except FileNotFoundError as e:
            print('visual',e)
            sys.exit()
            


def get_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--num_utts_per_shard',
                        type=int,
                        default=1000,
                        help='num utts per shard')
    parser.add_argument('--num_threads',
                        type=int,
                        default=1,
                        help='num threads for make shards')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--shuffle',
                        action='store_true',
                        help='whether to shuffle data')
    parser.add_argument('--video_data_direc')
    parser.add_argument('--mixture_direc')
    parser.add_argument('--mix_data_list')
    parser.add_argument('--output_dir', help='output npy dir')
    parser.add_argument('--output_folder_2', help='output npy dir')
    parser.add_argument('--mix_spk',
                        type=int,
                        default=2)

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    random.seed(args.seed)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')

    data=[]
    with open(args.mix_data_list, 'r', encoding='utf8') as fin:
        for line in fin:
            arr = line.strip().split()
            path_line = line.split(",")[0:-6]
            partition = line.split(',')[0]
            if partition=='test':
                continue
            visual_paths=[]
            for c in range(args.mix_spk):
                # read target audio
                visual_path = (
                    args.video_data_direc
                    + line.split(",")[1 + c * 4]
                    + "/"
                    + line.split(",")[2 + c * 4]
                    + "/"
                    + line.split(",")[3 + c * 4]
                    + ".mp4"
                )
                visual_paths.append(visual_path)
            data.append(visual_paths)


    if args.shuffle:
        random.shuffle(data)

    num = args.num_utts_per_shard
    chunks = [data[i:i + num] for i in range(0, len(data), num)]
    
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    output_folder_2 = args.output_folder_2
    os.makedirs(output_folder_2, exist_ok=True)

    # Using thread pool to speedup
    pool = multiprocessing.Pool(processes=args.num_threads)
    num_chunks = len(chunks)
    for i, chunk in enumerate(chunks):
        pool.apply_async(write_tar_file, (chunk, output_dir, i, num_chunks,output_folder_2))
        # write_tar_file(chunk, output_dir, i, num_chunks,output_folder_2)

    pool.close()
    pool.join()

if __name__ == '__main__':
    main()
