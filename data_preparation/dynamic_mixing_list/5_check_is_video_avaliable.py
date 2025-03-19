import librosa 
from tqdm import tqdm
import pdb 
import multiprocessing
import cv2
import numpy as np 
import argparse

import os 

def process_file(folder_path, broken_file):
    """检查文件夹中的视频文件是否损坏"""
    for root, dirs, files in os.walk(folder_path):
        for file in tqdm(files):
            file_path = os.path.join(root, file)
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                broken_file.append(file_path)  # 添加损坏文件路径
            else:
                ret, frame = cap.read()
                if not ret:
                    broken_file.append(file_path)
            cap.release()

def main(args):
    subdirectories = []
    for d in os.listdir(args.data_direc_mp4+'/train/'):
        subdirectories.append(os.path.join(args.data_direc_mp4+'/train/',d))
    for d in os.listdir(args.data_direc_mp4+'/test/'):
        subdirectories.append(os.path.join(args.data_direc_mp4+'/test/',d))

    # # single processing 
    # for folder_path in subdirectories:
    #     process_file(folder_path)

    manager = multiprocessing.Manager()
    broken_file = manager.list()  # 共享列表，存储损坏的视频文件

    with multiprocessing.Pool(processes=args.num_threads) as pool:
        # 使用 starmap 传递多个参数
        pool.starmap(process_file, [(folder_path, broken_file) for folder_path in subdirectories])

    # 进程结束后，转换为普通列表
    broken_file_list = list(broken_file)

    with open(args.output_file, "w", encoding="utf-8") as f:
        for bf in broken_file_list:
            f.write(bf + "\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Voxceleb2 dataset")
    parser.add_argument("--data_direc_mp4", type=str)
    parser.add_argument("--num_threads", type=int)
    parser.add_argument("--output_file", type=str)
    args = parser.parse_args()

    main(args)