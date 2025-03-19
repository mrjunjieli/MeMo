import argparse
import torch
# from dataload_dynamic import get_dataloader_dy
from dataload import get_dataloader

import os
from Model import get_model

# from solver import Solver
from solver_seanet import Solver
from tools import load_model

import torch.nn as nn
import pdb
import time

def main(args):
    if args.distributed:
        # dist configs
        torch.manual_seed(3407)
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")

    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # speaker id assignment
    mix_lst = open(args.mix_lst_path).read().splitlines()
    train_lst = list(filter(lambda x: x.split(",")[0] == "train", mix_lst))
    IDs = 0
    speaker_dict = {}
    for line in train_lst:
        for i in range(2):
            ID = line.split(",")[i * 4 + 2]
            if ID not in speaker_dict:
                speaker_dict[ID] = IDs
                IDs += 1
    args.speaker_dict = speaker_dict
    args.speakers = len(speaker_dict)

    # Model
    model = get_model(args.model_name)(
        sr=args.sample_rate,
        win=args.win,
        n_mels=args.n_mels,
        hop_length=args.hop_length,
    )

    if (args.distributed and args.local_rank == 0) or args.distributed == False:
        print("started on " + args.log_name + "\n")
        print(args)
        print(
            "\nTotal number of parameters: {} \n".format(
                sum(p.numel() for p in model.parameters())
            )
        )
        print(model)

    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # train_sampler, train_generator = get_dataloader_dy(args, partition="train")
    train_sampler, train_generator = get_dataloader(args, partition="train")
    _, val_generator = get_dataloader(args, partition="val")
    args.train_sampler = train_sampler
    solver = Solver(
        args=args,
        model=model,
        optimizer=optimizer,
        train_data=train_generator,
        validation_data=val_generator,
    )
    solver.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("online")

    # Dataloader
    parser.add_argument(
        "--mix_lst_path",
        type=str,
        default="/home/panzexu/datasets/LRS2/audio/2_mix_min/mixture_data_list_2mix.csv",
        help="directory including train data",
    )
    parser.add_argument(
        "--visual_direc",
        type=str,
        default="/home/panzexu/datasets/LRS2/lip/",
        help="directory including test data",
    )
    parser.add_argument(
        "--mixture_direc",
        type=str,
        default="/home/panzexu/datasets/LRS2/audio/2_mix_min/",
        help="directory of audio",
    )
    parser.add_argument(
        "--obj_dir",
        type=str,
        default="../../data_preparation/Asset/object_image_sr",
        help="location of occlusion patch mask",
    )
    parser.add_argument(
        "--obj_mask_dir",
        default="../../data_preparation/Asset/object_mask_x4",
        help="location of occlusion patch mask",
    )


    # Training
    parser.add_argument("--batch_size", default=8, type=int, help="Batch size")
    parser.add_argument(
        "--max_length", default=6, type=int, help="max_length of mixture in training",
    )
    parser.add_argument(
        "--num_workers",
        default=4,
        type=int,
        help="Number of workers to generate minibatch",
    )
    parser.add_argument(
        "--epochs", default=100, type=int, help="Number of maximum epochs"
    )
    parser.add_argument(
        "--effec_batch_size", default=8, type=int, help="effective Batch size"
    )
    parser.add_argument(
        "--accu_grad", default=0, type=int, help="whether to accumulate grad"
    )
    parser.add_argument("--C", default=2, type=int, help="number of speakrs")

    parser.add_argument(
        "--model_name",
        default="TDSE_P_SPKEMB",
        type=str,
        help="select separation model",
    )

    # optimizer
    parser.add_argument("--lr", default=1e-3, type=float, help="Init learning rate")
    parser.add_argument(
        "--max_norm", default=5, type=float, help="Gradient norm threshold to clip",
    )

    # Log and Visulization
    parser.add_argument(
        "--log_name", type=str, default=None, help="the name of the log"
    )
    parser.add_argument(
        "--use_tensorboard", type=int, default=0, help="Whether to use use_tensorboard",
    )
    parser.add_argument(
        "--continue_from", type=str, default="", help="Whether to use use_tensorboard",
    )

    # Distributed training
    parser.add_argument("--local-rank", default=0, type=int)

    # parameters for fbank of spekaker model
    parser.add_argument("--sample_rate", default=16000, type=int)
    parser.add_argument("--win", default=512, type=int)
    parser.add_argument("--hop_length", default=128, type=int)
    parser.add_argument("--n_mels", default=80, type=int)

    # self-training paprameters
    parser.add_argument("--shift_range", default="0,0.5", type=str)
    parser.add_argument("--Self_enroll_amplitude_scaling", default="0,1.5", type=str)
    parser.add_argument("--teacher_point", default=50, type=int)
    parser.add_argument("--num_slots", default='5,5', type=str)
    parser.add_argument("--loss_weight", default='226', type=str,help='the weight of loss for v_train a_train av_train,')

    args = parser.parse_args()
    args.distributed = False
    args.world_size = 1
    if "WORLD_SIZE" in os.environ:
        args.distributed = int(os.environ["WORLD_SIZE"]) > 1
        args.world_size = int(os.environ["WORLD_SIZE"])

    assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."

    main(args)
