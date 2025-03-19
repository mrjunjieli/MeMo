import os
import argparse
import json
import time
import torch
import torch.nn as nn
from Model import get_model
import torch.nn.functional as F
import pdb 

from thop import profile ,clever_format 
import numpy as np 
import math 

def calculate_audio_segment(j, a_chunk_samples, a_receptive_samples, mix_audio,j_start):
    
    if j==j_start:
        a_start = 0
    else:
        a_start = max(0, (j + 1) * a_chunk_samples - a_receptive_samples)

    if a_start + a_receptive_samples > mix_audio.shape[1]:
        a_start = mix_audio.shape[1] - a_receptive_samples
    if a_chunk_samples * (j + 1) > mix_audio.shape[1]:
        a_duration_seg = a_chunk_samples - (a_chunk_samples * (j + 1) - mix_audio.shape[1])
    else:
        a_duration_seg = a_chunk_samples
    a_end = a_chunk_samples * j + a_duration_seg
    return a_start, a_end, a_duration_seg

def calculate_video_segment(j, v_chunk_samples, v_receptive_samples, visual,j_start):
    if j==j_start:
        v_start = 0
    else:
        v_start = max(0, (j + 1) * v_chunk_samples - v_receptive_samples)
    if v_chunk_samples * (j + 1) > visual.shape[1]:
        v_duration_seg = v_chunk_samples - (v_chunk_samples * (j + 1) - visual.shape[1])
    else:
        v_duration_seg = v_chunk_samples
    v_end = v_chunk_samples * j + v_duration_seg
    return v_start, v_end, v_duration_seg

def process_model_segment(model, j, j_start, mix_audio_seg, v_clean_seg, v_seg, enroll=None,spk_enroll=None,clean_visual_init=1):
    if j == j_start:
        if clean_visual_init:
            return model(mix_audio_seg, v_clean_seg, torch.zeros_like(mix_audio_seg),torch.zeros_like(mix_audio_seg))
        else:
            return model(mix_audio_seg, v_seg, torch.zeros_like(mix_audio_seg),torch.zeros_like(mix_audio_seg))
    else:
        return model(mix_audio_seg, v_seg, enroll,spk_enroll)
    
def prepare_self_enroll(audio, a_end, a_duration_seg, mix_audio_seg):
    self_enroll = audio[:, : a_end - a_duration_seg]
    if self_enroll.shape[1] < mix_audio_seg.shape[1]:
        self_enroll = F.pad(
            self_enroll,
            (mix_audio_seg.shape[1] - self_enroll.shape[1], 0,),
        )
    else:
        self_enroll = self_enroll[:, -mix_audio_seg.shape[1] :]
    return self_enroll

def eval(args,model,mixture,visual):

    a_chunk_samples = int(args.chunk_time * args.sample_rate)
    a_receptive_samples = int(args.recep_field * args.sample_rate)
    # video
    v_chunk_samples = int(args.chunk_time * 25)
    v_receptive_samples = int(args.recep_field * 25)
    initilization_samples = int(args.initilization * args.sample_rate)

    j_start = max(0, (initilization_samples - a_chunk_samples) // a_chunk_samples,)

    start_time = 0
    for j in range(j_start, math.ceil((mixture.shape[1] - a_chunk_samples) / a_chunk_samples)+ 1,):
        a_start, a_end, a_duration_seg = calculate_audio_segment(j, a_chunk_samples, a_receptive_samples, mixture,j_start)
        mix_audio_seg = mixture[:, a_start:a_end]
        v_start, v_end, v_duration_seg = calculate_video_segment(j, v_chunk_samples, v_receptive_samples, visual,j_start)
        v_seg = visual[:, v_start:v_end, :, :]
        self_enroll_est_input_av=prepare_self_enroll(mixture, a_end, a_duration_seg, mix_audio_seg) 
        if j!=j_start:
            est_seg,_,_ = process_model_segment(model, j, j_start, mix_audio_seg, v_seg, v_seg, self_enroll_est_input_av, self_enroll_est_input_av)
        if j==j_start:
            start_time = time.time()
    infer_time = time.time() - start_time
    duration  = (mixture.shape[1]-initilization_samples)/args.sample_rate
    return infer_time,duration

def main(args):
    length = 20
    mixture = torch.randn(1, args.sample_rate*length).cuda()
    visual = torch.randn(1, 25*length, 112, 112).cuda()
    self_enroll = mixture
    s=0
    model = get_model(args.model_name)(
        num_speakers=800,
        sr=args.sample_rate,
        win=args.win,
        n_mels=args.n_mels,
        hop_length=args.hop_length,
    ).cuda()
    
    macs,params = profile(model,inputs=(mixture,visual,self_enroll),verbose=True)
    macs, params = clever_format([macs, params], "%.3f")

    print("MACs: ", macs)
    print("Params: ", params )

    for param in model.parameters():
        s += np.product(param.size())
    print("# of parameters: " + str(s / 1024.0 / 1024.0))
    # total_infer_time =0 
    # total_duration_time = 0
    # num=20
    # for i in range(num):
    #     infer_time,duration = eval(args,model,mixture,visual)
    #     total_infer_time+= infer_time
    #     total_duration_time+=duration

    # print("RTF:",total_infer_time/(total_duration_time))

if __name__ == "__main__":
    parser = argparse.ArgumentParser("avConv-tasnet")

    parser.add_argument(
            "--model_name",
            default="TDSE_P_SPKEMB",
            type=str,
            help="select separation model",
        )
    # parameters for fbank
    parser.add_argument("--win", default=512, type=int)
    parser.add_argument("--hop_length", default=128, type=int)
    parser.add_argument("--n_mels", default=80, type=int)
    # Online evaluation setting
    parser.add_argument(
        "--chunk_time", type=float, default=0.2, help="time length(s) of chunk_time",
    )
    parser.add_argument(
        "--recep_field",
        type=float,
        default=2.5,
        help="time length(s) of receptive filed",
    )
    parser.add_argument(
        "--sample_rate", type=int, default=16000, help="sampling_rate of audio"
    )
    parser.add_argument(
    "--initilization",
    type=float,
    default=2.0,
    help="time length(s) of initilization",
)
    args = parser.parse_args()
    main(args)