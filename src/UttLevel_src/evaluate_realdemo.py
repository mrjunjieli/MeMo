import argparse
import torch
import os
from Model import get_model
import numpy as np 
import torch.utils.data as data
import librosa
import python_speech_features
import cv2 as cv
import tqdm
import pdb 
# torch.set_printoptions(threshold=torch.inf)
from tools import audioread, audiowrite, cal_SISNR, load_model



MAX_INT16 = np.iinfo(np.int16).max
EPS = 10e-8



def main(args):
    # Model
    model = get_model(args.model_name)(
        sr=args.sampling_rate,
        win=args.win,
        n_mels=args.n_mels,
        hop_length=args.hop_length,
    )
    model = model.cuda()
    model = load_model(model, args.checkpoint)

    video_path = args.video_path
    #extract_audio 
    video_name = os.path.splitext(video_path)[0]
    audio_path = video_name+'_mix.wav'
    if not os.path.exists(audio_path): 
        os.system("ffmpeg -i %s %s"%(video_path,audio_path ))


    mixture,sr = audioread(audio_path)
    if sr != args.sampling_rate:
        mixture = librosa.resample(mixture,orig_sr=sr, target_sr=args.sampling_rate) 
    captureObj = cv.VideoCapture(video_path)
    roiSequence = []
    roiSize = 112
    while (captureObj.isOpened()):
        ret, frame = captureObj.read()
        if ret == True:
            grayed = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            grayed = cv.resize(grayed, (roiSize*2,roiSize*2))
            roi = grayed[int(roiSize-(roiSize/2)):int(roiSize+(roiSize/2)), int(roiSize-(roiSize/2)):int(roiSize+(roiSize/2))]
            roiSequence.append(roi)
        else:
            break
    captureObj.release()
    visual = np.asarray(roiSequence)
    visual = (visual -  0.4161) / 0.1688
    K = args.sampling_rate//int(args.visual_fps)
    length  = min(visual.shape[0],len(mixture)//K)
    length = min(length,int(args.visual_fps)*args.max_audio_length)  #<=30s 
    mixture = mixture[0:length*K]
    visual = visual[0:length,...]
    
    with torch.no_grad():
        a_mix = torch.from_numpy(mixture).cuda().float().unsqueeze(0)
        v_tgt = torch.from_numpy(visual).cuda().float().unsqueeze(0)
        a_zeros = torch.zeros_like(a_mix,device=a_mix.device)

        est_a_tgt = model(a_mix,v_tgt,a_zeros)
        est_a_tgt = est_a_tgt[0]/torch.max(torch.abs(est_a_tgt[0]))
        est_a_tgt = model(a_mix,v_tgt,est_a_tgt)
        est_a_tgt = est_a_tgt[0]/torch.max(torch.abs(est_a_tgt[0]))
        est_a_tgt = est_a_tgt.squeeze().cpu().numpy()
        save_path = video_name+'_est.wav'
        audiowrite(save_path,est_a_tgt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("avConv-tasnet")
    
    # Dataloader
    parser.add_argument('--video_path', type=str, help='path of vidoe data')
    parser.add_argument('--checkpoint', type=str,
                        help='the path of trained model')
    
    parser.add_argument('--model_name', type=str)

   # parameters for fbank
    parser.add_argument("--sampling_rate", default=16000, type=int)
    parser.add_argument("--win", default=512, type=int)
    parser.add_argument("--hop_length", default=128, type=int)
    parser.add_argument("--n_mels", default=80, type=int)

    parser.add_argument("--visual_fps", default=25, type=int)
    parser.add_argument("--max_audio_length", default=30, type=int,help='since the limit of GPU memory, audio length should be no more than 30s')
     

    args = parser.parse_args()

    main(args)