import argparse
import torch
import os
import sys

from Model import get_model
from pystoi import stoi
from pesq import pesq
import sys
import numpy as np
import soundfile as sf
import torch.utils.data as data
import librosa
import cv2 as cv
import tqdm
from tools import audioread, audiowrite, cal_SISNR, load_model
import fast_bss_eval
import pdb
from copy import deepcopy
from data_preparation.Visual_perturb import *
import torch.nn.functional as F

EPS = np.finfo(float).eps
MAX_INT16 = np.iinfo(np.int16).max


class dataset(data.Dataset):
    def __init__(
        self,
        obj_dir,
        obj_mask_dir,
        mix_lst_path,
        visual_direc,
        mixture_direc,
        batch_size=1,
        partition="test",
        sampling_rate=16000,
        mix_no=2,
    ):

        self.minibatch = []
        self.visual_direc = visual_direc
        self.mixture_direc = mixture_direc
        self.sampling_rate = sampling_rate
        self.partition = partition
        self.C = mix_no
        self.fps = 25
        self.normMean = 0.4161
        self.normStd = 0.1688


        self.obj_dir = obj_dir
        self.obj_mask_dir = obj_mask_dir


        mix_csv = open(mix_lst_path).read().splitlines()
        self.mix_lst = list(filter(lambda x: x.split(",")[0] == partition, mix_csv))

        spk2utt_dict = {}
        for line in self.mix_lst:
            path_line = line.split(",")[0:-6]
            for i in range(2):
                ID = line.split(",")[i * 4 + 2]
                utt_path = (
                    self.mixture_direc
                    + self.partition
                    + "/s%d/" % (i + 1)
                    + ",".join(path_line).replace(",", "_").replace("/", "_")
                    + ".wav"
                )
                if ID not in spk2utt_dict:
                    spk2utt_dict[ID] = [utt_path]
                else:
                    spk2utt_dict[ID].append(utt_path)
        self.spk2utt_dict = spk2utt_dict

    def __getitem__(self, index):

        line = self.mix_lst[index]
        mixtures = []
        audios = []
        visuals = []
        clean_visuals = []
        mask_types = []
        mask_starts = []
        visual_lengths = []
        mask_lengths = []
        enrolls = []
        tgt_utt_ids = []

        # read mix
        path_line = line.split(",")[0:-6]
        mixture_path = (
            self.mixture_direc
            + self.partition
            + "/mix/"
            + ",".join(path_line).replace(",", "_").replace("/", "_")
            + ".wav"
        )
        mixture, sr = audioread(mixture_path)
        if sr != self.sampling_rate:
            mixture = librosa.resample(
                mixture, orig_sr=sr, target_sr=self.sampling_rate
            )

        min_length = mixture.shape[0]
        min_length_tim = min_length / self.sampling_rate
        enroll_length = 3 * self.sampling_rate  #

        for c in range(2):

            # read tgt audio
            audio_path = (
                self.mixture_direc
                + self.partition
                + "/s%d/" % (c + 1)
                + ",".join(path_line).replace(",", "_").replace("/", "_")
                + ".wav"
            )
            spk_id = line.split(",")[c * 4 + 2]
            utt_id = line.split(",")[c * 4 + 3]
            tgt_utt_ids.append(spk_id + "_" + utt_id.replace("/", "_"))
            enroll_path = self.spk2utt_dict[spk_id][0]

            audio, sr = audioread(audio_path)
            audio = audio[0:min_length]
            if sr != self.sampling_rate:
                audio = librosa.resample(
                    audio, orig_sr=sr, target_sr=self.sampling_rate
                )

            if len(audio) < int(min_length):
                audio = np.pad(audio, (0, int(min_length) - len(audio)))
            audios.append(audio)

            enroll_audio, sr = audioread(enroll_path)
            if sr != self.sampling_rate:
                enroll_audio = librosa.resample(
                    enroll_audio, orig_sr=sr, target_sr=self.sampling_rate
                )

            if enroll_audio.shape[0] < enroll_length:
                enroll_audio = np.pad(
                    enroll_audio, (0, int(enroll_length) - len(enroll_audio))
                )
            if enroll_audio.shape[0] > enroll_length:
                enroll_audio = enroll_audio[0:enroll_length]
            enrolls.append(enroll_audio)

            # read video
            mask_start = int(line.split(",")[c * 3 + 10])
            mask_length = int(line.split(",")[c * 3 + 11])
            mask_type = int(
                line.split(",")[c * 3 + 12]
            )  # 0:full_mask 1: occluded 2: low resolution
            visual_path = (
                self.visual_direc
                + line.split(",")[1 + c * 4]
                + "/"
                + line.split(",")[2 + c * 4]
                + "/"
                + line.split(",")[3 + c * 4]
                + ".mp4"
            )
            captureObj = cv.VideoCapture(visual_path)
            roiSequence = []
            clean_roiSequence = []
            roiSize = 112
            start = 0
            if mask_type == 1:
                occlude_img, occluder_img, occluder_mask = get_occluders(
                    self.obj_dir, self.obj_mask_dir, state=self.partition
                )
                alpha_mask = np.expand_dims(occluder_mask, axis=2)
                alpha_mask = np.repeat(alpha_mask, 3, axis=2) / 255.0
            while captureObj.isOpened():
                ret, frame = captureObj.read()
                if ret == True:
                    grayed = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                    grayed = grayed / 255
                    grayed = cv.resize(grayed, (roiSize * 2, roiSize * 2))
                    roi = grayed[
                        int(roiSize - (roiSize / 2)) : int(roiSize + (roiSize / 2)),
                        int(roiSize - (roiSize / 2)) : int(roiSize + (roiSize / 2)),
                    ]
                    clean_roiSequence.append(roi)
                    if start >= mask_start and start < mask_start + mask_length:
                        if mask_type == 0:
                            roi = np.zeros_like(roi)
                        elif mask_type == 1:
                            frame = cv.resize(frame, (roiSize * 2, roiSize * 2))
                            roi = frame[
                                int(roiSize - (roiSize / 2)) : int(
                                    roiSize + (roiSize / 2)
                                ),
                                int(roiSize - (roiSize / 2)) : int(
                                    roiSize + (roiSize / 2)
                                ),
                            ]
                            offset_x = 15
                            offset_y = 15
                            x = 112 / 2
                            y = 112 / 2
                            roi = overlay_image_alpha(
                                roi,
                                occluder_img,
                                int(x - offset_x),
                                int(y - offset_y),
                                alpha_mask,
                            )
                            roi = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
                            # cv.imwrite('frames/'+'org_image_'+str(id_)+'.png',roi)
                            # pdb.set_trace()
                            roi = roi / 255
                        elif mask_type == 2:
                            frame = cv.resize(frame, (roiSize * 2, roiSize * 2))
                            roi = frame[
                                int(roiSize - (roiSize / 2)) : int(
                                    roiSize + (roiSize / 2)
                                ),
                                int(roiSize - (roiSize / 2)) : int(
                                    roiSize + (roiSize / 2)
                                ),
                            ]
                            times = 10
                            roi = cv.resize(roi, (roiSize // times, roiSize // times))
                            roi = cv.resize(roi, (roiSize, roiSize))
                            roi = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
                            # cv.imwrite('frames/'+'org_image_'+str(id_)+'.png',roi)
                            roi = roi / 255
                        else:
                            sys.exit("error: ", mask_type)
                    roiSequence.append(roi)
                    start += 1
                else:
                    break
            captureObj.release()
            clean_visual = np.asarray(clean_roiSequence)
            visual = np.asarray(roiSequence)
            visual_length = len(roiSequence)
            visual = visual[0 : int(min_length_tim * self.fps), ...]
            visual = (visual - self.normMean) / self.normStd
            if visual.shape[0] < int(min_length_tim * self.fps):
                visual = np.pad(
                    visual,
                    (
                        (0, int(min_length_tim * self.fps) - visual.shape[0]),
                        (0, 0),
                        (0, 0),
                    ),
                    mode="edge",
                )
            clean_visual = clean_visual[0 : int(min_length_tim * self.fps), ...]
            clean_visual = (clean_visual - self.normMean) / self.normStd
            if clean_visual.shape[0] < int(min_length_tim * self.fps):
                clean_visual = np.pad(
                    clean_visual,
                    (
                        (0, int(min_length_tim * self.fps) - clean_visual.shape[0],),
                        (0, 0),
                        (0, 0),
                    ),
                    mode="edge",
                )

            visuals.append(visual)
            clean_visuals.append(clean_visual)
            mask_lengths.append(mask_length)
            visual_lengths.append(visual_length)
            mask_types.append(mask_type)
            mask_starts.append(mask_start)

        mixtures.append(mixture)
        mixtures.append(mixture)

        np_mixtures = np.asarray(mixtures)
        np_audios = np.asarray(audios)
        np_visuals = np.asarray(visuals)
        np_clean_visuals = np.asarray(clean_visuals)
        np_enrolls = np.asarray(enrolls)

        return (
            tgt_utt_ids,
            np_mixtures,
            np_audios,
            np_visuals,
            np_clean_visuals,
            np_enrolls,
            mask_starts,
            mask_lengths,
            mask_types,
            visual_lengths,
        )

    def __len__(self):
        return len(self.mix_lst)


def main(args):

    # Model
    model = get_model(args.model_name)(
        sr=args.sample_rate,
        win=args.win,
        n_mels=args.n_mels,
        hop_length=args.hop_length,
    )
    model = model.cuda()
    model = load_model(model, "%s/model_dict_best.pt" % args.continue_from)

    print("model_name:", args.model_name)
    print("load from ", args.continue_from)
    datasets = dataset(
        obj_dir=args.obj_dir,
        obj_mask_dir=args.obj_mask_dir,
        mix_lst_path=args.mix_lst_path,
        visual_direc=args.visual_direc,
        mixture_direc=args.mixture_direc,
        mix_no=args.C,
        sampling_rate=args.sample_rate,
    )

    test_generator = data.DataLoader(
        datasets, batch_size=1, shuffle=False, num_workers=args.num_workers
    )

    model.eval()
    with torch.no_grad():
        # 打印表头
        print(
            f"{'UTT_ID':<30},{'Mask_type':<10},{'V_length(fps)':<15},{'Mask_start':<15},{'Mask_length':<15},{'Mix_SISNR':<10},{'Visual_only':<15},{'SelfEnro+V':<15},{'Tgt+V':<5},{'Pre+V':<5}"
        )
        print("=" * 170)
        for (
            i,
            (
                tgt_uttid,
                a_mix,
                a_tgt,
                v_tgt,
                v_clean_tgt,
                enroll_audio,
                mask_start,
                mask_length,
                mask_type,
                visual_length,
            ),
        ) in enumerate(tqdm.tqdm(test_generator)):
            a_mix = a_mix.cuda().squeeze().float()
            a_tgt = a_tgt.cuda().squeeze().float()
            v_tgt = v_tgt.cuda().squeeze().float()
            v_clean_tgt = v_clean_tgt.cuda().squeeze().float()
            enroll_audio = enroll_audio.cuda().squeeze().float()
            a_zeros = torch.zeros_like(a_mix,device=a_mix.device)
            v_zeros =  torch.zeros_like(v_tgt,device=v_tgt.device)

            if args.clean_visual:
                visual_tgt = v_clean_tgt
            else:
                visual_tgt = v_tgt

            est_audio = {}
            est_audio["Visual_only"] = torch.zeros_like(a_tgt)
            est_audio["Self_enroll+Visual"] = torch.zeros_like(a_tgt)
            est_audio["Tgt_enroll+Visual"] = torch.zeros_like(a_tgt)
            est_audio["Pre_enroll+Visual"] = torch.zeros_like(a_tgt)


            # visual only
            est_audio["Visual_only"] = model(a_mix, visual_tgt,a_zeros,a_zeros)[0]
            # est_audio["Visual_only"] = model(a_mix, visual_tgt,a_zeros,enroll_audio)[0]
            if 'self' in args.model_name.lower() or 'spk' in args.model_name.lower():
                self_enroll = deepcopy(est_audio["Visual_only"])
                self_enroll = self_enroll / torch.max(abs(self_enroll)) * 0.7

                #self_enroll + Visual
                if 'self' in args.model_name.lower() and 'spk' in args.model_name.lower(): 
                    est_audio['Self_enroll+Visual'] = model(a_mix, visual_tgt, self_enroll,self_enroll)[0]
                elif 'self' in args.model_name.lower():
                    est_audio['Self_enroll+Visual'] = model(a_mix, visual_tgt, self_enroll,a_zeros)[0]
                elif 'spk' in args.model_name.lower():
                    est_audio['Self_enroll+Visual'] = model(a_mix, visual_tgt, a_zeros, self_enroll)[0]

                # #tgt_enroll + Visual
                if 'self' in args.model_name.lower() and 'spk' in args.model_name.lower(): 
                    est_audio['Tgt_enroll+Visual'] = model(a_mix, visual_tgt, a_tgt,a_tgt)[0]
                elif 'self' in args.model_name.lower():
                    est_audio['Tgt_enroll+Visual'] = model(a_mix, visual_tgt, a_tgt,a_zeros)[0]
                elif 'spk' in args.model_name.lower():
                    est_audio['Tgt_enroll+Visual'] = model(a_mix, visual_tgt, a_zeros, a_tgt)[0]

                #pre_enroll + Visual
                pre_enroll = enroll_audio
                if pre_enroll.shape[1]<a_mix.shape[1]:
                    pre_enroll = F.pad(pre_enroll, (a_tgt.shape[1] - pre_enroll.shape[1], 0))
                else:
                    pre_enroll = pre_enroll[:,0:a_mix.shape[1]]

                if 'self' in args.model_name.lower() and 'spk' in args.model_name.lower(): 
                    est_audio['Pre_enroll+Visual'] = model(a_mix, visual_tgt, pre_enroll,enroll_audio)[0]
                elif 'self' in args.model_name.lower():
                    est_audio['Pre_enroll+Visual'] = model(a_mix, visual_tgt, pre_enroll,a_zeros)[0]
                elif 'spk' in args.model_name.lower():
                    est_audio['Pre_enroll+Visual'] = model(a_mix, visual_tgt, a_zeros, enroll_audio)[0]

            for c in range(2):
                mix_sisnr = 0
                utt_sisnr = {}
                utt_sisnr["Visual_only"] = 0
                utt_sisnr["Self_enroll+Visual"] = 0
                utt_sisnr["Tgt_enroll+Visual"] = 0
                utt_sisnr["Pre_enroll+Visual"] = 0
                for key in utt_sisnr.keys():
                    if torch.sum(abs(est_audio[key])) != 0:
                        utt_sisnr[key] = cal_SISNR(
                            a_tgt[c].unsqueeze(0), est_audio[key][c].unsqueeze(0)
                        ).item()
                mix_sisnr = cal_SISNR(
                            a_tgt[c].unsqueeze(0), a_mix[c].unsqueeze(0)
                        ).item()
                print(
                    f"{tgt_uttid[c][0]:<30},{mask_type[c].item():<10},{visual_length[c].item():<15},{int(mask_start[c].item()):<15},{int(mask_length[c].item()):<15},{mix_sisnr:<10.2f},{utt_sisnr['Visual_only']:<15.2f},{utt_sisnr['Self_enroll+Visual']:<15.2f},{utt_sisnr['Tgt_enroll+Visual']:<5.2f} ,{utt_sisnr['Pre_enroll+Visual']:<5.2f}"
                )
                if utt_sisnr["Self_enroll+Visual"] != 0 and args.save_audio:
                    est_source = est_audio["Self_enroll+Visual"][0].cpu().numpy()
                    tgt_audio = a_tgt[c].cpu().numpy()
                    mix_audio = a_mix[c].cpu().numpy()
                    if not os.path.exists(args.save_dir):
                        os.makedirs(args.save_dir)
                    audiowrite(
                        str(args.save_dir) + "/" + tgt_uttid[c][0]+"_mix.wav",mix_audio,
                    )
                    audiowrite(
                        str(args.save_dir) + "/" + tgt_uttid[c][0]+"_tgt.wav",tgt_audio,
                    )
                    audiowrite(
                        str(args.save_dir) + "/"+ tgt_uttid[c][0]+"_est.wav",est_source,
                    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("avConv-tasnet")

    parser.add_argument(
        "--mix_lst_path",
        type=str,
        default="../../data_preparation/mixture_data_list_2mix_with_occludded.csv",
        help="directory including train data",
    )
    parser.add_argument(
        "--visual_direc",
        type=str,
        default="/mntcephfs/lee_dataset/separation/voxceleb2/mp4/",
        help="directory including test data",
    )
    parser.add_argument(
        "--mixture_direc",
        type=str,
        default="/mntcephfs/lee_dataset/separation/voxceleb2/mixture/",
        help="directory of audio",
    )
    parser.add_argument(
        "--continue_from",
        type=str,
        default="/mntnfs/lee_data1/wwu/TSE_MEMBANK/src_memorybank/logs/TDSE_WENXUAN/",
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

    parser.add_argument(
        "--save_audio", default=0, type=int, help="whether to save audio"
    )
    parser.add_argument(
        "--save_dir", default="./save_audio/", type=str, help="audio_save_path"
    )

    # Training
    parser.add_argument(
        "--num_workers",
        default=4,
        type=int,
        help="Number of workers to generate minibatch",
    )

    parser.add_argument("--C", default=2, type=int, help="number of speakrs")

    parser.add_argument(
        "--model_name",
        default="TDSE_P_SPKEMB",
        type=str,
        help="select separation model",
    )

    # parameters for fbank
    parser.add_argument("--sample_rate", default=16000, type=int)
    parser.add_argument("--win", default=512, type=int)
    parser.add_argument("--hop_length", default=128, type=int)
    parser.add_argument("--n_mels", default=80, type=int)

    parser.add_argument("--clean_visual", default=0, type=int)


    args = parser.parse_args()

    main(args)
