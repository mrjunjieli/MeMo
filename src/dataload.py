import numpy as np
import math
import torch.distributed as dist
import torch
import torch.nn as nn
import torch.utils.data as data
import os
import cv2 as cv
import random
import soundfile as sf
import librosa
import sys
from tools import audioread

# sys.path.append("../../data_preparation/")
from data_preparation.Visual_perturb import *

import pdb

EPS = np.finfo(float).eps
MAX_INT16 = np.iinfo(np.int16).max
np.random.seed(0)
random.seed(0)


class dataset(data.Dataset):
    def __init__(
        self,
        obj_dir,
        obj_mask_dir,
        speaker_dict,
        mix_lst_path,
        visual_direc,
        mixture_direc,
        batch_size,
        partition="val",
        sampling_rate=16000,
        mix_no=2,
        max_length=6,
    ):

        self.minibatch = []
        self.visual_direc = visual_direc
        self.mixture_direc = mixture_direc
        self.sampling_rate = sampling_rate
        self.partition = partition
        self.C = mix_no
        self.fps = 25
        self.batch_size = batch_size
        self.speaker_id = speaker_dict
        self.max_length = max_length

        self.obj_dir = obj_dir
        self.obj_mask_dir = obj_mask_dir

        self.normMean = 0.4161
        self.normStd = 0.1688

        mix_lst = open(mix_lst_path).read().splitlines()
        mix_lst = list(filter(lambda x: x.split(",")[0] == partition, mix_lst))
        spk2utt_dict = {}
        for line in mix_lst:
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

        sorted_mix_lst = sorted(
            mix_lst, key=lambda data: float(data.split(",")[-1]), reverse=True
        )
        if self.partition == "train":
            random.shuffle(sorted_mix_lst)
        start = 0
        while True:
            end = min(len(sorted_mix_lst), start + self.batch_size)
            self.minibatch.append(sorted_mix_lst[start:end])
            if end == len(sorted_mix_lst):
                break
            start = end

    def __getitem__(self, index):
        if self.partition=='train' and index==0:
            random.shuffle(self.minibatch)
        batch_lst = self.minibatch[index]
        min_length = self.max_length
        for _ in range(len(batch_lst)):
            if float(batch_lst[_].split(",")[-7]) < min_length:
                min_length = float(batch_lst[_].split(",")[-7])

        enroll_length = 3

        mixtures = []
        audios = []
        enrolls = []
        visuals = []
        clean_visuals = []
        speakers = []
        mask_starts = []
        mask_lengths = []
        mask_types = []

        for line in batch_lst:
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

            # truncate
            mixture = mixture[0 : int(min_length * self.sampling_rate)]
            if len(mixture) < int(min_length * self.sampling_rate):
                mixture = np.pad(
                    mixture,
                    (0, int(min_length * self.sampling_rate) - len(mixture)),
                )

            for c in range(self.C):
                # read target audio
                audio_path = (
                    self.mixture_direc
                    + self.partition
                    + "/s%d/" % (c + 1)
                    + ",".join(path_line).replace(",", "_").replace("/", "_")
                    + ".wav"
                )
                spk_id = line.split(",")[c * 4 + 2]
                if self.partition != "train":
                    enroll_path = self.spk2utt_dict[spk_id][0]
                else:
                    enroll_path = random.choice(self.spk2utt_dict[spk_id])

                audio, sr = audioread(audio_path)
                audio = audio[0 : int(min_length * self.sampling_rate)]
                if sr != self.sampling_rate:
                    audio = librosa.resample(
                        audio, orig_sr=sr, target_sr=self.sampling_rate
                    )
                if len(audio) < int(min_length * self.sampling_rate):
                    audio = np.pad(
                        audio,
                        (0, int(min_length * self.sampling_rate) - len(audio)),
                    )
                audios.append(audio)

                enroll_audio, sr = audioread(enroll_path)
                if sr != self.sampling_rate:
                    enroll_audio = librosa.resample(
                        enroll_audio, orig_sr=sr, target_sr=self.sampling_rate
                    )
                if len(enroll_audio) < int(enroll_length * self.sampling_rate):
                    enroll_audio = np.pad(
                        enroll_audio,
                        (
                            0,
                            int(enroll_length * self.sampling_rate)
                            - len(enroll_audio),
                        ),
                    )
                else:
                    enroll_audio = enroll_audio[
                        0 : int(enroll_length * self.sampling_rate)
                    ]
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

                if mask_type == 1 and self.partition != "test":
                    occlude_img, occluder_img, occluder_mask = get_occluders(
                        self.obj_dir, self.obj_mask_dir, state=self.partition
                    )
                    alpha_mask = np.expand_dims(occluder_mask, axis=2)
                    alpha_mask = np.repeat(alpha_mask, 3, axis=2) / 255.0
                elif mask_type == 1 and self.partition == "test":
                    occlude_img, occluder_img, occluder_mask = get_occluders(
                        self.obj_dir + "_test",
                        self.obj_mask_dir + "_test",
                        state=self.partition,
                    )
                    alpha_mask = np.expand_dims(occluder_mask, axis=2)
                    alpha_mask = np.repeat(alpha_mask, 3, axis=2) / 255.0
                # id_=0
                ratio = mask_length / (min_length * 25)
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
                            ),
                        ]
                        clean_roiSequence.append(roi)
                        if (
                            start >= mask_start
                            and start < mask_start + mask_length
                        ):
                            if mask_type == 0:
                                roi = np.zeros_like(roi)
                            elif mask_type == 1:
                                frame = cv.resize(
                                    frame, (roiSize * 2, roiSize * 2)
                                )
                                roi = frame[
                                    int(roiSize - (roiSize / 2)) : int(
                                        roiSize + (roiSize / 2)
                                    ),
                                    int(roiSize - (roiSize / 2)) : int(
                                        roiSize + (roiSize / 2)
                                    ),
                                ]
                                if self.partition == "train":
                                    offset_x = random.uniform(13, 17)
                                    offset_y = random.uniform(13, 17)
                                    x = random.uniform(112 / 2 - 5, 112 / 2 + 5)
                                    y = random.uniform(112 / 2 - 5, 112 / 2 + 5)
                                else:
                                    offset_x = 15
                                    offset_y = 15
                                    x = 112 / 2
                                    y = 112 / 2
                                roi = overlay_image_alpha(
                                    roi,
                                    occluder_img,
                                    int(x - offset_x),
                                    int(y - offset_y),
                                    alpha_mask
                                )
                                roi = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
                                # cv.imwrite('frames/'+'org_image_'+str(id_)+'.png',roi)
                                # pdb.set_trace()
                                roi = roi / 255
                                # id_+=1
                            elif mask_type == 2:
                                frame = cv.resize(
                                    frame, (roiSize * 2, roiSize * 2)
                                )
                                roi = frame[
                                    int(roiSize - (roiSize / 2)) : int(
                                        roiSize + (roiSize / 2)
                                    ),
                                    int(roiSize - (roiSize / 2)) : int(
                                        roiSize + (roiSize / 2)
                                    ),
                                ]
                                if self.partition == "train":
                                    if random.random() < 0.5:
                                        var = random.uniform(0.02, 0.2)
                                        roi = random_noise(
                                                roi,
                                                mode="gaussian",
                                                mean=0,
                                                var=var,
                                                clip=True,
                                            )* 255
                    
                                        roi = np.uint8(roi)
                                    else:
                                        blur = (
                                            torchvision.transforms.GaussianBlur(
                                                kernel_size=(13, 13),
                                                sigma=(4, 8),
                                            )
                                        )
                                        roi = (
                                            blur(
                                                torch.tensor(roi)
                                                .unsqueeze(0)
                                                .permute(0, 3, 1, 2)
                                            )
                                            .permute(0, 2, 3, 1)
                                            .squeeze(0)
                                            .numpy()
                                        )
                                else:
                                    times = 10
                                    roi = cv.resize(
                                        roi,
                                        (roiSize // times, roiSize // times),
                                    )
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
                visual = np.asarray(roiSequence)
                visual = visual[0 : int(min_length * self.fps), ...]
                visual = (visual - self.normMean) / self.normStd
                if visual.shape[0] < int(min_length * self.fps):
                    visual = np.pad(
                        visual,
                        (
                            (0, int(min_length * self.fps) - visual.shape[0]),
                            (0, 0),
                            (0, 0),
                        ),
                        mode="edge",
                    )
                visuals.append(visual)

                clean_visual = np.asarray(clean_roiSequence)
                clean_visual = clean_visual[0 : int(min_length * self.fps), ...]
                clean_visual = (clean_visual - self.normMean) / self.normStd
                if clean_visual.shape[0] < int(min_length * self.fps):
                    clean_visual = np.pad(
                        clean_visual,
                        (
                            (
                                0,
                                int(min_length * self.fps)
                                - clean_visual.shape[0],
                            ),
                            (0, 0),
                            (0, 0),
                        ),
                        mode="edge",
                    )
                clean_visuals.append(clean_visual)

                # read speaker label
                speakers.append(self.speaker_id[line.split(",")[c * 4 + 2]])
                
                mask_types.append(mask_type)
                mask_starts.append(mask_start)
                mask_lengths.append(mask_length)
            mixtures.append(mixture)
            mixtures.append(mixture)
           
        np_mixtures = np.asarray(mixtures)
        np_audios = np.asarray(audios)
        np_visuals = np.asarray(visuals)
        np_speakers = np.asarray(speakers)
        np_enrolls = np.asarray(enrolls)
        np_clean_visuals = np.asarray(clean_visuals)

        return (
            np_mixtures,
            np_audios,
            np_visuals,
            np_clean_visuals,
            np_enrolls,
            np_speakers,)
            # mask_starts,
            # mask_lengths,
            # mask_types,

    def __len__(self):
        # return 10 
        if len(self.minibatch)* self.batch_size>21000:
            return 20000//self.batch_size
        else:
            return len(self.minibatch)
         


class DistributedSampler(data.Sampler):
    def __init__(
        self, dataset, num_replicas=None, rank=None, shuffle=True, seed=0
    ):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError(
                    "Requires distributed package to be available"
                )
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError(
                    "Requires distributed package to be available"
                )
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(
            math.ceil(len(self.dataset) * 1.0 / self.num_replicas)
        )
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            # indices = torch.randperm(len(self.dataset), generator=g).tolist()
            ind = (
                torch.randperm(
                    int(len(self.dataset) / self.num_replicas), generator=g
                )
                * self.num_replicas
            )
            indices = []
            for i in range(self.num_replicas):
                indices = indices + (ind + i).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        # indices = indices[self.rank:self.total_size:self.num_replicas]
        indices = indices[
            self.rank * self.num_samples : (self.rank + 1) * self.num_samples
        ]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


def get_dataloader(args, partition):
    datasets = dataset(
        obj_dir=args.obj_dir,
        obj_mask_dir=args.obj_mask_dir,
        speaker_dict=args.speaker_dict,
        mix_lst_path=args.mix_lst_path,
        visual_direc=args.visual_direc,
        mixture_direc=args.mixture_direc,
        batch_size=args.batch_size,
        partition=partition,
        mix_no=args.C,
        max_length=args.max_length,
    )

    sampler = (
        DistributedSampler(
            datasets, num_replicas=args.world_size, rank=args.local_rank
        )
        if args.distributed
        else None
    )

    generator = data.DataLoader(
        datasets,
        batch_size=1,
        shuffle=(sampler is None),
        num_workers=args.num_workers,
        sampler=sampler,
        pin_memory=True,
    )

    return sampler, generator
