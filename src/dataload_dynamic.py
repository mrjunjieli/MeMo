import os
import numpy as np
# np.set_printoptions(threshold=np.inf)
# import torch 
# torch.set_printoptions(threshold=np.inf)
import librosa
import random
import math
import torch.distributed as dist
import torch
import torch.utils.data as data
import os
import cv2
import sys 
sys.path.append('/data/junjie/AVTSE_momentum')
from data_preparation.Visual_perturb import *


from tools import audiowrite, audioread, segmental_snr_mixer


EPS = np.finfo(float).eps
MAX_INT16 = np.iinfo(np.int16).max
np.random.seed(1234)
random.seed(1234)


class dataset(data.Dataset):
    def __init__(self,
                 obj_dir,
                obj_mask_dir,
                audio_wav_list,
                batch_size,
                partition='train',
                sampling_rate=16000,
                fps = 25,
                max_length=6,
                max_spk=2,
                mix_db =10):

        self.minibatch =[]
        self.mix_db = mix_db
        self.sampling_rate = sampling_rate
        self.partition = partition
        self.max_length = int(max_length)
        self.C=max_spk
        self.fps = fps 
        self.batch_size = batch_size
        self.obj_dir = obj_dir
        self.obj_mask_dir = obj_mask_dir
    
        self.normMean = 0.4161
        self.normStd = 0.1688

        #read wav list from audio_direc 
        self.wav_list=[]
        with open(audio_wav_list,'r') as p:
            for line in p:
                self.wav_list.append(line.strip())
        print('wav_list_length:',len(self.wav_list))
        

        start = 0
        while True:
            end = min(len(self.wav_list), start + self.batch_size)
            self.minibatch.append(self.wav_list[start:end])
            if end == len(self.wav_list):
                break
            start = end

        

    def __getitem__(self, index):
        
        if self.partition=='train' and index==0:
            random.shuffle(self.minibatch)
        batch_lst = self.minibatch[index]

        mixtures=[]
        audios=[]
        visuals=[]
        mix_mfccs= []

        for idx, line in enumerate(batch_lst):
            
            tgt_audio_path = line
            tgt_audio,sr = audioread(tgt_audio_path)
            if sr != self.sampling_rate:
                tgt_audio = librosa.resample(tgt_audio,orig_sr=sr, target_sr=self.sampling_rate)

            tgt_audio_length_second = len(tgt_audio)//self.sampling_rate
            audio_start=0
            if tgt_audio_length_second >self.max_length:
                left_length = tgt_audio_length_second-self.max_length
                audio_start = int(random.uniform(0,max(left_length-1,0)))
                tgt_audio = tgt_audio[audio_start*self.sampling_rate:(audio_start+self.max_length)*self.sampling_rate]
                tgt_audio_len = self.max_length
            elif tgt_audio_length_second <self.max_length:
                tgt_audio = np.pad(tgt_audio, ((0, int(self.sampling_rate* self.max_length - len(tgt_audio)))), mode = 'edge')
                tgt_audio_len = tgt_audio_length_second
            else:
                tgt_audio = tgt_audio[0:self.max_length*self.sampling_rate]
                tgt_audio_len = self.max_length
            interference_audio = np.zeros_like(tgt_audio)

            for num in range(self.C-1):
                inf_audio_path = np.random.choice(self.wav_list)
                inf_audio,sr = audioread(inf_audio_path)
                if sr != self.sampling_rate:
                    inf_audio = librosa.resample(inf_audio,orig_sr=sr, target_sr=self.sampling_rate)
                inf_audio_length_second = len(inf_audio)//self.sampling_rate 
                tmp_start=0
                if inf_audio_length_second >self.max_length:
                    left_length = inf_audio_length_second-self.max_length
                    tmp_start = int(random.uniform(0,max(left_length-1,0)))
                    inf_audio = inf_audio[tmp_start*self.sampling_rate:(tmp_start+self.max_length)*self.sampling_rate]
                elif inf_audio_length_second <self.max_length:
                    inf_audio = np.pad(inf_audio, ((0, int(self.sampling_rate* self.max_length - len(tgt_audio)))), mode = 'edge')
                else:
                    inf_audio = inf_audio[0:self.max_length*self.sampling_rate]
                snr_speech = round(np.random.uniform(-self.mix_db,self.mix_db),2)
                _, _, interference_audio, _ = segmental_snr_mixer(interference_audio,inf_audio,snr_speech,min_option=False)
                             

            snr_speech = round(np.random.uniform(-self.mix_db,self.mix_db),2) 
            tgt_audio, inf_audio, mix_audio, _ = segmental_snr_mixer(tgt_audio,interference_audio,snr_speech,min_option=False)
            video_path = tgt_audio_path.replace('/wav/','/mp4/')
            video_path = video_path.replace('.wav','.mp4')

            captureObj = cv2.VideoCapture(video_path)
            roiSequence = []
            clean_roiSequence = []
            roiSize = 112
            start = 0
            mask_type = random.choice([0,1,2])
            mask_start = random.randint(0,tgt_audio_len-1)
            mask_length = random.randint(1,tgt_audio_len-mask_start)
            mask_end = mask_start+mask_length
            mask_end *=25 
            mask_start *=25 
            mask_length *=25
            # print(mask_start,mask_end,mask_type,tgt_audio_len*)
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
            while captureObj.isOpened():
                ret, frame = captureObj.read()
                if ret == True:
                    grayed = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    grayed = grayed / 255
                    grayed = cv2.resize(grayed, (roiSize * 2, roiSize * 2))
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
                            frame = cv2.resize(
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
                            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                            # cv.imwrite('frames/'+'org_image_'+str(id_)+'.png',roi)
                            # pdb.set_trace()
                            roi = roi / 255
                            # id_+=1
                        elif mask_type == 2:
                            frame = cv2.resize(
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
                                roi = cv2.resize(
                                    roi,
                                    (roiSize // times, roiSize // times),
                                )
                                roi = cv2.resize(roi, (roiSize, roiSize))
                            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
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
            visual = visual[0 : int(self.max_length * self.fps), ...]
            visual = (visual - self.normMean) / self.normStd
            if visual.shape[0] < int(self.max_length * self.fps):
                visual = np.pad(
                    visual,
                    (
                        (0, int(self.max_length * self.fps) - visual.shape[0]),
                        (0, 0),
                        (0, 0),
                    ),
                    mode="edge",
                )
            visuals.append(visual)
            mixtures.append(mix_audio)
            audios.append(tgt_audio)
            np_mixtures = np.asarray(mixtures)
            np_audios = np.asarray(audios)
            np_visuals = np.asarray(visuals)

        return np_mixtures,np_audios,np_visuals,1,1,1

    def __len__(self):
        if self.partition=='train':
            return len(self.minibatch)//200
            # return 10 




class DistributedSampler(data.Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, seed=0):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            # indices = torch.randperm(len(self.dataset), generator=g).tolist()
            ind = torch.randperm(int(len(self.dataset)/self.num_replicas), generator=g)*self.num_replicas
            indices = []
            for i in range(self.num_replicas):
                indices = indices + (ind+i).tolist()
        else:
            indices = list(range(len(self.dataset)))
        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        # indices = indices[self.rank:self.total_size:self.num_replicas]
        indices = indices[self.rank*self.num_samples:(self.rank+1)*self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch



def get_dataloader_dy(args, partition):
    datasets = dataset(
                obj_dir=args.obj_dir,
                obj_mask_dir=args.obj_mask_dir,
                audio_wav_list="/data/junjie/AVTSE_momentum/data_preparation/dynamic_mixing_list/data.lst",
                batch_size=args.batch_size,
                max_length=args.max_length,
                partition=partition,
                sampling_rate=16000,
                fps = 25,
                max_spk=3,
                mix_db =10)
    sampler = DistributedSampler(
        datasets,
        num_replicas=args.world_size,
        rank=args.local_rank) if args.distributed else None

    generator = data.DataLoader(datasets,
            batch_size = 1,
            shuffle = (sampler is None),
            num_workers = args.num_workers,
            sampler=sampler)

    return sampler, generator


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser("av-yolo")
    
    # Dataloader
    parser.add_argument('--max_length', default=6, type=int)
    # Training    
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Batch size')
    parser.add_argument('--num_workers', default=1, type=int,
                        help='Number of workers to generate minibatch')
    parser.add_argument('--C', type=int, default=3,
                        help='number of speakers to mix')

    # Distributed training
    parser.add_argument("--local_rank", default=0, type=int)

    parser.add_argument(
        "--obj_dir",
        type=str,
        default="/data/junjie/AVTSE_momentum/data_preparation/Asset/object_image_sr",
        help="location of occlusion patch mask",
    )
    parser.add_argument(
        "--obj_mask_dir",
        default="/data/junjie/AVTSE_momentum/data_preparation/Asset/object_mask_x4",
        help="location of occlusion patch mask",
    )

    args = parser.parse_args()

    args.distributed = True
    args.world_size = 1
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
        args.world_size = int(os.environ['WORLD_SIZE'])


    train_sampler, train_generator = get_dataloader_dy(args, 'train')
    
    mix_total = 0
    # save_folder = 'tempory_folder'
    # if not os.path.exists('./'+save_folder):
    #     os.makedirs('./'+save_folder)

    for i, (a_mix, a_tgt, v_tgt, clean_v_tgt, enroll_audio, speaker,) in enumerate(train_generator):  
        audiowrite('mix.wav',a_mix[0][0])
        audiowrite('tgt.wav',a_tgt[0][0])
        pdb.set_trace()
