import time
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
import torch.nn as nn
from datetime import datetime
import torch
import random
import numpy as np
import torch.nn.functional as F
from tools import audioread, audiowrite, load_model, cal_SISNR, normalize

import copy 
EPS = np.finfo(float).eps
import pdb

        

class Solver(object):
    def __init__(self, train_data, validation_data, model, optimizer, args):
        self.train_data = train_data
        self.validation_data = validation_data
        self.args = args
        self.ce_loss = nn.CrossEntropyLoss()

        self.print = False
        if (
            self.args.distributed and self.args.local_rank == 0
        ) or not self.args.distributed:
            self.print = True
            if self.args.use_tensorboard:
                self.writer = SummaryWriter("%s/tensorboard/" % args.log_name)
        self.model = model
        self.optimizer = optimizer

        if self.args.distributed:
            self.model = DDP(self.model, find_unused_parameters=True)
        self.shift_range = self.args.shift_range.split(",")
        self.Self_enroll_amplitude_scaling = args.Self_enroll_amplitude_scaling.split(",")
        self.num_slot_range = args.num_slots.split(',')
        self._reset()
        self.v_loss_weight = int(args.loss_weight[0])*0.1
        self.av_loss_weight = int(args.loss_weight[1])*0.1
        assert self.v_loss_weight+self.av_loss_weight == 1 

    def _reset(self):
        self.halving = False
        if self.args.continue_from and self.args.continue_from.lower() != "false":
            self.model = load_model(
                self.model, "%s/model_dict_last.pt" % self.args.continue_from
            )
            checkpoint = torch.load(
                "%s/model_dict_last.pt" % self.args.continue_from, map_location="cpu",
            )
            self.optimizer.load_state_dict(checkpoint["optimizer"])

            self.start_epoch = checkpoint["epoch"]
            self.best_val_loss = checkpoint["best_val_loss"]
            self.val_no_impv = checkpoint["val_no_impv"]

            if self.print:
                print("Resume training from epoch: {}".format(self.start_epoch))
        else:
            self.best_val_loss = float("inf")
            self.val_no_impv = 0
            self.start_epoch = 1
            if self.print:
                print("Start new training")

    def train(self):
        for epoch in range(self.start_epoch, self.args.epochs + 1):
            self.joint_loss_weight = epoch
            if self.args.distributed:
                self.args.train_sampler.set_epoch(epoch)
            # # Train
            self.model.train()
            start = time.time()
            tr_loss = self._run_one_epoch(
                data_loader=self.train_data,
                state="train",
                epoch=epoch,
                teacher_point=self.args.teacher_point,
            )
            reduced_tr_loss = self._reduce_tensor(tr_loss)

            if self.print:
                print(
                    "Train Summary | End of Epoch {0} | Time {1:.2f}s | Current time {2} |"
                    "Train Loss {3:.3f}| ".format(
                        epoch, time.time() - start, datetime.now(), reduced_tr_loss,
                    )
                )

            # Validation
            self.model.eval()
            start = time.time()
            with torch.no_grad():
                val_loss = self._run_one_epoch(
                    data_loader=self.validation_data, state="val", epoch=epoch
                )
                reduced_val_loss = self._reduce_tensor(val_loss)
                if self.print:
                    print(
                        "Valid Summary | End of Epoch {0} | Time {1:.2f}s | Current time {2} |"
                        "Valid Loss {3:.3f}| ".format(
                            epoch,
                            time.time() - start,
                            datetime.now(),
                            reduced_val_loss,
                        )
                    )

            # Check whether to adjust learning rate and early stop
            find_best_model = False
            if reduced_val_loss >= self.best_val_loss:
                self.val_no_impv += 1
                if self.val_no_impv >= 10:
                    if self.print:
                        print("No improvement for 10 epochs, early stopping.")
                    break
            else:
                self.val_no_impv = 0
                self.best_val_loss = reduced_val_loss
                find_best_model = True

            if self.val_no_impv == 6:
                self.halving = True

            # Halving the learning rate
            if self.halving:
                optim_state = self.optimizer.state_dict()
                optim_state["param_groups"][0]["lr"] = (
                    optim_state["param_groups"][0]["lr"] / 2
                )
                self.optimizer.load_state_dict(optim_state)
                if self.print:
                    print(
                        "Learning rate adjusted to: {lr:.6f}".format(
                            lr=optim_state["param_groups"][0]["lr"]
                        )
                    )
                self.halving = False

            if self.print:
                # Tensorboard logging
                if self.args.use_tensorboard:
                    self.writer.add_scalar("Train_loss", reduced_tr_loss, epoch)
                    self.writer.add_scalar("Validation_loss", reduced_val_loss, epoch)

                # Save model
                checkpoint = {
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "epoch": epoch + 1,
                    "best_val_loss": self.best_val_loss,
                    "val_no_impv": self.val_no_impv,
                }
                torch.save(checkpoint, self.args.log_name + "/model_dict_last.pt")
                if find_best_model:
                    torch.save(checkpoint, self.args.log_name + "/model_dict_best.pt")
                    print("Found new best model, dict saved")
                if epoch % 10 == 0:
                    torch.save(
                        checkpoint,
                        self.args.log_name + "/model_dict_" + str(epoch) + ".pt",
                    )

    def _run_one_epoch(
        self, data_loader, state, teacher_point=50, epoch=0
    ):
        step = 0
        total_step = len(data_loader)
        total_loss = 0
        stage_1_loss = 0 
        stage_2_loss = 0

        self.accu_count = 0
        self.optimizer.zero_grad()
        for i, (a_mix, a_tgt, v_tgt, clean_v_tgt, enroll_audio, speaker,) in enumerate(data_loader):              
            a_mix = a_mix.cuda().squeeze(0).float() # 1 x T  
            a_tgt = a_tgt.cuda().squeeze(0).float()
            v_tgt = v_tgt.cuda().squeeze(0).float()
            clean_v_tgt = clean_v_tgt.cuda().squeeze(0).float()
            enroll_audio = enroll_audio.cuda().squeeze(0).float()
            speaker = speaker.cuda().squeeze(0)
            a_zero = torch.zeros_like(a_tgt,device=a_tgt.device)
            v_zero = torch.zeros_like(v_tgt,device=v_tgt.device)
            
            noise_speech = torch.zeros_like(a_tgt)
            noise_speech[::2] = a_tgt[1::2]  # 偶数索引换成奇数索引的数据
            noise_speech[1::2] = a_tgt[::2]  # 奇数索引换成偶数索引的数据
            noise_speech = noise_speech.cuda()
            if state == "train":
                if 'spk' not in self.args.model_name.lower() and 'self' not in self.args.model_name.lower():
                    est_a_tgt, _, est_a_noise = self.model(a_mix, v_tgt)
                    B = a_mix.shape[0]
                    snr_s_main = 0-cal_SISNR(a_tgt, est_a_tgt[-B:,:])
                    snr_n = 0-cal_SISNR(noise_speech.repeat(6,1), est_a_noise)
                    snr_s_rest = 0-cal_SISNR(a_tgt.repeat(5,1), est_a_tgt[B:,:])
                    sisnr_loss = snr_s_main+0.1*(snr_s_rest+snr_n)
                    sisnr_loss.backward()
                    loss = sisnr_loss
                else:
                    loss = 0
                    if 'self' in self.args.model_name.lower() and 'spk' not in self.args.model_name.lower():
                        # v_only training 
                        sisnr_loss1 = 0
                        if self.v_loss_weight !=0:
                            est_a_tgt1,_,est_a_noise = self.model(a_mix, v_tgt,a_zero,a_zero)

                            B = a_mix.shape[0]
                            snr_s_main = 0-cal_SISNR(a_tgt, est_a_tgt1[-B:,:])
                            snr_n = 0-cal_SISNR(noise_speech.repeat(6,1), est_a_noise)
                            snr_s_rest = 0-cal_SISNR(a_tgt.repeat(5,1), est_a_tgt1[B:,:])
                            sisnr_loss1 = self.v_loss_weight *(snr_s_main+0.1*(snr_s_rest+snr_n))
                            sisnr_loss1.backward()

                            stage_1_loss += (snr_s_main+0.1*(snr_s_rest+snr_n)).data

                    #V+self_enroll training 
                    target_level = random.uniform(
                        float(self.Self_enroll_amplitude_scaling[0]),
                        float(self.Self_enroll_amplitude_scaling[1]))
                    self_enroll = copy.deepcopy(est_a_tgt1[-B:,:].detach())
                    if epoch < teacher_point:
                        self_enroll_rms = (self_enroll ** 2).mean() ** 0.5
                        a_tgt_rms = (a_tgt ** 2).mean() ** 0.5
                        weight = epoch / teacher_point
                        self_enroll = weight * self_enroll + (1 - weight) * (
                            a_tgt * self_enroll_rms / (a_tgt_rms + EPS)
                        )
                    # increase diversity of magnitude of self_enroll
                    self_enroll = self_enroll / (torch.max(abs(self_enroll))) * target_level
                    shift = int(random.uniform(
                                float(self.shift_range[0]),
                                float(self.shift_range[1])) * self.args.sample_rate)
                    self_enroll_shift = copy.deepcopy(self_enroll[:, :-shift])
                    self_enroll_shift = F.pad(
                        self_enroll_shift, (a_tgt.shape[1] - self_enroll_shift.shape[1], 0)
                    )
                    enroll_audio = enroll_audio / (torch.max(abs(enroll_audio))) * target_level

                    self_enroll_input = None 
                    pre_enroll_input = None 
                        
                    if 'selfmem' in  self.args.model_name.lower():
                        self_mem_bank = []
                        num_slots = random.randint(int(self.num_slot_range[0]),int(self.num_slot_range[1]))
                        self_mem_bank.append(self_enroll_shift)
                        for _ in range(num_slots-1):
                            shift = int(random.uniform(
                                        float(self.shift_range[0]),
                                        float(self.shift_range[1])) * self.args.sample_rate)
                            self_enroll_shift = copy.deepcopy(self_enroll[:, :-shift])
                            self_enroll_shift = F.pad(self_enroll_shift, (a_tgt.shape[1] - self_enroll_shift.shape[1], 0))
                            self_mem_bank.append(self_enroll_shift)  
                        random.shuffle(self_mem_bank)
                        self_enroll_input = self_mem_bank
                    
                    sisnr_loss2 =0
                    if self.av_loss_weight != 0:
                        est_a_tgt2,_,est_a_noise = self.model(a_mix, v_tgt,self_enroll=self_enroll_input, pre_enroll = pre_enroll_input)       
                        
                        B = a_mix.shape[0]
                        snr_s_main = 0-cal_SISNR(a_tgt, est_a_tgt2[-B:,:])
                        snr_n = 0-cal_SISNR(noise_speech.repeat(6,1), est_a_noise)
                        snr_s_rest = 0-cal_SISNR(a_tgt.repeat(5,1), est_a_tgt2[B:,:])
                        sisnr_loss2 = self.av_loss_weight *(snr_s_main+0.1*(snr_s_rest+snr_n))
                        sisnr_loss2.backward()
                        
                        stage_2_loss+=(snr_s_main+0.1*(snr_s_rest+snr_n)).data
                    loss = self.v_loss_weight * stage_1_loss + self.av_loss_weight * stage_2_loss

                self.accu_count += 1
                step += 1
                total_loss = loss

                # loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                if 'spk' not in self.args.model_name.lower() and 'self' not in self.args.model_name.lower():
                    if i % 1000 == 0:
                        print(
                            "step:{}/{} avg loss:{:.3f}".format(
                                step, total_step,total_loss / step
                            )
                        )
                else: 
                    if i % 1000 == 0:
                        print(
                            "step:{}/{} avg loss:{:.3f}, stage_1_loss:{:.3f}, stage_2_loss:{:.3f}".format(
                                step, total_step,total_loss / step, stage_1_loss / step,
                                stage_2_loss/step)
                                )
            else:
                sisnr_loss1=0
                sisnr_loss2=0
                if 'spk' not in self.args.model_name.lower() and 'self' not in self.args.model_name.lower():
                    est_a_tgt, _ ,_= self.model(a_mix, v_tgt)
                    B = a_mix.shape[0]
                    max_snr = cal_SISNR(a_tgt, est_a_tgt[-B:,:])
                    sisnr_loss = 0-torch.mean(max_snr)
                    loss = sisnr_loss
                else:
                    shift = int(0.2 * self.args.sample_rate)
                    if 'self' in self.args.model_name.lower() and 'spk' not in self.args.model_name.lower():
                        est_a_tgt1, _,_ = self.model(a_mix, clean_v_tgt,a_zero,a_zero)
                    B = a_mix.shape[0]
                    self_enroll = copy.deepcopy(est_a_tgt1[-B:,:])
                    max_snr1 = cal_SISNR(a_tgt, est_a_tgt1[-B:,:])
                    sisnr_loss1 =  (0 - torch.mean(max_snr1))
                    
                    self_enroll = self_enroll / torch.max(abs(self_enroll)) * 0.7
                    self_enroll_shift = copy.deepcopy(self_enroll[:, :-shift])
                    self_enroll_shift = F.pad(
                        self_enroll_shift, (a_tgt.shape[1] - self_enroll_shift.shape[1], 0)
                    )

                    est_a_tgt2, _,_ = self.model(a_mix, v_tgt, self_enroll_shift)
                    B = a_mix.shape[0]
                    max_snr2 = cal_SISNR(a_tgt, est_a_tgt2[-B:,:])
                    sisnr_loss2 =  (0 - torch.mean(max_snr2))

                    loss = self.v_loss_weight * sisnr_loss1 + self.av_loss_weight * sisnr_loss2
                step += 1
                total_loss += loss.data
                stage_1_loss += sisnr_loss1 
                stage_2_loss+=sisnr_loss2


                if 'spk' not in self.args.model_name.lower() and 'self' not in self.args.model_name.lower():
                    if i % 300 == 0:
                        print(
                            "step:{}/{} avg loss:{:.3f}".format(
                                step, total_step,total_loss / step
                            )
                        )
                else: 
                    if i % 300 == 0:
                        print(
                            "step:{}/{} avg loss:{:.3f}, stage_1_loss:{:.3f}, stage_2_loss:{:.3f}".format(
                                step, total_step,total_loss / step, stage_1_loss / step, 
                                stage_2_loss/step)
                                )
        return total_loss / (i + 1)

    def _reduce_tensor(self, tensor):
        if not self.args.distributed:
            return tensor
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt /= self.args.world_size
        return rt
