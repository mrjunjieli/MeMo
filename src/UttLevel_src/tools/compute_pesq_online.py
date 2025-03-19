import sys
import numpy as np
import pdb
import argparse
from tools import audioread
import fast_bss_eval
from pystoi import stoi
from pesq import pesq
from speechmos import dnsmos
import os 
from tqdm import tqdm 

       
def str2bool(value):
    if isinstance(value,bool):
        return value
    if value.lower() in {'true','1','yes','y'}:
        return 1
    elif value.lower() in {'false','0','no','n'}:
        return 0
    else: 
        argparse.ArgumentTypeError('Boolean value expected.')



def main(args):
    #Calculate SI_SNR 
    with open(args.sisnr_log, "r") as file:
        lines = file.readlines()
    data_lines = lines[4:]  
    data = []
    with open(args.save_path,'w') as file:
        file.write(f"{'UTT_ID':<30},{'Mask_type':<10},{'V_length(fps)':<15},{'Mask_start':<15},{'Mask_length':<15},{'sdr_mix':<10},{'sdr_est':<10},{'pesq_mix':<10},{'pesq_est':<10},{'stoi_mix':<10},{'stoi_est':<10},{'mix_ovrl_mos':<15},{'mix_sig_mos':<15},{'mix_bak_mos':<15},{'mix_p808_mos':<15},{'est_ovrl_mos':<15},{'est_sig_mos':<15},{'est_bak_mos':<15},{'est_p808_mos':<15}\n")
    for line in tqdm(data_lines):
        parts = [part.strip() for part in line.split(",")]
        assert len(parts)==9
        entry = {
            "UTT_ID": parts[0],
            "Mask_type": int(parts[1]),
            "V_length(fps)": int(parts[2]),
            "Mask_start": int(parts[3]),
            "Mask_length": int(parts[4])
        }
        est_audio,sr = audioread(args.audio_dir+'/'+parts[0]+'_est.wav')
        tgt_audio,sr = audioread(args.audio_dir+'/'+parts[0]+'_tgt.wav')
        mix_audio,sr = audioread(args.audio_dir+'/'+parts[0]+'_mix.wav')
        sdr_mix = 0
        sdr_est = 0 
        if args.use_sdr:
            # sdr_mix = fast_bss_eval.sdr(np.expand_dims(tgt_audio,0),np.expand_dims(mix_audio,0))[0]
            sdr_est = fast_bss_eval.sdr(np.expand_dims(tgt_audio,0),np.expand_dims(est_audio,0))[0]
        pesq_est = 0 
        pesq_mix = 0
        if args.use_pesq: 
            if sr==16000:
                pesq_est = pesq(sr, tgt_audio, est_audio, 'wb')
                # pesq_mix = pesq(sr, tgt_audio, mix_audio, 'wb')
            if sr==8000:
                pesq_est = pesq(sr, tgt_audio, est_audio, 'nb')
                # pesq_mix = pesq(sr, tgt_audio, mix_audio, 'nb')
        stoi_mix = 0 
        stoi_est = 0
        if args.use_stoi:
            # stoi_mix = stoi(tgt_audio, mix_audio, sr, extended=False)
            stoi_est = stoi(tgt_audio, est_audio, sr, extended=False)
        
        est_ovrl_mos,est_sig_mos,est_bak_mos,est_p808_mos = 0, 0, 0, 0
        mix_ovrl_mos,mix_sig_mos,mix_bak_mos,mix_p808_mos = 0, 0, 0, 0
        if args.use_dnsmos:
            assert sr==16000
            est_output = dnsmos.run(est_audio, sr=sr)
            # mix_output = dnsmos.run(mix_audio, sr=sr)
            est_ovrl_mos,est_sig_mos,est_bak_mos,est_p808_mos = \
            est_output['ovrl_mos'],est_output['sig_mos'],est_output['bak_mos'],est_output['p808_mos']
            # mix_ovrl_mos,mix_sig_mos,mix_bak_mos,mix_p808_mos = \
            # mix_output['ovrl_mos'],mix_output['sig_mos'],mix_output['bak_mos'],mix_output['p808_mos']
        
        with open(args.save_path,'a') as file:
            file.write(f"{parts[0]:<30},{parts[1]:<10},{parts[2]:<15},{parts[3]:<15},{parts[4]:<15},{sdr_mix:<10.2f},{sdr_est:<10.2f},{pesq_mix:<10.2f},{pesq_est:<10.2f},{stoi_mix:<10.2f},{stoi_est:<10.2f},{mix_ovrl_mos:<15.2f},{mix_sig_mos:<15.2f},{mix_bak_mos:<15.2f},{mix_p808_mos:<15.2f},{est_ovrl_mos:<15.2f},{est_sig_mos:<15.2f},{est_bak_mos:<15.2f},{est_p808_mos:<15.2f}\n")
 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sisnr_log",
        type=str,
        default="",
        help="directory including train data",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="",
        help="directory including train data",
    )
    parser.add_argument(
        "--audio_dir",
        type=str,
        default="",
        help="directory including train data",
    )
    parser.add_argument(
        "--use_pesq",
        type=str2bool,
        default="false",
        help="directory including train data",
    )
    parser.add_argument(
        "--use_stoi",
        type=str2bool,
        default="false",
        help="directory including train data",
    )
    parser.add_argument(
        "--use_sdr",
        type=str2bool,
        default="false",
        help="directory including train data",
    )
    parser.add_argument(
        "--use_dnsmos",
        type=str2bool,
        default="false",
        help="directory including train data",
    )
    parser.add_argument(
        "--dnsmos_use_gpu",
        type=str2bool,
        default="false",
        help="directory including train data",
    )
    
    args = parser.parse_args()
    main(args)
