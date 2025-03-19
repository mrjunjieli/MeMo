import sys
import numpy as np
import pdb
import argparse
import os 

# visual type: 
# 0: visual_missing 1: lip_occ 2: low_resolution 
mask_type_dict = {'0':'visual_missing','1':'lip_occ','2':'low_resolution'}
Sisnr_score = {
    'Visual_only':{
        '0':[[], [], [], [], [], [], [], [], [], []],
        '1':[[], [], [], [], [], [], [], [], [], []],
        '2':[[], [], [], [], [], [], [], [], [], []]
    },
    'SelfEnro+V':{
        '0':[[], [], [], [], [], [], [], [], [], []],
        '1':[[], [], [], [], [], [], [], [], [], []],
        '2':[[], [], [], [], [], [], [], [], [], []]
    },
    'Tgt+V':{
        '0':[[], [], [], [], [], [], [], [], [], []],
        '1':[[], [], [], [], [], [], [], [], [], []],
        '2':[[], [], [], [], [], [], [], [], [], []]
    },
    'Pre+V':{
        '0':[[], [], [], [], [], [], [], [], [], []],
        '1':[[], [], [], [], [], [], [], [], [], []],
        '2':[[], [], [], [], [], [], [], [], [], []]
    },
}

Other_scores = {
    'sdr_mix':{
    '0':[[], [], [], [], [], [], [], [], [], []],
    '1':[[], [], [], [], [], [], [], [], [], []],
    '2':[[], [], [], [], [], [], [], [], [], []]
    },
    'sdr_est':{
    '0':[[], [], [], [], [], [], [], [], [], []],
    '1':[[], [], [], [], [], [], [], [], [], []],
    '2':[[], [], [], [], [], [], [], [], [], []]
    },
    'pesq_mix':{
    '0':[[], [], [], [], [], [], [], [], [], []],
    '1':[[], [], [], [], [], [], [], [], [], []],
    '2':[[], [], [], [], [], [], [], [], [], []]
    },
    'pesq_est':{
    '0':[[], [], [], [], [], [], [], [], [], []],
    '1':[[], [], [], [], [], [], [], [], [], []],
    '2':[[], [], [], [], [], [], [], [], [], []]
    },
    'stoi_mix':{
    '0':[[], [], [], [], [], [], [], [], [], []],
    '1':[[], [], [], [], [], [], [], [], [], []],
    '2':[[], [], [], [], [], [], [], [], [], []]
    },
    'stoi_est':{
    '0':[[], [], [], [], [], [], [], [], [], []],
    '1':[[], [], [], [], [], [], [], [], [], []],
    '2':[[], [], [], [], [], [], [], [], [], []]
    },
    'mix_ovrl_mos':{
    '0':[[], [], [], [], [], [], [], [], [], []],
    '1':[[], [], [], [], [], [], [], [], [], []],
    '2':[[], [], [], [], [], [], [], [], [], []]
    },
    'mix_sig_mos':{
    '0':[[], [], [], [], [], [], [], [], [], []],
    '1':[[], [], [], [], [], [], [], [], [], []],
    '2':[[], [], [], [], [], [], [], [], [], []]
    },
    'mix_bak_mos':{
    '0':[[], [], [], [], [], [], [], [], [], []],
    '1':[[], [], [], [], [], [], [], [], [], []],
    '2':[[], [], [], [], [], [], [], [], [], []]
    },
    'mix_p808_mos':{
    '0':[[], [], [], [], [], [], [], [], [], []],
    '1':[[], [], [], [], [], [], [], [], [], []],
    '2':[[], [], [], [], [], [], [], [], [], []]
    },
    'est_ovrl_mos':{
    '0':[[], [], [], [], [], [], [], [], [], []],
    '1':[[], [], [], [], [], [], [], [], [], []],
    '2':[[], [], [], [], [], [], [], [], [], []]
    },
    'est_sig_mos':{
    '0':[[], [], [], [], [], [], [], [], [], []],
    '1':[[], [], [], [], [], [], [], [], [], []],
    '2':[[], [], [], [], [], [], [], [], [], []]
    },
    'est_bak_mos':{
    '0':[[], [], [], [], [], [], [], [], [], []],
    '1':[[], [], [], [], [], [], [], [], [], []],
    '2':[[], [], [], [], [], [], [], [], [], []]
    },
    'est_p808_mos':{
    '0':[[], [], [], [], [], [], [], [], [], []],
    '1':[[], [], [], [], [], [], [], [], [], []],
    '2':[[], [], [], [], [], [], [], [], [], []]
    },
}

def print_score(score,save_path):    
    with open(save_path,'w') as file:
        print('\n',file=file)
    
    # Calculate averages for each condition and mask_type
    for condition, mask_types in score.items():
        ratio_score = [[], [], [], [], [], [], [], [], [], []]
        all_values = []
        with open(save_path,'a') as file:
            print(f"\n{'=' * 40}",file=file)
            print(f"Condition: {condition}",file=file)
        
        for mask_type, mix_ratios in mask_types.items():
            masktype_all_values = []
            with open(save_path,'a') as file:
                print(f"\n  mask_type {mask_type_dict[mask_type]}:",file=file)
            for i, mix in enumerate(mix_ratios):
                if len(mix):  # Avoid calculating mean for empty lists
                    avg = np.mean(mix)
                else:
                    avg = 0.0
                start = i * 10
                end = (i + 1) * 10
                # if condition=='SelfEnro+V' and i ==6:
                #     pdb.set_trace()
                ratio_score[i].extend(mix)

                masktype_all_values.extend(mix)  # Combine all values for overall average
            if masktype_all_values:
                with open(save_path,'a') as file:
                    print(f"  Overall average for {mask_type_dict[mask_type]}: {np.mean(masktype_all_values):.2f}",file=file)
            else:
                with open(save_path,'a') as file:
                    print(f"  Overall average for {mask_type_dict[mask_type]}: N/A",file=file)
            all_values.extend(masktype_all_values)
        with open(save_path,'a') as file:
            print('\n',file=file)
        if ratio_score:
            for i in range(10):
                if len(ratio_score[i]):
                    avg = np.mean(ratio_score[i])
                else:
                    avg = 0.0
                start = i * 10
                end = (i + 1) * 10
                with open(save_path,'a') as file:
                    print(f"  Ratio {start}%-{end}%: {avg:.2f}",file=file)    
        with open(save_path,'a') as file:
            print('\n',file=file)
        if all_values:
            with open(save_path,'a') as file:
                print(f"Overall average for {condition}: {np.mean(all_values):.2f}",file=file)
        else:
            with open(save_path,'a') as file:
                print(f"Overall average for {condition}: N/A",file=file)
        

def main(args):
    #Calculate SI_SNR 
    with open(args.sisnr_log, "r") as file:
        lines = file.readlines()
    data_lines = lines[6:]  
    data = []
    # import pdb;pdb.set_trace()
    for line in data_lines:
        parts = [part.strip() for part in line.split(",")]
        assert len(parts)==9
        entry = {
            "UTT_ID": parts[0],
            "Mask_type": int(parts[1]),
            "V_length(fps)": int(parts[2]),
            "Mask_start": int(parts[3]),
            "Mask_length": int(parts[4]),
            "Visual_only": float(parts[5]),
            "SelfEnro+V": float(parts[6]),
            "Tgt+V": float(parts[7]),
            "Pre+V": float(parts[8]),
        }
        for key in Sisnr_score.keys():
            Sisnr_score[key][str(entry['Mask_type'])][
                min(int(10 * entry["Mask_length"] / entry["V_length(fps)"]), 9)].append(entry[key])
    print_score(Sisnr_score,args.sisnr_score_out_path)

    #Calculate pesq stoi sdr dnsmos 
    if os.path.exists(args.pesq_log):
            
        with open(args.pesq_log, "r") as file:
            lines = file.readlines()
        data_lines = lines[1:]  
        data = []
        for line in data_lines:
            parts = [part.strip() for part in line.split(",")]
            assert len(parts)==19
            entry = {
                "UTT_ID": parts[0],
                "Mask_type": int(parts[1]),
                "V_length(fps)": int(parts[2]),
                "Mask_start": int(parts[3]),
                "Mask_length": int(parts[4]),
                "sdr_mix": float(parts[5]),
                "sdr_est": float(parts[6]),
                "pesq_mix": float(parts[7]),
                "pesq_est": float(parts[8]),
                "stoi_mix": float(parts[9]),
                "stoi_est": float(parts[10]),
                'mix_ovrl_mos':float(parts[11]),
                'mix_sig_mos':float(parts[12]),
                'mix_bak_mos':float(parts[13]),
                'mix_p808_mos':float(parts[14]),
                'est_ovrl_mos':float(parts[15]),
                'est_sig_mos':float(parts[16]),
                'est_bak_mos':float(parts[17]),
                'est_p808_mos':float(parts[18]),
            }
            for key in Other_scores.keys():
                Other_scores[key][str(entry['Mask_type'])][
                    min(int(10 * entry["Mask_length"] / entry["V_length(fps)"]), 9)].append(entry[key])
        print_score(Other_scores,args.pesq_score_out_path)

 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sisnr_log",
        type=str,
        default="",
        help="directory including train data",
    )
    parser.add_argument(
        "--pesq_log",
        type=str,
        default="",
        help="directory including train data",
    )
    parser.add_argument(
        "--sisnr_score_out_path",
        type=str,
        default="",
        help="directory including train data",
    )
    parser.add_argument(
        "--pesq_score_out_path",
        type=str,
        default="",
        help="directory including train data",
    )
 
    args = parser.parse_args()
    main(args)
