import sys
import numpy as np
import pdb
import argparse

# visual type: 
# 0: visual_missing 1: lip_occ 2: low_resolution 
mask_type_dict = {'0':'visual_missing','1':'lip_occ','2':'low_resolution'}
Sisnr_score = {
    'Mix_SISNR':{
        '0':[[], [], [], [], [], [], [], [], [], []],
        '1':[[], [], [], [], [], [], [], [], [], []],
        '2':[[], [], [], [], [], [], [], [], [], []]
    },
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

def print_score(score,save_path):    
    
    with open(save_path,'w') as file:
        print('\n',file=file)
    # Calculate averages for each condition and mask_type
    for condition, mask_types in score.items():
        all_values = []
        with open(save_path,'a') as file:
            print(f"\n{'=' * 40}",file=file)
            print(f"Condition: {condition}",file=file)
        
        mask_type_avgs = {}
        for mask_type, mix_ratios in mask_types.items():
            masktype_all_values = []
            mask_type_avgs[mask_type] = 0
            with open(save_path,'a') as file:
                print(f"\n  mask_type {mask_type_dict[mask_type]}:",file=file)
            for i, mix in enumerate(mix_ratios):
                if len(mix):  # Avoid calculating mean for empty lists
                    avg = np.mean(mix)
                else:
                    avg = 0.0
                mask_type_avgs[mask_type] = avg
                start = i * 10
                end = (i + 1) * 10
                with open(save_path,'a') as file:
                    print(f"    Ratio {start}%-{end}%: {avg:.2f}",file=file)
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
    data_lines = lines[4:]  
    data = []
    for line in data_lines:
        parts = [part.strip() for part in line.split(",")]
        assert len(parts)==10
        entry = {
            "UTT_ID": parts[0],
            "Mask_type": int(parts[1]),
            "V_length(fps)": int(parts[2]),
            "Mask_start": int(parts[3]),
            "Mask_length": int(parts[4]),
            "Mix_SISNR": float(parts[5]),
            "Visual_only": float(parts[6]),
            "SelfEnro+V": float(parts[7]),
            "Tgt+V": float(parts[8]),
            "Pre+V": float(parts[9]),
        }
        for key in Sisnr_score.keys():
            Sisnr_score[key][str(entry['Mask_type'])][
                min(int(10 * entry["Mask_length"] / entry["V_length(fps)"]), 9)].append(entry[key])
    print_score(Sisnr_score,args.sisnr_score_out_path)

 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sisnr_log",
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

 
    args = parser.parse_args()
    main(args)
