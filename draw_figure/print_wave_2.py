import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import pdb

def load_audio(file_path):
    y, sr = librosa.load(file_path, sr=None)
    return y, sr

def create_mixture(wav_paths, save_path="waveform_plot.png"):
    waveforms = [load_audio(path) for path in wav_paths]
    lengths = [len(wav[0]) for wav in waveforms]
    
    # pdb.set_trace()
    # 找到最长的作为干扰语音
    interfering_idx = np.argmax(lengths)
    interfering_wav, sr = waveforms[interfering_idx]

    interfering_wav = interfering_wav/np.max(abs(interfering_wav)) *0.6

    
    # 选择另外两个语音，并确保时间顺序
    remaining_idxs = [i for i in range(3) if i != interfering_idx]
    first_wav, _ = waveforms[min(remaining_idxs)]
    first_wav = first_wav[0:64000]
    first_wav = first_wav/np.max(abs(first_wav)) *0.5 
    second_wav, _ = waveforms[max(remaining_idxs)]
    second_wav = second_wav[0:64000]
    second_wav = second_wav/np.max(abs(second_wav)) *0.6
    
    # 拼接并裁剪到与干扰语音相同长度
    concatenated_wav = np.concatenate([first_wav, second_wav])
    if len(concatenated_wav) > len(interfering_wav):
        concatenated_wav = concatenated_wav[:len(interfering_wav)]
    concatenated_wav[0:64000] = 0
    # 找到中间点进行切换
    switch_point = len(concatenated_wav) // 2
    
    # 绘制波形
    plt.figure(figsize=(12, 6))
    librosa.display.waveshow(interfering_wav, sr=sr, alpha=0.5, color='orange', label='Interfering Speech')
    librosa.display.waveshow(concatenated_wav, sr=sr, alpha=0.5, color='red', label='Concatenated Speech')
    librosa.display.waveshow(first_wav, sr=sr, alpha=0.5, color='blue', label='Mixed Speech')
    
    plt.axvline(x=switch_point / sr, color='black', linestyle='--', label='Switch Point')
    plt.savefig(save_path)
    plt.show()

# 使用示例
wav_files = ['/home/data1/voxceleb2/test/aac/id00017/8_a6O3vdlU0/00013.wav', '/home/data1/voxceleb2/test/aac/id00017/8_a6O3vdlU0/00021.wav','/home/data1/voxceleb2/test/aac/id04094/DRq5F2261Ko/00074.wav']  # 替换为你的实际文件路径
create_mixture(wav_files)
