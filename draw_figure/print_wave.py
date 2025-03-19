import numpy as np
import matplotlib.pyplot as plt
import wave
import struct

def read_wavefile(file_path):
    with wave.open(file_path, 'r') as wav_file:
        # 获取基本信息
        num_frames = wav_file.getnframes()
        framerate = wav_file.getframerate()
        
        # 读取音频数据
        frames = wav_file.readframes(num_frames)
        # 将音频数据解码为numpy数组
        waveform = np.array(struct.unpack("<" + str(num_frames) + "h", frames))
        
        return waveform, framerate

def plot_waveforms(wavefile1, wavefile2):
    # 读取两个波形文件
    waveform1, framerate1 = read_wavefile(wavefile1)
    waveform2, framerate2 = read_wavefile(wavefile2)
    
    # 创建一个时间轴
    time1 = np.linspace(0, len(waveform1) / framerate1, num=len(waveform1))
    time2 = np.linspace(0, len(waveform2) / framerate2, num=len(waveform2))
    
    # 创建绘图
    plt.figure(figsize=(10, 6))

    # 绘制第一个waveform（绿色）
    plt.plot(time1, waveform1, color='orange', label='Waveform 1', alpha=0.7)
    
    # 绘制第二个waveform（红色）
    plt.plot(time2, waveform2, color='blue', label='Waveform 2', alpha=0.7)
    
    # 添加标题和标签
    # plt.title('Speech Waveforms')
    # plt.xlabel('Time [s]')
    # plt.ylabel('Amplitude')
    
    # 显示图例
    # plt.legend()
    
    # 显示图形
    plt.show()
    plt.savefig('mix.png')

# 调用示例：输入两个wav文件的路径
plot_waveforms('/home/data1/voxceleb2/test/aac/id00017/8_a6O3vdlU0/00013.wav', '/home/data1/voxceleb2/test/aac/id00017/8_a6O3vdlU0/00021.wav')
