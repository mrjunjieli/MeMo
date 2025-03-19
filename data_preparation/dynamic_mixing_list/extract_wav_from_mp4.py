import os
import numpy as np
import argparse
import csv
import tqdm
import librosa
import scipy.io.wavfile as wavfile
import pdb 

EPS = np.finfo(float).eps
np.random.seed(0)


def extract_wav_from_mp4(line):
    # Extract .wav file from mp4
    video_from_path = (
        args.data_direc_mp4 + line[0] + "/" + line[1] + "/" + line[2] + ".mp4"
    )
    audio_save_path = (
        args.audio_data_direc + line[0] + "/" + line[1] + "/" + line[2] + ".wav"
    )
    if not os.path.exists(audio_save_path.rsplit("/", 1)[0]):
        os.makedirs(audio_save_path.rsplit("/", 1)[0])
    if not os.path.exists(audio_save_path):
        os.system(
            "ffmpeg -i %s %s" % (video_from_path, audio_save_path)
        )  # if audio not exist, then extract audio from video

    sr, audio = wavfile.read(audio_save_path)
    assert sr == args.sampling_rate, "sampling_rate mismatch"
    sample_length = audio.shape[0]
    return sample_length  # In seconds

def main(args):

    for path, dirs, files in os.walk(args.data_direc_mp4 + "/test/"):
        for filename in files:
            if filename[-4:] == ".mp4":
                ln = [
                    path.split("/")[-3],
                    path.split("/")[-2],
                    path.split("/")[-1] + "/" + filename.split(".")[0],
                ]
                sample_length = extract_wav_from_mp4(ln)


    for path, dirs, files in os.walk(args.data_direc_mp4 + "/train/"):
        for filename in files:
            if filename[-4:] == ".mp4":
                ln = [
                    path.split("/")[-3],
                    path.split("/")[-2],
                    path.split("/")[-1] + "/" + filename.split(".")[0],
                ]
                sample_length = extract_wav_from_mp4(ln)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Voxceleb2 dataset")
    parser.add_argument("--data_direc_mp4", type=str)
    parser.add_argument("--audio_data_direc", type=str)
    parser.add_argument("--sampling_rate", type=int)
    args = parser.parse_args()

    main(args)