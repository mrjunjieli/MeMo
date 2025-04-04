import os
import numpy as np
import soundfile as sf
import torch
import pdb


EPS = np.finfo(float).eps
np.random.seed(0)


def load_model(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model"]
    model_dict = model.state_dict()

    # 1. 检查是否有 'module.' 前缀
    new_state_dict = {}
    for k, v in state_dict.items():
        # 1.1 如果当前模型是多卡（有 'module.' 前缀）但加载的参数没有 'module.' 前缀
        if k.startswith("module.") and not any(
            key.startswith("module.") for key in model_dict
        ):
            new_key = k[len("module.") :]  # 去掉 'module.' 前缀
            new_state_dict[new_key] = v

        # 1.2 如果当前模型是单卡（没有 'module.' 前缀）但加载的参数有 'module.' 前缀
        elif not k.startswith("module.") and any(
            key.startswith("module.") for key in model_dict
        ):
            new_key = "module." + k  # 添加 'module.' 前缀
            new_state_dict[new_key] = v

        # 1.3 当前模型和加载的参数前缀一致
        else:
            new_state_dict[k] = v

    # 2. 检查模型结构是否一致
    for k, v in model_dict.items():
        if k in new_state_dict:
            try:
                model_dict[k].copy_(new_state_dict[k])
            except Exception as e:
                print(f"Error in copying parameter {k}: {e}")
        else:
            # pdb.set_trace()
            print(f"Parameter {k} not found in checkpoint. Skipping...")

    # 3. 更新模型参数
    model.load_state_dict(model_dict)

    return model


def cal_logpower(source):
    ratio = torch.sum(source**2, axis=-1)
    sdr = 10 * torch.log10(ratio / (source.shape[-1] / 16000) + EPS)
    return sdr


def cal_SISNR(source, estimate_source):
    """Calcuate Scale-Invariant Source-to-Noise Ratio (SI-SNR)
    Args:
        source: torch tensor, [batch size, sequence length]
        estimate_source: torch tensor, [batch size, sequence length]
    Returns:
        SISNR, [batch size]
    """
    assert source.size() == estimate_source.size()

    # Step 1. Zero-mean norm
    source = source - torch.mean(source, axis=-1, keepdim=True)
    estimate_source = estimate_source - torch.mean(
        estimate_source, axis=-1, keepdim=True
    )

    # Step 2. SI-SNR
    # s_target = <s', s>s / ||s||^2
    ref_energy = torch.sum(source**2, axis=-1, keepdim=True) + EPS
    proj = (
        torch.sum(source * estimate_source, axis=-1, keepdim=True)
        * source
        / ref_energy
    )
    # e_noise = s' - s_target
    noise = estimate_source - proj
    # SI-SNR = 10 * log_10(||s_target||^2 / ||e_noise||^2)
    ratio = torch.sum(proj**2, axis=-1) / (torch.sum(noise**2, axis=-1) + EPS)
    sisnr = 10 * torch.log10(ratio + EPS)

    return torch.mean(sisnr)


def audiowrite(
    destpath,
    audio,
    sample_rate=16000,
    norm=False,
    target_level=-25,
    clipping_threshold=0.99,
    clip_test=False,
):
    """Function to write audio"""

    if clip_test:
        if is_clipped(audio, clipping_threshold=clipping_threshold):
            raise ValueError(
                "Clipping detected in audiowrite()! "
                + destpath
                + " file not written to disk."
            )

    if norm:
        audio = normalize(audio, target_level)
        max_amp = max(abs(audio))
        if max_amp >= clipping_threshold:
            audio = audio / max_amp * (clipping_threshold - EPS)

    destpath = os.path.abspath(destpath)
    destdir = os.path.dirname(destpath)

    if not os.path.exists(destdir):
        os.makedirs(destdir)

    sf.write(destpath, audio, sample_rate)
    return


def is_clipped(audio, clipping_threshold=0.99):
    return any(abs(audio) > clipping_threshold)


def normalize(audio, target_level=-25):
    """Normalize the signal to the target level"""
    rms = (audio**2).mean() ** 0.5
    scalar = 10 ** (target_level / 20) / (rms + EPS)
    audio = audio * scalar
    return audio


def normalize_segmental_rms(audio, rms, target_level=-25):
    """Normalize the signal to the target level
    based on segmental RMS"""
    scalar = 10 ** (target_level / 20) / (rms + EPS)
    audio = audio * scalar
    return audio


def audioread(path, norm=False, start=0, stop=None, target_level=-25):
    """Function to read audio"""
    """taget_level dBFs"""

    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise ValueError("[{}] does not exist!".format(path))
    try:
        audio, sample_rate = sf.read(path, start=start, stop=stop)
    except RuntimeError:  # fix for sph pcm-embedded shortened v2
        print("WARNING: Audio type not supported")

    if len(audio.shape) == 1:  # mono
        if norm:
            rms = (audio**2).mean() ** 0.5
            scalar = 10 ** (target_level / 20) / (rms + EPS)
            audio = audio * scalar
    else:  # multi-channel
        audio = audio.T
        audio = audio.sum(axis=0) / audio.shape[0]
        if norm:
            audio = normalize(audio, target_level)
    return audio, sample_rate


def segmental_snr_mixer(
    clean,
    noise,
    snr,
    min_option=True,
    target_level_lower=-35,
    target_level_upper=-5,
    target_level=-25,
    clipping_threshold=0.99,
):
    """Function to mix clean speech and noise at various segmental SNR levels"""
    if min_option:
        length = min(len(clean), len(noise))
        if len(clean) > length:
            clean = clean[0:length]
        if len(noise) > length:
            noise = noise[0:length]
    else:
        if len(clean) > len(noise):
            noise = np.append(noise, np.zeros(len(clean) - len(noise)))
        else:
            noise = noise[0 : len(clean)]

    clean = clean / (max(abs(clean)) + EPS)
    noise = noise / (max(abs(noise)) + EPS)
    rmsclean, rmsnoise = active_rms(clean=clean, noise=noise)

    # Set the noise level for a given SNR
    if rmsclean == 0:
        noise = normalize_segmental_rms(
            noise, rms=rmsnoise, target_level=target_level
        )
        noisenewlevel = noise
    elif rmsnoise == 0:
        noisenewlevel = noise
        clean = normalize_segmental_rms(
            clean, rms=rmsclean, target_level=target_level
        )
    else:
        clean = normalize_segmental_rms(
            clean, rms=rmsclean, target_level=target_level
        )
        noise = normalize_segmental_rms(
            noise, rms=rmsnoise, target_level=target_level
        )
        noisescalar = rmsclean / (10 ** (snr / 20)) / (rmsnoise + EPS)
        noisenewlevel = noise * noisescalar

    # Mix noise and clean speech
    noisyspeech = clean + noisenewlevel
    # Randomly select RMS value between -15 dBFS and -35 dBFS and normalize noisyspeech with that value
    # There is a chance of clipping that might happen with very less probability, which is not a major issue.
    noisy_rms_level = np.random.randint(target_level_lower, target_level_upper)
    rmsnoisy = (noisyspeech**2).mean() ** 0.5
    scalarnoisy = 10 ** (noisy_rms_level / 20) / (rmsnoisy + EPS)
    noisyspeech = noisyspeech * scalarnoisy
    clean = clean * scalarnoisy
    noisenewlevel = noisenewlevel * scalarnoisy
    # Final check to see if there are any amplitudes exceeding +/- 1. If so, normalize all the signals accordingly
    if is_clipped(noisyspeech):
        noisyspeech_maxamplevel = max(abs(noisyspeech)) / (
            clipping_threshold - EPS
        )
        noisyspeech = noisyspeech / noisyspeech_maxamplevel
        clean = clean / noisyspeech_maxamplevel
        noisenewlevel = noisenewlevel / noisyspeech_maxamplevel
        noisy_rms_level = int(
            20
            * np.log10(scalarnoisy / noisyspeech_maxamplevel * (rmsnoisy + EPS))
        )

    return clean, noisenewlevel, noisyspeech, noisy_rms_level


def active_rms(clean, noise, fs=16000, energy_thresh=-50):
    """Returns the clean and noise RMS of the noise calculated only in the active portions"""
    window_size = 100  # in ms
    window_samples = int(fs * window_size / 1000)
    sample_start = 0
    noise_active_segs = []
    clean_active_segs = []

    while sample_start < len(noise):
        sample_end = min(sample_start + window_samples, len(noise))
        noise_win = noise[sample_start:sample_end]
        clean_win = clean[sample_start:sample_end]
        noise_seg_rms = 20 * np.log10((noise_win**2).mean() + EPS)
        clean_seg_rms = 20 * np.log10((clean_win**2).mean() + EPS)
        # Considering frames with energy
        if noise_seg_rms > energy_thresh:
            noise_active_segs = np.append(noise_active_segs, noise_win)
        if clean_seg_rms > energy_thresh:
            clean_active_segs = np.append(clean_active_segs, clean_win)
        sample_start += window_samples

    if len(noise_active_segs) != 0:
        noise_rms = (noise_active_segs**2).mean() ** 0.5
    else:
        noise_rms = 0

    if len(clean_active_segs) != 0:
        clean_rms = (clean_active_segs**2).mean() ** 0.5
    else:
        clean_rms = 0

    return clean_rms, noise_rms
