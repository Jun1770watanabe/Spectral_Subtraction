# -*- coding: utf-8 -*-

import soundfile as sf
import tools
import numpy as np
import librosa.core as lc

def noise_subtraction(wav, out_wav_pass, alpha, wav_ext='wav'):
    """
    Spectral subtraction

    Parameters
    ----------
    wav: str
        wavへのファイルパス
    wav: str
        拡張し

    Returns
    -------
    None

    """

    split_level = 32
    ite = 10

    # read wav file
    x, fs = sf.read(wav)

    # サンプル数を2のべき乗にする(末尾をゼロ詰め)
    for n in range(ite):
        sample_size = 1024 * 2**n
        if(x.shape[0] > sample_size):
            continue
        elif(x.shape[0] < sample_size):
            new_x = np.zeros(sample_size)
            new_x[:x.shape[0]] = x
            break

    # フレーム分割の除算が整数商になっているか確認
    mod = new_x.shape[0] % split_level
    if(mod != 0):
        print("サンプル数が正しくありません")
        exit()

    # 音声をフレームに分割
    frame_size = sample_size / split_level
    frame_x = np.reshape(new_x, (split_level, int(frame_size)))
    final_x = np.zeros((frame_x.shape[0], frame_x.shape[1]))

    # 半フレームずらしたフレームを作成
    half_frame_size = int(frame_size/2)
    half_new_x = new_x[half_frame_size:new_x.shape[0]-half_frame_size]
    half_frame_x = np.reshape(half_new_x,(split_level-1, int(frame_size)))

    #################################################
    # 雑音スペクトルの推定
    #################################################

    ham = np.hamming(frame_size)
    noise_frame = lc.stft(frame_x[0] * ham)
    noise_frame = (np.abs(noise_frame))**2

    # # 列毎の平均値で
    # noise_mean = np.mean(noise_frame, axis=1)
    # for i in range(noise_frame.shape[1]):
    #     noise_frame[:,i] = noise_mean

    # # 最大値で
    # noise_frame[:] = np.max(noise_frame)

    # 移動平均で
    b = np.ones(5)/5.0
    for i in range(noise_frame.shape[1]):
        noise_frame[:,i] = np.convolve(noise_frame[:,i], b, 'same')
        # temp = np.convolve(noise_frame[:,i], b, 'same')
        # tools.plot_features(temp)

    # 普通のフレームと半分ずらしたフレームとで別々に作ってしまう
    # 最後逆変換までした後に両者を足す？
    
    # normal frame part
    for i in range(split_level):
        FRAME_X = lc.stft(frame_x[i] * ham)

        # culcurate phase
        arg = np.angle(FRAME_X) 
        phase = np.cos(arg) + 1j * np.sin(arg)

        # spectral subtraction
        FRAME_X = (np.abs(FRAME_X))**2
        frame_sp = FRAME_X - (alpha * noise_frame)

        # negative number to zero
        frame_sp = np.clip(frame_sp, 0, None)

        frame_sp = np.sqrt(frame_sp)
        frame_sp = frame_sp * phase
        final_x[i] = lc.istft(frame_sp) 

    new_x = final_x.flatten()

    # half frame part
    for i in range(split_level-1):
        HALF_FRAME_X = lc.stft(half_frame_x[i] * ham)

        # culcurate half_phase
        half_arg = np.angle(HALF_FRAME_X) 
        half_phase = np.cos(half_arg) + 1j * np.sin(half_arg)

        # spectral subtraction
        HALF_FRAME_X = (np.abs(HALF_FRAME_X))**2
        frame_half_sp = HALF_FRAME_X - (alpha * noise_frame)

        # negative number to zero
        frame_half_sp = np.clip(frame_half_sp, 0, None)

        frame_half_sp = np.sqrt(frame_half_sp)
        frame_half_sp = frame_half_sp * half_phase
        half_frame_x[i] = lc.istft(frame_half_sp)

    half_new_x = half_frame_x.flatten()
    half_new_x = np.pad(half_new_x, (half_frame_size,half_frame_size), 'constant')
    new_x = new_x + half_new_x

    # delete tail
    new_x = new_x.tolist()
    del new_x[x.shape[0]:]
    new_x = np.array(new_x)

    # write wav file
    sf.write(out_wav_pass, new_x, fs)

def noise_subtraction_dir(wav_dir, alpha, wav_ext='wav'):
    """
    Spectral subtraction for directry

    Parameters
    ----------

    Returns
    -------

    """
    out_wav_dir = "{name}_ss_{alpha}".format(name=wav_dir,alpha=alpha)
    tools.MyMkdir(out_wav_dir)
    wav_files = tools.get_files_pass_from_folder(wav_dir, wav_ext)

    for wav_file in wav_files:
        basename, dir, name, ext = tools.get_fname_param(wav_file)
        out_wav_pass = "{dir}/{name}_ss.{ext}".format(dir=out_wav_dir, name=name, ext=wav_ext)

        noise_subtraction(wav_file, out_wav_pass, alpha)
        print("output >> {0}".format(out_wav_pass))

# noise_subtraction_dir("MJW_wn/MJW_NSE_wn20", alpha=4)
noise_subtraction("MJW_NSD_pn10_J01.wav", "x+y.wav", 16)