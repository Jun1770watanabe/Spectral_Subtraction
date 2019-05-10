# -*- coding: utf-8 -*-

import soundfile as sf
import tools
import numpy as np

def noise_subtruction(x, wav_ext='wav'):
    """
    Spectral Subtruction

    Parameters
    ----------
    x: ndarray
        wav data
    wav: str
        拡張子

    Returns
    -------
    output_x/2: ndarray
        SSed x

    None

    """

    #################################################
    # パラメータ
    #################################################

    frame_len = 1024
    half_fl = int(frame_len / 2)
    frame_shift = int(frame_len / 4)
    ite = 10

    alpha = 2.0
    beta = 0.9
    win = np.hanning(frame_len)
    K = 32


    #################################################
    # 前処理
    #################################################

    # サンプル数を2のべき乗にする(末尾をゼロ詰め)
    for n in range(ite):
        sample_size = 1024 * 2**n
        if(x.shape[0] > sample_size):
            continue
        elif(x.shape[0] < sample_size):
            new_x = np.zeros(sample_size)
            new_x[:x.shape[0]] = x
            break

    new_x = np.insert(new_x, 0, np.zeros(half_fl))
    new_x_size = new_x.shape[0]
    output_x = np.zeros(new_x_size)


    #################################################
    # 雑音スペクトルの推定
    #################################################

    noise_spctl = np.zeros(frame_len)
    for i in range(2,K):
        noise_fft = np.zeros(frame_len, dtype='complex')
        # noise_fft = np.fft.fft(new_x[int(i*(frame_shift-1)):int(i*(frame_shift-1)+frame_len)] * win)
        noise_fft = np.fft.fft(new_x[(i*frame_shift):((i*frame_shift)+frame_len)])
        noise_spctl = noise_spctl + (np.abs(noise_fft))**2
    noise_spctl = noise_spctl / (K-2)


    #################################################
    # 減算部
    #################################################
    index = (new_x_size / frame_shift) - 1
    for i in range(int(index)):

        if((i*frame_shift)+frame_len > new_x_size):
            break

        # fft
        obs_fft = np.zeros(frame_len, dtype='complex')
        obs_fft = np.fft.fft(new_x[(i*frame_shift):((i*frame_shift)+frame_len)] * win)

        # culcurate phase
        arg = np.angle(obs_fft)
        phase = np.cos(arg) + 1j * np.sin(arg)

        # subtraction
        obs_spctl = (np.abs(obs_fft))**2
        tar_spctl = obs_spctl - (alpha * noise_spctl)
        tar_spctl = np.where(tar_spctl < 0, (beta*obs_spctl), tar_spctl)

        # ifft
        tar_spctl = np.sqrt(tar_spctl)
        tar_fft = np.zeros(frame_len, dtype='complex')
        tar_fft = tar_spctl * phase
        process_x = np.fft.ifft(tar_fft)
        process_x = process_x.real

        # modulate length
        left_zeros = frame_shift * i
        temp_x = np.insert(process_x, 0, np.zeros(left_zeros))
        if(new_x_size-frame_len-left_zeros < 0):
            break
        modulate_x = np.append(temp_x, np.zeros(new_x_size-frame_len-left_zeros))

        output_x = output_x + modulate_x

    # delete head and tail
    output_x = output_x.tolist()
    del output_x[:half_fl]
    del output_x[x.shape[0]:]
    output_x = np.array(output_x)

    return output_x / 2

def noise_subtruction_repetition(input_wav, output_wav, num_rep):
    # read wav file
    x, fs = sf.read(input_wav)

    rep_x = x
    for i in range(num_rep):
        rep_x = noise_subtruction(rep_x)

    # write wav file
    print("output >> {0}".format(output_wav))
    sf.write(output_wav, rep_x, fs)


def noise_subtruction_dir(wav_dir, num_rep, wav_ext='wav'):
    """
    Spectral Subtruction for directry

    Parameters
    ----------

    Returns
    -------

    """
    out_wav_dir = "{name}_ss".format(name=wav_dir)
    tools.MyMkdir(out_wav_dir)
    wav_files = tools.get_files_pass_from_folder(wav_dir, wav_ext)

    for wav_file in wav_files:
        basename, dir, name, ext = tools.get_fname_param(wav_file)
        out_wav_pass = "{dir}/{name}.{ext}".format(dir=out_wav_dir, name=name, ext=wav_ext)

        noise_subtruction_repetition(wav_file, out_wav_pass, num_rep)

noise_subtruction_repetition("MSA_SE_X01.wav", "aaa.wav", 10)
# noise_subtruction_dir("MSA_SE", 10)
