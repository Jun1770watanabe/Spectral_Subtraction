# -*- coding: utf-8 -*-

import soundfile as sf
import utils
import numpy as np
import argparse
import logging
import utils
import config as cf


def spectral_subtraction_process(new_x, output_x, alpha, beta, win, K, frame_len, frame_shift, new_x_size):

    #################################################
    # 雑音スペクトルの推定
    #################################################
    noise_spctl = np.zeros(frame_len)
    for i in range(K):
        noise_fft = np.zeros(frame_len, dtype='complex')
        noise_fft = np.fft.fft(new_x[(i*frame_shift):((i*frame_shift)+frame_len)] * win)
        # noise_fft = np.fft.fft(new_x[(i*frame_shift):((i*frame_shift)+frame_len)])
        noise_spctl = noise_spctl + (np.abs(noise_fft))**2
    noise_spctl = noise_spctl / K


    #################################################
    # 減算部
    #################################################
    index = int(new_x_size / frame_shift) - 1
    for i in range(index):

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

    return output_x / 2

def spectral_subtraction(x, ite, wav_ext='wav'):
    """
    Spectral subtraction

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
    frame_shift = int(frame_len / 4)
    ite = cf.ite
    win = np.hanning(frame_len)
    K = 16
    alpha = cf.alpha
    beta = cf.beta

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

    new_x_size = new_x.shape[0]
    output_x = np.zeros(new_x_size)

    for n in range(cf.ite):
        print("iteration....{n}".format(n=n+1))
        output_x = spectral_subtraction_process(new_x, output_x, alpha, beta, win,
                                                K, frame_len, frame_shift, new_x_size)

    # delete head and tail
    output_x = output_x.tolist()
    del output_x[x.shape[0]:]
    output_x = np.array(output_x)

    return output_x / 2

def spectral_subtraction_iteration(input_wav, output_wav, ite):
    print("input << {0}".format(input_wav))
    # read wav file
    x, fs = sf.read(input_wav)
    print(x[:10])

    out_x = spectral_subtraction(x, ite)

    print(out_x[:10])

    # write wav file
    print("output >> {0}".format(output_wav))
    sf.write(output_wav, out_x, fs)


def spectral_subtraction_dir(wav_dir, out_wav_dir, ite, wav_ext='wav'):
    """
    Spectral subtraction for directry

    Parameters
    ----------
    wav_dir: str
        wav directory name

    num_ite: int


    Returns
    -------

    """
    # out_wav_dir = "{name}_ss".format(name=wav_dir)
    utils.MyMkdir(out_wav_dir)
    wav_files = utils.get_files_pass_from_folder(wav_dir, wav_ext)

    for wav_file in wav_files:
        basename, dir, name, ext = utils.get_filepath_param(wav_file)
        out_wav_pass = "{dir}/{name}.{ext}".format(dir=out_wav_dir, name=name, ext=wav_ext)

        spectral_subtraction_iteration(wav_file, out_wav_pass, ite)

def main():
    parser = argparse.ArgumentParser(description="spectral subtraction")

    parser.add_argument(
        "type", choices=["spectral_subtraction"], help="type of process")
    parser.add_argument(
        "-i", "--in_path", default=None, help="path for input files")
    parser.add_argument(
        "-o", "--out_path", default=None, help="path for output files")
    parser.add_argument(
        "-alpha", type=float, default=None, help="subtraction coefficient")
    parser.add_argument(
        "-beta", type=float, default=None, help="flooring coefficient")
    parser.add_argument(
        "-ni", type=int, default=None, help="number of iteration")
    parser.add_argument(
        '-ow', action='store_true', help="allow over write")

    args = parser.parse_args()

    # set log level
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S')

    # show argmument
    for key, value in vars(args).items():
        logging.info("%s = %s" % (key, str(value)))

    if args.alpha is not None:
        cf.alpha = args.alpha
    if args.beta is not None:
        cf.beta = args.beta
    if args.ni is not None:
        cf.ite = args.ni

    if args.type == "spectral_subtraction":
        spectral_subtraction_dir(args.in_path, args.out_path, cf.ite)

if __name__ == "__main__":
    main()
