# -*- coding: utf-8 -*-

import config as cf

import os
import glob
import numpy as np
import shutil
from scipy.io import wavfile
from scipy.signal import firwin, lfilter
import h5py
import logging
import sys
import soundfile as sf
from inspect import currentframe
import pickle
import argparse
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.cm as cm  # カラーマップ
from matplotlib.colors import LinearSegmentedColormap  # カラーマップ自作
from sprocket.util.hdf5 import HDF5


def queue(src, a):
    dst = np.roll(src, -1)
    dst[-1] = a
    return dst


def MyMkdir(dr, disp=True):
    if not os.path.exists(dr):
        os.makedirs(dr)
        if(disp):
            logging.info("mkdir >> {0}".format(dr))


def get_filepath_param(filepath):
    """
    Ex: home/test.wav
    basename = "test.wav", dir = "home", name = "test", ext = "wav"
    """
    basename = os.path.basename(filepath)
    dr = os.path.dirname(filepath)
    if(dr == ""):
        dr = "."
    name, ext = os.path.splitext(basename)
    ext = ext.replace(".", "")
    return basename, dr, name, ext

def get_files_pass_from_folder(directory, ext):
    """
    directory中の拡張子extを探索し，名前順にソートしたリストを返す

    Parameter
    ---------
    directory: str
    ディレクトリへのパス
    ext: str
    拡張子

    Return
    ------
    files: list
    結果のリスト
    """

    files = []
    for file in glob.glob("{0}/*.{1}".format(directory, ext)):
        file = file.replace('\\','/')
        files.extend([file])
    files.sort()
    return files

def get_files_path_from_dir(path, sort=True):
    """
    get file path in the directory

    Args:
        path (str): path of directory ex."./root/*.wav"
        sort (bool): if True,  sort file list
    Returns:
        files_list (list): file list
    """

    files = []
    for file in glob.glob(path):
        file = file.replace('\\', '/')
        files.extend([file])
    if sort:
        files.sort()
    return files


def get_dirs_pass_from_folder(directory, sort=True):
    """
    get file path in the directory

    Args:
        directory (str): path of directory
        sort (bool): if True,  sort file list
    Returns:
        dirs (list): directory list
    """

    dirs = []
    for file in glob.glob("{0}/*".format(directory)):
        file = file.replace('\\', '/')
        if(os.path.isdir(file)):
            dirs.extend([file])
    if sort:
        dirs.sort()
    return dirs


def print_output(fname):
    logging.info("output >> {name}".format(name=fname))


def print_Phase(string):
    print("")
    print("#==============================================================================")
    print("# " + string)
    print("#==============================================================================")


def print_config(arg_list):
    print_Phase("Parameter config")
    names = {id(v): k for k, v in currentframe().f_back.f_locals.items()}
    print('\n'.join(names.get(id(arg), '???') + ' = ' + repr(arg)
                    for arg in arg_list))
    print("")


def print_config_dict(dic):
    print_Phase("Parameter config")
    logs = ""
    for k, v in dic.items():
        log = "{0:<12}: {1:<15}".format(k, v)
        print(log)
        logs = logs + log + "\n"
    return logs


def write_log(log_path, log, disp=True):
    print(log)
    fp = open(log_path, "a")
    fp.write(log)
    fp.close()


def check_the_file_exist(f_path):
    if os.path.exists(f_path):
        return True
    else:
        return False


def check_the_number_of_files(N1, N2):
    flag = False
    if N1 != N2:
        flag=True
    elif N1==0 or N2==0:
        flag=True

    if flag:
        logging.error("FileNuberError: the number of files is incorrect.")
        logging.error("({0}, {1})".format(N1, N2))
        sys.exit()


def read_wav(wav_path):
    x, fs = sf.read(wav_path)
    x = np.array(x, dtype=np.float64)
    assert fs == cf.fs
    return x, fs


def write_wav(out_path, x, fs, disp=True):
    sf.write(out_path, x, fs)
    if(disp):
        print_output(out_path)


def save_as_float(x, f_path, disp=True):
    """
    save the data as float32
    Args:
        x (ndarray): data
        out_path (str): output path
    """
    x = np.array(x).astype(np.float32)
    x.tofile(f_path)
    if(disp):
        print_output(f_path)


def load_as_float(f_path, dim):
    """
    load the data as float32
    Args:
        out_path (str): output path
    """
    x = np.fromfile(f_path, np.float32)
    return x


def save_as_ascii(x, f_path, disp=True):
    # save the data as ascii
    np.savetxt(f_path, x, delimiter='\t')
    if(disp):
        print_output(f_path)


def load_as_ascii(f_path, disp=True):
    # load the data as ascii
    return np.loadtxt(f_path, delimiter='\t')


def save_as_pickle(f_pass, obj):
    # save the data as pickle
    f = open(f_pass, 'wb')
    pickle.dump(obj, f)
    f.close()


def load_as_pickle(f_pass):
    # load the data as pickle
    f = open(f_pass, 'rb')
    return pickle.load(f)


def save_as_hdf5(x, f_path, disp=True):
    _, _, _, ext = get_filepath_param(f_path)
    h5 = HDF5(f_path, mode='w')
    h5.save(x, ext)
    if disp:
        print_output(f_path)
    h5.close()


def load_as_hdf5(f_path):
    _, _, _, ext = get_filepath_param(f_path)
    h5 = HDF5(f_path, mode='r')
    x = h5.read(ext)
    h5.close()
    return x


# def load_as_hdf5_files(files_list):
#     datas = load_as_hdf5(files_list[0])
    
#     for i in tqdm(range(1, len(files_list))):
#         x = load_as_hdf5(files_list[i])
        
#         if(i==1):
#             x = np.append(x, np.zeros(datas.shape[0]-x.shape[0]))
#             datas = np.vstack((datas, x))
#             continue

#         if(x.shape[0] < datas.shape[1]):
#             x = np.append(x, np.zeros(datas.shape[1] - x.shape[0]))
#             datas = np.vstack((datas, x))
#         elif(x.shape[0] > datas.shape[1]):
#             datas = np.hstack((datas, np.zeros([datas.shape[0], x.shape[0]-datas.shape[1]])))
#             datas = np.vstack((datas, x))
#         else:
#             datas = np.vstack((datas, x))            
#     return datas

def load_as_hdf5_files(files_list):
    datas = load_as_hdf5(files_list[0])

    if datas.ndim == 1:
        for i in tqdm(range(1, len(files_list))):
            x = load_as_hdf5(files_list[i])
            datas = np.hstack((datas, x))

    elif datas.ndim == 2:
        for i in tqdm(range(1, len(files_list))):
            x = load_as_hdf5(files_list[i])
            datas = np.vstack((datas, x))

    return datas


def load_as_hdf5_files_list(files_list):
    datas = []

    for i in range(len(files_list)):
        x = load_as_hdf5(files_list[i])
        datas.append(x)
    return datas


def remove_file(in_path, disp=True):
    if check_the_file_exist(in_path) is False:
        raise FileNotFoundError("'{0}' is not found.".format(in_path))
        sys.exit()
    os.remove(in_path)
    logging.info("rm '{0}'".format(in_path))


def low_cut_filter(x, fs, cutoff=70):
    """Low cut filter

    Args:
        x (array): Waveform sequence
        fs (int): Sampling frequency
        cutoff (float): Cutoff frequency of low cut filter

    Returns:
        lcf_x (array): Low cut filtered waveform sequence
    """

    nyquist = fs // 2
    norm_cutoff = cutoff / nyquist

    # low cut filter
    fil = firwin(255, norm_cutoff, pass_zero=False)
    lcf_x = lfilter(fil, 1, x)

    return lcf_x


def create_histogram(data, figure_path, range_min=-70, range_max=20,
                     step=10, xlabel='Power [dB]', disp=False):
    """Create histogram

    Parameters
    ----------
    data : list,
        List of several data sequences
    figure_path : str,
        Filepath to be output figure
    range_min : int, optional,
        Minimum range for histogram
        Default set to -70
    range_max : int, optional,
        Maximum range for histogram
        Default set to -20
    step : int, optional
        Stap size of label in horizontal axis
        Default set to 10
    xlabel : str, optional
        Label of the horizontal axis
        Default set to 'Power [dB]'
    """

    # plot histgram
    plt.hist(data, bins=200, range=(range_min, range_max),
             normed=True, histtype="stepfilled")
    plt.xlabel(xlabel)
    plt.ylabel("Probability")
    plt.xticks(np.arange(range_min, range_max, step))

    if(figure_path != None):
        plt.savefig(figure_path)

    if disp:
        plt.show()

    plt.close()


def zscore(x, xmean, xstd):
    # normalization
    zscore = (x - xmean) / xstd
    return zscore


def zscore_reverse(x, xmean, xstd):
    # inverse normalization
    return (x * xstd) + xmean


def normalization(x, axis=None):
    """
    z-score normalization
    convert data to 'average=0 and standard_deviation=1'

    Args:
        x (ndarray): data
        axis [None,0,1]: axis direction

    Returns:
    z (ndarray): normalized data
    xmean (ndarray): mean vector
    xstd (ndarray): standard deviation vector
    """
    xmean = np.mean(x, axis=axis, keepdims=True)
    xstd = np.std(x, axis=axis, keepdims=True)
    z = zscore(x, xmean, xstd)
    return z, xmean, xstd


def get_dataset_path(in_ext, src_or_tgt="src", train_or_test="train"):
    """
    get path to the dataset

    Args:
        src_or_tgt (str): "src" or "tgt"
        train_or_test (str): "train" or "test"
    """
    path = "{dr}/{sot}_{tot}_{src}_to_{tgt}.{in_ext}".format(
        dr=cf.scripts_dir, src=cf.src_name, tgt=cf.tgt_name, in_ext=in_ext,
        sot=src_or_tgt, tot=train_or_test)
    return path


def get_norms_path(sot):
    """
    Args:
        sot (str): "src" or "tgt"
    """
    if sot == "src":
        name = cf.src_name
    else:
        name = cf.tgt_name

    path = "{dr}/{name}.{ext}".format(
        dr=cf.scripts_dir, name=name, ext=cf.norm_ext)
    return path


def concatenate_files(in1_path, in2_path, out_path,
                      in1_ext=None, in2_ext=None, out_ext=None):
    x1 = load_as_hdf5(in1_path)
    x2 = load_as_hdf5(in2_path)

    x12 = np.hstack((x1, x2))

    save_as_hdf5(x12, out_path)


def in_check_file(in_files_list, in_dir_list, in_ext_list):
    flag = False

    # check directory existance
    for i in range(len(in_dir_list)):
        if os.path.exists(in_dir_list[i]) is False:
            logging.error(
                "'{0}' directory is not exist.".format(in_dir_list[i]))
            sys.exit(1)

    # check existance
    for i in range(len(in_files_list)):
        if len(in_files_list[i]) == 0:
            logging.warning("'{0}' files is not found in '{1}' directory".format(
                in_ext_list[i], in_dir_list[i]))
            flag = True

    # check the number of files
    lengths = list(map(len, in_files_list))
    if not (all([e == lengths[0] for e in lengths[1:]]) if lengths else False):
        logging.warning(
            "the number of datas is not correct: {0}".format(lengths))
        flag = True

    if flag:
        return True
    else:
        return False


def out_check_file(out_paths, ow):
    if ow:
        return False

    flag = False
    for out_path in out_paths:
        if check_the_file_exist(out_path):
            logging.warning("{0} is already exist".format(out_path))
            flag = True
    if flag:
        return True
    else:
        return False


def apply_func_inN_outN(func, in_path_list, out_path_list, ow=False):
    in_N = len(in_path_list)
    out_N = len(out_path_list)

    # preprocess
    in_dirs, out_dirs = [], []
    in_exts, out_exts, in_files = [], [], []
    for i in range(in_N):
        _, in_dr, _, in_ext = get_filepath_param(in_path_list[i])
        in_dirs.extend([in_dr])
        in_exts.extend([in_ext])
        in_files.append(get_files_path_from_dir(in_path_list[i]))

    for i in range(out_N):
        _, out_dr, _, out_ext = get_filepath_param(out_path_list[i])
        out_dirs.extend([out_dr])
        out_exts.extend([out_ext])

    # check the number of input files
    if in_check_file(in_files, in_dirs, in_exts):
        return 0

    # mkdir
    for i in range(len(out_dirs)):
        MyMkdir(out_dirs[i])

    for i in range(len(in_files[0])):
        in_paths, in_names, out_paths = [], [], []
        for j in range(in_N):
            in_paths.extend([in_files[j][i]])
            _, _, in_name, _ = get_filepath_param(in_files[j][i])
            in_names.extend([in_name])

        for j in range(out_N):
            if in_N == 1:
                out_path = "{0}/{1}.{2}".format(
                    out_dirs[j], in_names[0], out_exts[j])
            elif in_N == 2:
                out_path = "{0}/{1}_{2}.{3}".format(
                    out_dirs[j], in_names[0], in_names[1], out_exts[j])
            else:
                out_path = "{0}/{1}.{2}".format(
                    out_dirs[j], in_names[0], out_exts[j])
            out_paths.extend([out_path])

        # check output file
        if out_check_file(out_paths, ow):
            continue

        if in_N == 1 and out_N == 0:
            func(in_paths[0])
        elif in_N == 1 and out_N == 1:
            func(in_paths[0], out_paths[0])
        elif in_N == 1 and out_N == 2:
            func(in_paths[0], out_paths[0], out_paths[1])
        elif in_N == 1 and out_N == 3:
            func(in_paths[0], out_paths[0], out_paths[1], out_paths[2])
        elif in_N == 2 and out_N == 1:
            func(in_paths[0], in_paths[1], out_paths[0])
        elif in_N == 2 and out_N == 2:
            func(in_paths[0], in_paths[1], out_paths[0], out_paths[1])
        elif in_N == 3 and out_N == 1:
            func(in_paths[0], in_paths[1], in_paths[2], out_paths[0])
        else:
            logging.error(
                "in_N={0}, out_N={1} is not implemented.".format(in_N, out_N))


def main():
    parser = argparse.ArgumentParser(description="Utils")

    parser.add_argument(
        "type", choices=["concatenate", "remove"], help="type of process")
    parser.add_argument(
        "-i", "--in_path", default=None, help="directory path of input files")
    parser.add_argument(
        "-i1", "--in1_path", default=None, help="directory path of input1 files")
    parser.add_argument(
        "-i2", "--in2_path", default=None, help="directory path of input2 files")
    parser.add_argument(
        "-o", "--out_path", default=None, help="directory path to save features")
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

    if args.type == "concatenate":
        apply_func_inN_outN(concatenate_files,
                            [args.in1_path, args.in2_path],
                            [args.out_path], args.ow)

    if args.type == "remove":
        apply_func_inN_outN(remove_file, [args.in_path], [], args.ow)

if __name__ == "__main__":
    main()
