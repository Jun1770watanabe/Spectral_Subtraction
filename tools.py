# -*- coding: utf-8 -*-

#==============================================================================
# 【Information】
#==============================================================================
# @ Title     : Useful Tools
# @ Program   : Python (3.xx is recommended)
# @ Author    : Hiroki Murakami
# @ Univ      : Graduate School of Natural Science, Okayama University, Okayama, Japan
# @ Lab       : Abe Laboratory (Human Centric Information Processing)
# @ Date      : 2017.07.29
# @ Copyright : © 2017 Hiroki Murakami. All rights reserved.
# @ License   : http://www.opensource.org/licenses/mit-license.html  MIT License
# This source code or any portion thereof must not be
# reproduced or used in any manner whatsoever.
#
# @ Description
# 便利なツールを雑多に記述
#
#==============================================================================

import pickle
import os
import glob
import numpy as np
import shutil
import math

import matplotlib.pyplot as plt
import matplotlib.cm as cm # カラーマップ
from matplotlib.colors import LinearSegmentedColormap # カラーマップ自作
from sklearn.decomposition import PCA
from scipy.fftpack import rfft
from scipy import ifft

ATR_lbls = ["A01","A02","A03","A04","A05","A06","A07","A08","A09","A10","A11","A12","A13","A14","A15","A16","A17","A18","A19","A20","A21","A22","A23","A24","A25","A26","A27","A28","A29","A30","A31","A32","A33","A34","A35","A36","A37","A38","A39","A40","A41","A42","A43","A44","A45","A46","A47","A48","A49","A50","B01","B02","B03","B04","B05","B06","B07","B08","B09","B10","B11","B12","B13","B14","B15","B16","B17","B18","B19","B20","B21","B22","B23","B24","B25","B26","B27","B28","B29","B30","B31","B32","B33","B34","B35","B36","B37","B38","B39","B40","B41","B42","B43","B44","B45","B46","B47","B48","B49","B50","C01","C02","C03","C04","C05","C06","C07","C08","C09","C10","C11","C12","C13","C14","C15","C16","C17","C18","C19","C20","C21","C22","C23","C24","C25","C26","C27","C28","C29","C30","C31","C32","C33","C34","C35","C36","C37","C38","C39","C40","C41","C42","C43","C44","C45","C46","C47","C48","C49","C50","D01","D02","D03","D04","D05","D06","D07","D08","D09","D10","D11","D12","D13","D14","D15","D16","D17","D18","D19","D20","D21","D22","D23","D24","D25","D26","D27","D28","D29","D30","D31","D32","D33","D34","D35","D36","D37","D38","D39","D40","D41","D42","D43","D44","D45","D46","D47","D48","D49","D50","E01","E02","E03","E04","E05","E06","E07","E08","E09","E10","E11","E12","E13","E14","E15","E16","E17","E18","E19","E20","E21","E22","E23","E24","E25","E26","E27","E28","E29","E30","E31","E32","E33","E34","E35","E36","E37","E38","E39","E40","E41","E42","E43","E44","E45","E46","E47","E48","E49","E50","F01","F02","F03","F04","F05","F06","F07","F08","F09","F10","F11","F12","F13","F14","F15","F16","F17","F18","F19","F20","F21","F22","F23","F24","F25","F26","F27","F28","F29","F30","F31","F32","F33","F34","F35","F36","F37","F38","F39","F40","F41","F42","F43","F44","F45","F46","F47","F48","F49","F50","G01","G02","G03","G04","G05","G06","G07","G08","G09","G10","G11","G12","G13","G14","G15","G16","G17","G18","G19","G20","G21","G22","G23","G24","G25","G26","G27","G28","G29","G30","G31","G32","G33","G34","G35","G36","G37","G38","G39","G40","G41","G42","G43","G44","G45","G46","G47","G48","G49","G50","H01","H02","H03","H04","H05","H06","H07","H08","H09","H10","H11","H12","H13","H14","H15","H16","H17","H18","H19","H20","H21","H22","H23","H24","H25","H26","H27","H28","H29","H30","H31","H32","H33","H34","H35","H36","H37","H38","H39","H40","H41","H42","H43","H44","H45","H46","H47","H48","H49","H50","I01","I02","I03","I04","I05","I06","I07","I08","I09","I10","I11","I12","I13","I14","I15","I16","I17","I18","I19","I20","I21","I22","I23","I24","I25","I26","I27","I28","I29","I30","I31","I32","I33","I34","I35","I36","I37","I38","I39","I40","I41","I42","I43","I44","I45","I46","I47","I48","I49","I50","J01","J02","J03","J04","J05","J06","J07","J08","J09","J10","J11","J12","J13","J14","J15","J16","J17","J18","J19","J20","J21","J22","J23","J24","J25","J26","J27","J28","J29","J30","J31","J32","J33","J34","J35","J36","J37","J38","J39","J40","J41","J42","J43","J44","J45","J46","J47","J48","J49","J50","J51","J52","J53"]

def queue(src, a):
    dst = np.roll(src, -1)
    dst[-1] = a
    return dst


def MyMkdir(dir, disp=True):
    # ディレクトリ生成
    if not os.path.exists(dir):
        os.mkdir(dir)
        if(disp):
            print("mkdir >> {0}".format(dir))


def get_fname_param(fname):
    """
    ファイル名のパラメータ取得
    Parameter
    ---------
    fname: str
    ファイルのパス

    Return
    ------
    例: home/test.wav
    basename = "test.wav"
    dir = "home"
    name = "test"
    ext = ".wav"

    Example
    -------
    basename, dir, name, ext = tools.get_fname_param(fname)
    """

    basename  = os.path.basename(fname)
    dir       = os.path.dirname(fname)
    if(dir == ""):
        dir = "."
    name, ext = os.path.splitext(basename)
    return basename, dir, name, ext


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


def get_dirs_pass_from_folder(directory):
    """
    directory中のディレクトリを探索し，名前順にソートしたリストを返す

    Parameter
    ---------
    directory: str
    ディレクトリへのパス

    Return
    ------
    files: list
    結果のリスト
    """

    files = []
    for file in glob.glob("{0}/*".format(directory)):
        file = file.replace('\\','/')
        if(os.path.isdir(file)):
            files.extend([file])
    files.sort()
    return files


def check_the_number_of_datas(src_n, tgt_n):
    """
    データ数のチェック
    """
    if(src_n == 0 or tgt_n == 0 or (src_n != tgt_n)):
        raise ValueError("the number of data is not correct!!")


def check_the_file_exist(f_pass):
    if os.path.exists(f_pass):
        return True
    else:
        return False


def print_output(fname):
    print("output >> {name}".format(name=fname))


def print_Phase(string):
    """
    強調表示
    """
    print("")
    print("#==============================================================================")
    print("# " + string)
    print("#==============================================================================")


def remove_dir_list(dir_list):
    """
    dir_list: list(str)
    dir_listのディレクトリを完全に削除する
    """

    for dir in dir_list:
        if(os.path.exists(dir)):
            print("rm "+dir)
            shutil.rmtree(dir)


def remove_files_ext(dir_pass, ext, disp=True):
    """
    dir_pass内に存在する拡張子extのファイルを全削除する

    Parameters
    ----------
    dir_pass: str
        ディレクトリへのパス
    ext: str
        拡張子
    disp: boolean
        削除結果を表示
    """

    files = get_files_pass_from_folder(dir_pass, ext)
    for f in files:
        print("rm " + f)
        os.remove(f)


def save_as_pickle(f_pass, obj, disp=True, over_write=True):
    # pickleでオブジェクトを保存
    f = open(f_pass, 'wb')

    if os.path.isfile(f_pass):
        # ファイルが既に存在する場合
        if over_write:
            pickle.dump(obj, f)
        else:
            print("{} is already exist.".format(f_pass))
    else:
        # ファイルが存在しない場合
        pickle.dump(obj, f)

    f.close()

    # 出力表示
    if disp:
        print_output(f_pass)


def load_as_pickle(f_pass):
    # pickleでオブジェクトを読み込み
    f = open(f_pass, 'rb')
    return pickle.load(f)


def save_as_float(x, f_pass, disp=True):
    """
    特徴量xをf_passでFloat32型で書き込み
    SPTKとの連携などで使用
    """
    x = np.array(x).astype(np.float32) # float型に変換
    x.tofile(f_pass)                    # float型で出力
    if(disp): print_output(f_pass)


def load_as_float(f_pass, order):
    """
    f_passの特徴量をFloat32型で読み込み
    SPTKとの連携などで使用
    """
    x = np.fromfile(f_pass, np.float32)
    if(order != 1):
        x = x.reshape(-1, order)
    return x


def save_as_ascii(x, f_pass, disp=True):
    """
    特徴量xをf_passでascii型で書き込み
    デバッグ用
    """

    np.savetxt(f_pass, x, delimiter='\t')
    if(disp): print_output(f_pass)


def load_as_ascii(f_pass, disp=True):
    """
    f_passの特徴量をFloat32型で読み込み
    デバッグ用
    """

    return np.loadtxt(f_pass, delimiter='\t')


def cat_files(dir_pass, ext):
    """
    dir_pass内にある拡張子extのファイルデータを全結合する
    """
    files = get_files_pass_from_folder(dir_pass, ext)

    cat_data = []
    for file_pass in files:
        data = load_as_pickle(file_pass)
        cat_data.extend(data)

    return cat_data


def replication_datas(dir_pass, ext, N):
    """
    dir_pass内の拡張子extのデータをN個複製する
    主に一対多声質変換などのために使用する．

    Example
    -------
    dir_pass => A01.wav,A02.wav,A03.wav

    tools.replication_datas(dir_pass, 'wav', 3)
    N=3のとき，以下のように複製される
    dir_pass => 0001_A01.wav,0001_A02.wav,0001_A03.wav
             => 0002_A01.wav,0002_A02.wav,0002_A03.wav
             => 0003_A01.wav,0003_A02.wav,0003_A03.wav

    """

    files = get_files_pass_from_folder(dir_pass, ext)

    # データ複製
    for file in files:
        for i in range(N):
            basename, dir, name, ext = get_fname_param(file)
            out_file = "{dir}/{num:0>4}_{name}{ext}".format(
                    dir=dir_pass, num=i+1, name=name, ext=ext)
            shutil.copy(file, out_file)

            print_output(out_file)

    # 元データの除去
    for file in files:
        os.remove(file)


# カラーマップ作製
def generate_cmap(colors):
    """
    自分で定義したカラーマップを返す
    """
    values = range(len(colors))

    vmax = np.ceil(np.max(values))
    color_list = []
    for v, c in zip(values, colors):
        color_list.append( ( v/ vmax, c) )
    return LinearSegmentedColormap.from_list('custom_cmap', color_list)


def plot_spectrogram(sp, fs, frame_period, out_img_pass="", fig_width=15, fig_height=5,
                     color_min=-15, color_max=0, disp=True):
    """
    スペクトログラムのプロット
    プロットの範囲は固定する

    Parameter
    ---------
    sp: ndarray
        スペクトル包絡
    fs: float
        サンプリング周波数[Hz]
    frame_period: float
        フレームシフト長 [ms]
    out_img_pass: str
        出力する画像のパス
    disp: boolean
        ウィンドウで表示するかどうか
    fig_width: 画像の横幅
    fig_height: 画像の高さ
    """

    # カラー設定
    color = cm.gist_heat

    # フォントの設定
    plt.rcParams['font.family'] = 'Arial'

    # ログスケール
    sp = np.log(sp)

    # グラフのサイズを変更
    plt.figure(figsize=(fig_width, fig_height))

    # プロット設定
    plt.imshow(sp.T, aspect='auto', cmap=color, vmin = color_min, vmax = color_max)

    # カラーバー設定
    plt.colorbar()

    # 軸ラベル
    plt.xlabel("Time [sec]", fontsize=25)
    plt.ylabel("Frequency [kHz]", fontsize=25)

    # X軸の設定
    x_N = 6 # X軸目盛の数
    x_A = np.linspace(0, sp.shape[0], x_N)
    x_B = np.linspace(0, sp.shape[0]*(frame_period*1e-3), x_N)
    x_B = np.round(x_B, decimals=2) # 有効数字を2桁にする
    plt.xticks(x_A,x_B)

    # Y軸の設定
    y_N = 6 # Y軸目盛の数
    plt.ylim([0,sp.shape[1]])
    y_A = np.linspace(0, sp.shape[1], y_N)
    y_B = np.linspace(0, (fs*1e-3)/2, y_N)
    y_B = np.round(y_B, decimals=1) # 有効数字を1桁にする
    plt.yticks(y_A,y_B)

    # 目盛りの文字サイズ設定
    plt.tick_params(labelsize=20)

    # グラフの位置調整(画像のはみ出しを自動修正)
    plt.tight_layout()

    if(out_img_pass):
        plt.savefig(out_img_pass)
        print("output >> {0}".format(out_img_pass))

    if(disp):
        plt.show()

    plt.clf()
    plt.close()


def plot_features(x, out_img_pass="", fig_width=15, fig_height=5,
                  color_min=2, color_max=2.3, disp=True, rm_pow=True):
    """
    時系列データのプロット
    プロットの範囲は固定する

    Example
    -------
    メルケプストラム推奨設定
    tools.plot_features(src_train, color_min=0, color_max=1)


    Parameter
    ---------
    x: ndarray
        特徴量
    out_img_pass: str
        出力する画像のパス
    disp: boolean
        ウィンドウで表示するかどうか
    fig_width: int
        画像の横幅
    fig_height: int
        画像の高さ
    rm_pow:boolean
        パワー成分の除去
    """

    x = np.array(x)

    fre = np.arange(x.shape[0])

    plt.figure(figsize=(fig_width, fig_height))

    plt.plot(x)

    # # カラー設定
    # color = cm.gist_heat

    # # フォントの設定
    # plt.rcParams['font.family'] = 'Arial'

    # # ログスケール
    # x_log = np.log(x+9)

    # if(color_min == None):
    #     color_min = np.min(x_log)

    # if(color_max == None):
    #     color_max = np.max(x_log)

    # # グラフのサイズを変更
    # plt.figure(figsize=(fig_width, fig_height))

    # # プロット設定
    # plt.imshow(x_log.T, aspect='auto', cmap=color, vmin = color_min, vmax = color_max)

    # # カラーバー設定
    # plt.colorbar()

    # # 軸ラベル
    # plt.xlabel("Time [sec]", fontsize=25)
    # plt.ylabel("Frequency [kHz]", fontsize=25)

    # # Y軸の設定
    # plt.ylim([0,x_log.shape[1]])

    # # 目盛りの文字サイズ設定
    # plt.tick_params(labelsize=20)

    # # グラフの位置調整(画像のはみ出しを自動修正)
    # plt.tight_layout()

    if(out_img_pass):
        plt.savefig(out_img_pass)
        print("output >> {0}".format(out_img_pass))

    if(disp):
        plt.show()

    plt.clf()
    plt.close()


#==============================================================================
# 主成分分析(PCA：Principal Component. Analysis)
#==============================================================================
def PCA_make(X, n_components):
    """
    Xに対してPCA変換行列を算出

    Parameters
    ----------
    X: array
        入力特徴量
    n_components: int
        分析後の成分の次元数
    """
    pca = PCA(n_components=n_components)
    pca.fit(X)
    return pca


def PCA_transform(pca, X):
    """
    Xに対してPCA変換行列を適用し，変換
    """
    X_pca = pca.transform(X)
    return X_pca


def convert_IntList_to_OneHotVec(int_list, n_labels):
    """
    整数値のベクトルをone hot表現に変換

    Parameters
    ----------
    int_list: list
    整数値配列

    n_labels: int
    ラベル数

    Returns
    -------
    one-hot-vector list

    Example
    -------
    n_labels = 5
    int_list = [0,2,1,4]

    array([[ 1.,  0.,  0.,  0.,  0.],   # 0
           [ 0.,  0.,  1.,  0.,  0.],   # 2
           [ 0.,  1.,  0.,  0.,  0.],   # 1
           [ 0.,  0.,  0.,  0.,  1.]])  # 4
    """

    return np.eye(n_labels)[int_list]


def convert_OneHotVec_to_IntList(one_hot_vec):
    """
    整数値のベクトルをone hot表現に変換

    Parameters
    ----------
    one_hot_vec: array
    one-hot-vector array

    Returns
    -------
    integer list

    Example
    -------
    one_hot_vec =
    array([[ 1.,  0.,  0.,  0.,  0.],   # 0
           [ 0.,  0.,  1.,  0.,  0.],   # 2
           [ 0.,  1.,  0.,  0.,  0.],   # 1
           [ 0.,  0.,  0.,  0.,  1.]])  # 4

    return [0,2,1,4]

    """

    index_list = []
    for one_hot in one_hot_vec:
        cnt=0
        for one in one_hot:
            if(one == 1.0):
                index_list.extend([cnt])
                break
            cnt+=1
    return index_list

def add_speaker_code_to_mcep(mcep, speaker_code, speakers_N):
    """
    mcepに対して話者コード(one-hot-vector)を付与

    Parameters
    ----------
    mcep: array
    メルケプストラム

    speaker_code: int
    話者コード

    speakers_N: int
    全体の話者数

    Returns
    -------
    mcep_ohv: array
    one-hot-vecを付与したメルケプストラム
    """

    index_list = [speaker_code]*mcep.shape[0]
    ohv = convert_IntList_to_OneHotVec(index_list, speakers_N)

    mcep_ohv = np.hstack((mcep, ohv))

    return mcep_ohv

def culc_SNR(x1, x2, SNR):
    """
    任意のSNRになるよう雑音データの振幅を変更

    Parameters
    ----------
    x1: array
    雑音データ

    x2: array
    音声データ

    SNR: int
    任意のSNR

    Returns
    -------
    new_x1: array
    振幅変更後の雑音データ
    """

    # # RMS
    # A_noise = np.sqrt((sum(np.power(x1,2)))/x1.shape[0])
    # A_signal = np.sqrt((sum(np.power(x2,2)))/x2.shape[0])
    # print(A_signal)

    # MAX
    A_noise = max(x1)
    A_signal = max(x2)

    effect = (A_signal / 10**(SNR/20)) / A_noise
    
    return x1 * effect


