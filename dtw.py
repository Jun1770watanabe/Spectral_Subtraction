# -*- coding: utf-8 -*-

import config as cf
import numpy as np
import utils
import logging
import argparse
from cdtw import pydtw

from sprocket.util import melcd
from fastdtw import fastdtw

dtw_flag = True


def dtw(src, tgt, stflag=None):
    """
    DTW for the glossectomy patients

    Args:
        stflag (str): None or 'src' or 'tgt'
            if set None, align freely
            if set 'src', align according to the number of frames of src
            if set 'tgt', align according to the number of frames of tgt
    """

    # extract mcep only
    src_mc = src[:, 0:cf.mcep_dim + 1]
    tgt_mc = tgt[:, 0:cf.mcep_dim + 1]

    # DTW calc only power component for distance cost
    # when gllossectomy cases, it's difficult to matching by Mel-CD cost
    src_pow = src_mc[:, 0]
    tgt_pow = tgt_mc[:, 0]

    """
    dist=距離計算関数
    step=ステップ計算の形　["dp1","dp2","dp3","p05sym","p05asym","p1sym","p1asym","p2sym","p2asym"]
    window=制限窓の種類
    param=制限窓の幅

    試行錯誤の結果，"p05asym"が最も良く動作することが分かった
    p05asym: Sakoe-Chiba classification p = 0.5, asymmetric step pattern
    https://maxwell.ict.griffith.edu.au/spl/publications/papers/sigpro82_kkp_dtw.pdf
    """
    d = pydtw.dtw(src_pow, tgt_pow, pydtw.Settings(dist='euclid',
                                                   step="p05asym",
                                                   window='nowindow',
                                                   compute_path=True))
    twf = np.array(d.get_path()).T

    if stflag == 'src':
        sf, index = np.unique(twf[0], return_index=True)
        twf = np.c_[sf, twf[1][index]].T
    elif stflag == 'tgt':
        tf, index = np.unique(twf[1], return_index=True)
        twf = np.c_[twf[0][index], tf].T

    # # DTW by melcd
    # def distance_func(x, y): return melcd(x, y)
    #
    # _, path = fastdtw(src_mc, tgt_mc, dist=distance_func)
    # twf = np.array(path).T

    # alignment
    src_aligned = src[twf[0]]
    tgt_aligned = tgt[twf[1]]

    # debug: check whether dtw works well
    # plot matching path
    # d.plot_alignment()
    # exit()

    global dtw_flag
    if dtw_flag:
        import matplotlib.pyplot as plt
        plt.plot(src[:, 0])
        plt.plot(tgt[:, 0])
        plt.show()
        plt.clf()

        plt.plot(src_aligned[:, 0])
        plt.plot(tgt_aligned[:, 0])
        plt.show()
        plt.clf()
        dtw_flag = False

    # import matplotlib.pyplot as plt
    # plt.plot(src[:,0])
    # plt.plot(tgt[:,0])
    # plt.show()
    # plt.clf()
    #
    # plt.plot(src_aligned[:,0])
    # plt.plot(tgt_aligned[:,0])
    # plt.show()
    # plt.clf()
    # exit()

    return src_aligned, tgt_aligned


def dtw_file(src_inpath, tgt_inpath, src_outpath, tgt_outpath,
             src_ext="mcep", tgt_ext="mcep"):
    logging.info("{0} <=> {1}".format(src_inpath, tgt_inpath))

    src_data = utils.load_as_hdf5(src_inpath)
    tgt_data = utils.load_as_hdf5(tgt_inpath)

    src_aligned, tgt_aligned = dtw(src_data, tgt_data, stflag=None)

    utils.save_as_hdf5(src_aligned, src_outpath, disp=True)
    utils.save_as_hdf5(tgt_aligned, tgt_outpath, disp=True)


def main():
    parser = argparse.ArgumentParser(description="Dynamic Time Warping")

    parser.add_argument(
        "type", choices=["dtw"], help="type of process")
    parser.add_argument(
        "-si", "--src_in_path", default=None, help="path for source input files")
    parser.add_argument(
        "-ti", "--tgt_in_path", default=None, help="path for target input files")
    parser.add_argument(
        "-so", "--src_out_path", default=None, help="path for source output files")
    parser.add_argument(
        "-to", "--tgt_out_path", default=None, help="path for target output files")
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

    if args.type == "dtw":
        utils.apply_func_inN_outN(dtw_file, [args.src_in_path, args.tgt_in_path],
                                  [args.src_out_path, args.tgt_out_path], args.ow)


if __name__ == "__main__":
    main()
