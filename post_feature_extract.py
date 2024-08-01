import copy
import datetime as dt
import os
import sys
from os.path import join, dirname, abspath
from typing import List
from datetime import timedelta
import itertools
import shutil
import textwrap
import inspect
import re
import more_itertools as mit
from typing import Tuple
from tqdm import tqdm
from operator import itemgetter
import pickle as pkl

import numpy as np

import constants

root_dir = dirname(abspath(__file__))
sys.path.append(root_dir)

from abaw5_pre_processing.dlib.utils.tools import fmsg


def get_home_root_data():
    baseurl = None  # path where all datasets are located.
    if "HOST_XXX" in os.environ.keys():
        if os.environ['HOST_XXX'] == 'tay':
            baseurl = f"{os.environ['SBHOME']}/workspace/datasets/abaw7"
        else:
            raise NotImplementedError(os.environ['HOST_XXX'])

    else:
        raise NotImplementedError

    msg_unknown_host = "Sorry, it seems we are unable to recognize the " \
                       "host. You seem to be new to this code. " \
                       "We recommend you to add your baseurl on your own."
    if baseurl is None:
        raise ValueError(msg_unknown_host)

    return baseurl


def merge_dataset_info(fd: str, pairs: list, split: str):
    """
    trial <class 'list'> # list ids of videos
    subject_no <class 'list'>  # sequential number for the list of trials
    trial_no <class 'list'>  # a list of 1 (int)
    length <class 'list'>  # list: length of video (number of frames).
    partition <class 'list'>  # list of splits: train, valid, test
    pseudo_partition <class 'list'>  # list of split

    # constant:
    trial_list <class 'dict'>  # dict ['EXPR'][split] --> list of entries in the csv fold for the split
    data_folder <class 'str'>  # compacted_48
    """

    out = None
    s = f"dataset_info_{ds}_{split}"

    for i, p in enumerate(pairs):
        f = p[0]
        f = join(fd, f)
        assert os.path.isfile(f), f

        with open(f, 'rb') as fx:
            data = pkl.load(fx)

        if i == 0:
            out = copy.deepcopy(data)

        else:
            out['trial'].extend(data['trial'])
            out['trial_no'].extend(data['trial_no'])
            out['length'].extend(data['length'])
            out['partition'].extend(data['partition'])
            out['pseudo_partition'].extend(data['pseudo_partition'])
            out['subject_no'].extend(data['subject_no'])

    with open(join(fd, f'{s}.pkl'), 'wb') as fx:
        pkl.dump(out, fx, protocol=pkl.HIGHEST_PROTOCOL)


def merge_processing_records(fd: str, pairs: list):
    out = []
    s = f"processing_records_{ds}_{split}"
    for p in pairs:
        f = p[1]
        f = join(fd, f)
        assert os.path.isfile(f), f

        with open(f, 'rb') as fx:
            data = pkl.load(fx)
            assert isinstance(data, list), type(data)

        out = out + data

    with open(join(fd, f'{s}.pkl'), 'wb') as fx:
        pkl.dump(out, fx, protocol=pkl.HIGHEST_PROTOCOL)


def merge_results(fd: str, ds: str, split: str):

    pairs = get_pairs_dataset_info_preprocessing_records(fd, ds, split)
    merge_processing_records(fd, pairs)
    merge_dataset_info(fd, pairs, split)


def get_pairs_dataset_info_preprocessing_records(fd: str, ds: str,
                                                 split: str) -> list:
    l_files = os.listdir(fd)
    s = f'dataset_info_{ds}_{split}_'
    # base-names only
    l_files = [f for f in l_files if f.startswith(s) and f.endswith('.pkl')]

    # order them
    idx = [f.split('_')[-1].split('.')[0] for f in l_files]
    idx = [int(d) for d in idx]
    holder = list(zip(l_files, idx))
    # holder = sorted(holder, key=itemgetter(1), reverse=)
    holder.sort(key=lambda x: x[1], reverse=False)

    l_files = [z[0] for z in holder]

    l_f_d_info = l_files
    s2 = f"processing_records_{ds}_{split}_"
    l_f_p_re = [f.replace(s, s2) for f in l_f_d_info]

    for f in l_f_d_info:
        p = join(features_path, f)
        assert os.path.isfile(p), p

    for f in l_f_p_re:
        p = join(features_path, f)
        assert os.path.isfile(p), p

    pairs = list(zip(l_f_d_info, l_f_p_re))
    print(f"Found: {len(pairs)} files for {ds} {split}")
    return pairs


if __name__ == "__main__":
    ds = constants.C_EXPR_DB
    splits = [constants.TRAINSET, constants.TESTSET, constants.VALIDSET]
    if ds == constants.C_EXPR_DB_CHALLENGE:
        splits = [constants.TRAINSET]

    elif ds == constants.C_EXPR_DB:
        splits = [constants.TRAINSET, constants.VALIDSET]

    elif ds == constants.MELD:
        splits = [constants.TRAINSET, constants.VALIDSET, constants.TESTSET]

    else:
        raise NotImplementedError(ds)

    features_path = join(get_home_root_data(), ds, 'features')

    for split in splits:
        merge_results(features_path, ds, split)
