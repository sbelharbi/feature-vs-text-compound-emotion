"""
Create csv file of RAF-DB dataset.
"""
import math
import random
import copy
import argparse
import csv
import sys
import os
from os.path import join, dirname, abspath, basename
from typing import List

import numpy as np
import tqdm
import yaml
from PIL import Image
import cv2

root_dir = dirname(dirname(abspath(__file__)))
sys.path.append(root_dir)
master_root = dirname(root_dir)
sys.path.append(master_root)

from abaw5_pre_processing.dlib.face_landmarks.retinaface_align import RetinaFaceAlign
from abaw5_pre_processing.dlib.utils.tools import chunks_into_n


if "HOST_XXX" in os.environ.keys():
    if os.environ['HOST_XXX'] == 'gsys':
        # CUDA_VISIBLE_DEVICES="" python dlib/face_landmarks/face_align.py
        import tensorflow
        path = join(os.environ['CUSTOM_CUDNN'],
                    'cudnn-10.1-linux-x64-v7.6.0.64/lib64/libcudnn.so')
        tensorflow.load_library(path)
        tensorflow.config.set_visible_devices([], 'GPU')
        print(path, 'Tensorflow has been loaded early, '
                    'and gpu-usage has been disabled')


# from abaw5_pre_processing.dlib.face_landmarks import FaceAlign
from abaw5_pre_processing.dlib.face_landmarks.faceevolve_align import FaceEvolveAlign
from abaw5_pre_processing.dlib.utils.tools import Dict2Obj
from abaw5_pre_processing.dlib.utils.tools import get_root_wsol_dataset
from abaw5_pre_processing.dlib.utils.shared import find_files_pattern
from abaw5_pre_processing.dlib.utils.tools import check_box_convention

from abaw5_pre_processing.dlib.utils.shared import announce_msg

# from abaw5_pre_processing.project.abaw5 import _constants
import constants

from abaw5_pre_processing.dlib.utils.reproducibility import set_seed
from abaw5_pre_processing.dlib.datasets.default_labels_order import ORDERED_EMOTIONS

from abaw5_pre_processing.dlib.datasets.ds_shared import CROP_SIZE
from abaw5_pre_processing.dlib.datasets.ds_shared import SEED
from abaw5_pre_processing.dlib.datasets.ds_shared import ALIGN_DEEP_FACE
from abaw5_pre_processing.dlib.datasets.ds_shared import ALIGN_FACE_EVOLVE

from abaw5_pre_processing.dlib.datasets.ds_shared import class_balance_stat
from abaw5_pre_processing.dlib.datasets.ds_shared import dump_set
from abaw5_pre_processing.dlib.datasets.ds_shared import print_args_info
from abaw5_pre_processing.dlib.datasets.ds_shared import per_cl_stats


def get_all_videos(p: str) -> List[dict]:
    l = find_files_pattern(p, '*.mp4')
    out_f = dict()
    out_dia_utt = dict()

    for f in l:
        if not basename(f).startswith('dia'):
            continue

        b = basename(f).split('.')[0]
        dia_id = b.split('_')[0].split('dia')[1]
        utt_id = b.split('_')[1].split('utt')[1]

        try:
            i = int(dia_id)
        except:
            print(f"Found issues with {f} @ dialogue id.")

        try:
            i = int(utt_id)
        except:
            print(f"Found issues with {f} @ utterance id.")

        assert f not in out_f, f
        out_f[f] = [dia_id, utt_id]
        k = f"{dia_id}_{utt_id}"
        assert k not in out_dia_utt, k
        out_dia_utt[k] = f

    results = [out_f, out_dia_utt]
    return results


def is_ascii(s):
    return all(ord(c) < 128 for c in s)


def is_video_corrupted(p_vid: str, log_file) -> bool:
    corrupt_video = False

    try:
        cap = cv2.VideoCapture(p_vid)
        if not cap.isOpened():
            corrupt_video = True
    except cv2.error as e:
        corrupt_video = True
    except Exception as e:
        corrupt_video = True

    if corrupt_video:
        msg = f"Corrupted: {p_vid}"
        log_file.write(f"{msg}\n")
        print(msg)

    return corrupt_video


def simplify_csv():
    baseurl = get_root_wsol_dataset()
    ds = constants.MELD
    _dir_meld = join(baseurl, ds)

    _map = {
        constants.VALIDSET: 'dev_sent_emo.csv',
        constants.TRAINSET: 'train_sent_emo.csv',
        constants.TESTSET: 'test_sent_emo.csv'
    }

    keep = {
        'Utterance': 1,
        'Emotion': 3,
        'Dialogue_ID': 5,
        'Utterance_ID': 6
    }

    _remap_labels = {
        'surprise': constants.SURPRISE,
        'fear': constants.FEAR,
        'disgust': constants.DISGUST,
        'joy': constants.HAPPINESS,
        'sadness': constants.SADNESS,
        'anger': constants.ANGER,
        'neutral': constants.NEUTRAL
    }

    ordered_labels = ORDERED_EMOTIONS[ds]

    folds_dir = join(master_root, 'folds', ds, f"split-0")
    os.makedirs(folds_dir, exist_ok=True)
    log_file = open(join(_dir_meld, 'log.txt'), 'w')

    log_file.write(f"Dataset: {ds}\n")

    with open(join(folds_dir, "class_id.yaml"), 'w') as f:
        yaml.dump(ordered_labels, f)

    with open(join(_dir_meld, "class_id.yaml"), 'w') as f:
        yaml.dump(ordered_labels, f)


    for k in _map:
        f_b_name = _map[k]
        f_name = join(_dir_meld, f_b_name)
        out_file = join(_dir_meld, f"{k}.txt")
        fout = open(out_file, 'w', encoding='utf-8')

        msg = f"split: {k}"
        log_file.write(f"{msg}\n")
        print(msg)

        n_vid_missing = 0

        out_f, out_dia_utt = get_all_videos(join(_dir_meld, 'original', k))
        nvids = 0
        n_corpt = 0  # number of corrpted videos.

        with open(f_name, 'r', encoding="UTF-8") as f:

            csv_reader = csv.reader(f, delimiter=',')
            line_count = 0

            for row in csv_reader:

                # non-ascii.
                row[1] = row[1].replace("\x92", "'")
                row[1] = row[1].replace("\x85", " ")
                row[1] = row[1].replace("\x97", " ")
                row[1] = row[1].replace("\x91", " ")
                row[1] = row[1].replace("\x93", " ")
                row[1] = row[1].replace("\x94", " ")
                row[1] = row[1].replace("\x96", " ")
                row[1] = row[1].replace("\xa0", " ")
                row[1] = row[1].replace("’", "'")
                row[1] = row[1].replace("…", "...")

                if line_count == 0:
                    line_count += 1
                    continue

                ut = row[keep['Utterance']]
                emo = row[keep['Emotion']]
                d_id = row[keep['Dialogue_ID']]
                u_id = row[keep['Utterance_ID']]

                key1 = f"{d_id}_{u_id}"
                if not (key1 in out_dia_utt):
                    n_vid_missing += 1
                    continue  # some files are missing such as dial 110 utt 7
                    # at valid.
                file_name: str = out_dia_utt[key1]
                # check if file is not corrupted.
                if is_video_corrupted(file_name, log_file):
                    n_corpt += 1

                    continue

                file_id = file_name.replace(join(_dir_meld, 'original'), '')

                if file_id.startswith(os.sep):
                    file_id = file_id[1:]

                file_id = file_id.split('.')[0]
                label_str: str = _remap_labels[emo]
                label_int: int = ordered_labels[label_str]
                new_row = f"{file_id},{label_int},{ut}\n"
                fout.write(new_row)
                line_count += 1

                nvids += 1

        fout.close()
        cmd = f"cp {out_file} {folds_dir}"
        os.system(cmd)

        msg = f"split: {k}: missing videos: {n_vid_missing}."
        log_file.write(f"{msg}\n")
        print(msg)

        msg = f"split: {k}: corrupted videos: {n_corpt}."
        log_file.write(f"{msg}\n")
        print(msg)

        msg = f"split: {k}: total number of OK videos: {nvids}."
        log_file.write(f"{msg}\n")
        print(msg)

    log_file.close()


def store_faces(faces: List[np.ndarray], out_dir: str, bname: str, f_cnt: int):
    os.makedirs(out_dir, exist_ok=True)

    _out_frame_dir = join(out_dir, f"frame-{f_cnt}")
    os.makedirs(_out_frame_dir, exist_ok=True)

    for i, f in enumerate(faces):
        f = Image.fromarray(f, 'RGB')
        f.save(join(_out_frame_dir, f"v-{bname}-f-{f_cnt}-face-{i}.jpg"),
               format='JPEG')


def process_one_video(p_video, out_dir: str, store_top_n_faces: int,
                      log_file, nblocks: int, process_block: int, ds: str,
                      split: str):
    assert os.path.isfile(p_video), p_video

    assert isinstance(store_top_n_faces, int), type(store_top_n_faces)
    assert store_top_n_faces > 0, store_top_n_faces

    os.makedirs(out_dir, exist_ok=True)

    issue_log_dir = join(master_root, 'face-crop-issues')
    os.makedirs(issue_log_dir, exist_ok=True)
    issue_log_path = join(issue_log_dir, f"{ds}-{split}-"
                                         f"{nblocks}-{process_block}.txt")

    f_cnt = 0
    corrupt_video = False
    try:
        cap = cv2.VideoCapture(p_video)
        if not cap.isOpened():
            corrupt_video = True
    except cv2.error as e:
        corrupt_video = True
    except Exception as e:
        corrupt_video = True

    if corrupt_video:
        log_file.write(f"video: {p_video}. "
                       f"N-frames: {0}. "
                       f"N-faces: {0}  XXXX CORRUPTED\n")
        log_file.flush()

        with open(issue_log_path, 'a') as fissue:
            fissue.write(f"video {p_video} is corrupted.")

        return -1

    success, frame = cap.read()  # bool, np.ndarray h, w, 3 uint8.
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    bname = basename(p_video).split('.')[0]
    if store_top_n_faces == 1:
        return_all_faces = False
    else:
        return_all_faces = True

    face_extractor = RetinaFaceAlign(out_size=constants.SZ256,
                                     verbose=False,
                                     no_warnings=False,
                                     return_all_faces=return_all_faces,
                                     confidence_threshold=0.9
                                     )
    faces = face_extractor.align(img_path=None, img=frame)
    faces = faces[:store_top_n_faces]

    previous_faces = faces
    if not face_extractor.success:
        with open(issue_log_path, 'a') as fissue:
            fissue.write(f"video {p_video} has no faces at the first frame.\n")

        faces_cnt = 0
    else:
        faces_cnt = len(faces)

    store_faces(faces, join(out_dir, bname), bname, f_cnt)
    f_cnt = 1

    while success:
        success, frame = cap.read()

        if success:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = face_extractor.align(img_path=None, img=frame)
            faces = faces[:store_top_n_faces]

            # take the previous faces when there are no faces detected at the
            # current frame.
            if not face_extractor.success:
                faces = previous_faces
            else:
                previous_faces = faces
                faces_cnt += len(faces)

            store_faces(faces, join(out_dir, bname), bname, f_cnt)

            f_cnt += 1

    log_file.write(f"video: {p_video}. "
                   f"N-frames: {f_cnt}. "
                   f"N-faces: {faces_cnt}\n")
    log_file.flush()


def crop_faces_align(ds: str, split: str, nblocks: int, process_block: int):
    baseurl = get_root_wsol_dataset()
    _dir_ds = join(baseurl, ds)
    _dir_in = join(baseurl, ds, 'original')
    _dir_out = join(baseurl, ds, 'cropped_aligned', split)
    os.makedirs(_dir_out, exist_ok=True)

    assert nblocks > 0, nblocks
    assert process_block <= nblocks, f"{process_block}, {nblocks}"

    path_fold = join(_dir_ds, f"{split}.txt")
    assert os.path.isfile(path_fold), path_fold

    with open(path_fold, 'r') as f:
        content = f.readlines()
        lines = []
        for l in content:
            lines.append(l.strip())

    assert nblocks <= len(lines), f"{nblocks}, {lines}"
    blocks = list(chunks_into_n(lines, nblocks))
    block = blocks[process_block]
    store_top_n_faces = 1
    if split == constants.TESTSET:
        store_top_n_faces = 10

    logs_dir = join(_dir_ds, 'logs-crop-face', split)
    os.makedirs(logs_dir, exist_ok=True)
    _dir_log_faces = join(logs_dir, 'log_face_crop_align')
    os.makedirs(_dir_log_faces, exist_ok=True)
    log_file_path = join(_dir_log_faces, f'log-nblocks-{nblocks}-'
                                         f'process-block-{process_block}.txt')
    log_file = open(log_file_path, 'w')

    for l in block:
        _id = l.split(',')[0]
        path_video = join(_dir_in, f"{_id}.mp4")
        assert os.path.isfile(path_video), path_video

        process_one_video(path_video, _dir_out, store_top_n_faces, log_file,
                          nblocks, process_block, ds, split)

    log_file.close()

    print(f"Done cropping faces: {ds}, {split}, nblocks: {nblocks},"
          f" for block {process_block}.")


if __name__ == "__main__":
    root_dir = dirname(dirname(abspath(__file__)))
    sys.path.append(root_dir)

    # 1: simplify csv files.
    # simplify_csv()

    # 2: crop and align faces.
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds", type=str, default=constants.MELD,
                        help="dataset name.")
    parser.add_argument("--split", type=str, default=constants.TRAINSET,
                        help="Name of the split: train, valid, test.")
    parser.add_argument("--nblocks", type=int, default=100,
                        help="Total number of blocks to divide the split into."
                             "You can call this code for each block "
                             "separately.")
    parser.add_argument("--process_block", type=int, default=0,
                        help="Among all the blocks, which block to process.")
    # a block is a list of videos from the split subset.

    parsedargs = parser.parse_args()
    ds = parsedargs.ds
    split = parsedargs.split
    nblocks = parsedargs.nblocks
    process_block = parsedargs.process_block

    crop_faces_align(ds=ds, split=split, nblocks=nblocks,
                     process_block=process_block)
