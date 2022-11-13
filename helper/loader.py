
from itertools import groupby
import logging
import natsort as ns
import numpy as np
import os
import glob
import cv2


def load_all(imp, recursive, steps, ext=".png"):
    """
    Loads all valid samples in a provided directory
    :param imp: import directory
    :param recursive: whether to recursively search all subdirectories
    :param steps: number of reverse process steps per sample
    :param ext: image extension to use

    :return: a 2d numpy array consisting of reverse processes and single steps
    """
    assert os.path.exists(imp)
    to_search = [d[0] for d in os.walk(imp)] if recursive else [""]
    samples = []

    for d in to_search:
        ims = glob.glob('{sub}/*{ext}'.format(sub=d, ext=ext))
        ims = ns.natsorted(ims, alg=ns.PATH)  # first sort path names for groupby to work
        ims = [list(v) for k, v in groupby(ims, lambda l: l.partition('_')[0])]
        [samples.append(load_sample(s, steps)) for s in ims]
        logging.info("Finished loading directory {dir}".format(dir=d))
    return samples


def load_sample(sample, steps):
    """
    Loads a single sample reverse process in a provided directory
    :param sample: all files in the directory included in this sample
    :param steps: number of reverse process steps per sample
    :return:
    """
    assert len(sample) == steps
    steps = [cv2.imread(s) for s in sample]
    steps = np.roll(np.array(steps), 1, axis=0)
    return steps
