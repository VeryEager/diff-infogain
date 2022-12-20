"""
Main script for Information analysis of the reverse process of Diffusion models
python main.py --inp /local/scratch/samples/celeba/ddim50/runs/ --recursive True --mode info --steps 51 --expr infoim50
"""
import os.path

from helper.loader import load_all
from mode import info, classifier, consistency
import argparse
import logging
# import tqdm
import PIL
import numpy as np


def main():
    """
    Configures valid arguments

    :return: initialized ArgumentParser
    """
    args = argparse.ArgumentParser()
    args.add_argument(
        "--inp",
        type=str,
        default="samples/",
        help="path to input directory; should contain single folder of reverse process images"
    )
    args.add_argument(
        "--recursive",
        nargs='?',
        type=bool,
        default=True,
        help="whether to load images from the input directory recursively"
    )
    args.add_argument(
        "--out",
        nargs='?',
        type=str,
        default="output/",
        help="path to output directory"
    )
    args.add_argument(
        "--mode",
        type=str,
        choices=['info', 'classifier', 'consistency'],
        help="type of analysis to perform on the data"
    )
    args.add_argument(
        "--expr",
        nargs='?',
        type=str,
        default="expr",
        help="experiment name"
    )
    args.add_argument(
        "--steps",
        type=int,
        default=0,
        help="number of reverse process steps"
    )

    return args.parse_args()


if __name__ == "__main__":
    arg = main()
    if not os.path.exists("logs/"):
        os.mkdir("logs/")
    logging.basicConfig(filename="logs/{name}.txt".format(name=arg.expr), level=logging.DEBUG,
                        format="%(asctime)s: %(message)s", filemode="w")

    # Delegate to correct mode file
    if arg.mode == "info":
        logging.info("Loading data")
        data = load_all(arg.inp, arg.recursive, arg.steps)
        info.delegate(data, arg.expr)
    elif arg.mode == "classifier":
        classifier.delegate(data)
    elif arg.mode == "consistency":
        consistency.delegate(data)
