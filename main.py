#  Main script for Information analysis of the reverse process of Diffusion models
#  Asher Stout

import tqdm
import argparse
import logging
import PIL
import seaborn
import pandas as pd
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
        "--out",
        nargs='?',
        type=str,
        default="output/",
        help="path to output directory"
    )
    args.add_argument(
        "--analysis",
        type=str,
        default="info",
        help="type of analysis to perform on the data"
    )
    args.add_argument(
        "--log",
        nargs='?',
        type=str,
        default="log.txt",
        help="file to print logs to"
    )

    return args.parse_args()


if __name__ == "__main__":
    arg = main()
    logging.basicConfig(filename=arg.log, level=logging.DEBUG,
                        format="%(asctime)s: %(message)s", filemode="w")

    # Delegate to correct mode file
    
