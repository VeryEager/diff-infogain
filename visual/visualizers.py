import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
# plt.rcParams["font.family"] = "Cantarell-VF"


def v_row_img():
    """
    Displays a sequence of images in row format

    :return:
    """

    files = []
    for file in os.listdir("../temp"):
        print(file)
        with Image.open("../temp/" + file) as im:
            files.append(im)


def lines(fnames):
    """
    Displays multiple time-series data from a local directory in a single figure
    """
    # load data
    arrays = []
    for f in fnames:
        arrays.append(np.load("../" + f))

    max_len = max([len(a) for a in arrays])

    # iteratively add lines to figure
    for data, col, label in zip(arrays, ("blue", "green", "red"), ("DDIM, T=50", "DDIM, T=100", "DDPM", "Improved DDPM")):
        plt.plot(data, color=col, label=label)
    plt.legend()
    plt.xticks(ticks=[0, 25, 49], labels=["0%", "50%", "100%"])
    plt.xlabel("Denoising Process T/t")
    plt.ylabel("Stepwise Entropy")
    plt.show()


lines(fnames=["infoim50.npy"])
