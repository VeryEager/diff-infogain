import glob
import os
import seaborn_image as isns
from PIL import Image


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

    img = isns.ImageGrid([files[0:5:1]])

v_row_img()

