"""
Command-line run:
python export.py --files ../files/iddpm --output /local/scratch/samples/cifar10/iddpm/ --batch_size 2 --diffusion_steps 4000

"""


import numpy as np
from PIL import Image
import argparse
import glob
from pathlib import Path


def main():
    """
    Configures valid arguments

    :return: initialized ArgumentParser
    """
    args = argparse.ArgumentParser()
    args.add_argument(
        "--files",
        type=str,
        default="samples",
        help="path to input directory; should contain single folder of .npy files"
    )
    args.add_argument(
        "--output",
        type=str,
        default="images",
        help="directory where images are exported to"
    )
    args.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="batch size of .npy files"
    )
    args.add_argument(
        "--diffusion_steps",
        type=int,
        default=16,
        help="number of steps per sample in .npy files"
    )
    return args.parse_args()


class IDDPMExporter:
    """
    Extracts batch & intermediate results produced by IDDPM in .npy format to a jpg directory
    """
    def __init__(self, bs, ds, od):
        self.batch_size = bs
        self.steps = ds
        self.output_dir = od
        self.global_total = 0

    def extract_from(self, array):
        """
        Extracts intermediate samples from numpy array and saves as images

        :param array: contents of a .npy file
        :return:
        """
        prefix = self.output_dir + "/" + str(self.global_total) + "/"
        Path(prefix).mkdir(parents=True, exist_ok=True)

        # Export first 1/4 denoising steps
        for i, step in enumerate(array[0:self.steps//4]):
            for j, sample in enumerate(step):
                fname = "_noise.jpg" if i == 0 else "_" + str(i) + ".jpg"
                img = Image.fromarray(sample)
                img.save(prefix+str(j)+fname)

        # And every 10th step thereafter
        for i, step in enumerate(array[self.steps//4::10]):
            for j, sample in enumerate(step):
                fname = "_" + str(self.steps//4+i*10) + ".jpg"
                img = Image.fromarray(sample)
                img.save(prefix + str(j) + fname)

        self.global_total += 1  # increments export directory


if __name__ == "__main__":
    arg = main()
    exporter = IDDPMExporter(arg.batch_size, arg.diffusion_steps, arg.output)

    for file in glob.glob(arg.files+'/*.np[yz]'):
        raw = np.load(file)
        raw = raw['arr_0']
        print(file, " ", str(raw.shape))
        exporter.extract_from(raw)
