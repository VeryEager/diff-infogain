import numpy as np
from scipy.stats import entropy
import matplotlib.pyplot as plt


def delegate(data, exp_name):
    """
    Configures run settings and begins script run

    :return:
    """
    num_imgs = len(data)
    num_steps = len(data[0])
    img_size = len(data[0][0])**2

    to_joint_histogram([i[-1] for i in data], [i[45] for i in data])

    # Obtain probabilities for pixel intensities at each timestep
    xts = []
    for i in range(0, num_steps-1):
        xts.append(to_histogram([n[i] for n in data]))

    # Compute entropies
    entropies = []
    for t in range(0, num_steps-1):
        xt = xts[t]
        xt_en = entropy(xt, base=2)
        joint = to_joint_histogram([i[-1] for i in data], [i[t] for i in data])
        joint_en = entropy(joint, base=2)
        joint_en = np.mean(joint_en)
        cond_en = joint_en-xt_en
        entropies.append(cond_en)
        print(str(num_steps-t-1), ": ", cond_en)
    entropies = np.array(entropies)
    np.save(exp_name, entropies)


def to_histogram(ims):
    """
    Converts a single step from all samples to a greyscale histogram (bin counts)
    :param ims: images to merge into a single histogram
    :return:
    """
    histograms = []
    [histograms.append(np.histogram(im, bins=256, range=[0, 255])[0]/(len(im[0])**2)) for im in ims]  # norm
    histogram = np.mean(histograms, axis=0)
    return histogram


def to_joint_histogram(x0, xt):
    """
    Computes the joint probabilities of 2 sequences of images from identical timesteps

    :param x0: final denoised images
    :param xt: images from the current step t
    :return:
    """
    histograms = []
    for x0i, xti in zip(x0, xt):
        _hist = np.array(np.ones(shape=(256, 256)))
        for x in range(0, len(x0i)):
            for y in range(0, len(x0i)):
                _hist[x0i[x][y]][xti[x][y]] += 1
        histograms.append(_hist)
    histogram = np.divide(np.sum(histograms, axis=0), (len(x0)*len(x0[0]**2)))
    plt.imshow(histogram, cmap='hot', interpolation='nearest')
    plt.show()
    plt.clf()
    return histogram

# if i == 48:
#     import matplotlib.pyplot as plt
#     plt.plot(list(range(0, 256)), imds[47])
#     plt.show()
#     plt.clf()
