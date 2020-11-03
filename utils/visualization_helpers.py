from math import ceil

import numpy as np
from matplotlib import pyplot as plt
import torch
from time import gmtime, strftime
import os
from matplotlib.image import imsave


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


def load_images(filepaths):
    images = []
    for filepath in filepaths:
        rgba_image = plt.imread(filepath)
        rgb_image = rgba_image[:, :, :3]
        images.append(rgb_image)
    return images


def tensor_to_image(tensor):
    image = np.transpose(tensor.numpy(), (1, 2, 0))
    return image


def image_to_tensor(image):
    tensor = torch.from_numpy(np.transpose(image, (2, 0, 1)))
    return tensor


def show_images(gibbs_image, model_image, target_image, model_description=None, save_plot=False):
    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(131)
    ax.imshow(gibbs_image, "gray")
    ax.set_title("Gibbs Image")
    ax.axis("off")

    ax = fig.add_subplot(132)
    ax.imshow(model_image, "gray")
    if model_description is not None:
        ax.set_title(str(model_description))
    else:
        ax.set_title("Model Image")
    ax.axis("off")

    ax = fig.add_subplot(133)
    ax.imshow(target_image, "gray")
    ax.set_title("Target Image")
    ax.axis("off")
    plt.tight_layout()

    if save_plot:
        path = "data/results/" + strftime("%Y-%m-%d-%H:%M" + ".png", gmtime())
        plt.savefig(path)

    plt.show()


def plot_image_grid(imgs, titles, nrows=4):
    ncols = ceil(len(imgs) / nrows)
    nrows = min(nrows, len(imgs))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True, figsize=(ncols*4, nrows*5), squeeze=False)
    for i, row in enumerate(axes):
        for j, ax in enumerate(row):
            if j * nrows + i < len(imgs):
                ax.imshow(imgs[j * nrows + i], cmap='Greys_r', interpolation='none')
                ax.set_title(titles[j * nrows + i])
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    fig.tight_layout(pad=0.1)
    return fig
