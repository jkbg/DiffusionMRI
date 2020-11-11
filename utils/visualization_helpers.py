from math import ceil

import numpy as np
from matplotlib import pyplot as plt
import torch
from time import gmtime, strftime


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


def load_image(filepath):
    rgba_image = plt.imread(filepath)
    rgb_image = rgba_image[:, :, :3]
    return rgb_image


def load_images(filepaths):
    images = []
    for filepath in filepaths:
        rgb_image = load_image(filepath)
        images.append(rgb_image)
    return images


def load_noisy_and_target_image(fit_model_configuration):
    paths = [fit_model_configuration.noisy_image_path, fit_model_configuration.target_image_path]
    return load_images(paths)


def tensor_to_image(tensor):
    image = np.transpose(tensor.numpy(), (1, 2, 0))
    return image


def image_to_tensor(image):
    tensor = torch.from_numpy(np.transpose(image, (2, 0, 1)))
    return tensor


def prepare_for_plot(image):
    image = np.array(image)
    image = image - np.min(image)
    image = image / np.max(image)
    return (image * 255).astype(np.uint8)


def show_images(noisy_image, model_image, target_image, result_path=None, model_description=None):
    model_image = prepare_for_plot(model_image)

    fig = plt.figure(figsize=(12, 5))

    ax = fig.add_subplot(131)
    ax.imshow(noisy_image, "gray")
    ax.set_title("Noisy Image")
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

    if result_path is not None:
        path = result_path + strftime("%Y-%m-%d-%H:%M" + ".png", gmtime())
        plt.savefig(path)
        print('saved at', path)
        plt.show()


def plot_image_grid(imgs, titles=None, nrows=4):
    clipped_imgs = []
    for img in imgs:
        clipped_imgs.append(prepare_for_plot(img))
    print(len(clipped_imgs))
    ncols = ceil(len(clipped_imgs) / nrows)
    print('ncols', ncols)
    nrows = min(nrows, len(clipped_imgs))
    print('nrows', nrows)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True, figsize=(ncols * 4, nrows * 5),
                             squeeze=False)
    for i, column in enumerate(axes.T):
        for j, ax in enumerate(column):
            if j * ncols + i < len(clipped_imgs):
                ax.imshow(clipped_imgs[j * ncols + i], cmap='Greys_r', interpolation='none')
                if titles is not None:
                    ax.set_title(titles[j * ncols + i])
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    fig.tight_layout(pad=0.1)
    return fig
