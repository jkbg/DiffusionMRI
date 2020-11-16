from math import ceil
from torchvision import datasets, transforms
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


def plot_image_grid(imgs, titles=None, ncols=4):
    clipped_imgs = []
    for img in imgs:
        clipped_imgs.append(prepare_for_plot(img))
    ncols = min(ncols, len(clipped_imgs))
    nrows = ceil(len(clipped_imgs) / ncols)
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


def get_images(path, max_amount):
    transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                    transforms.RandomResizedCrop(size=256, scale=(1.0, 1.0), ratio=(1.0, 1.0)),
                                    transforms.ToTensor()])

    image_dataset = datasets.ImageFolder(root=path, transform=transform)
    print('Number of Images:', len(image_dataset), 'in', path)

    image_data_loader = torch.utils.data.DataLoader(image_dataset, batch_size=1, shuffle=True)
    data_iter = iter(image_data_loader)
    images = []
    for _ in range(max_amount):
        try:
            images.append(tensor_to_image(next(data_iter)[0][0, ...]))
        except StopIteration:
            break
    return [np.squeeze(image) for image in images]