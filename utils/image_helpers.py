from math import ceil
from torchvision import datasets, transforms
import numpy as np
from matplotlib import pyplot as plt
import torch


def tensor_to_image(tensor):
    image = np.transpose(tensor.numpy(), (1, 2, 0))
    return image


def image_to_tensor(image):
    tensor = torch.from_numpy(np.transpose(image, (2, 0, 1)))
    return tensor


def prepare_for_plot(image):
    image = np.array(image)
    #image = image - np.min(image)
    #image = image / np.max(image)
    return (image * 255).astype(np.uint8)


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


def get_images(path, max_amount, size=256):
    transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                    transforms.RandomResizedCrop(size=size, interpolation=2),
                                    transforms.ToTensor()])

    image_dataset = datasets.ImageFolder(root=path, transform=transform)
    print(len(image_dataset), ' images found in ', path)

    image_data_loader = torch.utils.data.DataLoader(image_dataset, batch_size=1, shuffle=True)
    data_iter = iter(image_data_loader)
    images = []
    for _ in range(max_amount):
        try:
            images.append(tensor_to_image(next(data_iter)[0][0, ...]))
        except StopIteration:
            data_iter = iter(image_data_loader)
            images.append(tensor_to_image(next(data_iter)[0][0, ...]))
    return [np.squeeze(image)[:, :, None] for image in images]
