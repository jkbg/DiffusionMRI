import pickle
import glob
from matplotlib import pyplot as plt
from skimage.metrics import peak_signal_noise_ratio

from utils.visualization_helpers import plot_image_grid, rgb2gray
from utils.evaluation_helpers import vifp_mscale


def get_most_recent_gridsearch():
    file_paths = glob.glob("data/results/*gridsearch.pkl")
    file_paths.sort()
    return file_paths[-1]


if __name__ == "__main__":
    file_path = get_most_recent_gridsearch()
    results = []
    with open(file_path, 'rb') as input:
        while True:
            try:
                results.append(pickle.load(input))
            except EOFError:
                break
    results.sort(key=lambda x: x.loss_wrt_target.item())
    losses_wrt_target = list(map(lambda x: x.loss_wrt_target.item(), results))
    # vif = vifp_mscale(rgb2gray(target_image), rgb2gray(noisy_image))
    # print(f'Noisy Image VIF: {vif:.4f}')
    # psnr = psnr(results[0].target_image, results[0].noisy_image)
    # print(f'Noisy Image PSNR: {psnr:.4f}')
    # for result in results:
    #     vif = vifp_mscale(rgb2gray(result.target_image), rgb2gray(result.model_image))
    #     result.vif = vif
    #
    # for result in results:
    #     model_image = result.model_image
    #     psnr = peak_signal_noise_ratio(result.target_image, result.model_image)
    #     print(psnr)
    #     result.psnr = psnr

    results.sort(key=lambda x: x.vif, reverse=True)
    images = list(map(lambda x: x.model_image, results))
    descriptions = list(map(lambda x: x.model_parameters, results))
    fig = plot_image_grid(images, descriptions, nrows=4)
    plt.show()
