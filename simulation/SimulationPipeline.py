import numpy as np
from matplotlib import pyplot as plt

from utils.image_helpers import plot_image_grid, get_images


def fft(image):
    image = np.fft.fft2(image)
    return np.fft.fftshift(image)


def ifft(image):
    image = np.fft.ifftshift(image)
    return np.fft.ifft2(image)


def normalize(array):
    array = array - np.min(array)
    return array / np.max(array)


def gibbs_crop(image_spectrum, k_factor):
    input_size = image_spectrum.shape
    cropped_size = tuple(map(lambda x: int(np.sqrt(k_factor) * x), input_size))
    start_index = tuple(map(lambda x, y: (x - y) // 2, input_size, cropped_size))
    end_index = tuple(map(lambda x, y: x + y, start_index, cropped_size))
    slice_indices = tuple(map(slice, start_index, end_index))
    pad = [(start_index[0], input_size[0] - end_index[0]), (start_index[1], input_size[1] - end_index[1])]
    cropped_image_spectrum = image_spectrum[slice_indices]
    return cropped_image_spectrum, pad


def generate_noise(shape, snr, avg_signal_power):
    sigma = avg_signal_power / snr / np.sqrt(2)
    noise = sigma * (np.random.normal(size=shape) + 1j * np.random.normal(size=shape))
    return noise


def pad_cropped_spectrum(image, pad_parameters):
    return np.pad(image, pad_parameters, constant_values=0 + 0j)


def apply_partial_fourier(array, pf_factor):
    cutoff = int(array.shape[1] * (1 - pf_factor))
    array[:, :cutoff] = 0 + 0j
    return array


class SimulationPipeline:
    def __init__(self, k_factor=0.5, snr=None, pf_factor=1.0, absolute_output=True):
        self.k_factor = k_factor
        self.snr = snr
        self.pf_factor = pf_factor
        self.absolute_output = absolute_output

    def simulate(self, image):
        image_spectrum = fft(np.squeeze(image))

        image_spectrum, pad_parameters = gibbs_crop(image_spectrum, self.k_factor)

        avg_signal_power = np.sqrt(np.mean(np.absolute(image_spectrum) ** 2))

        if self.snr is not None:
            noise_spectrum = generate_noise(image_spectrum.shape, self.snr, avg_signal_power)
        else:
            noise_spectrum = np.zeros(image_spectrum.shape)

        image_spectrum += noise_spectrum

        image_spectrum = pad_cropped_spectrum(image_spectrum, pad_parameters)

        image_spectrum = apply_partial_fourier(image_spectrum, self.pf_factor)

        gibbs_image = ifft(image_spectrum)

        noisy_image = normalize(gibbs_image)

        if self.absolute_output:
            noisy_image = np.absolute(noisy_image)

        noisy_image = noisy_image[:, :, None]
        return noisy_image

    def simulate_list(self, images):
        noisy_images = []
        for image in images:
            noisy_image = self.simulate(image)
            noisy_images.append(noisy_image)
        return noisy_images


def test_pipeline():
    size = (130, 130)
    vertical_image = np.concatenate((np.ones((size[0], size[1] // 2)), np.zeros((size[0], size[1] // 2))), axis=1)

    cnr_range = [0.1, 0.5, 1, 2, 4, 8, 16, 1024]
    number_of_runs_per_cnr = 1

    vertical_noisy_images = []
    vertical_target_images = []
    for cnr in cnr_range:
        snr = cnr / 2
        pipeline = SimulationPipeline(k_factor=1.0, snr=snr, pf_factor=1.0)
        for index in range(number_of_runs_per_cnr):
            vertical_noisy_image, vertical_target_image = pipeline.simulate(vertical_image)
            vertical_noisy_images.append(vertical_noisy_image)
            vertical_target_images.append(vertical_target_image)
            print(f'{snr} SNR: {index + 1}/{number_of_runs_per_cnr}', end='\r')
        print('')

    titles = [str(x) for x in cnr_range for _ in range(number_of_runs_per_cnr)]
    plot_image_grid(vertical_noisy_images, titles=titles, ncols=2)
    plt.show()


def test_gibbscrop():
    image = np.ones((100, 100)) + 1j * np.ones((100, 100))
    cropped_image, pad = gibbs_crop(image, 0.3)
    padded_image = pad_cropped_spectrum(cropped_image, pad)
    plot_image_grid([np.absolute(image)[:, :], np.absolute(padded_image)[:, :]])
    plt.show()


if __name__ == '__main__':
    test_pipeline()
