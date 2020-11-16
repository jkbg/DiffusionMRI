import numpy as np
import simulation.dldegibbs as transforms

from matplotlib import pyplot as plt

from utils.image_helpers import load_image, rgb2gray, plot_image_grid


class Simulation:
    def __init__(self,
                 input_size=(1024, 1024),
                 cropped_size=(256, 256),
                 flip=False,
                 transpose=False,
                 ellipse=False,
                 snr_range=(0, 6),
                 noise_sigma=None,
                 pf_factor=5,
                 mask_target=False,
                 pe_factor=1,
                 absolute_output=True):
        self.transform_set = []
        if flip:
            self.transform_set.append(transforms.RandFlip(targ_op=True))

        if transpose:
            self.transform_set.append(transforms.RandTranspose(targ_op=True))

        if any(np.array(input_size) > 256):
            phase_transform = transforms.MriRandPhaseBig(targ_op=True)
        else:
            phase_transform = transforms.MriRandPhase(targ_op=True)
        self.transform_set.append(phase_transform)

        if ellipse:
            self.transform_set.append(transforms.MriRandEllipse(targ_op=True))

        self.transform_set.append(transforms.MriFFT(unitary=True, targ_op=True))

        scale = np.prod(np.sqrt(np.array(cropped_size) / np.array(input_size)))
        self.transform_set.append(transforms.MriCrop(crop_sz=cropped_size, scale=scale))

        self.transform_set.append(transforms.MriNoise(snr_range=snr_range, sigma=noise_sigma))

        mask = np.ones(shape=cropped_size)
        pe_max = np.floor(cropped_size[1] * (1 - (1 - pe_factor) / 2)).astype(np.int)
        num_keep = (pe_max * pf_factor) // 8
        mask[:, (num_keep + 1):] = 0
        mask[:, :-pe_max] = 0
        self.transform_set.append(transforms.MriMask(mask=mask))

        if mask_target:
            target_mask = np.ones(shape=input_size)
            pe_max = np.floor(input_size[1] * (1 - (1 - pe_factor) / 2)).astype(np.int)
            print('orig pe_max is {}'.format(pe_max))
            num_keep = (pe_max * pf_factor) // 8
            target_mask[:, (num_keep + 1):] = 0
            target_mask[:, :-pe_max] = 0
            self.transform_set.append(transforms.MriMask(dat_op=False, targ_op=True, mask=target_mask))

        self.transform_set.append(transforms.MriInverseFFT(unitary=True, targ_op=True))

        self.transform_set.append(transforms.MriNormalize(percentile_norm='Max', targ_op=True))

        self.transform_set.append(transforms.MriAbsolute(targ_op=True, dat_op=absolute_output))

        self.transform_set.append(transforms.MriResize(output_sz=cropped_size, targ_op=True, dat_op=False))

    def __call__(self, input_image):
        sample = {'dat': input_image, 'target': input_image.copy()}
        for t in self.transform_set:
            sample = t(sample)
        return np.transpose(sample['dat'], (1, 2, 0)), sample['target'][:, :, None]


if __name__ == '__main__':
    diagonal_image = load_image("data/raw_images/MRI_Test.png")
    vertical_image = load_image("data/raw_images/cameraman.png")
    diagonal_image = rgb2gray(diagonal_image)
    vertical_image = rgb2gray(vertical_image)

    snr_range = [1, 2, 4, 8, 16, 32, 64]
    number_of_runs_per_snr = 3

    diagonal_noisy_images = []
    diagonal_target_images = []
    vertical_noisy_images = []
    vertical_target_images = []
    for snr in snr_range:
        simulation = Simulation(cropped_size=(256, 256), snr_range=(snr, snr))
        for index in range(number_of_runs_per_snr):
            diagonal_noisy_image, diagonal_target_image = simulation(diagonal_image)
            diagonal_noisy_images.append(diagonal_noisy_image)
            diagonal_target_images.append(diagonal_target_image)
            vertical_noisy_image, vertical_target_image = simulation(vertical_image)
            vertical_noisy_images.append(vertical_noisy_image)
            vertical_target_images.append(vertical_target_image)
            print(f'{snr} SNR: {index + 1}/{number_of_runs_per_snr}', end='\r')
        print('')

    noisy_images = diagonal_noisy_images + vertical_noisy_images
    plot = plot_image_grid(noisy_images, nrows=2 * len(snr_range))
    plt.show()