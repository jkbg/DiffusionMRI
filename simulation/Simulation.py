import numpy as np
import simulation.dldegibbs as transforms

from matplotlib import pyplot as plt

from utils.image_helpers import plot_image_grid


class Simulation:
    def __init__(self,
                 input_size=(1024, 1024),
                 cropped_size=(256, 256),
                 flip=False,
                 transpose=False,
                 ellipse=False,
                 snr_range=(0, 6),
                 noise_sigma=None,
                 pf_factor=8,
                 mask_target=False,
                 pe_factor=1,
                 absolute_output=True,
                 snr_logflag=True):
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

        self.transform_set.append(transforms.MriNoise(snr_range=snr_range, sigma=noise_sigma, logflag=snr_logflag))

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

        if absolute_output:
            self.transform_set.append(transforms.MriAbsolute(targ_op=True, dat_op=True))

        self.transform_set.append(transforms.MriResize(output_sz=cropped_size, targ_op=True, dat_op=False, complex_flag=not absolute_output))

    def __call__(self, input_image):
        sample = {'dat': input_image, 'target': input_image.copy()}
        dat = sample['dat']
        dat = np.reshape(dat, (1,) + dat.shape)
        sample['dat'] = dat
        sample['siglevel'] = np.mean(np.absolute(sample['target']))
        for t in self.transform_set:
            sample = t(sample)
        return np.transpose(sample['dat'], (1, 2, 0)), sample['target'][:, :, None]

    def simulate_list_of_images(self, original_images):
        noisy_images = []
        target_images = []
        for original_image in original_images:
            noisy_image, target_image = self(original_image)
            noisy_images.append(noisy_image)
            target_images.append(target_image)
        return noisy_images, target_images


if __name__ == '__main__':
    size = (256, 256)
    diagonal_image = np.triu(np.ones(shape=size))
    vertical_image = np.concatenate((np.ones((size[0], size[1]//2)), np.zeros((size[0], size[1]//2))), axis=1)

    cnr_range = [0.25, 1, 5, 10]
    number_of_runs_per_cnr = 1

    vertical_noisy_images = []
    vertical_target_images = []
    for cnr in cnr_range:
        sigma = 1/cnr
        simulation = Simulation(input_size=size, cropped_size=(100, 100), noise_sigma=sigma, absolute_output=False)
        for index in range(number_of_runs_per_cnr):
            vertical_noisy_image, vertical_target_image = simulation(vertical_image)
            vertical_noisy_images.append(vertical_noisy_image)
            vertical_target_images.append(vertical_target_image)
            print(f'{cnr} SNR: {index + 1}/{number_of_runs_per_cnr}', end='\r')
        print('')

    titles = [str(x) for x in cnr_range for _ in range(number_of_runs_per_cnr)]
    plot = plot_image_grid(vertical_noisy_images, titles=titles, ncols=2)
    plt.show()