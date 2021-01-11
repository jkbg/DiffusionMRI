import numpy as np
from skimage.metrics import structural_similarity
import scipy


def mse(target, noisy):
    return (np.square(target - noisy)).mean(axis=None)


def psnr(target, noisy):
    return 10 * np.log10(np.square(np.max(target)) / mse(target, noisy))


def ssim(target, noisy):
    return structural_similarity(target, noisy, multichannel=True)


def vif(target, noisy):
    return vifp_mscale(target, noisy)


def vifp_mscale(ref, dist, sigma_nsq=1, eps=1e-10):
    ### from https://github.com/aizvorski/video-quality/blob/master/vifp.py
    sigma_nsq = sigma_nsq  ### tune this for your dataset to get reasonable numbers
    eps = eps

    num = 0.0
    den = 0.0
    for scale in range(1, 5):

        N = 2 ** (4 - scale + 1) + 1
        sd = N / 5.0

        if (scale > 1):
            ref = scipy.ndimage.gaussian_filter(ref, sd)
            dist = scipy.ndimage.gaussian_filter(dist, sd)
            ref = ref[::2, ::2]
            dist = dist[::2, ::2]

        mu1 = scipy.ndimage.gaussian_filter(ref, sd)
        mu2 = scipy.ndimage.gaussian_filter(dist, sd)
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = scipy.ndimage.gaussian_filter(ref * ref, sd) - mu1_sq
        sigma2_sq = scipy.ndimage.gaussian_filter(dist * dist, sd) - mu2_sq
        sigma12 = scipy.ndimage.gaussian_filter(ref * dist, sd) - mu1_mu2

        sigma1_sq[sigma1_sq < 0] = 0
        sigma2_sq[sigma2_sq < 0] = 0

        g = sigma12 / (sigma1_sq + eps)
        sv_sq = sigma2_sq - g * sigma12

        g[sigma1_sq < eps] = 0
        sv_sq[sigma1_sq < eps] = sigma2_sq[sigma1_sq < eps]
        sigma1_sq[sigma1_sq < eps] = 0

        g[sigma2_sq < eps] = 0
        sv_sq[sigma2_sq < eps] = 0

        sv_sq[g < 0] = sigma2_sq[g < 0]
        g[g < 0] = 0
        sv_sq[sv_sq <= eps] = eps

        num += np.sum(np.log10(1 + g * g * sigma1_sq / (sv_sq + sigma_nsq)))
        den += np.sum(np.log10(1 + sigma1_sq / sigma_nsq))

    vifp = num / den

    return vifp


def performance_from_images(reconstructed_image, target_image, id):
    return generate_performance(id=id,
                                mse=mse(target_image, reconstructed_image),
                                psnr=psnr(target_image, reconstructed_image),
                                vif=vif(target_image, reconstructed_image),
                                ssim=ssim(target_image, reconstructed_image))


def generate_performance(id=None, mse=None, psnr=None, vif=None, ssim=None):
    performance = {}
    if id is not None:
        performance['id'] = id
    if mse is not None:
        performance['mse'] = mse
    if psnr is not None:
        performance['psnr'] = psnr
    if vif is not None:
        performance['vif'] = vif
    if ssim is not None:
        performance['ssim'] = ssim
    return performance


def filter_duplicates(list):
    filtered_list = []
    for element in list:
        if not any([np.array_equal(element, x) for x in filtered_list]):
            filtered_list.append(element)
    return filtered_list


def split_performances(performances, keys):
    if len(performances) == 0:
        print('Empty list given as input.')
        return performances
    if len(keys) == 0:
        return performances
    elif len(keys) >= 1:
        key = keys[-1]
        splits = []
        values = list(map(lambda x: x['id'][key], performances))
        unique_values = filter_duplicates(values)
        for unique_value in unique_values:
            filtered_performances = list(filter(lambda x: x['id'][key] == unique_value, performances))
            recursive = split_performances(filtered_performances, keys[:-1])
            if isinstance(recursive[0], dict):
                splits.append(recursive)
            else:
                splits.extend(recursive)
        return splits


def average_performances(performances, keys):
    id_keys = [key for key in performances[0]['id'].keys()]
    performance_parameters = [key for key in performances[0].keys() if key != 'id']
    splits = split_performances(performances, [k for k in id_keys if k not in keys])
    avg_performances = []
    for split in splits:
        avg_performance = {'id': {k: split[0]['id'][k] for k in id_keys if k not in keys}}
        for performance_parameter in performance_parameters:
            avg_performance[performance_parameter] = np.mean([x[performance_parameter] for x in split])
        avg_performances.append(avg_performance)
    return avg_performances



