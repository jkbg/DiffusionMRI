import numpy as np
import scipy.signal
import scipy.ndimage
from skimage.metrics import peak_signal_noise_ratio


def mse(a, b):
    return (np.square(a - b)).mean(axis=None)


def vifp_mscale(ref, dist):
    # from https://github.com/aizvorski/video-quality/blob/master/vifp.py

    sigma_nsq = 2
    eps = 1e-10

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

    if np.isnan(vifp):
        return 1.0
    else:
        return vifp


def get_average_performance_per_parameter_combination(results):
    # Generate list containing every parameter combination once
    models = list(map(lambda x: str(x.model_parameters), results))
    models = list(dict.fromkeys(models))
    models.sort()

    # Average Performance Indicators for each model setup
    average_performances = []
    for model in models:
        model_results = list(filter(lambda x: str(x.model_parameters) == model, results))
        average_performance = {'model_parameters': model,
                               'mse_noisy': np.mean([x.best_loss_wrt_noisy.cpu() for x in model_results]),
                               'mse_target': np.mean([x.loss_wrt_target for x in model_results]),
                               'vif': np.mean([x.vif for x in model_results]),
                               'psnr': np.mean([x.psnr for x in model_results])}
        average_performances.append(average_performance)
    return average_performances


def calculate_average_noisy_performance(results):
    # Generate list containing every noisy_image once
    images = list(map(lambda x: (x.noisy_image, x.target_image), results))
    images = np.unique(np.array(images), axis=0)

    # Average Performance Indicators for each model setup
    average_performance = {'mse_target': np.mean([mse(x[0], x[1]) for x in images]),
                           'vif': np.mean([vifp_mscale(x[1], x[0]) for x in images]),
                           'psnr': np.mean([peak_signal_noise_ratio(x[1], x[0]) for x in images])}
    return average_performance
