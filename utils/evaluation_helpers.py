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


def calculate_model_performances(results):
    splitted_results = split_result_list(results, model_split=True, image_split=False)
    performances = []
    for model_results in splitted_results:
        performance = generate_performance(description=model_results[0].model_parameters,
                                           mse_noisy=np.mean([x.best_loss_wrt_noisy for x in model_results]),
                                           mse_target=np.mean([x.loss_wrt_target for x in model_results]),
                                           psnr=np.mean([x.psnr for x in model_results]),
                                           vif=np.mean([x.vif for x in model_results]))
        performances.append(performance)
    return performances


def calculate_noisy_performance(results):
    given_image_pairs = get_given_image_pairs(results)
    performance = generate_performance(description='Average Noisy Performance',
                                       mse_target=np.mean([mse(x[0], x[1]) for x in given_image_pairs]),
                                       vif=np.mean([vifp_mscale(x[1], x[0]) for x in given_image_pairs]),
                                       psnr=np.mean([peak_signal_noise_ratio(x[1], x[0]) for x in given_image_pairs]))
    return performance


def filter_duplicates(list):
    filtered_list = []
    for element in list:
        if not any([np.array_equal(element, x) for x in filtered_list]):
            filtered_list.append(element)
    return filtered_list


def get_model_parameters_used(results):
    all_model_parameters = list(map(lambda x: x.model_parameters, results))
    return filter_duplicates(all_model_parameters)


def get_noisy_images_used(results):
    all_noisy_images = list(map(lambda x: x.noisy_image, results))
    return filter_duplicates(all_noisy_images)


def get_given_image_pairs(results):
    all_given_image_pairs = list(map(lambda x: (x.noisy_image, x.target_image), results))
    return filter_duplicates(all_given_image_pairs)


def split_result_list(results, model_split=True, image_split=False):
    splitted_result_list = []
    if model_split:
        model_parameters_used = get_model_parameters_used(results)
        for model_parameters in model_parameters_used:
            results_per_model = list(filter(lambda x: x.model_parameters == model_parameters, results))
            splitted_result_list.append(results_per_model)
    else:
        splitted_result_list.append(results)

    further_splitted_result_list = []
    if image_split:
        for results_per_model in splitted_result_list:
            noisy_images_used = get_noisy_images_used(results_per_model)
            for noisy_image in noisy_images_used:
                results_per_image = list(
                    filter(lambda x: np.array_equal(x.noisy_image, noisy_image), results_per_model))
                further_splitted_result_list.append(results_per_image)
    else:
        further_splitted_result_list = splitted_result_list

    return further_splitted_result_list


def generate_performance(description=None, mse_noisy=None, mse_target=None, psnr=None, vif=None):
    performance = {}
    if description is not None:
        performance['description'] = description
    if mse_noisy is not None:
        performance['mse_noisy'] = mse_noisy
    if mse_target is not None:
        performance['mse_target'] = mse_target
    if psnr is not None:
        performance['psnr'] = psnr
    if vif is not None:
        performance['vif'] = vif
    return performance


def split_performances(performances, split_type='number_of_channels'):
    performances_split_per_type = {}
    index = 0
    if split_type == 'model_type':
        index = 0
    elif split_type == 'input_shape':
        index = 1
    elif split_type == 'number_of_layers':
        index = 2
    elif split_type == 'number_of_channels':
        index = 3
    all_types = list(map(lambda x: x['description'][index], performances))
    unique_types = filter_duplicates(all_types)
    for unique_type in unique_types:
        performances_per_type = list(filter(lambda x: np.array_equal(x['description'][index], unique_type), performances))
        performances_split_per_type[unique_type] = performances_per_type
    return performances_split_per_type


if __name__ == '__main__':
    performances = []
    description = ['deep', [10, 10], 2, 64]
    performances.append({'description': description})
    description = ['deep', [12, 12], 2, 64]
    performances.append({'description': description})
    description = ['deep', [10, 10], 4, 64]
    performances.append({'description': description})
    description = ['deep', [10, 10], 2, 128]
    performances.append({'description': description})
    description = ['conv', [10, 10], 2, 32]
    performances.append({'description': description})
    description = ['deep', [10, 10], 2, 32]
    performances.append({'description': description})

    splitted_performances = split_performances(performances, 'number_of_channels')
    print(splitted_performances)