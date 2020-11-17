import numpy as np
import scipy.signal
import scipy.ndimage
from skimage.metrics import peak_signal_noise_ratio

from fitting.Result import generate_rudimentary_result


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
        performance = generate_performance(description=str(model_results[0].model_parameters),
                                           mse_noisy=np.mean([x.best_loss_wrt_noisy.cpu() for x in model_results]),
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


def calculate_combination_results(results, combine_function=lambda x: np.mean(x, axis=0), include_noisy=False):
    splitted_results = split_result_list(results, split_model=True, split_image=True)
    combination_results = []
    for run_results in splitted_results:
        noisy_image = run_results[0].noisy_image
        target_image = run_results[0].target_image
        model_parameters = run_results[0].model_parameters
        images_to_combine = [x.model_image for x in run_results]
        if include_noisy:
            images_to_combine.append(noisy_image)
        combined_image = combine_function(images_to_combine)
        result = generate_rudimentary_result(model_parameters, noisy_image, combined_image, target_image)
        combination_results.append(result)
    return combination_results


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
                results_per_image = list(filter(lambda x: x.noisy_image == noisy_image, results_per_model))
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


class Test:
    def __init__(self, model_parameters, noisy_image):
        self.model_parameters = model_parameters
        self.noisy_image = noisy_image

    def __str__(self):
        return str(self.model_parameters) + ' ' + str(self.noisy_image)


if __name__ == '__main__':
    # results = [Test('a', 1), Test('a', 2), Test('a', 3), Test('b', 1), Test('b', 2), Test('b', 3), Test('c', 2)]
    # splitted_results = split_result_list(results, model_split=True, image_split=False)
    test_list = [np.array((1, 2)), np.array((1, 2)), np.array((1, 3)), np.array((1, 3)), np.array((2, 5))]
    test_list = ['a', 'a', 'b']
    print(filter_duplicates(test_list))
