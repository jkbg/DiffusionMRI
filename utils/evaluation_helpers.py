import numpy as np
from skimage.metrics import structural_similarity
from sewar.full_ref import vifp


def mse(target, noisy):
    return (np.square(target - noisy)).mean(axis=None)


def psnr(target, noisy):
    return 10 * np.log10(np.square(np.max(target)) / mse(target, noisy))


def ssim(target, noisy):
    return structural_similarity(target, noisy, multichannel=True)


def vif(target, noisy):
    return vifp(target, noisy)


def performance_from_images(reconstructed_image, target_image, id):
    return generate_performance(id=id,
                                mse=mse(target_image, reconstructed_image),
                                psnr=psnr(target_image, reconstructed_image),
                                vif=vif(target_image, reconstructed_image),
                                ssim=ssim(target_image, reconstructed_image))


def generate_performance(id=None, mse=None, psnr=None, vif=None, ssim=None):
    performance = {}
    if id is not None:
        performance['description'] = id
    if mse is not None:
        performance['mse_target'] = mse
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
    performance_parameters = [key for key in performances[0].keys() if key != 'id']
    splits = split_performances(performances, keys)
    avg_performances = []
    for split in splits:
        avg_performance = {'id': {x: split[0]['id'][x] for x in keys}}
        for performance_parameter in performance_parameters:
            avg_performance[performance_parameter] = np.mean([x[performance_parameter] for x in split])
        avg_performances.append(avg_performance)
    return avg_performances



def calculate_model_performances(results):
    splitted_results = split_result_list(results, model_split=True, image_split=False)
    performances = []
    for model_results in splitted_results:
        performance = generate_performance(id=model_results[0].model_parameters,
                                           mse_noisy=np.mean([x.best_loss_wrt_noisy for x in model_results]),
                                           mse=np.mean([x.loss_wrt_target for x in model_results]),
                                           psnr=np.mean([x.psnr for x in model_results]),
                                           vif=np.mean([x.vif for x in model_results]),
                                           ssim=np.mean([x.ssim for x in model_results]))
        performances.append(performance)
    return performances


def calculate_noisy_performance(results):
    given_image_pairs = get_given_image_pairs(results)
    performance = generate_performance(id='Average Noisy Performance',
                                       mse=np.mean([mse(x[0], x[1]) for x in given_image_pairs]),
                                       vif=np.mean([vif(x[1], x[0]) for x in given_image_pairs]),
                                       psnr=np.mean([psnr(x[1], x[0]) for x in given_image_pairs]),
                                       ssim=np.mean([ssim(x[1], x[0]) for x in given_image_pairs]))
    return performance


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

#
# def split_performances(performances, split_type='number_of_channels'):
#     performances_split_per_type = {}
#     index = 0
#     if split_type == 'model_type':
#         index = 0
#     elif split_type == 'input_shape':
#         index = 1
#     elif split_type == 'number_of_layers':
#         index = 2
#     elif split_type == 'number_of_channels':
#         index = 3
#     all_types = list(map(lambda x: x['description'][index], performances))
#     unique_types = filter_duplicates(all_types)
#     for unique_type in unique_types:
#         performances_per_type = list(
#             filter(lambda x: np.array_equal(x['description'][index], unique_type), performances))
#         performances_split_per_type[str(unique_type)] = performances_per_type
#     return performances_split_per_type
