import numpy as np
from skimage.metrics import peak_signal_noise_ratio
from utils.evaluation_helpers import vifp_mscale, mse, split_result_list


def generate_rudimentary_result(model_parameters, noisy_image, model_image, target_image):
    loss_wrt_target = mse(target_image, model_image)
    best_loss_wrt_noisy = mse(noisy_image, model_image).cuda()
    return Result(model_parameters, noisy_image, model_image, target_image, loss_wrt_target, 0, best_loss_wrt_noisy)


class Result:
    def __init__(self, model_parameters, noisy_image, model_image, target_image, loss_wrt_target, number_of_iterations,
                 best_loss_wrt_noisy):
        self.model_parameters = model_parameters
        self.noisy_image = noisy_image
        self.model_image = model_image
        self.target_image = target_image
        self.best_loss_wrt_noisy = best_loss_wrt_noisy
        self.loss_wrt_target = loss_wrt_target
        self.number_of_iterations = number_of_iterations
        self.vif = vifp_mscale(target_image, model_image)
        self.psnr = peak_signal_noise_ratio(target_image, model_image)

    def __str__(self):
        output_string = str(self.loss_wrt_target)
        output_string += ", "
        output_string += str(self.number_of_iterations)
        output_string += ", "
        output_string += str(self.model_parameters)
        return output_string


def calculate_combination_results(results, combine_function=lambda x: np.mean(x, axis=0), include_noisy=False):
    splitted_results = split_result_list(results, model_split=True, image_split=True)
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
