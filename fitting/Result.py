from skimage.metrics import peak_signal_noise_ratio

from utils.evaluation_helpers import vifp_mscale


class Result:
    def __init__(self, model_description, noisy_image, model_image, target_image, loss_wrt_target, number_of_iterations, best_loss_wrt_noisy):
        self.model_parameters = model_description
        self.noisy_image = noisy_image
        self.model_image = model_image
        self.target_image = target_image
        self.best_loss_wrt_noisy = best_loss_wrt_noisy
        self.loss_wrt_target = loss_wrt_target.item()
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