import torch
import copy
import os
import numpy as np
from matplotlib import pyplot as plt

from utils.image_helpers import image_to_tensor, tensor_to_image
from models.model_creation import create_model_from_configuration


def fit_model(noisy_image, configuration, log_prefix='', filename=None):
    fitter = create_fitter_from_configuration(configuration)
    run_images = []
    for run_index in range(configuration.number_of_runs):
        model = create_model_from_configuration(configuration)
        extended_log_prefix = log_prefix + f'Run {run_index+1}/{configuration.number_of_runs}, '
        fitter(model, noisy_image, log_prefix=extended_log_prefix)
        run_images.append(fitter.get_best_image())
    fitted_image = np.median(run_images, axis=0)
    if filename is not None:
        if not os.path.exists(configuration.result_path):
            os.makedirs(configuration.result_path)
        plt.imsave(configuration.result_path + filename, fitted_image[:, :, 0], cmap='gray')
    return fitted_image


def create_fitter_from_configuration(configuration):
    fitter = Fitter(number_of_iterations=configuration.number_of_iterations,
                    learning_rate=configuration.learning_rate,
                    convergence_check_length=configuration.convergence_check_length,
                    log_frequency=configuration.log_frequency,
                    find_best=configuration.find_best,
                    data_type=configuration.data_type,
                    save_losses=configuration.save_losses,
                    constant_fixed_input=configuration.constant_input,
                    number_of_runs=configuration.number_of_runs)
    return fitter


class Fitter:
    def __init__(self, number_of_iterations, learning_rate=0.01, convergence_check_length=None, log_frequency=10,
                 find_best=False, data_type=torch.FloatTensor, save_losses=False, constant_fixed_input=False, number_of_runs=10):
        self.loss_function = torch.nn.MSELoss().type(data_type)
        self.number_of_iterations = number_of_iterations
        self.learning_rate = learning_rate
        self.convergence_check_length = convergence_check_length
        self.log_frequency = log_frequency
        self.find_best = find_best
        self.data_type = data_type
        self.save_losses = save_losses
        self.constant_fixed_input = constant_fixed_input
        self.fixed_net_input = None
        self.number_of_runs = number_of_runs

    def __call__(self, model, original_image, target_image=None, log_prefix=None):
        self.model = model.type(self.data_type)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        self.step_counter = 0
        if (self.fixed_net_input is None) or (not self.constant_fixed_input):
            self.fixed_net_input = 2 * torch.rand(self.model.get_input_shape()) - 1
            self.fixed_net_input = self.fixed_net_input.type(self.data_type)
        self.noisy_image = image_to_tensor(original_image).type(self.data_type)
        if target_image is not None:
            self.target_image = image_to_tensor(target_image).type(self.data_type)
        else:
            self.target_image = None
        self.best_model = copy.deepcopy(self.model)
        self.best_model_step = 0
        self.best_model_loss = 1000
        if self.save_losses:
            self.losses_wrt_noisy = []
            self.losses_wrt_target = []
        self.current_loss_wrt_noisy = 1000
        self.current_loss_wrt_target = 1000
        if log_prefix is None:
            self.log_prefix = ''
        else:
            self.log_prefix = log_prefix
        self.fit()

    def fit(self):
        while self.has_not_converged() and self.step_counter < self.number_of_iterations:

            def closure():
                self.optimizer.zero_grad()
                output = self.model(self.fixed_net_input)
                loss = self.loss_function(self.noisy_image, output)
                loss.backward()
                self.update_loss_metrics_and_best_model(loss, output)

            self.optimizer.step(closure)

            self.step_counter += 1

            if self.should_log():
                self.log()

    def has_not_converged(self):
        if self.convergence_check_length is None:
            return True
        elif self.step_counter < self.convergence_check_length:
            return True
        else:
            if self.best_model_step < self.step_counter - self.convergence_check_length:
                print(self.log_prefix + f'Converged at step {self.step_counter}.' + ' '*50, end='\r')
                return False
        return True

    def update_loss_metrics_and_best_model(self, current_loss_wrt_noisy, current_output):
        self.current_loss_wrt_noisy = current_loss_wrt_noisy.data

        if self.save_losses:
            self.losses_wrt_noisy.append(self.current_loss_wrt_noisy)

        if self.target_image is not None:
            current_loss_wrt_target = self.loss_function(current_output, self.target_image)
            self.current_loss_wrt_target = current_loss_wrt_target.data
            if self.save_losses:
                self.losses_wrt_target.append(self.current_loss_wrt_target.data)

        if self.find_best:
            if self.step_counter > 0:
                if self.best_model_loss > 1.005 * current_loss_wrt_noisy.data:
                    self.best_model = copy.deepcopy(self.model)
                    self.best_model_step = self.step_counter
                    self.best_model_loss = current_loss_wrt_noisy.data
        elif self.step_counter == self.number_of_iterations - 1:
            self.best_model = copy.deepcopy(self.model)
            self.best_model_step = self.step_counter
            self.best_model_loss = current_loss_wrt_noisy.data

    def should_log(self):
        if self.step_counter % self.log_frequency == 0:
            return True
        elif self.step_counter == self.number_of_iterations:
            return True
        else:
            return False

    def log(self):
        log_string = self.log_prefix
        log_string += f"Step: {self.step_counter:05d}"
        log_string += ", "
        log_string += f"Loss: {self.current_loss_wrt_noisy:.6f}"
        if self.target_image is not None:
            log_string += ", "
            log_string += f"Target Loss: {self.current_loss_wrt_target:.6f}"
        if self.find_best:
            log_string += ', '
            log_string += f'Minimum Loss at: {self.best_model_step} with {self.best_model_loss:.6f}'
        print(log_string, end='\r')

    def get_best_image(self):
        return tensor_to_image(self.best_model(self.fixed_net_input).detach().cpu())
