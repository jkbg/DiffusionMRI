import torch
import numpy as np
import copy

from fitting.Result import Result
from utils.visualization_helpers import image_to_tensor, tensor_to_image


def create_fitter_from_configuration(fit_model_configuration):
    fitter = Fitter(number_of_iterations=fit_model_configuration.number_of_iterations,
                    learning_rate=fit_model_configuration.learning_rate,
                    convergence_check_length=fit_model_configuration.convergence_check_length,
                    log_frequency=fit_model_configuration.log_frequency,
                    find_best=fit_model_configuration.find_best,
                    data_type=fit_model_configuration.data_type)
    return fitter


class Fitter:
    def __init__(self, number_of_iterations, learning_rate=0.01, convergence_check_length=40, log_frequency=10,
                 find_best=False, data_type=torch.FloatTensor):
        self.loss_fn = torch.nn.MSELoss().type(data_type)
        self.number_of_iterations = number_of_iterations
        self.learning_rate = learning_rate
        self.convergence_check_length = convergence_check_length
        self.log_frequency = log_frequency
        self.find_best = find_best
        self.data_type = data_type

    def __call__(self, model, original_image, target_image=None):
        self.model = model.type(self.data_type)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        self.step_counter = 0
        self.fixed_net_input = 2 * torch.rand(self.model.get_input_shape()) - 1
        self.fixed_net_input = self.fixed_net_input.type(self.data_type)
        self.noisy_image = image_to_tensor(original_image).type(self.data_type)
        self.target_image = image_to_tensor(target_image).type(self.data_type)
        self.best_model = copy.deepcopy(self.model)
        self.best_model_step = 0
        self.best_model_loss = 1000
        self.losses_wrt_noisy = []
        self.losses_wrt_target = []
        return self.fit()

    def fit(self):
        while self.has_not_converged() and self.step_counter < self.number_of_iterations:

            def closure():
                self.optimizer.zero_grad()
                output = self.model(self.fixed_net_input)
                loss = self.loss_fn(output, self.noisy_image)
                self.update_loss_metrics_and_best_model(loss, output)
                loss.backward()

            self.optimizer.step(closure)

            if self.should_log():
                self.log()

            self.step_counter += 1

    def has_not_converged(self):
        if len(self.losses_wrt_noisy) < self.convergence_check_length:
            return True
        else:
            if np.argmin(self.losses_wrt_noisy) < len(self.losses_wrt_noisy) - self.convergence_check_length:
                print('')
                print(f"Adam has converged at step {self.step_counter}.")
                return False
        if not self.find_best:
            self.best_model = copy.deepcopy(self.model)
        return True

    def update_loss_metrics_and_best_model(self, current_loss_wrt_noisy, current_output):
        self.losses_wrt_noisy.append(current_loss_wrt_noisy.data)
        if self.target_image is not None:
            current_loss_wrt_target = self.loss_fn(current_output, self.target_image)
            self.losses_wrt_target.append(current_loss_wrt_target.data)

        if self.find_best:
            if self.step_counter > 0:
                if self.best_model_loss > 1.005 * current_loss_wrt_noisy.data:
                    self.best_model = copy.deepcopy(self.model)
                    self.best_model_step = self.step_counter
                    self.best_model_loss = current_loss_wrt_noisy.data
        elif self.step_counter == self.number_of_iterations - 1:
            self.best_model = copy.deepcopy(self.model)

    def should_log(self):
        if self.step_counter % self.log_frequency == 0:
            return True
        elif self.step_counter == self.number_of_iterations - 1:
            return True
        else:
            return False

    def log(self):
        log_string = f"Step: {self.step_counter:05d}"
        log_string += ", "
        log_string += f"Loss: {self.losses_wrt_noisy[-1]:.6f}"
        if len(self.losses_wrt_target) > 0:
            log_string += ", "
            log_string += f"Target Loss: {self.losses_wrt_target[-1]:.6f}"
        if self.find_best:
            log_string += ', '
            log_string += f'Minimum Loss at: {self.best_model_step} with {self.best_model_loss:.6f}'
        if self.data_type == torch.cuda.FloatTensor:
            print(log_string, end='\r')
        else:
            print(log_string)

    def get_best_image(self):
        return tensor_to_image(self.best_model(self.fixed_net_input).detach().cpu())

    def get_final_target_loss(self):
        return self.loss_fn(self.best_model(self.fixed_net_input).detach(), self.target_image).data

    def get_step_counter(self):
        return self.step_counter

    def get_result(self):
        result = Result(model_parameters=str(self.model),
                        noisy_image=tensor_to_image(self.noisy_image),
                        model_image=self.get_best_image(),
                        target_image=tensor_to_image(self.target_image),
                        loss_wrt_target=self.get_final_target_loss(),
                        number_of_iterations=self.step_counter)
        return result
