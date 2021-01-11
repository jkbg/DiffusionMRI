import torch
import os


class ModelFittingConfiguration:
    def __init__(self):
        self.image_dimensions = [100, 100, 1]
        self.result_path = 'data/results'

        self.model_type = 'deep'
        self.input_shape = [14, 14]
        self.number_of_layers = 5
        self.number_of_hidden_channels = 40

        self.number_of_iterations = 3000
        self.number_of_runs = 10
        self.learning_rate = 0.1
        self.convergence_check_length = None
        self.log_frequency = 100
        self.find_best = False
        self.save_losses = False
        self.constant_input = False

        self.data_type = torch.cuda.FloatTensor

    def __str__(self):
        dictionary = self.__dict__
        result = ""
        for key in dictionary:
            result += key + ": " + str(dictionary[key]) + "  " + os.linesep
        return result
