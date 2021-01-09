import torch
import os
from itertools import product
from time import gmtime, strftime


class GridsearchConfiguration:
    def __init__(self, ):
        self.image_dimensions = [100, 100, 1]
        self.result_path = 'data_results' + strftime("%Y-%m-%d-%H:%M-gridsearch.pkl", gmtime())

        self.model_types = ['deep']
        self.input_shapes = [[2, 2], [4, 4], [8, 8], [16, 16]]
        self.numbers_of_layers = [4, 6, 8]
        self.numbers_of_hidden_channels = [32, 64, 128]
        self.number_of_runs = 10

        self.number_of_iterations = 3000
        self.learning_rate = 0.1
        self.convergence_check_length = None
        self.log_frequency = 100
        self.find_best = False
        self.save_losses = False
        self.constant_input = False

        self.data_type = torch.cuda.FloatTensor

    def generate_parameter_combinations(self):
        model_types = self.model_types
        input_shapes = self.input_shapes
        numbers_of_layers = self.numbers_of_layers
        numbers_of_hidden_channels = self.numbers_of_hidden_channels
        parameter_combinations = list(product(*[model_types,
                                                input_shapes,
                                                numbers_of_layers,
                                                numbers_of_hidden_channels]))
        return parameter_combinations * self.number_of_runs

    def __str__(self):
        dictionary = self.__dict__
        result = ""
        for key in dictionary:
            result += key + ": " + str(dictionary[key]) + "  " + os.linesep
        return result


if __name__ == "__main__":
    print(get_gridsearch_configuration())
