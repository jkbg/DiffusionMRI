import argparse
import torch
import os
from itertools import product
from time import gmtime, strftime

from utils.visualization_helpers import load_image


def get_gridsearch_configuration():
    command_line_arguments = parse_command_line_arguments()
    return GridsearchConfiguration(command_line_arguments)


def parse_command_line_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--noisy-image-path', type=str, default='data/raw_images/canonical_noisy.png')
    parser.add_argument('--target-image-path', type=str, default='data/raw_images/canonical_target.png')
    parser.add_argument('--result-path', type=str, default='data/results/')
    parser.add_argument('--model-types', type=str, nargs='+', default=['deep', 'conv'])
    parser.add_argument('--input-shapes', type=input_shape, nargs='+', default=[[2, 2], [4, 4], [8, 8]])
    parser.add_argument('--numbers-of-layers', type=int, nargs='+', default=[4, 6, 8])
    parser.add_argument('--numbers-of-hidden-channels', type=int, nargs='+', default=[128])
    parser.add_argument('--number-of-iterations', type=int, default=30000)
    parser.add_argument('--learning-rate', type=float, default=0.1)
    parser.add_argument('--convergence-check-length', type=int, default=100)
    parser.add_argument('--log-frequency', type=int, default=1)
    parser.add_argument('--find-best', type=bool, default=True)
    parser.add_argument('--cpu', action='store_true')
    parsed_arguments, non_parsed_arguments = parser.parse_known_args()
    return parsed_arguments


class GridsearchConfiguration:
    def __init__(self, command_line_arguments):
        self.noisy_image_path = command_line_arguments.noisy_image_path
        self.target_image_path = command_line_arguments.target_image_path
        self.result_path = command_line_arguments.result_path + strftime("%Y-%m-%d-%H:%M-gridsearch.pkl", gmtime())
        self.image_shape = load_image(self.noisy_image_path).shape

        self.model_types = command_line_arguments.model_types
        self.input_shapes = command_line_arguments.input_shapes
        self.numbers_of_layers = command_line_arguments.numbers_of_layers
        self.numbers_of_hidden_channels = command_line_arguments.numbers_of_hidden_channels

        self.number_of_iterations = command_line_arguments.number_of_iterations
        self.learning_rate = command_line_arguments.learning_rate
        self.convergence_check_length = command_line_arguments.convergence_check_length
        self.log_frequency = command_line_arguments.log_frequency
        self.find_best = command_line_arguments.find_best

        if command_line_arguments.cpu:
            self.data_type = torch.FloatTensor
        else:
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            self.data_type = torch.cuda.FloatTensor
            os.environ['CUDA_VISIBLE_DEVICES'] = '3'
            print("number of GPUs: ", torch.cuda.device_count())

    def generate_parameter_combinations(self):
        model_types = self.model_types
        input_shapes = self.input_shapes
        numbers_of_layers = self.numbers_of_layers
        numbers_of_hidden_channels = self.numbers_of_hidden_channels
        parameter_combinations = list(product(*[model_types,
                                                input_shapes,
                                                numbers_of_layers,
                                                numbers_of_hidden_channels]))
        return parameter_combinations

    def __str__(self):
        dictionary = self.__dict__
        result = ""
        for key in dictionary:
            result += key + ": " + str(dictionary[key]) + "  " + os.linesep
        return result


def input_shape(string):
    try:
        x, y = map(int, string.split(','))
        return [x, y]
    except:
        raise argparse.ArgumentTypeError("Input Shapes must be x, y")


if __name__ == "__main__":
    print(get_gridsearch_configuration())
