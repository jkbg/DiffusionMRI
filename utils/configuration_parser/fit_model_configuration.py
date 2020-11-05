import argparse
import torch
import os

from utils.visualization_helpers import load_image


def get_fit_model_configuration():
    command_line_arguments = parse_command_line_arguments()
    return ModelFittingConfiguration(command_line_arguments)


def parse_command_line_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--noisy-image-path', type=str, default='data/raw_images/canonical_noisy.png')
    parser.add_argument('--target-image-path', type=str, default='data/raw_images/canonical_target.png')
    parser.add_argument('--result-path', type=str, default='data/results/')
    parser.add_argument('--model-type', type=str, default='deep')
    parser.add_argument('--input-shape', type=int, nargs=2, default=(8, 8))
    parser.add_argument('--number-of-layers', type=int, default=6)
    parser.add_argument('--number-of-hidden-channels', type=int, default=128)
    parser.add_argument('--number-of-iterations', type=int, default=30000)
    parser.add_argument('--learning-rate', type=float, default=0.1)
    parser.add_argument('--convergence-check-length', type=int, default=100)
    parser.add_argument('--log-frequency', type=int, default=1)
    parser.add_argument('--find-best', type=bool, default=True)
    parser.add_argument('--cpu', action='store_true')
    parsed_arguments, non_parsed_arguments = parser.parse_known_args()
    return parsed_arguments


class ModelFittingConfiguration:
    def __init__(self, command_line_arguments):
        self.noisy_image_path = command_line_arguments.noisy_image_path
        self.target_image_path = command_line_arguments.target_image_path
        self.result_path = command_line_arguments.result_path
        self.image_shape = load_image(self.noisy_image_path).shape

        self.model_type = command_line_arguments.model_type
        self.input_shape = list(command_line_arguments.input_shape)
        self.number_of_layers = command_line_arguments.number_of_layers
        self.number_of_hidden_channels = command_line_arguments.number_of_hidden_channels

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
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            print("Number of GPUs: ", torch.cuda.device_count())