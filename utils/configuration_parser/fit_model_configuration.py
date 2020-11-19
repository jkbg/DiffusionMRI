import argparse
import torch
import os


def get_fit_model_configuration():
    command_line_arguments = parse_command_line_arguments()
    return ModelFittingConfiguration(command_line_arguments)


def parse_command_line_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-dimensions', type=int,  nargs='+', default=[256, 256, 1])
    parser.add_argument('--result-path', type=str, default='data/results/')
    parser.add_argument('--model-type', type=str, default='deep')
    parser.add_argument('--input-shape', type=input_shape, default=(8, 8))
    parser.add_argument('--number-of-layers', type=int, default=6)
    parser.add_argument('--number-of-hidden-channels', type=int, default=128)
    parser.add_argument('--number-of-iterations', type=int, default=30000)
    parser.add_argument('--learning-rate', type=float, default=0.1)
    parser.add_argument('--convergence-check-length', type=int, default=100)
    parser.add_argument('--log-frequency', type=int, default=1)
    parser.add_argument('--find-best', type=bool, default=True)
    parser.add_argument('--save-losses', type=bool, default=True)
    parser.add_argument('--cpu', action='store_true')
    parsed_arguments, non_parsed_arguments = parser.parse_known_args()
    return parsed_arguments


class ModelFittingConfiguration:
    def __init__(self, command_line_arguments):
        self.image_dimensions = command_line_arguments.image_dimensions
        self.result_path = command_line_arguments.result_path

        self.model_type = command_line_arguments.model_type
        self.input_shape = list(command_line_arguments.input_shape)
        self.number_of_layers = command_line_arguments.number_of_layers
        self.number_of_hidden_channels = command_line_arguments.number_of_hidden_channels

        self.number_of_iterations = command_line_arguments.number_of_iterations
        self.learning_rate = command_line_arguments.learning_rate
        self.convergence_check_length = command_line_arguments.convergence_check_length
        self.log_frequency = command_line_arguments.log_frequency
        self.find_best = command_line_arguments.find_best
        self.save_losses = command_line_arguments.save_losses

        if command_line_arguments.cpu:
            self.data_type = torch.FloatTensor
        else:
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            self.data_type = torch.cuda.FloatTensor
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            print("number of GPUs: ", torch.cuda.device_count())

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
