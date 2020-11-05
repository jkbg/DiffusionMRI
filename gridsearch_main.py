from time import gmtime, strftime
from itertools import product

from fitting.Fitter import create_fitter_from_configuration
from fitting.Result import Result
from models.ConvolutionDecoder import ConvolutionDecoder
from models.DeepDecoder import DeepDecoder
from utils.configuration_parser.gridsearch_configuration import get_gridsearch_configuration
from utils.pickle_utils import save_gridsearch_result
from utils.visualization_helpers import load_noisy_and_target_image

IMAGE_PATHS = ["data/raw_images/canonical_noisy.png", "data/raw_images/canonical_target.png"]
MODEL_TYPES = ['conv', 'deep']
INPUT_SHAPES = [[8, 8], [4, 8], [4, 4], [2, 4], [2, 2]]
NUMBERS_OF_HIDDEN_LAYERS = [4, 6, 8]  # , 5, 6, 7, 8
NUMBERS_OF_HIDDEN_CHANNELS = [32, 64, 128, 256]


def test_parameter_combination(parameters, fitter, noisy_image, target_image):
    if parameters[0] == "conv":
        model = ConvolutionDecoder(parameters[1], noisy_image.shape, parameters[2], parameters[3])
    elif parameters[0] == "deep":
        model = DeepDecoder(parameters[1], noisy_image.shape, parameters[2], parameters[3])
    fitter(model, noisy_image, target_image)
    result = Result(parameters, noisy_image, fitter.get_best_image(), target_image, fitter.get_final_target_loss(),
                    fitter.get_step_counter())
    return result


def generate_parameter_combinations(gridsearch_configuration):
    model_types = gridsearch_configuration.model_types
    input_shapes = gridsearch_configuration.input_shapes
    numbers_of_layers = gridsearch_configuration.numbers_of_layers
    numbers_of_hidden_channels = gridsearch_configuration.numbers_of_hidden_channels
    parameter_combinations = list(product(*[model_types,
                                            input_shapes,
                                            numbers_of_layers,
                                            numbers_of_hidden_channels]))
    return parameter_combinations


if __name__ == '__main__':
    gridsearch_configuration = get_gridsearch_configuration()
    noisy_image, target_image = load_noisy_and_target_image(gridsearch_configuration)
    file_path = gridsearch_configuration.result_path + strftime("%Y-%m-%d-%H:%M-gridsearch.pkl", gmtime())
    fitter = create_fitter_from_configuration(gridsearch_configuration)
    parameter_combinations = generate_parameter_combinations(gridsearch_configuration)
    for parameters in parameter_combinations:
        print("+++" + str(parameters) + "+++")
        result = test_parameter_combination(parameters, fitter, noisy_image, target_image)
        save_gridsearch_result(result, file_path)
