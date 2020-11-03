import os
import pickle
from time import gmtime, strftime
from itertools import product

from fitting.Result import Result
from models.ConvolutionDecoder import ConvolutionDecoder
from models.DeepDecoder import DeepDecoder
from fitting.Fitter import Fitter
from utils.visualization_helpers import load_images

IMAGE_PATHS = ["data/raw_images/canonical_noisy.png", "data/raw_images/canonical_target.png"]
MODEL_TYPES = ['deep', 'conv']
INPUT_SHAPES = [[8, 8], [4, 8], [4, 4], [2, 4], [2, 2]]
NUMBERS_OF_HIDDEN_LAYERS = [4, 5, 6, 7, 8] #, 5, 6, 7, 8
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


def generate_file_path():
    file_path = "data/results/" + strftime("%Y-%m-%d-%H:%M-gridsearch.pkl", gmtime())
    return file_path


def load_results(file_path):
    results = []
    if os.path.isfile(file_path):
        with open(file_path, 'rb') as input:
            while True:
                try:
                    results.append(pickle.load(input))
                except EOFError:
                    break
    return results


def save_gridsearch_result(result, file_path):
    results = load_results(file_path)
    results.append(result)
    with open(file_path, 'wb') as output:
        for result in results:
            pickle.dump(result, output, pickle.HIGHEST_PROTOCOL)


def run_gridsearch(image_paths, model_types, input_shapes, numbers_of_hidden_layers, numbers_of_hidden_channels, results_path=None):
    noisy_image, target_image = load_images(image_paths)
    if results_path is None:
        file_path = generate_file_path()
    else:
        file_path = results_path
    parameter_combinations = list(
        product(*[model_types, input_shapes, numbers_of_hidden_layers, numbers_of_hidden_channels]))
    fitter = Fitter(5000, convergence_check_length=10, log_frequency=50, find_best=True)
    for parameters in list(parameter_combinations):
        print("+++" + str(parameters) + "+++")
        result = test_parameter_combination(parameters, fitter, noisy_image, target_image)
        save_gridsearch_result(result, file_path)


if __name__ == '__main__':
    run_gridsearch(IMAGE_PATHS, MODEL_TYPES, INPUT_SHAPES, NUMBERS_OF_HIDDEN_LAYERS, NUMBERS_OF_HIDDEN_CHANNELS)