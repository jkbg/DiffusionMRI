from time import gmtime, strftime

from fitting.Fitter import create_fitter_from_configuration
from fitting.Result import Result
from models.ConvolutionDecoder import ConvolutionDecoder
from models.DeepDecoder import DeepDecoder
from models.model_creation import create_model_from_parameter_combination
from utils.configuration_parser.gridsearch_configuration import get_gridsearch_configuration
from utils.pickle_utils import save_gridsearch_result
from utils.visualization_helpers import load_noisy_and_target_image


def test_parameter_combination(parameters, fitter, noisy_image, target_image):
    if parameters[0] == "conv":
        model = ConvolutionDecoder(parameters[1], noisy_image.shape, parameters[2], parameters[3])
    elif parameters[0] == "deep":
        model = DeepDecoder(parameters[1], noisy_image.shape, parameters[2], parameters[3])

    return result


if __name__ == '__main__':
    gridsearch_configuration = get_gridsearch_configuration()
    noisy_image, target_image = load_noisy_and_target_image(gridsearch_configuration)
    fitter = create_fitter_from_configuration(gridsearch_configuration)
    parameter_combinations = gridsearch_configuration.generate_parameter_combinations()
    for parameter_combination in parameter_combinations:
        print("+++" + str(parameter_combination) + "+++")
        model = create_model_from_parameter_combination(parameter_combination, gridsearch_configuration.image_shape)
        fitter(model, noisy_image, target_image)
        result = fitter.get_result()
        save_gridsearch_result(result, gridsearch_configuration.result_path)
