from time import gmtime, strftime

from fitting.Fitter import create_fitter_from_configuration
from fitting.Result import Result
from models.ConvolutionDecoder import ConvolutionDecoder
from models.DeepDecoder import DeepDecoder
from utils.configuration_parser.gridsearch_configuration import get_gridsearch_configuration
from utils.pickle_utils import save_gridsearch_result
from utils.visualization_helpers import load_noisy_and_target_image


def test_parameter_combination(parameters, fitter, noisy_image, target_image):
    if parameters[0] == "conv":
        model = ConvolutionDecoder(parameters[1], noisy_image.shape, parameters[2], parameters[3])
    elif parameters[0] == "deep":
        model = DeepDecoder(parameters[1], noisy_image.shape, parameters[2], parameters[3])
    fitter(model, noisy_image, target_image)
    result = Result(parameters, noisy_image, fitter.get_best_image(), target_image, fitter.get_final_target_loss(),
                    fitter.get_step_counter())
    return result


if __name__ == '__main__':
    gridsearch_configuration = get_gridsearch_configuration()
    noisy_image, target_image = load_noisy_and_target_image(gridsearch_configuration)
    fitter = create_fitter_from_configuration(gridsearch_configuration)
    parameter_combinations = gridsearch_configuration.generate_parameter_combinations()
    for parameters in parameter_combinations:
        print("+++" + str(parameters) + "+++")
        result = test_parameter_combination(parameters, fitter, noisy_image, target_image)
        save_gridsearch_result(result, gridsearch_configuration.result_path)
