from fitting.Fitter import create_fitter_from_configuration
from models.model_creation import create_model_from_parameter_combination
from utils.configuration_parser.gridsearch_configuration import get_gridsearch_configuration
from utils.pickle_utils import save_gridsearch_result
from utils.image_helpers import load_images

if __name__ == '__main__':
    gridsearch_configuration = get_gridsearch_configuration()
    noisy_image, target_image = load_images(['data/raw_images/canonical_noisy.png', 'data/raw_images/canonical_target.png'])
    fitter = create_fitter_from_configuration(gridsearch_configuration)
    parameter_combinations = gridsearch_configuration.generate_parameter_combinations()
    for parameter_combination in parameter_combinations:
        print("+++" + str(parameter_combination) + "+++")
        model = create_model_from_parameter_combination(parameter_combination, gridsearch_configuration.image_dimensions)
        fitter(model, noisy_image, target_image)
        result = fitter.get_result()
        save_gridsearch_result(result, gridsearch_configuration.result_path)
        print('')