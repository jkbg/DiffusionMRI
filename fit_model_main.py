from utils.configuration_parser.fit_model_configuration import get_fit_model_configuration
from utils.visualization_helpers import load_noisy_and_target_image, show_images
from models.model_creation import create_model_from_configuration
from fitting.Fitter import create_fitter_from_configuration

if __name__ == '__main__':
    fit_model_configuration = get_fit_model_configuration()
    noisy_image, target_image = load_noisy_and_target_image(fit_model_configuration)
    model = create_model_from_configuration(fit_model_configuration)
    fitter = create_fitter_from_configuration(fit_model_configuration)

    fitter(model, noisy_image, target_image)

    model_image = fitter.get_best_image()
    show_images(noisy_image, model_image, target_image, str(model), fit_model_configuration.result_path)
