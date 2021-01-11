from models.ConvolutionDecoder import ConvolutionDecoder
from models.DeepDecoder import DeepDecoder


def create_model_from_configuration(fit_model_configuration):
    if fit_model_configuration.model_type == "conv":
        model = ConvolutionDecoder(input_shape=fit_model_configuration.input_shape,
                                   image_dimensions=fit_model_configuration.image_dimensions,
                                   number_of_layers=fit_model_configuration.number_of_layers,
                                   number_of_hidden_channels=fit_model_configuration.number_of_hidden_channels)
    elif fit_model_configuration.model_type == "deep":
        model = DeepDecoder(input_shape=fit_model_configuration.input_shape,
                            image_dimensions=fit_model_configuration.image_dimensions,
                            number_of_layers=fit_model_configuration.number_of_layers,
                            number_of_hidden_channels=fit_model_configuration.number_of_hidden_channels)
    else:
        raise Exception(fit_model_configuration.model_type + 'is not a known model type.')
    return model