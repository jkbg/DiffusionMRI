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
    return model


def create_model_from_parameter_combination(parameter_combination, image_dimensions):
    if parameter_combination[0] == "conv":
        model = ConvolutionDecoder(input_shape=parameter_combination[1],
                                   image_dimensions=image_dimensions,
                                   number_of_layers=parameter_combination[2],
                                   number_of_hidden_channels=parameter_combination[3])
    elif parameter_combination[0] == "deep":
        model = DeepDecoder(input_shape=parameter_combination[1],
                            image_dimensions=image_dimensions,
                            number_of_layers=parameter_combination[2],
                            number_of_hidden_channels=parameter_combination[3])
    return model
