import torch.nn as nn
import numpy as np

from models.DeepModule import DeepModule


class DeepDecoder(nn.Module):
    def __init__(self, input_shape, image_dimensions, number_of_layers, number_of_hidden_channels):
        super(DeepDecoder, self).__init__()
        self.input_shape = [1, number_of_hidden_channels] + input_shape
        self.image_dimensions = image_dimensions[:2]
        self.number_of_layers = number_of_layers
        self.number_of_hidden_channels = number_of_hidden_channels
        self.number_of_output_channels = image_dimensions[2]
        upsample_sizes = calculate_upsample_sizes(input_shape, self.image_dimensions, number_of_layers)

        # Initialize Module List to be filled with Module consisting of layers
        self.module_list = nn.ModuleList()

        # Fill Module List with Modules consisting of Reflection, Convolution, Upsampling, ReLU and Batch Normalization
        for upsample_size in upsample_sizes:
            self.module_list.append(DeepModule(number_of_hidden_channels, number_of_hidden_channels, upsample_size))

        # Add final module
        self.module_list.append(nn.ReflectionPad2d((0, 0, 0, 0)))
        self.module_list.append(nn.Conv2d(in_channels=number_of_hidden_channels,
                                          out_channels=self.number_of_output_channels,
                                          kernel_size=1,
                                          stride=1,
                                          bias=False))
        self.module_list.append(nn.Sigmoid())

    def forward(self, x):
        for module in self.module_list:
            x = module(x)
        return x.reshape(self.number_of_output_channels, self.image_dimensions[0], self.image_dimensions[1])

    def get_input_shape(self):
        return self.input_shape

    def __str__(self):
        output_string = "Deep Decoder "
        output_string += str(self.input_shape[-2:])
        output_string += ", "
        output_string += str(self.number_of_layers)
        output_string += ", "
        output_string += str(self.number_of_hidden_channels)
        return output_string

    def get_model_parameters(self):
        return ['deep', self.input_shape[-2:], self.number_of_layers, self.number_of_hidden_channels]


def calculate_upsample_sizes(input_shape, output_shape, number_of_layers):
    scale = (np.array(output_shape) / np.array(input_shape)) ** (1 / number_of_layers)
    upsample_sizes = [np.ceil(np.array(input_shape) * (scale ** n)).astype(int).tolist() for n in
                      range(1, number_of_layers)] + [output_shape]
    return upsample_sizes


if __name__ == '__main__':
    deep_decoder = DeepDecoder([8, 8], [256, 256, 3], 6, 128)
    # summary(deep_decoder, (128, 8, 8))
    print(deep_decoder.module_list)
    print(deep_decoder)
