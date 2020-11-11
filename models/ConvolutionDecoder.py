import torch
import torch.nn as nn
import numpy as np
from torchsummary import summary

from models.ConvolutionModule import ConvolutionModule


class ConvolutionDecoder(nn.Module):
    def __init__(self, input_shape, image_dimensions, number_of_layers, number_of_hidden_channels):
        super(ConvolutionDecoder, self).__init__()
        self.input_shape = [1, 1] + input_shape
        self.output_shape = image_dimensions[:2]
        self.number_of_layers = number_of_layers
        self.number_of_hidden_channels = number_of_hidden_channels
        self.number_of_output_channels = image_dimensions[2]
        upsample_sizes = calculate_upsample_sizes(input_shape, self.output_shape, number_of_layers)

        # Initialize Module List to be filled with Module consisting of layers
        self.module_list = nn.ModuleList()

        # Add layer module consisting of Up-Sampling, Convolution, ReLU and Batch Normalization layer
        for upsample_size in upsample_sizes:
            if not self.module_list:
                self.module_list.append(
                    ConvolutionModule(upsample_size, number_of_hidden_channels, number_of_input_channels=1))
            else:
                self.module_list.append(ConvolutionModule(upsample_size, number_of_hidden_channels))

        # Add final module consisting of Convolution, ReLU, Batch Normalization and a channels to image layer
        self.module_list.append(
            nn.Conv2d(in_channels=number_of_hidden_channels, out_channels=number_of_hidden_channels, kernel_size=3,
                      stride=1, padding=1, bias=False))
        self.module_list.append(nn.ReLU())
        self.module_list.append(nn.BatchNorm2d(num_features=number_of_hidden_channels))
        self.module_list.append(
            nn.Conv2d(in_channels=number_of_hidden_channels, out_channels=self.number_of_output_channels, kernel_size=1,
                      stride=1, padding=0, bias=False))

    def forward(self, x):
        for module in self.module_list:
            x = module(x)
        x.resize(self.number_of_output_channels, self.image_dimensions[0], self.image_dimensions[1])
        return x

    def get_input_shape(self):
        return self.input_shape

    def __str__(self):
        output_string = "Convolution Decoder"
        output_string += str(self.input_shape[-2:])
        output_string += ", "
        output_string += str(self.number_of_layers)
        output_string += ", "
        output_string += str(self.number_of_hidden_channels)
        return output_string

    def get_model_parameters(self):
        return ['conv', self.input_shape[-2:], self.number_of_layers, self.number_of_hidden_channels]


def calculate_upsample_sizes(input_shape, output_shape, number_of_layers):
    scale = (np.array(output_shape) / np.array(input_shape)) ** (1 / (number_of_layers - 1))
    upsample_sizes = [np.ceil(np.array(input_shape) * (scale ** n)).astype(int).tolist() for n in
                      range(1, number_of_layers - 1)] + [output_shape]
    return upsample_sizes


if __name__ == '__main__':
    model = ConvolutionDecoder([5, 2], [160, 64, 3], 6, 160)
    print(model.module_list)
    summary(model, (1, 5, 2))
    print(model(torch.ones([1, 1, 5, 2])).shape)
