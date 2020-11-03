import torch.nn as nn


class ConvolutionModule(nn.Module):
    def __init__(self, upsample_size, number_of_output_channels, number_of_input_channels=None):
        if number_of_input_channels is None:
            number_of_input_channels = number_of_output_channels
        super(ConvolutionModule, self).__init__()
        self.upsample_layer = nn.Upsample(size=upsample_size, mode='nearest')
        self.convolution_layer = nn.Conv2d(in_channels=number_of_input_channels, out_channels=number_of_output_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.activation_layer = nn.ReLU()
        self.batch_normalization = nn.BatchNorm2d(num_features=number_of_output_channels)

    def forward(self, module_input):
        x = self.upsample_layer(module_input)
        x = self.convolution_layer(x)
        x = self.activation_layer(x)
        module_output = self.batch_normalization(x)
        return module_output


if __name__ == '__main__':
    convolution_block = ConvolutionModule([20, 29], 160)
    print(convolution_block)