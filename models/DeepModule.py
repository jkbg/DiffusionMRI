import torch.nn as nn
from torchsummary import summary


class DeepModule(nn.Module):
    def __init__(self, number_of_input_channels, number_of_output_channels, upsample_size,
                 upsample_mode='bilinear'):
        if number_of_input_channels is None:
            number_of_input_channels = number_of_output_channels
        super(DeepModule, self).__init__()
        self.reflection_layer = nn.ReflectionPad2d((0, 0, 0, 0))
        self.convolution_layer = nn.Conv2d(in_channels=number_of_input_channels, out_channels=number_of_output_channels,
                                           kernel_size=1, stride=1, bias=False)
        self.upsample_layer = nn.Upsample(size=upsample_size, mode=upsample_mode, align_corners=False)
        self.activation_layer = nn.ReLU()
        self.batch_normalization = nn.BatchNorm2d(num_features=number_of_output_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, module_input):
        x = self.reflection_layer(module_input)
        x = self.convolution_layer(x)
        x = self.upsample_layer(x)
        x = self.activation_layer(x)
        module_output = self.batch_normalization(x)
        return module_output


if __name__ == '__main__':
    deep_module = DeepModule(128, 3, [10, 2])
    print(deep_module)
    summary(deep_module, (128, 5, 2))