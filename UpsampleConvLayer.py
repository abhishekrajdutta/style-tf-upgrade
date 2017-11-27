import torch
import torch.nn as nn


class UpsampleConvLayer(torch.nn.Module):
          def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
              super(UpsampleConvLayer, self).__init__()
              self.upsample = upsample
              if upsample:
                  self.upsample_layer = torch.nn.Upsample(scale_factor=upsample)
              reflection_padding = kernel_size // 2
              self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
              self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

          def forward(self, x):
              x_in = x
              if self.upsample:
                  x_in = self.upsample_layer(x_in)
              out = self.reflection_pad(x_in)
              out = self.conv2d(out)
              return out