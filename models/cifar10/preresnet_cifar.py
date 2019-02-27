#
# Copyright (c) 2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""Pre-Activation ResNet for CIFAR10

Pre-Activation ResNet for CIFAR10, based on "Identity Mappings in Deep Residual Networks".
This is based on TorchVision's implementation of ResNet for ImageNet, with appropriate
changes for pre-activation and the 10-class Cifar-10 dataset.
This ResNet also has layer gates, to be able to dynamically remove layers.

@article{
  He2016,
  author = {Kaiming He and Xiangyu Zhang and Shaoqing Ren and Jian Sun},
  title = {Identity Mappings in Deep Residual Networks},
  journal = {arXiv preprint arXiv:1603.05027},
  year = {2016}
}
"""
import torch
import torch.nn as nn
import math

__all__ = ['preact_resnet20_cifar', 'preact_resnet32_cifar', 'preact_resnet44_cifar', 'preact_resnet56_cifar',
           'preact_resnet110_cifar', 'preact_resnet20_cifar_conv_ds', 'preact_resnet32_cifar_conv_ds',
           'preact_resnet44_cifar_conv_ds', 'preact_resnet56_cifar_conv_ds', 'preact_resnet110_cifar_conv_ds',
		   'preact_resnet20_cifar100'] #ADD at 2/22

NUM_CLASSES = 10

def _prep_saturation_val_tensor(sat_val):
    is_scalar = not isinstance(sat_val, torch.Tensor)
    out = torch.tensor(sat_val)
    if not out.is_floating_point():
        out = out.to(torch.float32)
    if out.dim() == 0:
        out = out.unsqueeze(0)
    return is_scalar, out


def symmetric_linear_quantization_params(num_bits, saturation_val):
    is_scalar, sat_val = _prep_saturation_val_tensor(saturation_val)

    if any(sat_val < 0):
        raise ValueError('Saturation value must be >= 0')

    # Leave one bit for sign
    n = 2 ** (num_bits - 1) - 1

    # If float values are all 0, we just want the quantized values to be 0 as well. So overriding the saturation
    # value to 'n', so the scale becomes 1
    sat_val[sat_val == 0] = n
    scale = n / sat_val
    zero_point = torch.zeros_like(scale)

    if is_scalar:
        # If input was scalar, return scalars
        return scale.item(), zero_point.item()
    return scale, zero_point


def asymmetric_linear_quantization_params(num_bits, saturation_min, saturation_max,
                                          integral_zero_point=True, signed=False):
    scalar_min, sat_min = _prep_saturation_val_tensor(saturation_min)
    scalar_max, sat_max = _prep_saturation_val_tensor(saturation_max)
    is_scalar = scalar_min and scalar_max

    if scalar_max and not scalar_min:
        sat_max = sat_max.to(sat_min.device)
    elif scalar_min and not scalar_max:
        sat_min = sat_min.to(sat_max.device)

    if any(sat_min > sat_max):
        raise ValueError('saturation_min must be smaller than saturation_max')

    n = 2 ** num_bits - 1

    # Make sure 0 is in the range
    sat_min = torch.min(sat_min, torch.zeros_like(sat_min))
    sat_max = torch.max(sat_max, torch.zeros_like(sat_max))

    diff = sat_max - sat_min
    # If float values are all 0, we just want the quantized values to be 0 as well. So overriding the saturation
    # value to 'n', so the scale becomes 1
    diff[diff == 0] = n

    scale = n / diff
    zero_point = scale * sat_min
    if integral_zero_point:
        zero_point = zero_point.round()
    if signed:
        zero_point += 2 ** (num_bits - 1)
    if is_scalar:
        return scale.item(), zero_point.item()
    return scale, zero_point


def clamp(input, min, max, inplace=False):
    if inplace:
        input.clamp_(min, max)
        return input
    return torch.clamp(input, min, max)


def linear_quantize(input, scale, zero_point, inplace=False):
    if inplace:
        input.mul_(scale).sub_(zero_point).round_()
        return input
    return torch.round(scale * input - zero_point)


def linear_quantize_clamp(input, scale, zero_point, clamp_min, clamp_max, inplace=False):
    output = linear_quantize(input, scale, zero_point, inplace)
    return clamp(output, clamp_min, clamp_max, inplace)


def linear_dequantize(input, scale, zero_point, inplace=False):
    if inplace:
        input.add_(zero_point).div_(scale)
        return input
    return (input + zero_point) / scale


def get_tensor_min_max(t, per_dim=None):
    if per_dim is None:
        return t.min(), t.max()
    if per_dim > t.dim():
        raise ValueError('Got per_dim={0}, but tensor only has {1} dimensions', per_dim, t.dim())
    view_dims = [t.shape[i] for i in range(per_dim + 1)] + [-1]
    tv = t.view(*view_dims)
    return tv.min(dim=-1)[0], tv.max(dim=-1)[0]


def get_tensor_avg_min_max(t, across_dim=None):
    min_per_dim, max_per_dim = get_tensor_min_max(t, per_dim=across_dim)
    return min_per_dim.mean(), max_per_dim.mean()


def get_tensor_max_abs(t, per_dim=None):
    min_val, max_val = get_tensor_min_max(t, per_dim=per_dim)
    return torch.max(min_val.abs_(), max_val.abs_())


def get_tensor_avg_max_abs(t, across_dim=None):
    avg_min, avg_max = get_tensor_avg_min_max(t, across_dim=across_dim)
    return torch.max(avg_min.abs_(), avg_max.abs_())


def get_quantized_range(num_bits, signed=True):
    if signed:
        n = 2 ** (num_bits - 1)
        return -n, n - 1
    return 0, 2 ** num_bits - 1


class LinearQuantizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale, zero_point, dequantize, inplace):
        if inplace:
            ctx.mark_dirty(input)
        output = linear_quantize(input, scale, zero_point, inplace) #Become integer(0,1,2,3,4,....)
        if dequantize:
            output = linear_dequantize(output, scale, zero_point, inplace) #Become float(0,0.333,0.667,1)...
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through estimator
        return grad_output, None, None, None, None

class Demolition_Conv2d(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(Demolition_Conv2d, self).__init__()

        self.in_channels = in_channels
        self.scale, self.zero_point = symmetric_linear_quantization_params(5, 9)# bit:5 9MAC:9
        self.conv = nn.ModuleList([nn.Conv2d(1, out_channels, kernel_size, stride=stride, 
                    padding=padding, dilation=dilation, groups=groups, bias=bias) for i in range(0, in_channels)])
    def forward(self, x):
        out = LinearQuantizeSTE.apply(self.conv[0](x[:,0:1:,:,:]), self.scale, self.zero_point, True, False)
        for i in range(1, self.in_channels):
            out += LinearQuantizeSTE.apply(self.conv[i](x[:,i:i+1,:,:]), self.scale, self.zero_point, True, False)
        return out

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
    #return Demolition_Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     #padding=1, bias=False)


class PreactBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, block_gates, inplanes, planes, stride=1, downsample=None, preact_downsample=True):
        super(PreactBasicBlock, self).__init__()
        self.block_gates = block_gates
        self.pre_bn = nn.BatchNorm2d(inplanes)
        self.pre_relu = nn.ReLU(inplace=False)  # To enable layer removal inplace must be False
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(planes, planes)
        self.downsample = downsample
        self.stride = stride
        self.preact_downsample = preact_downsample

    def forward(self, x):
        need_preact = self.block_gates[0] or self.block_gates[1] or self.downsample and self.preact_downsample
        if need_preact:
            preact = self.pre_bn(x)
            preact = self.pre_relu(preact)
            out = preact
        else:
            preact = out = x

        if self.block_gates[0]:
            out = self.conv1(out)
            out = self.bn(out)
            out = self.relu(out)

        if self.block_gates[1]:
            out = self.conv2(out)

        if self.downsample is not None:
            if self.preact_downsample:
                residual = self.downsample(preact)
            else:
                residual = self.downsample(x)
        else:
            residual = x

        out += residual

        return out


class PreactResNetCifar(nn.Module):
    def __init__(self, block, layers, num_classes=NUM_CLASSES, conv_downsample=False):
        self.nlayers = 0
        # Each layer manages its own gates
        self.layer_gates = []
        for layer in range(3):
            # For each of the 3 layers, create block gates: each block has two layers
            self.layer_gates.append([])  # [True, True] * layers[layer])
            for blk in range(layers[layer]):
                self.layer_gates[layer].append([True, True])

        self.inplanes = 16  # 64
        super(PreactResNetCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(self.layer_gates[0], block, 16, layers[0],
                                       conv_downsample=conv_downsample)
        self.layer2 = self._make_layer(self.layer_gates[1], block, 32, layers[1], stride=2,
                                       conv_downsample=conv_downsample)
        self.layer3 = self._make_layer(self.layer_gates[2], block, 64, layers[2], stride=2,
                                       conv_downsample=conv_downsample)
        self.final_bn = nn.BatchNorm2d(64 * block.expansion)
        self.final_relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, layer_gates, block, planes, blocks, stride=1, conv_downsample=False):
        downsample = None
        outplanes = planes * block.expansion
        if stride != 1 or self.inplanes != outplanes:
            if conv_downsample:
                downsample = nn.Conv2d(self.inplanes, outplanes,
                                       kernel_size=1, stride=stride, bias=False)
            else:
                # Identity downsample uses strided average pooling + padding instead of convolution
                pad_amount = int(self.inplanes / 2)
                downsample = nn.Sequential(
                    nn.AvgPool2d(2),
                    nn.ConstantPad3d((0, 0, 0, 0, pad_amount, pad_amount), 0)
                )

        layers = []
        layers.append(block(layer_gates[0], self.inplanes, planes, stride, downsample, conv_downsample))
        self.inplanes = outplanes
        for i in range(1, blocks):
            layers.append(block(layer_gates[i], self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.final_bn(x)
        x = self.final_relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def preact_resnet20_cifar(**kwargs):
    model = PreactResNetCifar(PreactBasicBlock, [3, 3, 3], **kwargs)
    return model


def preact_resnet32_cifar(**kwargs):
    model = PreactResNetCifar(PreactBasicBlock, [5, 5, 5], **kwargs)
    return model


def preact_resnet44_cifar(**kwargs):
    model = PreactResNetCifar(PreactBasicBlock, [7, 7, 7], **kwargs)
    return model


def preact_resnet56_cifar(**kwargs):
    model = PreactResNetCifar(PreactBasicBlock, [9, 9, 9], **kwargs)
    return model


def preact_resnet110_cifar(**kwargs):
    model = PreactResNetCifar(PreactBasicBlock, [18, 18, 18], **kwargs)
    return model


def preact_resnet182_cifar(**kwargs):
    model = PreactResNetCifar(PreactBasicBlock, [30, 30, 30], **kwargs)
    return model


def preact_resnet20_cifar_conv_ds(**kwargs):
    return preact_resnet20_cifar(conv_downsample=True)


def preact_resnet32_cifar_conv_ds(**kwargs):
    return preact_resnet32_cifar(conv_downsample=True)


def preact_resnet44_cifar_conv_ds(**kwargs):
    return preact_resnet44_cifar(conv_downsample=True)


def preact_resnet56_cifar_conv_ds(**kwargs):
    return preact_resnet56_cifar(conv_downsample=True)


def preact_resnet110_cifar_conv_ds(**kwargs):
    return preact_resnet110_cifar(conv_downsample=True)


def preact_resnet182_cifar_conv_ds(**kwargs):
    return preact_resnet182_cifar(conv_downsample=True)

def preact_resnet20_cifar100(**kwargs): #Add at 2/22
    model = PreactResNetCifar(PreactBasicBlock, [3, 3, 3], 100, **kwargs)
    return model