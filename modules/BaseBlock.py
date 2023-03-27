import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor

import numpy as np


class BaseConv2d(nn.Cell):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1):
        super(BaseConv2d, self).__init__()
        self.basicconv = nn.SequentialCell(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, pad_mode="pad", has_bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU()
        )

    def construct(self, x):
        return self.basicconv(x)

def _weight_variable(shape, factor=0.01):
    init_value = np.random.randn(*shape).astype(np.float32) * factor
    return Tensor(init_value)

def _fc(in_channel, out_channel):
    weight_shape = (out_channel, in_channel)
    weight = _weight_variable(weight_shape)
    return nn.Dense(in_channel, out_channel, has_bias=True, weight_init=weight, bias_init=0)

class ChannelAttention(nn.Cell):
    """
    The implementation of channel attention mechanism.
    """

    def __init__(self, channel: int, ratio: int = 2) -> None:
        """
        Args:
            channel: Number of channels for the input features.
            ratio: The node compression ratio in the full connection layer.
        """
        super(ChannelAttention, self).__init__()
        self.avg_pool = ops.ReduceMean(keep_dims=True)
        self.reshape = ops.Reshape()
        self.fc = nn.SequentialCell(
            _fc(channel, channel // ratio),
            nn.ReLU(),
            _fc(channel // ratio, channel)
        )
        self.sigmoid = nn.Sigmoid()

    def construct(self, x: Tensor) -> Tensor:
        """
        Returns the channel attention tensor.
        """
        b, c, _, _ = x.shape
        y = self.avg_pool(x, (2, 3))
        y = self.reshape(y, (b, c))
        y = self.fc(y)
        y = self.reshape(y, (b, c, 1, 1))
        y = self.sigmoid(y)

        return y


class SpatialAttention(nn.Cell):
    """
    spatial attention, return weight map(default)
    """

    def __init__(self, kernel_size: int = 7) -> None:
        super(SpatialAttention, self).__init__()
        # padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(1, 1, kernel_size, has_bias=False)
        self.sigmoid = nn.Sigmoid()
        self.max = ops.ReduceMean(keep_dims=True)

    def construct(self, x: Tensor) -> Tensor:
        """
        Args:
            x: The input feature.

        Returns: A weight map of spatial attention, the size is H×W.

        """
        max_out = self.max(x, 1)
        x = self.conv1(max_out)
        weight_map = self.sigmoid(x)
        return weight_map
