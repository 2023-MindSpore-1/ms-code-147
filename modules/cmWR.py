import numpy as np
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
# from BaseBlock import BaseConv2d

class cmWR(nn.Cell):
    def __init__(self, in_channels:int, squeeze_ratio:int=2) -> None:
        super(cmWR, self).__init__()
        inter_channels = in_channels // squeeze_ratio

        self.conv_r = nn.Conv2d(in_channels, inter_channels, kernel_size=1)
        self.conv_d = nn.Conv2d(in_channels, inter_channels, kernel_size=1)
        self.conv_rd1 = nn.Conv2d(in_channels, inter_channels, kernel_size=1)
        self.conv_rd2 = nn.Conv2d(in_channels, inter_channels, kernel_size=1)
        self.reshape = ops.Reshape()
        self.tranpose = ops.Transpose()
        self.batmatmul = ops.BatchMatMul()
        self.softmax = ops.Softmax(axis=-1)

    def construct(self, rgb: Tensor, depth: Tensor, rgbd: Tensor) -> Tensor:
        B, C, H, W = rgb.shape
        P = H*W

        rgb_t = self.conv_r(rgb)
        rgb_t = self.tranpose(self.reshape(rgb_t, (B, -1, P)), (0, 2, 1))
        depth_t = self.conv_d(depth)
        depth_t = self.reshape(depth_t, (B, -1, P))
        rd_matrix = self.softmax(self.batmatmul(rgb_t, depth_t)) # [B, HW, HW]

        rgbd_t1 = self.conv_rd1(rgbd)
        rgbd_t1 = self.tranpose(self.reshape(rgbd_t1, (B, -1, P)), (0, 2, 1))
        rgbd_t2 = self.conv_rd2(rgbd)
        rgbd_t2 = self.reshape(rgbd_t2, (B, -1, P))
        rgbd_matrix = self.softmax(self.batmatmul(rgbd_t1, rgbd_t2)) # [B, HW, HW]

        weight_com = self.softmax(ops.mul(rd_matrix, rgbd_matrix))

        rgb_m = self.reshape(rgb, (B, -1, P))
        rgb_refine = self.reshape(self.batmatmul(rgb_m, weight_com), (B, C, H, W))
        rgb_final = rgb + rgb_refine

        depth_m = self.reshape(depth, (B, -1, P))
        depth_refine = self.reshape(self.batmatmul(depth_m, weight_com), (B, C, H, W))
        depth_final = depth + depth_refine

        rgbd_m = self.reshape(rgbd, (B, -1, P))
        rgbd_refine = self.reshape(self.batmatmul(rgbd_m, weight_com), (B, C, H, W))
        rgbd_final = rgbd + rgbd_refine

        return rgb_final, depth_final, rgbd_final

if __name__ == '__main__':
    mindspore.context.set_context(device_target="GPU")
    rgb = np.random.randn(2, 128, 256, 256)
    rgb = mindspore.Tensor(rgb, mindspore.float32)
    depth = np.random.randn(2, 128, 256, 256)
    depth = mindspore.Tensor(depth, mindspore.float32)
    rgbd = np.random.randn(2, 128, 256, 256)
    rgbd = mindspore.Tensor(rgbd, mindspore.float32)

    net = cmWR(128)
    out = net(rgb, depth, rgbd)
    # print(out.shape)








