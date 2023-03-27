# from backbone.resnet_ms import resnet50
from backbone.resnet_ms import resnet50
from modules.cmWR import cmWR
from modules.BaseBlock import BaseConv2d, SpatialAttention, ChannelAttention
from modules.Decoder import Decoder

import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.ops.function as F
from mindspore import Tensor

class CIRNet_R50(nn.Cell):
    def __init__(self, backbone='resnet50', norm_layer=nn.BatchNorm2d):
        super(CIRNet_R50, self).__init__()
        # backbone
        self.resnet_rgb = resnet50(pretrained=True)
        self.resnet_depth = resnet50(pretrained=True)

        res_channels = [64, 256, 512, 1024, 2048]
        channels = [64, 128, 256, 512, 512]

        # layer 1
        self.re1_r = BaseConv2d(res_channels[0], channels[0], kernel_size=1)
        self.re1_d = BaseConv2d(res_channels[0], channels[0], kernel_size=1)

        # layer 2
        self.re2_r = BaseConv2d(res_channels[1], channels[1], kernel_size=1)
        self.re2_d = BaseConv2d(res_channels[1], channels[1], kernel_size=1)

        # layer 3
        self.re3_r = BaseConv2d(res_channels[2], channels[2], kernel_size=1)
        self.re3_d = BaseConv2d(res_channels[2], channels[2], kernel_size=1)
         
        self.conv1 = BaseConv2d(2 * channels[2], channels[2], kernel_size=1)
        self.sa1 = SpatialAttention(kernel_size=7)

        # layer 4
        self.re4_r = BaseConv2d(res_channels[3], channels[3], kernel_size=1)
        self.re4_d = BaseConv2d(res_channels[3], channels[3], kernel_size=1)

        self.conv2 = BaseConv2d(2 * channels[3], channels[3], kernel_size=1)
        self.sa2 = SpatialAttention(kernel_size=7)

        # layer 5
        self.re5_r = BaseConv2d(res_channels[4], channels[4], kernel_size=1)
        self.re5_d = BaseConv2d(res_channels[4], channels[4], kernel_size=1)

        self.conv3 = BaseConv2d(2 * channels[4], channels[4], kernel_size=1)

        self.ca_rgb = ChannelAttention(channels[4])
        self.ca_depth = ChannelAttention(channels[4])
        self.ca_rgbd = ChannelAttention(channels[4])

        self.sa_rgb = SpatialAttention(kernel_size=7)
        self.sa_depth = SpatialAttention(kernel_size=7)
        self.sa_rgbd = SpatialAttention(kernel_size=7)

        # cross-modality weighting refinement
        self.cmWR = cmWR(channels[4], squeeze_ratio=1)

        self.conv_rgb = BaseConv2d(channels[4], channels[4], kernel_size=3, padding=1)
        self.conv_depth = BaseConv2d(channels[4], channels[4], kernel_size=3, padding=1)
        self.conv_rgbd = BaseConv2d(channels[4], channels[4], kernel_size=3, padding=1)

        self.reshape = ops.Reshape()
        self.tranpose = ops.Transpose()
        self.batmatmul = ops.BatchMatMul()
        
        self.decoder = Decoder()
    
    def construct(self, rgb:Tensor, depth:Tensor):
        decoder_rgb_list = []
        decoder_depth_list = []
        # rgb = rgb.squeeze()
        # depth = depth.squeeze()
        depth = ops.concat((depth, depth, depth), axis=1)

        # encoder layer 1
        # conv1_res_r, conv2_res_r, conv3_res_r, conv4_res_r, conv5_res_r = self.resnet_rgb(rgb)
        # conv1_res_d, conv2_res_d, conv3_res_d, conv4_res_d, conv5_res_d = self.resnet_rgb(depth)

        conv1_res_r = self.resnet_rgb.conv1(rgb)
        conv1_res_r = self.resnet_rgb.bn1(conv1_res_r)
        conv1_res_r = self.resnet_rgb.relu(conv1_res_r)
        
        conv2_res_r = self.resnet_rgb.maxpool(conv1_res_r)
        conv2_res_r = self.resnet_rgb.layer1(conv2_res_r)

        conv3_res_r = self.resnet_rgb.layer2(conv2_res_r)
        conv4_res_r = self.resnet_rgb.layer3(conv3_res_r)
        conv5_res_r = self.resnet_rgb.layer4(conv4_res_r)

        conv1_res_d = self.resnet_depth.conv1(depth)
        conv1_res_d = self.resnet_rgb.bn1(conv1_res_d)
        conv1_res_d = self.resnet_rgb.relu(conv1_res_d)
        
        conv2_res_d = self.resnet_rgb.maxpool(conv1_res_d)
        conv2_res_d = self.resnet_rgb.layer1(conv2_res_d)

        conv3_res_d = self.resnet_rgb.layer2(conv2_res_d)
        conv4_res_d = self.resnet_rgb.layer3(conv3_res_d)
        conv5_res_d = self.resnet_rgb.layer4(conv4_res_d)


        conv1_r = self.re1_r(conv1_res_r)
        conv1_d = self.re1_d(conv1_res_d)
        decoder_rgb_list.append(conv1_r)
        decoder_depth_list.append(conv1_d)

        # encoder layer 2
        conv2_r = self.re2_r(conv2_res_r)
        conv2_d = self.re2_d(conv2_res_d)
        decoder_rgb_list.append(conv2_r)
        decoder_depth_list.append(conv2_d)

        # encoder layer 3
        # progressive attention guided integration unit
        conv3_r = self.re3_r(conv3_res_r)
        conv3_d = self.re3_d(conv3_res_d)
        conv3_rgbd = self.conv1(ops.concat((conv3_r, conv3_d), axis=1))
        conv3_rgbd = F.interpolate(conv3_rgbd, scales=(1.,1.,0.5,0.5), mode='bilinear')
        conv3_rgbd_map = self.sa1(conv3_rgbd)
        decoder_rgb_list.append(conv3_r)
        decoder_depth_list.append(conv3_d)

        # encoder layer 4
        conv4_r = self.re4_r(conv4_res_r)
        conv4_d = self.re4_d(conv4_res_d)
        conv4_rgbd = self.conv2(ops.concat((conv4_r, conv4_d), axis=1))
        conv4_rgbd = conv4_rgbd * conv3_rgbd_map + conv4_rgbd
        conv4_rgbd = F.interpolate(conv4_rgbd, scales=(1.,1.,0.5,0.5), mode='bilinear')
        conv4_rgbd_map = self.sa2(conv4_rgbd)
        decoder_rgb_list.append(conv4_r)
        decoder_depth_list.append(conv4_d)

        # encoder layer 5
        conv5_r = self.re5_r(conv5_res_r)
        conv5_d = self.re5_d(conv5_res_d)
        conv5_rgbd = self.conv3(ops.concat((conv5_r, conv5_d), axis=1))
        conv5_rgbd = conv5_rgbd * conv4_rgbd_map + conv5_rgbd
        decoder_rgb_list.append(conv5_r)
        decoder_depth_list.append(conv5_d)

        # self-modality attention refinement
        B, C, H, W = conv5_r.shape
        P = H * W

        rgb_SA = self.reshape(self.sa_rgb(conv5_r), (B, -1, P))            # B * 1 * H * W
        depth_SA = self.reshape(self.sa_depth(conv5_d), (B, -1, P))
        rgbd_SA = self.reshape(self.sa_rgbd(conv5_rgbd), (B, -1, P))

        rgb_CA = self.reshape(self.ca_rgb(conv5_r), (B, C, -1))           # B * C * 1 * 1
        depth_CA = self.reshape(self.ca_depth(conv5_d), (B, C, -1))
        rgbd_CA = self.reshape(self.ca_rgbd(conv5_rgbd), (B, C, -1))


        rgb_M = self.reshape(self.batmatmul(rgb_CA, rgb_SA), (B, C, H, W))
        depth_M = self.reshape(self.batmatmul(depth_CA, depth_SA), (B, C, H, W))
        rgbd_M = self.reshape(self.batmatmul(rgbd_CA, rgbd_SA), (B, C, H, W))

        rgb_smAR = conv5_r *  rgb_M + conv5_r 
        depth_smAR = conv5_d * depth_M + conv5_d 
        rgbd_smAR = conv5_rgbd * rgbd_M + conv5_rgbd 
        
        # 
        rgb_smAR = self.conv_rgb(rgb_smAR)
        depth_smAR = self.conv_depth(depth_smAR)
        rgbd_smAR = self.conv_rgbd(rgbd_smAR)

        # cross-modality weighting refinement
        rgb_cmWR, depth_cmWR, rgbd_cmWR = self.cmWR(rgb_smAR, depth_smAR, rgbd_smAR)
        
        decoder_rgb_list.append(rgb_cmWR)
        decoder_depth_list.append(depth_cmWR)
        
        # decoder
        rgb_map, depth_map, rgbd_map = self.decoder(decoder_rgb_list, decoder_depth_list, rgbd_cmWR)
        return rgb_map, depth_map, rgbd_map