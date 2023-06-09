import argparse

parser = argparse.ArgumentParser()

# train settings
parser.add_argument('--epoch', type=int, default=50, help='epoch number')
parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')         # 1e-4
parser.add_argument('--batchsize', type=int, default=16, help='training batch size')     # 16
parser.add_argument('--trainsize', type=int, default=352, help='training image size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float, default=0.2, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=40, help='every n epochs decay learning rate')

parser.add_argument('--load', type=str, default="./CIRNet_cpts/CIRNet.ckpt", help='train from checkpoints')
parser.add_argument("--device_id", type=str, default='7', help="Device id")
parser.add_argument('--device_target', type=str, default="GPU",
                    choices=['Ascend', 'GPU', 'CPU'], help='device target')
parser.add_argument('--backbone', type=str, default='R50', help='backbone networks:R50 or V16')

parser.add_argument('--rgb_root', type=str, default='../data/RGBD_for_train/RGB/',
                    help='the training rgb images root')
parser.add_argument('--depth_root', type=str, default='../data/RGBD_for_train/depth/',
                    help='the training depth images root')
parser.add_argument('--gt_root', type=str, default='../data/RGBD_for_train/GT/',
                    help='the training gt images root')
parser.add_argument('--savepath', type=str, default='./CIRNet_cpts/', help='the path to save models and logs')

# test set
parser.add_argument('--testsize', type=int, default=352, help='testing image size')
parser.add_argument('--test_path', type=str, default='../data/RGBD_for_test/', help='test dataset path')
parser.add_argument('--test_model', type=str, default='./CIRNet_cpts/CIRNet.ckpt',
                    help='load the model for testing')

opt = parser.parse_args()
