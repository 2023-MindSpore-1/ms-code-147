import os
import cv2
import time
import random
import numpy as np
from datetime import datetime

import mindspore
# import moxing as mox
from mindspore import nn,context
from mindspore.nn.dynamic_lr import piecewise_constant_lr
import mindspore.ops as ops

from settings.options import opt
from settings.dataLoad import TestDataset
from model.CIRNet_R50 import CIRNet_R50

def seed_mindspore(seed=42):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

class ComputeLoss(nn.Cell):
    def __init__(self, network, loss_fn):
        super(ComputeLoss, self).__init__(auto_prefix=False)
        self.network = network
        self._loss_fn = loss_fn

    def construct(self, rgb, depth, label):
        label = label.squeeze(axis=1)
        out = self.network(rgb, depth)
        return self._loss_fn(out, label)

def test(opt):
    # load model
    print('load model')
    net = CIRNet_R50()
    mindspore.load_param_into_net(net, mindspore.load_checkpoint(opt.test_model))
    model = mindspore.Model(net)
    # train
    sigmoid = mindspore.ops.Sigmoid()
    print("==================Starting Testing==================")
    test_datasets = ['DUT', 'NJU2K', 'NLPR', 'STERE', 'SIP', 'LFSD']
    dataset_path = opt.test_path
    for dataset in test_datasets:
        print("Testing {} ...".format(dataset))
        save_path = 'test_maps/' + dataset + '/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        image_root = dataset_path + dataset + '/RGB/'
        depth_root = dataset_path + dataset + '/depth/'
        gt_root = dataset_path + dataset + '/GT/'
        test_loader = TestDataset(image_root, depth_root, gt_root, opt.testsize)
        for i in range(test_loader.size):
            image, depth, gt, name = test_loader.load_data()
            name = name.split('/')[-1]
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)

            _, _, pre = model.predict(mindspore.Tensor(image), mindspore.Tensor(depth))
            resizer = ops.ResizeNearestNeighbor((gt.shape[-2], gt.shape[-1]))
            pre = resizer(pre)

            pre = sigmoid(pre).squeeze().asnumpy()
            print(save_path + name)

            pre = (pre - pre.min()) / (pre.max() - pre.min() + 1e-8)
            cv2.imwrite(save_path + name, pre*255)
        print("Dataset:{} testing completed.".format(dataset))
    print("==================Ending Testinging==================")


if __name__ == '__main__':
    from settings.options import parser
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_id
    context.set_context(mode=context.PYNATIVE_MODE, device_target=args.device_target)
    # Train
    seed_mindspore()
    test(args)

