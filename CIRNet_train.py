import os
from pickletools import optimize
from statistics import mode
import time
import random
import numpy as np
from datetime import datetime



import mindspore
# import moxing as mox
from mindspore import nn, dataset, context
from mindspore.nn.dynamic_lr import piecewise_constant_lr
from mindspore.ops import functional as F

# from settings.options import opt
from settings.dataLoad import get_iterator
# from settings.utils import clip_gradient, adjust_lr
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

        rgb_, depth_, rgbd = self.network(rgb, depth)
        loss_r = self._loss_fn(rgb_, label)
        loss_d = self._loss_fn(depth_, label)
        loss_rgbd = self._loss_fn(rgbd, label)
        return loss_r,  loss_d,  loss_rgbd

class MultiLossTrainOneStepCell(nn.TrainOneStepCell):
    def __init__(self, network, optimizer, sens=1.0):
        super(MultiLossTrainOneStepCell, self).__init__(network, optimizer, sens)

    def construct(self, *inputs):
        loss_r,  loss_d,  loss_rgbd = self.network(*inputs)
        sens1 = F.fill(loss_r.dtype, loss_r.shape, self.sens)
        sens2 = F.fill(loss_d.dtype, loss_d.shape, self.sens)
        sens3 = F.fill(loss_rgbd.dtype, loss_rgbd.shape, self.sens)
        sens = (sens1,sens2,sens3)

        grads = self.grad(self.network, self.weights)(*inputs, sens)
        grads = self.grad_reducer(grads)

        return F.depend(loss_r, self.optimizer(grads)), F.depend(loss_d, self.optimizer(grads)), F.depend(loss_rgbd, self.optimizer(grads))


def train(opt):
    # load data
    print('load data')
    train_iterator, iterations_epoch = get_iterator(opt.rgb_root, opt.depth_root, opt.gt_root,\
         opt.batchsize,opt.trainsize)
    # model
    print('load model')
    model = CIRNet_R50()
    if opt.load is not None:
        mindspore.load_checkpoint(opt.load, net=model)
        print('load model from', opt.load)
    # optimizer
    optimizer = mindspore.nn.optim.Adam(model.trainable_params(), opt.lr, eps=1e-08, weight_decay=0.0)
    bce_loss = mindspore.nn.BCEWithLogitsLoss()
    # setting 
    net = ComputeLoss(model, bce_loss)
    train_net = MultiLossTrainOneStepCell(net, optimizer)
    model.set_train()
    epoch = opt.epoch
    print("==================Starting Training==================")
    for i in range(epoch):
        loss_all = 0
        epoch_num = i + 1
        epoch_step = 0

        time_begin_epoch = time.time()
        for iteration, data in enumerate(train_iterator, start=1):
            rgb, depth, label = data["rgb"], data["depth"], data["label"]
            rgb = F.squeeze(rgb, axis=(1))
            depth = F.squeeze(depth, axis=(1))
            label = F.squeeze(label, axis=(1))
            # forward
            loss_r, loss_d, loss_rgbd = train_net(rgb, depth, label)
            loss_all += loss_rgbd.asnumpy()
            epoch_step += 1

            if iteration % 50 == 0:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss: {:.4f}'.
                      format(datetime.now(), epoch_num, epoch, iteration, iterations_epoch, loss_all / epoch_step))
        time_end_epoch = time.time()
        loss_all /= epoch_step
        print('Epoch [{:03d}/{:03d}]:Loss_AVG={:.4f}, Time:{:.2f}'.
              format(epoch_num, epoch, loss_all, time_end_epoch - time_begin_epoch))
        if epoch_num > 30 and epoch_num % 5 == 0:
            mindspore.save_checkpoint(model, './CIRNet_cpts/CIRNet_epoch_{}_checkpoint.ckpt'.format(epoch_num))
    print("==================Ending Training==================")


if __name__ == '__main__':
    from settings.options import parser
    import warnings
    warnings.filterwarnings("ignore")
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']= args.device_id
    context.set_context(device_target=args.device_target)
    # Train
    seed_mindspore()
    train(args)

