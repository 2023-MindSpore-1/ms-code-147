#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os
import random
import numpy as np
from PIL import Image

import mindspore
from mindspore import dataset
from mindspore.dataset.vision import Inter
import mindspore.dataset.vision as vision
from mindspore.dataset.transforms.py_transforms import Compose


class GetDatasetGenerator:
    def __init__(self, image_root, depth_root, gt_root, trainsize):
        """
        :param image_root: The path of RGB training images.
        :param depth_root: The path of depth training images.
        :param gt_root: The path of training ground truth.
        :param trainsize: The size of training images.
        """
        # load path
        self.trainsize = trainsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.depths = [depth_root + f for f in os.listdir(depth_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')]
        # sort
        self.images = sorted(self.images)
        self.depths = sorted(self.depths)
        self.gts = sorted(self.gts)
        # read Image
        self._filter_files()
        self.size = len(self.images)

        self.img_transform = Compose([
            vision.Resize((self.trainsize, self.trainsize)),
            vision.ToTensor(),
            vision.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], is_hwc=False)
        ])
        self.depth_transform = Compose([
            vision.Resize((self.trainsize, self.trainsize)),
            vision.ToTensor(),
            vision.Normalize([0.485, ], [0.229, ], is_hwc=False)
        ])
        self.gt_transform = Compose([
            vision.Resize((self.trainsize, self.trainsize)),
            vision.ToTensor()
        ])

    def __getitem__(self, index):
        image = self._rgb_loader(self.images[index])
        depth = self._binary_loader(self.depths[index])
        gt = self._binary_loader(self.gts[index])

        image, depth, gt = self._randomFlip(image, depth, gt)
        image, depth, gt = self._randomRotation(image, depth, gt)
        # multiScale
        scale_flag = random.randint(0,2)
        if scale_flag == 1:
            self.trainsize = 128
        elif scale_flag == 2:
            self.trainsize = 256
        else:
            self.trainsize = 352

        image = self.img_transform(image)
        depth = self.depth_transform(depth)
        gt = self.gt_transform(gt)

        return image, depth, gt

    def __len__(self):
        return self.size

    def _filter_files(self):
        """ Check whether a set of images match in size. """
        assert len(self.images) == len(self.depths) == len(self.gts)
        images = []
        depths = []
        gts = []
        for img_path, depth_path, gt_path in zip(self.images, self.depths, self.gts):
            # Notes: On DUT dataset, the size of training depth images are [256, 256],
            # it is not matched with RGB images and GT [600, 400].
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                depths.append(depth_path)
                gts.append(gt_path)
            else:
                raise Exception("Image sizes do not match, please check.")
        self.images = images
        self.depths = depths
        self.gts = gts

    def _rgb_loader(self, path):
        with open(path, 'rb') as f:
            # Removing alpha channel.
            return Image.open(f).convert('RGB')

    def _binary_loader(self, path):
        with open(path, 'rb') as f:
            return Image.open(f).convert('L')

    def _randomFlip(self, img, depth, gt):
        flip_flag = random.randint(0, 2)
        if flip_flag == 1:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            depth = depth.transpose(Image.FLIP_LEFT_RIGHT)
            gt = gt.transpose(Image.FLIP_LEFT_RIGHT)
        elif flip_flag == 2:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            depth = depth.transpose(Image.FLIP_TOP_BOTTOM)
            gt = gt.transpose(Image.FLIP_TOP_BOTTOM)
        return img, depth, gt

    def _randomRotation(self, image, depth, gt):
        mode = Image.BICUBIC
        if random.random() > 0.8:
            random_angle = np.random.randint(-15, 15)
            image = image.rotate(random_angle, mode)
            depth = depth.rotate(random_angle, mode)
            gt = gt.rotate(random_angle, mode)
        return image, depth, gt


def get_iterator(image_root, depth_root, gt_root, batchsize, trainsize, shuffle=True, num_parallel_workers=4):
    dataset_generator = GetDatasetGenerator(image_root, depth_root, gt_root, trainsize)
    dataset_train = dataset.GeneratorDataset(
                    dataset_generator, ["rgb", "depth", "label"], shuffle=shuffle, num_parallel_workers=num_parallel_workers)
    dataset_train = dataset_train.batch(batchsize)
    iterations_epoch = dataset_train.get_dataset_size()
    train_iterator = dataset_train.create_dict_iterator()
    return train_iterator, iterations_epoch

class TestDataset:
    def __init__(self, image_root, depth_root, gt_root, testsize):
        self.testsize = testsize
        # load root
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.depths = [depth_root + f for f in os.listdir(depth_root) if f.endswith('.jpg') or f.endswith('.png')
                      or f.endswith('.bmp')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')]
        # sort
        self.images = sorted(self.images)
        self.depths = sorted(self.depths)
        self.gts = sorted(self.gts)
        # 
        self._filter_files()
        self.size = len(self.images)
        self.img_transform = Compose([
            vision.Resize((self.testsize, self.testsize)),
            vision.ToTensor(),
            vision.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], is_hwc=False)
        ])
        self.depth_transform = Compose([
            vision.Resize((self.testsize, self.testsize)),
            vision.ToTensor(),
            vision.Normalize([0.485, ], [0.229, ], is_hwc=False)
        ])
        self.gt_transform = Compose([
            vision.ToTensor()
        ])
        # 
        self.index = 0

    
    def load_data(self):
        image = self._rgb_loader(self.images[self.index])
        depth = self._binary_loader(self.depths[self.index])
        gt = self._binary_loader(self.gts[self.index])
        image = self.img_transform(image)
        depth = self.depth_transform(depth)
        gt = self.gt_transform(gt)
        # get name
        name_rgb = self.images[self.index].split('/')[-1]
        if name_rgb.endswith('.jpg'):
            name_rgb = name_rgb.split('.jpg')[0] + '.png'
        self.index += 1
        self.index = self.index % self.size
        return image, depth, gt, name_rgb

    def _filter_files(self):
        """ Check whether a set of images match in size. """
        assert len(self.images) == len(self.depths) == len(self.gts)
        images = []
        depths = []
        gts = []
        for img_path, depth_path, gt_path in zip(self.images, self.depths, self.gts):
            # Notes: On DUT dataset, the size of training depth images are [256, 256],
            # it is not matched with RGB images and GT [600, 400].
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                depths.append(depth_path)
                gts.append(gt_path)
            else:
                raise Exception("Image sizes do not match, please check.")
        self.images = images
        self.depths = depths
        self.gts = gts
    
    def _rgb_loader(self, path):
        with open(path, 'rb') as f:
            # Removing alpha channel.
            return Image.open(f).convert('RGB')

    def _binary_loader(self, path):
        with open(path, 'rb') as f:
            return Image.open(f).convert('L')


