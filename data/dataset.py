# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
from PIL import Image
import json
import numpy as np
import torchvision.transforms as transforms
import os
from torch.utils.data import Dataset
import random
import copy
import cv2
from .transforms import *
import PIL

identity = lambda x: x

class SimpleDataset:
    def __init__(self, data_file, image_size, transform, target_transform=identity, is_train=True):
        with open(data_file, 'r') as f:
            self.meta = json.load(f)
        self.transform = transform
        self.target_transform = target_transform
        self.flip = is_train
        self.image_size = image_size
        self.is_train = is_train

    def __getitem__(self, i):
        image_path = os.path.join(self.meta['image_names'][i])
        # data_numpy = cv2.imread(image_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

        # used for SUN, cv2.imread returns None type
        data_numpy = np.array(PIL.Image.open(image_path).convert('RGB'))[:, :, ::-1]  # to BGR

        if data_numpy is None:
            raise ValueError('Fail to read {}'.format(image_path))

        r = 0
        c = np.array([data_numpy.shape[1], data_numpy.shape[0]]) // 2
        s = np.array([data_numpy.shape[1], data_numpy.shape[0]]) // 160

        if self.is_train:
            sf = 0.25
            rf = 30
            s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
            r = np.clip(np.random.randn() * rf, -rf * 2, rf * 2) \
                if random.random() <= 0.6 else 0

            if self.flip and random.random() <= 0.5:
                data_numpy = data_numpy[:, ::-1, :]
                c[0] = data_numpy.shape[1] - c[0] - 1

        trans = get_affine_transform(c, s, r, [self.image_size, self.image_size])
        input = cv2.warpAffine(
            data_numpy,
            trans,
            (int(self.image_size), int(self.image_size)),
            flags=cv2.INTER_LINEAR)
        input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
        input = Image.fromarray(input.transpose((1, 0, 2)))

        if self.transform:
            input = self.transform(input)
        target = self.target_transform(self.meta['image_labels'][i])
        return input, target

    def __len__(self):
        return len(self.meta['image_names'])


class SetDataset:
    def __init__(self, data_file, batch_size, image_size, transform, is_train=True, attr_loc=False):
        with open(data_file, 'r') as f:
            self.meta = json.load(f)

        self.cl_list = np.unique(self.meta['image_labels']).tolist()

        self.sub_meta = {}
        for cl in self.cl_list:
            self.sub_meta[cl] = []

        if 'part' in self.meta:
            for x, y, z in zip(self.meta['image_names'], self.meta['image_labels'], self.meta['part']):
                self.sub_meta[y].append({'path': x, 'part': z})
        else:
            print("not use attribute location or attribute location is unavailable!")
            for x, y in zip(self.meta['image_names'], self.meta['image_labels']):
                self.sub_meta[y].append({'path': x})

        self.sub_dataloader = []
        sub_data_loader_params = dict(batch_size=batch_size,
                                      shuffle=True,
                                      num_workers=0,  # use main thread only or may receive multiple batches
                                      pin_memory=False)

        for cl in self.cl_list:
            sub_dataset = SubDataset(self.sub_meta[cl], cl, image_size, attr_loc, transform=transform, is_train=is_train)
            self.sub_dataloader.append(torch.utils.data.DataLoader(sub_dataset, **sub_data_loader_params))

    def __getitem__(self, i):
        return next(iter(self.sub_dataloader[i]))

    def __len__(self):
        return len(self.cl_list)


class EpisodicBatchSampler(object):
    def __init__(self, n_classes, n_way, n_episodes):
        self.n_classes = n_classes
        self.n_way = n_way
        self.n_episodes = n_episodes

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for i in range(self.n_episodes):
            yield torch.randperm(self.n_classes)[:self.n_way]


class SubDataset(Dataset):
    def __init__(self, sub_meta, cl, image_size, attr_loc=False, transform=transforms.ToTensor(), target_transform=identity,
                 is_train=True):
        self.num_joints = 15

        self.is_train = is_train
        self.sub_meta = sub_meta
        self.cl = cl
        self.transform = transform
        self.target_transform = target_transform

        self.flip = is_train
        self.attr_loc = attr_loc

        self.image_size = image_size

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self, ):
        return len(self.sub_meta)

    def __getitem__(self, idx):
        image_file = os.path.join(self.sub_meta[idx]['path'])

        #data_numpy = cv2.imread(image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

        # used for SUN, cv2.imread returns None type
        data_numpy = np.array(PIL.Image.open(image_file).convert('RGB'))[:, :, ::-1]  # to BGR

        if data_numpy is None:
            raise ValueError('Fail to read {}'.format(image_file))

        if self.attr_loc:
            joints_vis = self.sub_meta[idx]['part']
            joints_vis = np.array(joints_vis)

        r = 0
        c = np.array([data_numpy.shape[1], data_numpy.shape[0]]) // 2
        s = np.array([data_numpy.shape[1], data_numpy.shape[0]]) // 160

        if self.is_train:
            sf = 0.25
            rf = 30
            s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
            r = np.clip(np.random.randn() * rf, -rf * 2, rf * 2) \
                if random.random() <= 0.6 else 0

            if self.flip and random.random() <= 0.5:
                data_numpy = data_numpy[:, ::-1, :]
                if self.attr_loc:
                    for i in range(self.num_joints):
                        if joints_vis[i, 2] > 0.0:
                            joints_vis[i, 0] = data_numpy.shape[1] - joints_vis[i, 0]
                c[0] = data_numpy.shape[1] - c[0] - 1

        trans = get_affine_transform(c, s, r, [self.image_size, self.image_size])
        input = cv2.warpAffine(
            data_numpy,
            trans,
            (int(self.image_size), int(self.image_size)),
            flags=cv2.INTER_LINEAR)
        input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
        input = Image.fromarray(input.transpose((1, 0, 2)))

        if self.transform:
            input = self.transform(input)

        target = self.target_transform(self.cl)

        if self.attr_loc is False:
            return input, target
        else:
            for i in range(self.num_joints):
                if joints_vis[i, 2] > 0.0:
                    joints_vis[i, 0:2] = affine_transform(joints_vis[i, 0:2], trans)

            joints_vis = self.target_transform(joints_vis)
            return input, target, joints_vis