import json
import os
from collections import namedtuple

import torch
import torch.utils.data as data
from PIL import Image
import numpy as np


class Camvid(data.Dataset):
    """Cityscapes <http://www.cityscapes-dataset.com/> Dataset.

    **Parameters:**
        - **root** (string): Root directory of dataset where directory 'leftImg8bit' and 'gtFine' or 'gtCoarse' are located.
        - **split** (string, optional): The image split to use, 'train', 'test' or 'val' if mode="gtFine" otherwise 'train', 'train_extra' or 'val'
        - **mode** (string, optional): The quality mode to use, 'gtFine' or 'gtCoarse' or 'color'. Can also be a list to output a tuple with all specified target types.
        - **transform** (callable, optional): A function/transform that takes in a PIL image and returns a transformed version. E.g, ``transforms.RandomCrop``
        - **target_transform** (callable, optional): A function/transform that takes in the target and transforms it.
    """

    # Based on https://github.com/mcordts/cityscapesScripts
    CamvidClass = namedtuple('CamvidClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                     'has_instances', 'ignore_in_eval', 'color'])
    classes = [
        CamvidClass('sky', 0, 0, 'sky', 0, False, False, (128, 128, 128)),
        CamvidClass('building', 1, 1, 'building', 1, False, False, (128, 0, 0)),
        CamvidClass('pole', 2, 2, 'pole', 2, False, False, (192, 192, 128)),
        CamvidClass('road', 3, 3, 'road', 3, False, False, (128, 64, 128)),
        CamvidClass('sidewalk', 4, 4, 'sidewalk', 4, False, False, (0, 0, 192)),
        CamvidClass('tree', 5, 5, 'tree', 5, False, False, (128, 128, 0)),
        CamvidClass('signsymbol', 6, 6, 'signsymbol', 6, False, False, (192, 128, 128)),
        CamvidClass('fence', 7, 7, 'fence', 7, False, False, (64, 64, 128)),
        CamvidClass('car', 8, 8, 'car', 8, False, False, (64, 0, 128)),
        CamvidClass('pedestrian', 9, 9, 'pedestrian', 9, False, False, (64, 64, 0)),
        CamvidClass('bicyclist', 10, 10, 'bicyclist', 10, False, False, (0, 128, 192)),
        CamvidClass('unlabelled', -1, 255, 'void', 11, False, True, (0, 0, 0)),
    ]

    train_id_to_color = [c.color for c in classes if (c.train_id != -1 and c.train_id != 255)]
    train_id_to_color.append([0, 0, 0])
    train_id_to_color = np.array(train_id_to_color)
    id_to_train_id = np.array([c.train_id for c in classes])

    # train_id_to_color = [(0, 0, 0), (128, 64, 128), (70, 70, 70), (153, 153, 153), (107, 142, 35),
    #                      (70, 130, 180), (220, 20, 60), (0, 0, 142)]
    # train_id_to_color = np.array(train_id_to_color)
    # id_to_train_id = np.array([c.category_id for c in classes], dtype='uint8') - 1

    def __init__(self, root, root_blur, split='train', mode='fine', target_type='semantic', transform=None): ## 추가 1 root_blur
        self.root = os.path.expanduser(root)
        self.root_blur = os.path.expanduser(root_blur)  ## 추가 2
        self.mode = 'gtFine'
        self.target_type = target_type
        self.images_dir = os.path.join(self.root + '/', 'leftImg8bit/', split)
        self.images_dir_blur = os.path.join(self.root_blur + '/', 'leftImg8bit/', split)  ##추가 3

        self.targets_dir = os.path.join(self.root + '/', self.mode + '/', split)
        self.targets_dir_blur = os.path.join(self.root_blur + '/', self.mode + '/', split)
        self.transform = transform

        self.split = split
        self.images = []
        self.images_blur = []  ## 추가 4
        self.targets = []
        self.targets_blur = []

        if split not in ['train', 'test', 'val']:
            raise ValueError('Invalid split for mode! Please use split="train", split="test"'
                             ' or split="val"')

        if not os.path.isdir(self.images_dir) or not os.path.isdir(self.targets_dir):
            raise RuntimeError('Dataset not found or incomplete. Please make sure all required folders for the'
                               ' specified "split" and "mode" are inside the "root" directory')

        if not os.path.isdir(self.images_dir_blur) or not os.path.isdir(self.targets_dir):
            raise RuntimeError('Dataset not found or incomplete. Please make sure all required folders for the'
                               ' specified "split" and "mode" are inside the "root" directory')

        for city in os.listdir(self.images_dir):
            img_dir = os.path.join(self.images_dir + '/', city)
            target_dir = os.path.join(self.targets_dir + '/', city)

            for file_name in os.listdir(img_dir):
                self.images.append(os.path.join(img_dir + '/', file_name))
                # target_name = '{}_{}'.format(file_name.split('_leftImg8bit')[0],
                #                              self._get_target_suffix(self.mode, self.target_type))
                # self.targets.append(os.path.join(target_dir + '/', target_name))
                self.targets.append(os.path.join(target_dir + '/', file_name))

        for city in os.listdir(self.images_dir_blur):   ##추가 5
            img_dir_blur = os.path.join(self.images_dir_blur + '/', city)
            target_dir_blur = os.path.join(self.targets_dir_blur + '/', city)

            for file_name in os.listdir(img_dir_blur):
                self.images_blur.append(os.path.join(img_dir_blur + '/', file_name))
                # target_name = '{}_{}'.format(file_name.split('_leftImg8bit')[0],
                #                              self._get_target_suffix(self.mode, self.target_type))
                # self.targets.append(os.path.join(target_dir + '/', target_name))
                self.targets_blur.append(os.path.join(target_dir_blur + '/', file_name))

    @classmethod
    def encode_target(cls, target):
        return cls.id_to_train_id[np.array(target)]

    @classmethod
    def decode_target(cls, target):
        # target[target == 255] = 19
        target[target == 255] = 11
        # target = target.astype('uint8') + 1
        return cls.train_id_to_color[target]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item. Otherwise target is a json object if target_type="polygon", else the image segmentation.
        """
        image = Image.open(self.images[index]).convert('RGB')
        image_blur = Image.open(self.images_blur[index]).convert('RGB')  ## 추가 6
        target = Image.open(self.targets[index])
        target_blur = Image.open(self.targets_blur[index])

        if self.transform:
            image, target = self.transform(image, target) ## 추가 7
            image_blur, target_blur = self.transform(image_blur, target_blur)

        target = self.encode_target(target)
        target_blur = self.encode_target(target_blur)

        return image, target, image_blur, target_blur

    def __len__(self):
        return len(self.images)

    def _load_json(self, path):
        with open(path, 'r') as file:
            data = json.load(file)
        return data

    def _get_target_suffix(self, mode, target_type):
        if target_type == 'instance':
            return '{}_instanceIds.png'.format(mode)
        elif target_type == 'semantic':
            return '{}_labelIds.png'.format(mode)
        elif target_type == 'color':
            return '{}_color.png'.format(mode)
        elif target_type == 'polygon':
            return '{}_polygons.json'.format(mode)
        elif target_type == 'depth':
            return '{}_disparity.png'.format(mode)