import os
import numpy as np
from torch.utils.data import Dataset
import torch
from PIL import Image
import torchvision.transforms.functional as TF
from pdb import set_trace as stx
import random
from collections import namedtuple

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif'])

class DataLoaderTrain(Dataset):

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

    def __init__(self, rgb_dir, img_options=None, transform=None):
        super(DataLoaderTrain, self).__init__()

        inp_files = sorted(os.listdir(os.path.join(rgb_dir, 'input')))
        tar_files = sorted(os.listdir(os.path.join(rgb_dir, 'target')))

        self.inp_filenames = [os.path.join(rgb_dir, 'input', x)  for x in inp_files if is_image_file(x)]
        self.tar_filenames = [os.path.join(rgb_dir, 'target', x) for x in tar_files if is_image_file(x)]
        self.tar_filenames_for_segmentation = [os.path.join(rgb_dir, 'target_For_segmentation', x) for x in tar_files if is_image_file(x)]

        self.img_options = img_options
        self.sizex       = len(self.tar_filenames)  # get the size of target

        self.ps = self.img_options['patch_size']

        self.transform = transform
        self.images = []
        self.targets = []

    @classmethod
    def encode_target(cls, target):
        return cls.id_to_train_id[np.array(target)]

    @classmethod
    def decode_target(cls, target):
        # target[target == 255] = 19
        target[target == 255] = 11
        # target = target.astype('uint8') + 1
        return cls.train_id_to_color[target]

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex
        ps = self.ps

        inp_path = self.inp_filenames[index_]
        tar_path = self.tar_filenames[index_]

        inp_img = Image.open(inp_path)
        tar_img = Image.open(tar_path)
        target = Image.open(self.tar_filenames_for_segmentation[index_])


        w,h = tar_img.size
        padw = ps-w if w<ps else 0
        padh = ps-h if h<ps else 0

        # Reflect Pad in case image is smaller than patch_size
        if padw!=0 or padh!=0:
            inp_img = TF.pad(inp_img, (0,0,padw,padh), padding_mode='reflect')
            tar_img = TF.pad(tar_img, (0,0,padw,padh), padding_mode='reflect')
            target = TF.pad(target, (0, 0, padw, padh), padding_mode='reflect')


        inp_img = TF.to_tensor(inp_img)
        tar_img = TF.to_tensor(tar_img)
        target = torch.from_numpy(np.array( target, dtype=np.uint8))

        hh, ww = tar_img.shape[1], tar_img.shape[2]

        rr     = random.randint(0, hh-ps)
        cc     = random.randint(0, ww-ps)
        aug    = random.randint(0, 8)

        # Crop patch
        inp_img = inp_img[:, rr:rr+ps, cc:cc+ps]
        tar_img = tar_img[:, rr:rr+ps, cc:cc+ps]
        target = target[rr:rr + ps, cc:cc + ps]


        # Data Augmentations
        if aug==1:
            inp_img = inp_img.flip(1)
            tar_img = tar_img.flip(1)
            target = target.flip(0)
        elif aug==2:
            inp_img = inp_img.flip(2)
            tar_img = tar_img.flip(2)
            target = target.flip(1)
        elif aug==3:
            inp_img = torch.rot90(inp_img,dims=(1,2))
            tar_img = torch.rot90(tar_img,dims=(1,2))
            target = torch.rot90(target, dims=(0, 1))
        elif aug==4:
            inp_img = torch.rot90(inp_img,dims=(1,2), k=2)
            tar_img = torch.rot90(tar_img,dims=(1,2), k=2)
            target = torch.rot90(target, dims=(0, 1), k=2)
        elif aug==5:
            inp_img = torch.rot90(inp_img,dims=(1,2), k=3)
            tar_img = torch.rot90(tar_img,dims=(1,2), k=3)
            target = torch.rot90(target, dims=(0, 1), k=3)
        elif aug==6:
            inp_img = torch.rot90(inp_img.flip(1),dims=(1,2))
            tar_img = torch.rot90(tar_img.flip(1),dims=(1,2))
            target = torch.rot90(target.flip(0), dims=(0, 1))
        elif aug==7:
            inp_img = torch.rot90(inp_img.flip(2),dims=(1,2))
            tar_img = torch.rot90(tar_img.flip(2),dims=(1,2))
            target = torch.rot90(target.flip(1), dims=(0, 1))
        
        filename = os.path.splitext(os.path.split(tar_path)[-1])[0]

        # target = self.encode_target(target)

        return tar_img, inp_img, target

class DataLoaderVal(Dataset):

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

    def __init__(self, rgb_dir, img_options=None, transform=None):
        super(DataLoaderVal, self).__init__()

        inp_files = sorted(os.listdir(os.path.join(rgb_dir, 'input')))
        tar_files = sorted(os.listdir(os.path.join(rgb_dir, 'target')))

        self.inp_filenames = [os.path.join(rgb_dir, 'input', x)  for x in inp_files if is_image_file(x)]
        self.tar_filenames = [os.path.join(rgb_dir, 'target', x) for x in tar_files if is_image_file(x)]
        self.tar_filenames_for_segmentation = [os.path.join(rgb_dir, 'target_For_segmentation', x) for x in tar_files if is_image_file(x)]

        self.img_options = img_options
        self.sizex       = len(self.tar_filenames)  # get the size of target

        self.ps = self.img_options['patch_size']

        self.transform = transform
        self.images = []
        self.targets = []

    @classmethod
    def encode_target(cls, target):
        return cls.id_to_train_id[np.array(target)]

    @classmethod
    def decode_target(cls, target):
        # target[target == 255] = 19
        target[target == 255] = 11
        # target = target.astype('uint8') + 1
        return cls.train_id_to_color[target]


    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex
        ps = self.ps

        inp_path = self.inp_filenames[index_]
        tar_path = self.tar_filenames[index_]

        inp_img = Image.open(inp_path)
        tar_img = Image.open(tar_path)
        target = Image.open(self.tar_filenames_for_segmentation[index_])

        # Validate on center crop
        if self.ps is not None:
            inp_img = TF.center_crop(inp_img, (ps,ps))
            tar_img = TF.center_crop(tar_img, (ps,ps))
            target = TF.center_crop(target, (ps, ps))

        inp_img = TF.to_tensor(inp_img)
        tar_img = TF.to_tensor(tar_img)
        target = torch.from_numpy(np.array( target, dtype=np.uint8))

        filename = os.path.splitext(os.path.split(tar_path)[-1])[0]

        return tar_img, inp_img, target

class DataLoaderTest(Dataset):
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

    def __init__(self, rgb_dir, img_options=None, transform=None):
        super(DataLoaderTest, self).__init__()

        inp_files = sorted(os.listdir(os.path.join(rgb_dir, 'input')))
        tar_files = sorted(os.listdir(os.path.join(rgb_dir, 'target')))

        self.inp_filenames = [os.path.join(rgb_dir, 'input', x) for x in inp_files if is_image_file(x)]
        self.tar_filenames = [os.path.join(rgb_dir, 'target', x) for x in tar_files if is_image_file(x)]
        self.tar_filenames_for_segmentation = [os.path.join(rgb_dir, 'target_For_segmentation', x) for x in tar_files if
                                               is_image_file(x)]

        self.img_options = img_options
        self.sizex = len(self.tar_filenames)  # get the size of target

        self.ps = self.img_options['patch_size']

        self.transform = transform
        self.images = []
        self.targets = []

    @classmethod
    def encode_target(cls, target):
        return cls.id_to_train_id[np.array(target)]

    @classmethod
    def decode_target(cls, target):
        # target[target == 255] = 19
        target[target == 255] = 11
        # target = target.astype('uint8') + 1
        return cls.train_id_to_color[target]

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex
        ps = self.ps

        inp_path = self.inp_filenames[index_]
        tar_path = self.tar_filenames[index_]

        inp_img = Image.open(inp_path)
        tar_img = Image.open(tar_path)
        target = Image.open(self.tar_filenames_for_segmentation[index_])

        # Validate on center crop
        if self.ps is not None:
            inp_img = TF.center_crop(inp_img, (ps,ps))
            tar_img = TF.center_crop(tar_img, (ps,ps))
            target = TF.center_crop(target, (ps, ps))

        inp_img = TF.to_tensor(inp_img)
        tar_img = TF.to_tensor(tar_img)
        target = torch.from_numpy(np.array( target, dtype=np.uint8))

        filename = os.path.splitext(os.path.split(tar_path)[-1])[0]

        return tar_img, inp_img, target
