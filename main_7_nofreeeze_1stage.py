from tqdm import tqdm
import network
import utils
import os
import random
import argparse
import numpy as np
import pandas as pd

import torchvision.transforms as transforms
from torch.utils import data
from datasets import VOCSegmentation, Cityscapes, Camvid, Camvid_60,  Camvid_sample, Camvid_sample_3input, Camvid_focus, Kitti_sample, Camvid_deblur85_v2_blending, Camvid_proposed
from utils import ext_transforms as et
from metrics import StreamSegMetrics
from torchsummary import summary
import torch
import torch.nn as nn
from utils.visualizer import Visualizer

from torchvision.transforms.functional import to_pil_image
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from network.unet import UNet
from network.unet import UNet_attnetion_decoder_concat
from network.unet import UNet_attnetion_encoder_concat
from network.unet import UNet_attnetion_encoder_each_sum
from network.unet import UNet_attnetion_encoder_decoder_concat
from utils.loss import DiceBCELoss
from torch import optim
from typing import Optional
from glob import glob

# print(torch.cuda.is_available())

#python -m visdom.server

#
class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    # parser.add_argument("--data_root", type=str, default='./datasets/data',
    #                     help="path to Dataset")
    parser.add_argument("--data_root", type=str, default='D:/Dataset/Camvid/camvid_sample_2/blur/psf_15_3',   ##Deeplab으로 바꿀때 model backbone, optimizer 설정
                        help="path to Dataset")  ##crop size 바꿔주기
    # parser.add_argument("--data_root", type=str,
    #                     default='D:/Dataset/Camvid/camvid_original_240',
    #                      help="path to Dataset")   ##crop size 바꿔주기

    parser.add_argument("--dataset", type=str, default='camvid_sample_3input',
                        choices=['voc', 'cityscapes, camvid', 'camvid_sample', 'camvid_focus', 'kitti_sample', 'camvid_proposed', 'camvid_sample_3input'], help='Name of dataset')

    parser.add_argument("--num_classes", type=int, default=None,
                        help="num classes (default: None)")

    # Deeplab Options
    parser.add_argument("--model", type=str, default='deeplabv3plus_resnet50',
                        choices=['deeplabv3_resnet50',  'deeplabv3plus_resnet50',
                                 'deeplabv3_resnet101', 'deeplabv3plus_resnet101',
                                 'deeplabv3_mobilenet', 'deeplabv3plus_mobilenet'], help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    # Train Options
    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--save_val_results", action='store_true', default=False,
                        help="save segmentation results to \"./results\"")
    parser.add_argument("--total_itrs", type=int, default=60e3,
                        help="epoch number (default: 30k)")
    # parser.add_argument("--total_itrs", type=int, default=200,
    #                     help="epoch number (default: 30k)")
    parser.add_argument("--lr", type=float, default=0.0001,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--lr_policy", type=str, default='poly', choices=['poly', 'step'],
                        help="learning rate scheduler policy")
    parser.add_argument("--step_size", type=int, default=10000)
    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: False)')
    parser.add_argument("--batch_size", type=int, default=10,
                        help='batch size (default: 16)')
    parser.add_argument("--val_batch_size", type=int, default=1,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=240) ##513
    # parser.add_argument("--crop_size", type=int, default=176)  ##513
    # parser.add_argument("--ckpt", default='D:/Second_Paper/Checkpoint/Camvid/Segmentation/Original_using_binary/Subcom_Firstfold_UNet_BCE_Focal_alpha0.25_r2_lr0.001_500epoch_BlackWhite_/best_deeplabv3plus_resnet50_camvid_proposed_os16.pth', type=str,
    #                     help="restore from checkpoint")
    parser.add_argument("--ckpt",default=None, type=str,
                        help="restore from checkpoint")
    # parser.add_argument("--ckpt_pretrained",default='E:/pretrained/unet_carvana_scale1.0_epoch2.pth', type=str,
    #                     help="restore from checkpoint")

    parser.add_argument("--continue_training", action='store_true', default=False)

    parser.add_argument("--overlay",  default=True)

    parser.add_argument("--loss_type_pre", type=str, default='focal_loss',
                        choices=['cross_entropy', 'focal_loss'], help="loss type (default: False)")

    parser.add_argument("--loss_type", type=str, default='cross_entropy',
                        choices=['cross_entropy', 'focal_loss'], help="loss type (default: False)")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--random_seed", type=int, default=15,
                        help="random seed (default: 1)")
    parser.add_argument("--print_interval", type=int, default=10,
                        help="print interval of loss (default: 10)")
    parser.add_argument("--val_interval", type=int, default=87,
                        help="epoch interval for eval (default: 100)")
    parser.add_argument("--download", action='store_true', default=False,
                        help="download datasets")

    # PASCAL VOC Options
    parser.add_argument("--year", type=str, default='2012',
                        choices=['2012_aug', '2012', '2011', '2009', '2008', '2007'], help='year of VOC')

    # Visdom options
    parser.add_argument("--enable_vis", action='store_true', default=False,
                        help="use visdom for visualization")
    parser.add_argument("--vis_port", type=str, default='8097', #13570
                        help='port for visdom')
    parser.add_argument("--vis_env", type=str, default='main',
                        help='env for visdom')
    parser.add_argument("--vis_num_samples", type=int, default=8,
                        help='number of samples for visualization (default: 8)')
    return parser


def label_to_one_hot_label(
        labels: torch.Tensor,
        num_classes: int,
        device: Optional[torch.device] = 'cuda',
        dtype: Optional[torch.dtype] = None,
        eps: float = 1e-6,
        ignore_index=255,
) -> torch.Tensor:
    r"""Convert an integer label x-D tensor to a one-hot (x+1)-D tensor.

    Args:
        labels: tensor with labels of shape :math:`(N, *)`, where N is batch size.
          Each value is an integer representing correct classification.
        num_classes: number of classes in labels.
        device: the desired device of returned tensor.
        dtype: the desired data type of returned tensor.

    Returns:
        the labels in one hot tensor of shape :math:`(N, C, *)`,

    Examples:
        >>> labels = torch.LongTensor([
                [[0, 1],
                [2, 0]]
            ])
        >>> one_hot(labels, num_classes=3)
        tensor([[[[1.0000e+00, 1.0000e-06],
                  [1.0000e-06, 1.0000e+00]],

                 [[1.0000e-06, 1.0000e+00],
                  [1.0000e-06, 1.0000e-06]],

                 [[1.0000e-06, 1.0000e-06],
                  [1.0000e+00, 1.0000e-06]]]])

    """
    shape = labels.shape
    # one hot : (B, C=ignore_index+1, H, W)
    one_hot = torch.zeros((shape[0], ignore_index + 1) + shape[1:], device=device, dtype=dtype)

    # labels : (B, H, W)
    # labels.unsqueeze(1) : (B, C=1, H, W)
    # one_hot : (B, C=ignore_index+1, H, W)
    one_hot = one_hot.scatter_(1, labels.unsqueeze(1), 1.0) + eps

    # ret : (B, C=num_classes, H, W)
    ret = torch.split(one_hot, [num_classes, ignore_index + 1 - num_classes], dim=1)[0]

    return ret

def get_dataset(opts):
    """ Dataset And Augmentation
    """
    if opts.dataset == 'voc':
        train_transform = et.ExtCompose([
            #et.ExtResize(size=opts.crop_size),
            et.ExtRandomScale((0.5, 2.0)),
            et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size), pad_if_needed=True),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
        if opts.crop_val:
            val_transform = et.ExtCompose([
                et.ExtResize(opts.crop_size),
                et.ExtCenterCrop(opts.crop_size),
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
        else:
            val_transform = et.ExtCompose([
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
        train_dst = VOCSegmentation(root=opts.data_root, year=opts.year,
                                    image_set='train', download=opts.download, transform=train_transform)
        val_dst = VOCSegmentation(root=opts.data_root, year=opts.year,
                                  image_set='val', download=False, transform=val_transform)

    if opts.dataset == 'cityscapes':
        train_transform = et.ExtCompose([
            #et.ExtResize( 512 ),
            et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size)),
            et.ExtColorJitter( brightness=0.5, contrast=0.5, saturation=0.5 ),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        val_transform = et.ExtCompose([
            #et.ExtResize( 512 ),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
        train_dst = Cityscapes(root=opts.data_root,
                               split='train', transform=train_transform)
        val_dst = Cityscapes(root=opts.data_root,
                             split='val', transform=val_transform)

    if opts.dataset == 'camvid':
        train_transform = et.ExtCompose([
            # et.ExtResize( 512 ),
            et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size)),
            et.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        val_transform = et.ExtCompose([
            # et.ExtResize( 512 ),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        train_dst = Camvid(root=opts.data_root, root_blur=opts.data_root_blur, split='train', transform=train_transform)

        val_dst = Camvid_60(root=opts.data_root_blur, split='val', transform=val_transform)
        # val_dst = Camvid(root=opts.data_root, root_blur=opts.data_root_blur, split='val', transform=val_transform)

    if opts.dataset == 'camvid_sample':
        train_transform = et.ExtCompose([
            # et.ExtResize( 512 ),
            et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size)),
            et.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        val_transform = et.ExtCompose([
            # et.ExtResize( 512 ),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        test_transform = et.ExtCompose([
            # et.ExtResize( 512 ),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        train_dst = Camvid_sample(root=opts.data_root, split='train', transform=train_transform)

        val_dst = Camvid_sample(root=opts.data_root, split='val', transform=val_transform)

        test_dst = Camvid_sample(root=opts.data_root, split='test', transform=test_transform)


    if opts.dataset == 'kitti_sample':
        train_transform = et.ExtCompose([
            # et.ExtResize( 512 ),
            et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size)),
            et.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        val_transform = et.ExtCompose([
            # et.ExtResize( 512 ),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        train_dst = Kitti_sample(root=opts.data_root, split='train', transform=train_transform)

        val_dst = Kitti_sample(root=opts.data_root, split='val', transform=val_transform)


    if opts.dataset == 'camvid_focus':
        train_transform = et.ExtCompose([
            # et.ExtResize( 512 ),
            et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size)),
            et.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        val_transform = et.ExtCompose([
            # et.ExtResize( 512 ),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        train_dst = Camvid_sample(root=opts.data_root, split='train', transform=train_transform)

        val_dst = Camvid_sample(root=opts.data_root, split='val', transform=val_transform)

    if opts.dataset == 'camvid_sample_3input':
        train_transform = et.ExtCompose([
            # et.ExtResize( 512 ),
            et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size)),
            et.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        val_transform = et.ExtCompose([
            # et.ExtResize( 512 ),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        test_transform = et.ExtCompose([
            # et.ExtResize( 512 ),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        train_dst = Camvid_sample_3input(root=opts.data_root, split='train', transform=train_transform)

        val_dst = Camvid_sample_3input(root=opts.data_root, split='val', transform=val_transform)

        test_dst = Camvid_sample_3input(root=opts.data_root, split='test', transform=test_transform)

    return train_dst, val_dst, test_dst

def validate_pre(opts, model, loader, device, metrics, ret_samples_ids=None):
    """Do validation and return specified samples"""
    metrics.reset()
    ret_samples = []
    if opts.overlay:
        if not os.path.exists('results'):
            os.mkdir('results')
        denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
        img_id = 0

    with torch.no_grad():
        for images, labels, labels_pre in loader:

            images = images.to(device, dtype=torch.float32)
            labels_pre = labels_pre.to(device, dtype=torch.long)

            outputs = model(images)

            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels_pre.cpu().numpy()

            metrics.update(targets, preds)
            if ret_samples_ids is not None and i in ret_samples_ids:  # get vis samples
                ret_samples.append(
                    (images[0].detach().cpu().numpy(), targets[0], preds[0]))

            if opts.save_val_results:
                for i in range(len(images)):
                    image = images[i].detach().cpu().numpy()
                    target = targets[i]
                    pred = preds[i]

                    image = (denorm(image) * 255).transpose(1, 2, 0).astype(np.uint8)
                    target = loader.dataset.decode_target(target).astype(np.uint8)
                    pred = loader.dataset.decode_target(pred).astype(np.uint8)

                    Image.fromarray(image).save('results/%d_image.png' % img_id)
                    Image.fromarray(target).save('results/%d_target.png' % img_id)
                    Image.fromarray(pred).save('results/%d_pred.png' % img_id)

                    fig = plt.figure()
                    plt.imshow(image)
                    plt.axis('off')
                    plt.imshow(pred, alpha=0.7)
                    ax = plt.gca()
                    ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    plt.savefig('results/%d_overlay.png' % img_id, bbox_inches='tight', pad_inches=0)
                    plt.close()
                    img_id += 1

        score = metrics.get_results()
    return score, ret_samples

def validate(opts, model1, model2, loader, device, metrics, ret_samples_ids=None):
    """Do validation and return specified samples"""
    metrics.reset()
    ret_samples = []
    if opts.overlay:
        if not os.path.exists('results'):
            os.mkdir('results')
        denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
        img_id = 0

    with torch.no_grad():
        for images, labels, labels_pre in loader:

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)
            labels_pre = labels_pre.to(device, dtype=torch.long)

            outputs_pre = model1(images)
            outputs_pre = nn.Sigmoid()(outputs_pre)

            outputs = model2(images, outputs_pre)

            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()

            metrics.update(targets, preds)
            if ret_samples_ids is not None and i in ret_samples_ids:  # get vis samples
                ret_samples.append(
                    (images[0].detach().cpu().numpy(), targets[0], preds[0]))

            if opts.save_val_results:
                for i in range(len(images)):
                    image = images[i].detach().cpu().numpy()
                    target = targets[i]
                    pred = preds[i]

                    image = (denorm(image) * 255).transpose(1, 2, 0).astype(np.uint8)
                    target = loader.dataset.decode_target(target).astype(np.uint8)
                    pred = loader.dataset.decode_target(pred).astype(np.uint8)

                    Image.fromarray(image).save('results/%d_image.png' % img_id)
                    Image.fromarray(target).save('results/%d_target.png' % img_id)
                    Image.fromarray(pred).save('results/%d_pred.png' % img_id)

                    fig = plt.figure()
                    plt.imshow(image)
                    plt.axis('off')
                    plt.imshow(pred, alpha=0.7)
                    ax = plt.gca()
                    ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    plt.savefig('results/%d_overlay.png' % img_id, bbox_inches='tight', pad_inches=0)
                    plt.close()
                    img_id += 1

        score = metrics.get_results()
    return score, ret_samples


def main():
    opts = get_argparser().parse_args()
    if opts.dataset.lower() == 'voc':
        opts.num_classes = 21
    elif opts.dataset.lower() == 'cityscapes':
        opts.num_classes = 19
    elif opts.dataset.lower() == 'camvid':
        opts.num_classes = 12
    elif opts.dataset.lower() == 'camvid_sample':
        opts.num_classes = 12
    elif opts.dataset.lower() == 'kitti_sample':
        opts.num_classes = 12
    elif opts.dataset.lower() == 'camvid_proposed':
        opts.num_classes = 2
    elif opts.dataset.lower() == 'camvid_sample_3input':
        opts.num_classes_pre = 2
        # opts.num_classes = 12

    # Setup visualization
    vis = Visualizer(port=opts.vis_port,
                     env=opts.vis_env) if opts.enable_vis else None
    if vis is not None:  # display options
        vis.vis_table("Options", vars(opts))

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Setup random seed
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Setup dataloader
    if opts.dataset=='voc' and not opts.crop_val:
        opts.val_batch_size = 1

    train_dst, val_dst, test_dst = get_dataset(opts)

    train_loader = data.DataLoader(
        train_dst, batch_size=opts.batch_size, shuffle=True, num_workers=2)

    val_loader = data.DataLoader(
        val_dst, batch_size=opts.val_batch_size, shuffle=True, num_workers=2)

    test_loader = data.DataLoader(
        test_dst, batch_size=opts.val_batch_size, shuffle=False, num_workers=2)

    print("Dataset: %s, Train set: %d, Val set: %d, Test set: %d" %
          (opts.dataset, len(train_dst), len(val_dst), len(test_dst)))

    ## https://github.com/jfzhang95/pytorch-deeplab-xception/blob/master/utils/calculate_weights.py
    def calculate_weigths_labels(dataloader, num_classes):
        # Create an instance from the data loader
        z = np.zeros((num_classes,))
        # Initialize tqdm
        # tqdm_batch = tqdm(dataloader)
        print('Calculating classes weights')
        # for sample in tqdm_batch:
        for (_, label, _) in train_loader:
            # y = sample['label']
            y = label
            y = y.detach().cpu().numpy()
            mask = (y >= 0) & (y < num_classes)
            labels = y[mask].astype(np.uint8)
            count_l = np.bincount(labels, minlength=num_classes)
            z += count_l
        # tqdm_batch.close()
        total_frequency = np.sum(z)
        class_weights = []
        for frequency in z:
            class_weight = 1 / (np.log(1.02 + (frequency / total_frequency)))
            class_weights.append(class_weight)
        ret = np.array(class_weights)
        # classes_weights_path = os.path.join(opts.data_root, '_classes_weights.npy')
        # np.save(classes_weights_path, ret)

        return ret

    def calc_weights(class_num):
        z = np.zeros((class_num,))
        # (_, label) = enumerate(train_loader)
        # for sample in label:
        for (_, label, _) in train_loader:
            sample = label
            sample = sample.to('cpu').detach().numpy()
            # sample = sample.argmax(0)
            labels = sample.astype(np.uint8)
            # count = np.bincount(labels.reshape(-1), minlength=class_num)
            count = np.bincount(labels, minlength=class_num)
            z += count
        total_freq = np.sum(z)
        class_weights = []
        for freq in z:
            class_weight = 1 / (np.log(1.02 + (freq / total_freq)))
            class_weights.append(class_weight)

        ret = np.array(class_weights)

        return ret

    # def calc_class_weights(data_loader, class_nums):
    #     z = np.zeros((class_nums,))
    #     for data in data_loader:
    #         sample = data[1].detach().numpy()
    #         labels = sample.astype(np.uint8)
    #
    #         count = np.bincount(labels.reshape(-1), minlength=class_nums)
    #         z += count
    #         total_freq = np.sum(z)
    #         class_weights = []
    #         for freq in z:
    #             class_weight = 1 / (np.log(1.02 + (freq / total_freq)))
    #             class_weights.append(class_weight)
    #
    #         ret = np.array(class_weights)
    #
    #     return ret

    train_weight = calculate_weigths_labels(train_loader, 12)
    train_weight = torch.from_numpy(train_weight.astype(np.float32)).cuda()
    print(train_weight)

    # print(train_weight.is_cuda)

    def inverse_normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        """
        :param img: numpy array. shape (height, width, channel). [-1~1]
        :return: numpy array. shape (height, width, channel). [0~1]
        """
        img[:,:, :, 0] = ((img[:,:, :, 0]) * std[0]) + mean[0]
        img[:,:, :, 1] = ((img[:,:, :, 1]) * std[1]) + mean[1]
        img[:,:, :, 2] = ((img[:,:, :, 2]) * std[2]) + mean[2]
        return img

    def normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        """
        :param img: numpy array. shape (height, width, channel). [-1~1]
        :return: numpy array. shape (height, width, channel). [0~1]
        """
        img[:,:, :, 0] = ((img[:,:, :, 0]) - mean[0]) * std[0]
        img[:,:, :, 1] = ((img[:,:, :, 1]) - mean[1]) * std[1]
        img[:,:, :, 2] = ((img[:,:, :, 2]) - mean[2]) * std[2]
        return img

    # Set up model
    model_map = {
        'deeplabv3_resnet50': network.deeplabv3_resnet50,
        'deeplabv3plus_resnet50': network.deeplabv3plus_resnet50,
        'deeplabv3_resnet101': network.deeplabv3_resnet101,
        'deeplabv3plus_resnet101': network.deeplabv3plus_resnet101,
        'deeplabv3_mobilenet': network.deeplabv3_mobilenet,
        'deeplabv3plus_mobilenet': network.deeplabv3plus_mobilenet
    }

    # model = model_map[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)

    tf = transforms.ToPILImage()
    m_32 = nn.Conv2d(3, 2, kernel_size=1).cuda()
    m_23 = nn.Conv2d(2, 3, kernel_size=1).cuda()

    model_pre = UNet(n_channels=3, n_classes=2, bilinear=True)
    # checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
    # model_pre.load_state_dict(checkpoint["model_state"])
    model_pre.cuda()
    # model_pre.eval()

    optimizer_pre = optim.RMSprop(model_pre.parameters(), lr=opts.lr, weight_decay=1e-8, momentum=0.9)
    scheduler_pre = optim.lr_scheduler.ReduceLROnPlateau(optimizer_pre, 'max', patience=2)  # goal: maximize Dice score

    model_pretrained = UNet(n_channels=3, n_classes=2, bilinear=False)
    # model_pretrained.load_state_dict((torch.load(opts.ckpt_pretrained, map_location=device)))
    # pretrained_dict = model_pretrained.state_dict()

    model = UNet_attnetion_encoder_decoder_concat(n_channels=3, n_classes=12, bilinear=True)
    # model_dict = model.state_dict()
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # model_dict.update(pretrained_dict)
    # model.load_state_dict(model_dict)



    model.cuda()

    # Set up metrics
    metrics = StreamSegMetrics(12)
    metrics_pre = StreamSegMetrics(opts.num_classes_pre)


    optimizer = optim.RMSprop(model.parameters(), lr=opts.lr, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score

    #optimizer = torch.optim.SGD(params=model.parameters(), lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    #torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.lr_decay_step, gamma=opts.lr_decay_factor)

    if opts.lr_policy=='poly':
        scheduler = utils.PolyLR(optimizer, opts.total_itrs, power=0.9)
        scheduler_pre = utils.PolyLR(optimizer_pre, opts.total_itrs, power=0.9)
    elif opts.lr_policy=='step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.1)
        scheduler_pre = torch.optim.lr_scheduler.StepLR(optimizer_pre, step_size=opts.step_size, gamma=0.1)


    if opts.loss_type_pre == 'focal_loss':
        criterion_pre = utils.FocalLoss(ignore_index=255, size_average=True)
    elif opts.loss_type_pre == 'cross_entropy':
        criterion_pre = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')

    # Set up criterion
    #criterion = utils.get_loss(opts.loss_type)
    if opts.loss_type == 'focal_loss':
        criterion = utils.FocalLoss(ignore_index=255, size_average=True)
    elif opts.loss_type == 'cross_entropy':
        criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean', weight=train_weight)

    def save_ckpt(path):
        """ save current model
        """
        torch.save({
            "cur_itrs": cur_itrs,
            # "model_state": model.module.state_dict(),  ##Data.parraell 이 있으면 module 이 생긴다.
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_score": best_score,
        }, path)
        print("Model saved as %s" % path)

    def save_ckpt_pre(path):
        """ save current model
        """
        torch.save({
            "cur_itrs": cur_itrs,
            # "model_state": model.module.state_dict(),  ##Data.parraell 이 있으면 module 이 생긴다.
            "model_state": model_pre.state_dict(),
            "optimizer_state": optimizer_pre.state_dict(),
            "scheduler_state": scheduler_pre.state_dict(),
            "best_score": best_score_pre,
        }, path)
        print("Model saved as %s" % path)

    utils.mkdir('checkpoints')
    # Restore
    best_score = 0.0
    best_score_pre = 0.0
    cur_itrs = 0
    cur_epochs = 0

    total_train_miou = []
    total_train_loss = []
    total_val_miou = []
    total_val_loss = []
    # if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        # del checkpoint  # free memory
    if opts.ckpt is not None and opts.test_only:
        # pass
        # https://github.com/VainF/DeepLabV3Plus-Pytorch/issues/8#issuecomment-605601402, @PytaichukBohdan

        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])

        # PATH_1 = 'C:/checkpoint_void/pytorch/segmentation/camvid_original_model/model.pt'
        # model = torch.load(PATH_1)
        model = nn.DataParallel(model)
        model.to(device)
        # summary(model, (3,256,256))
        # if opts.continue_training:
        #     optimizer.load_state_dict(checkpoint["optimizer_state"])
        #     scheduler.load_state_dict(checkpoint["scheduler_state"])
        #     cur_itrs = checkpoint["cur_itrs"]
        #     best_score = checkpoint['best_score']
        #     print("Training state restored from %s" % opts.ckpt)
        print("Model restored from %s" % opts.ckpt)
        # del checkpoint  # free memory
    else:
        # pass
        print("[!] Retrain")
        # model = nn.DataParallel(model)
        model.to(device)

    #==========   Train Loop   ==========#
    vis_sample_id = np.random.randint(0, len(val_loader), opts.vis_num_samples,
                                      np.int32) if opts.enable_vis else None  # sample idxs for visualization
    denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # denormalization for ori images

    # if opts.test_only:
    #
    #     path = 'D:/checkpoint/Segmentation/new_torch_deeplabv3plus/original'
    #     ckp_list = os.listdir(path)
    #
    #     for i in range(len(ckp_list[:-2])):
    #         test_model = model
    #
    #         ckp_name = f'_{i+1}_deeplabv3plus_resnet50_camvid_sample_os16.pth'
    #         ckp = path + '/' + ckp_name
    #         print(ckp)
    #
    #         checkpoint = torch.load(str(ckp), map_location=torch.device('cpu'))
    #         test_model.load_state_dict(checkpoint["model_state"])
    #
    #         test_model = nn.DataParallel(test_model)
    #         test_model.to(device)
    #         # print("Model restored from %s" % opts.ckpt)
    #
    #         test_model.eval()
    #         val_score, ret_samples, val_loss = val_validate(
    #             opts=opts, model=test_model, loader=val_loader, device=device, metrics=metrics, ret_samples_ids=vis_sample_id)
    #
    #         val_loss = val_loss / 350
    #         print(metrics.to_str(val_score))
    #         print(val_loss)
    #         total_val_loss.append(val_loss)
    #         total_val_miou.append(val_score['Mean IoU'])
    #
    #         val_df_train_loss = pd.DataFrame(total_val_loss)
    #         val_df_train_miou = pd.DataFrame(total_val_miou)
    #
    #         val_df_train_miou.to_csv('D:/plt/segmentation/Camvid_original/original/val_miou.csv', index=False)
    #         val_df_train_loss.to_csv('D:/plt/segmentation/Camvid_original/original/val_loss.csv', index=False)
    #
    #         # PATH = 'C:/checkpoint_void/pytorch/segmentation/camvid_original_model_2/'
    #         # torch.save(model, PATH + 'model.pt' )
    #     return

    if opts.test_only:
        model.eval()
        val_score, ret_samples = validate(
            opts=opts, model1=model_pre, model2=model, loader=val_loader, device=device, metrics=metrics, ret_samples_ids=vis_sample_id)
        print(metrics.to_str(val_score))
        # PATH = 'C:/checkpoint_void/pytorch/segmentation/camvid_original_model_2/'
        # torch.save(model, PATH + 'model.pt' )
        return

    else:
        interval_loss = 0
        internval_loss_plt = 0

######### last checkpoint 받아서 이어서 학습 ##############
        # checkpoint = torch.load(
        #     f'D:/checkpoint/Segmentation/kitti/original/firstfold/_{cur_epochs + 286}_deeplabv3plus_resnet50_kitti_sample_os16.pth',
        #     map_location=torch.device('cpu'))
        # print("===>Testing using weights: ",
        #       f'D:/checkpoint/Segmentation/kitti/original/firstfold/_{cur_epochs + 286}_deeplabv3plus_resnet50_kitti_sample_os16')
        # model.load_state_dict(checkpoint["model_state"])

        if opts.continue_training:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            scheduler.max_iters = opts.total_itrs
            cur_itrs = checkpoint["cur_itrs"]
            best_score = checkpoint['best_score']
            # print("Training state restored from %s" % opts.ckpt)
        ######### ##############

        while True: #cur_itrs < opts.total_itrs:
            # =====  Train  =====
            # PATH_1 = 'C:/checkpoint_void/pytorch/segmentation/camvid_original_model/model.pt'
            # model = torch.load(PATH_1)
            # print(model)

            model.train()
            model_pre.train()
            cur_epochs += 1

            for (images, labels, labels_pre) in train_loader:
                cur_itrs += 1

                images = images.to(device, dtype=torch.float32)
                labels_pre = labels_pre.to(device, dtype=torch.long)

                optimizer.zero_grad()
                optimizer_pre.zero_grad()

                outputs_pre = model_pre(images)

                loss_1 = criterion_pre(outputs_pre, labels_pre)

                labels_pre_onehot = label_to_one_hot_label(labels_pre, num_classes=2, ignore_index=255)
                loss_2 = DiceBCELoss()(outputs_pre, labels_pre_onehot)

                loss_pre = loss_1 + loss_2
                loss_pre.backward()
                optimizer_pre.step()

                if (cur_itrs) % opts.val_interval == 0:
                    train_val_score_pre, ret_samples_pre = validate_pre(
                        opts=opts, model=model_pre, loader=train_loader, device=device, metrics=metrics_pre,
                        ret_samples_ids=vis_sample_id)
                    print(train_val_score_pre['Mean IoU'])
                    print('-----------------------1_stage_miou-------------------------')
                    print("---------Epoch %d, Itrs %d/%d, train_Miou_pre=%f----------" %
                          (cur_epochs, cur_itrs, opts.total_itrs, train_val_score_pre['Mean IoU']))


                labels = labels.to(device, dtype=torch.long)

                with torch.no_grad():
                    outputs_pre = model_pre(images)

                ## attention
                outputs_pre = nn.Sigmoid()(outputs_pre)
                # tf(outputs_pre[1]).show()

                optimizer.zero_grad()
                outputs = model(images, outputs_pre)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                np_loss = loss.detach().cpu().numpy()
                interval_loss += np_loss
                internval_loss_plt += np_loss

                if vis is not None:
                    vis.vis_scalar('Loss', cur_itrs, np_loss)

                if (cur_itrs) % 10 == 0:
                    interval_loss = interval_loss/10
                    print("Epoch %d, Itrs %d/%d, Loss=%f" %
                          (cur_epochs, cur_itrs, opts.total_itrs, interval_loss))

                    interval_loss = 0.0

                ## train loss
                if (cur_itrs) % opts.val_interval == 0:
                    internval_loss_plt = internval_loss_plt / opts.val_interval
                    print("---------Epoch %d, Itrs %d/%d, train_Loss=%f----------" %
                          (cur_epochs, cur_itrs, opts.total_itrs, internval_loss_plt))
                    total_train_loss.append(internval_loss_plt)

                ## train miou
                if (cur_itrs) % opts.val_interval == 0:
                    train_val_score, ret_samples = validate(
                        opts=opts, model1=model_pre, model2=model, loader=train_loader, device=device, metrics=metrics,
                        ret_samples_ids=vis_sample_id)
                    print(train_val_score['Mean IoU'])
                    print('-----------------------2_stage_miou-------------------------')
                    print("---------Epoch %d, Itrs %d/%d, train_Miou=%f----------" %
                          (cur_epochs, cur_itrs, opts.total_itrs, train_val_score['Mean IoU']))

                    total_train_miou.append(train_val_score['Mean IoU'])

                if (cur_itrs) % opts.val_interval == 0:
                    # save_ckpt('checkpoints/latest_%s_%s_os%d.pth' %
                    #           (opts.model, opts.dataset, opts.output_stride))
                    save_ckpt('E:/Second_paper/Checkpoint/Camvid/Blur_firstfold/main_7_binaryTrue_nopretrained_lr_0.0001/SecondStage/latest_%s_%s_os%d.pth' %
                              (opts.model, opts.dataset, opts.output_stride))
                    print("validation...")
                    model.eval()
                    val_score, ret_samples = validate(
                        opts=opts, model1=model_pre, model2=model, loader=test_loader, device=device, metrics=metrics, ret_samples_ids=vis_sample_id)
                    print(metrics.to_str(val_score))

                    save_ckpt_pre('E:/Second_paper/Checkpoint/Camvid/Blur_firstfold/main_7_binaryTrue_nopretrained_lr_0.0001/FirstStage/latest_%s_%s_os%d.pth' %
                              (opts.model, opts.dataset, opts.output_stride))
                    print("2-stage-validation...")
                    model_pre.eval()
                    val_score_pre, ret_samples_pre = validate_pre(
                        opts=opts, model=model_pre, loader=test_loader, device=device, metrics=metrics_pre, ret_samples_ids=vis_sample_id)
                    print(metrics_pre.to_str(val_score_pre))


                    if val_score['Mean IoU'] > best_score:  # save best model
                        best_score = val_score['Mean IoU']
                        # save_ckpt('checkpoints/best_%s_%s_os%d.pth' %
                        #           (opts.model, opts.dataset,opts.output_stride))
                        save_ckpt('E:/Second_paper/Checkpoint/Camvid/Blur_firstfold/main_7_binaryTrue_nopretrained_lr_0.0001/SecondStage/best_%s_%s_os%d.pth' %
                                  (opts.model, opts.dataset, opts.output_stride))

                    if val_score_pre['Mean IoU'] > best_score_pre:  # save best model
                        best_score_pre = val_score_pre['Mean IoU']
                        # save_ckpt('checkpoints/best_%s_%s_os%d.pth' %
                        #           (opts.model, opts.dataset,opts.output_stride))
                        save_ckpt_pre('E:/Second_paper/Checkpoint/Camvid/Blur_firstfold/main_7_binaryTrue_nopretrained_lr_0.0001/FirstStage/best_%s_%s_os%d.pth' %
                                  (opts.model, opts.dataset, opts.output_stride))

                    if vis is not None:  # visualize validation score and samples
                        vis.vis_scalar("[Val] Overall Acc", cur_itrs, val_score['Overall Acc'])
                        vis.vis_scalar("[Val] Mean IoU", cur_itrs, val_score['Mean IoU'])
                        vis.vis_table("[Val] Class IoU", val_score['Class IoU'])

                        for k, (img, target, lbl) in enumerate(ret_samples):
                            img = (denorm(img) * 255).astype(np.uint8)
                            target = train_dst.decode_target(target).transpose(2, 0, 1).astype(np.uint8)
                            lbl = train_dst.decode_target(lbl).transpose(2, 0, 1).astype(np.uint8)
                            concat_img = np.concatenate((img, target, lbl), axis=2)  # concat along width
                            vis.vis_image('Sample %d' % k, concat_img)

                    save_ckpt('E:/Second_paper/Checkpoint/Camvid/Blur_firstfold/main_7_binaryTrue_nopretrained_lr_0.0001/SecondStage/_%s_%s_%s_os%d.pth' %
                              (cur_epochs, opts.model, opts.dataset, opts.output_stride))
                    save_ckpt_pre('E:/Second_paper/Checkpoint/Camvid/Blur_firstfold/main_7_binaryTrue_nopretrained_lr_0.0001/FirstStage/_%s_%s_%s_os%d.pth' %
                              (cur_epochs, opts.model, opts.dataset, opts.output_stride))

                    model.train()
                scheduler.step()
                scheduler_pre.step()

                if cur_itrs >= opts.total_itrs:
                    df_train_loss = pd.DataFrame(total_train_loss)
                    df_train_miou = pd.DataFrame(total_train_miou)

                    df_train_miou.to_csv('E:/Second_paper/CSV/Camvid/Blur_firstfold/main_7_binaryTrue_nopretrain_lr_0.0001/train_miou_2.csv', index=False)
                    df_train_loss.to_csv('E:/Second_paper/CSV/Camvid/Blur_firstfold/main_7_binaryTrue_nopretrain_lr_0.0001/train_loss_2.csv', index=False)


                    # plt.plot(total_train_miou)
                    # plt.xlabel('epoch')
                    # plt.ylabel('miou')
                    plt.rcParams['axes.xmargin'] = 0
                    plt.rcParams['axes.ymargin'] = 0
                    plt.plot(total_train_miou)
                    plt.xlabel('epoch')
                    plt.ylabel('miou')
                    plt.show()

                    # plt.rcParams['axes.xmargin'] = 0
                    # plt.rcParams['axes.ymargin'] = 0
                    plt.plot(total_train_loss)
                    plt.xlabel('epoch')
                    plt.ylabel('loss')
                    plt.show()

                    return



if __name__ == '__main__':
    main()
