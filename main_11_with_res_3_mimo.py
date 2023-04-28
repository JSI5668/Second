import torchvision.transforms
from tqdm import tqdm
import network
import utils
import os
import random
import argparse
import numpy as np
import pandas as pd

from warmup_scheduler import GradualWarmupScheduler
from data_RGB import get_training_data, get_validation_data
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
from torchvision.utils import save_image

from dataset_RGB import DataLoaderTrain
from dataset_RGB import DataLoaderVal
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils import data
from datasets import VOCSegmentation, Cityscapes, Camvid, Camvid_sample, Camvid_sample_3input, Kitti_sample, Camvid_proposed, Camvid_MHNet_withresto
from utils import ext_transforms as et
from metrics import StreamSegMetrics
from torchsummary import summary
import torch
import torch.nn as nn
from utils.visualizer import Visualizer
import losses

from torchvision.transforms.functional import to_pil_image
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from network.unet import UNet
from network.unet import UNet_attnetion_decoder_concat
from network.unet import UNet_attnetion_encoder_concat
from network.unet import UNet_attnetion_encoder_each_sum
from network.unet import UNet_attnetion_encoder_decoder_concat
from network.unet import MHNet_my, MHNet_my_2, UNet_my_2, MHNet, MIMOUNet, MIMOUNetPlus

from utils.loss import DiceBCELoss
from torch import optim
from typing import Optional
from glob import glob

# print(torch.cuda.is_available())

# python -m visdom.server

from collections import OrderedDict


def load_checkpoint(model, weights):
    checkpoint = torch.load(weights)
    try:
        model.load_state_dict(checkpoint["state_dict"], strict=False)
    except:
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)


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
    parser.add_argument("--data_root", type=str,
                        default='E:/Second_paper/Data/CamVid/Blurred/Firstfold/For_Segmentation_With_Resto_typedeeplab',
                        ##Deeplab으로 바꿀때 model backbone, optimizer 설정
                        help="path to Dataset")  ##crop size 바꿔주기
    # parser.add_argument("--data_root", type=str,
    #                     default='D:/Dataset/Camvid/camvid_original_240',
    #                      help="path to Dataset")   ##crop size 바꿔주기

    parser.add_argument("--dataset", type=str, default='kitti_MHNet_withresto',
                        choices=['voc', 'cityscapes, camvid', 'camvid_sample', 'camvid_focus', 'kitti_sample',
                                 'camvid_proposed', 'camvid_MHNet_withresto', 'kitti_MHNet_withresto'], help='Name of dataset')

    parser.add_argument("--num_classes", type=int, default=None,
                        help="num classes (default: None)")

    # Deeplab Options
    parser.add_argument("--model", type=str, default='deeplabv3plus_resnet50',
                        choices=['deeplabv3_resnet50', 'deeplabv3plus_resnet50',
                                 'deeplabv3_resnet101', 'deeplabv3plus_resnet101',
                                 'deeplabv3_mobilenet', 'deeplabv3plus_mobilenet'], help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    # Train Options
    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--save_val_results", action='store_true', default=False,
                        help="save segmentation results to \"./results\"")
    parser.add_argument("--total_itrs", type=int, default=223000,
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
    parser.add_argument("--batch_size", type=int, default=1,
                        help='batch size (default: 16)')
    parser.add_argument("--val_batch_size", type=int, default=1,
                        help='batch size for validation (default: 4)')
    # parser.add_argument("--crop_size", type=int, default=240)  ##513
    parser.add_argument("--crop_size", type=int, default=176)  ##513
    parser.add_argument("--ckpt_pre",
                        default='E:/Second_paper/Checkpoint/Camvid/Blur_firstfold/main_11_with_res_MHNet_my_2/model_pre_best_deeplabv3plus_resnet50_camvid_MHNet_withresto_os16.pth',
                        type=str,
                        help="restore from checkpoint")
    parser.add_argument("--ckpt",
                        default='E:/Second_paper/Checkpoint/Camvid/Blur_firstfold/main_11_with_res_MHNet_my_2/best_deeplabv3plus_resnet50_camvid_MHNet_withresto_os16.pth',
                        type=str,
                        help="restore from checkpoint")
    # parser.add_argument("--ckpt",default=None, type=str,
    #                     help="restore from checkpoint")
    # parser.add_argument("--ckpt_pretrained",default='E:/pretrained/unet_carvana_scale1.0_epoch2.pth', type=str,
    #                     help="restore from checkpoint")

    parser.add_argument("--continue_training", action='store_true', default=False)

    parser.add_argument("--overlay", default=True)

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
    parser.add_argument("--val_interval", type=int, default=223,
                        help="epoch interval for eval (default: 100)")
    parser.add_argument("--download", action='store_true', default=False,
                        help="download datasets")

    # PASCAL VOC Options
    parser.add_argument("--year", type=str, default='2012',
                        choices=['2012_aug', '2012', '2011', '2009', '2008', '2007'], help='year of VOC')

    # Visdom options
    parser.add_argument("--enable_vis", action='store_true', default=False,
                        help="use visdom for visualization")
    parser.add_argument("--vis_port", type=str, default='8097',  # 13570
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

        test_dst = Kitti_sample(root=opts.data_root, split='test', transform=val_transform)

    if opts.dataset == 'camvid_MHNet_withresto':
        train_dst = get_training_data('E:/Second_paper/Data/CamVid/Blurred/Firstfold/For_Segmentation_With_Resto/train',
                                      {'patch_size': opts.crop_size})

        val_dst = get_validation_data('E:/Second_paper/Data/CamVid/Blurred/Firstfold/For_Segmentation_With_Resto/test',
                                      {'patch_size': opts.crop_size})

        test_dst = get_validation_data('E:/Second_paper/Data/CamVid/Blurred/Firstfold/For_Segmentation_With_Resto/test',
                                       {'patch_size': opts.crop_size})

    if opts.dataset == 'kitti_MHNet_withresto':
        train_dst = get_training_data('E:/Second_paper/Data/Kitti/Firstfold/Blurred/For_Segmentation_With_Resto/train',
                                      {'patch_size': opts.crop_size})

        val_dst = get_validation_data('E:/Second_paper/Data/Kitti/Firstfold/Blurred/For_Segmentation_With_Resto/test',
                                      {'patch_size': opts.crop_size})

        test_dst = get_validation_data('E:/Second_paper/Data/Kitti/Firstfold/Blurred/For_Segmentation_With_Resto/test',
                                       {'patch_size': opts.crop_size})

    return train_dst, val_dst, test_dst


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


def validate_With_resto(opts, model1, model2, loader, device, metrics, ret_samples_ids=None):
    """Do validation and return specified samples"""
    metrics.reset()
    ret_samples = []
    if opts.overlay:
        if not os.path.exists('results'):
            os.mkdir('results')
        denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
        img_id = 0

    val_transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_dst, val_dst, test_dst = get_dataset(opts)

    tf = transforms.ToPILImage()

    with torch.no_grad():
        for restore_target, blurred_image, segment_target in loader:
            images_resto = blurred_image.to(device, dtype=torch.float32)

            outputs_pre_resto = model1(images_resto)[2]
            # tf(images_resto[0]).show()
            # save_image(outputs_pre_resto, 'E:/Second_paper/Checkpoint/Camvid/Blur_firstfold/main_11_with_res_MHNet_my_2/image/save_img.png')

            outputs_pre_resto = val_transform(outputs_pre_resto)
            segment_target = val_dst.encode_target(segment_target)
            segment_target = torch.tensor(segment_target, dtype=torch.uint8)

            outputs_pre_resto.to(device, dtype=torch.float32)
            labels_seg = segment_target.to(device, dtype=torch.long)

            outputs_seg = model2(outputs_pre_resto)

            preds_seg = outputs_seg.detach().max(dim=1)[1].cpu().numpy()
            targets_seg = labels_seg.cpu().numpy()

            metrics.update(targets_seg, preds_seg)

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
    if opts.dataset == 'voc' and not opts.crop_val:
        opts.val_batch_size = 1

    train_transform = transforms.Compose([
        transforms.RandomCrop(size=(opts.crop_size, opts.crop_size)),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_transform_lbl = transforms.Compose([
        transforms.RandomCrop(size=(opts.crop_size, opts.crop_size)),
        transforms.RandomHorizontalFlip(),
    ])

    ####################resto#################################
    train_dataset_resto = get_training_data(
        'E:/Second_paper/Data/Kitti/Firstfold/Blurred/For_Segmentation_With_Resto/train',
        {'patch_size': opts.crop_size})
    train_loader_resto = DataLoader(dataset=train_dataset_resto, batch_size=opts.batch_size, shuffle=True,
                                    num_workers=2,
                                    drop_last=False, pin_memory=True)
    val_dataset_resto = get_validation_data(
        'E:/Second_paper/Data/Kitti/Firstfold/Blurred/For_Segmentation_With_Resto/test',
        {'patch_size': opts.crop_size})
    val_loader_resto = DataLoader(dataset=val_dataset_resto, batch_size=opts.val_batch_size, shuffle=False,
                                  num_workers=2)

    train_dst, val_dst, test_dst = get_dataset(opts)

    criterion_mse = losses.PSNRLoss()
    criterion_char = losses.CharbonnierLoss()
    criterion_edge = losses.EdgeLoss()
    criterion_mimo = torch.nn.L1Loss()
    ####################resto#################################
    # train_dst, val_dst, test_dst = get_dataset(opts)
    #
    # train_loader = DataLoader(
    #     train_dst, batch_size=opts.batch_size, shuffle=True, num_workers=2)
    #
    # val_loader = DataLoader(
    #     val_dst, batch_size=opts.val_batch_size, shuffle=False, num_workers=2)
    #
    # test_loader = DataLoader(
    #     test_dst, batch_size=opts.val_batch_size, shuffle=False, num_workers=2)
    #
    # print("Dataset: %s, Train set: %d, Val set: %d, Test set: %d" %
    #       (opts.dataset, len(train_dst), len(val_dst), len(test_dst)))

    ## https://github.com/jfzhang95/pytorch-deeplab-xception/blob/master/utils/calculate_weights.py
    # def calculate_weigths_labels(dataloader, num_classes):
    #     # Create an instance from the data loader
    #     z = np.zeros((num_classes,))
    #     # Initialize tqdm
    #     # tqdm_batch = tqdm(dataloader)
    #     print('Calculating classes weights')
    #     # for sample in tqdm_batch:
    #     for (_, label, _) in train_loader:
    #         # y = sample['label']
    #         y = label
    #         y = y.detach().cpu().numpy()
    #         mask = (y >= 0) & (y < num_classes)
    #         labels = y[mask].astype(np.uint8)
    #         count_l = np.bincount(labels, minlength=num_classes)
    #         z += count_l
    #     # tqdm_batch.close()
    #     total_frequency = np.sum(z)
    #     class_weights = []
    #     for frequency in z:
    #         class_weight = 1 / (np.log(1.02 + (frequency / total_frequency)))
    #         class_weights.append(class_weight)
    #     ret = np.array(class_weights)
    #     # classes_weights_path = os.path.join(opts.data_root, '_classes_weights.npy')
    #     # np.save(classes_weights_path, ret)
    #
    #     return ret
    #
    # def calc_weights(class_num):
    #     z = np.zeros((class_num,))
    #     # (_, label) = enumerate(train_loader)
    #     # for sample in label:
    #     for (_, label, _) in train_loader:
    #         sample = label
    #         sample = sample.to('cpu').detach().numpy()
    #         # sample = sample.argmax(0)
    #         labels = sample.astype(np.uint8)
    #         # count = np.bincount(labels.reshape(-1), minlength=class_num)
    #         count = np.bincount(labels, minlength=class_num)
    #         z += count
    #     total_freq = np.sum(z)
    #     class_weights = []
    #     for freq in z:
    #         class_weight = 1 / (np.log(1.02 + (freq / total_freq)))
    #         class_weights.append(class_weight)
    #
    #     ret = np.array(class_weights)
    #
    #     return ret
    #
    # # def calc_class_weights(data_loader, class_nums):
    # #     z = np.zeros((class_nums,))
    # #     for data in data_loader:
    # #         sample = data[1].detach().numpy()
    # #         labels = sample.astype(np.uint8)
    # #
    # #         count = np.bincount(labels.reshape(-1), minlength=class_nums)
    # #         z += count
    # #         total_freq = np.sum(z)
    # #         class_weights = []
    # #         for freq in z:
    # #             class_weight = 1 / (np.log(1.02 + (freq / total_freq)))
    # #             class_weights.append(class_weight)
    # #
    # #         ret = np.array(class_weights)
    # #
    # #     return ret
    #
    # train_weight = calculate_weigths_labels(train_loader, 12)
    # train_weight = torch.from_numpy(train_weight.astype(np.float32)).cuda()
    # print(train_weight)

    # print(train_weight.is_cuda)

    def inverse_normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        """
        :param img: numpy array. shape (height, width, channel). [-1~1]
        :return: numpy array. shape (height, width, channel). [0~1]
        """
        img[:, :, :, 0] = ((img[:, :, :, 0]) * std[0]) + mean[0]
        img[:, :, :, 1] = ((img[:, :, :, 1]) * std[1]) + mean[1]
        img[:, :, :, 2] = ((img[:, :, :, 2]) * std[2]) + mean[2]
        return img

    def normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        """
        :param img: numpy array. shape (height, width, channel). [-1~1]
        :return: numpy array. shape (height, width, channel). [0~1]
        """
        img[:, :, :, 0] = ((img[:, :, :, 0]) - mean[0]) * std[0]
        img[:, :, :, 1] = ((img[:, :, :, 1]) - mean[1]) * std[1]
        img[:, :, :, 2] = ((img[:, :, :, 2]) - mean[2]) * std[2]
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
    # m_32 = nn.Conv2d(3, 2, kernel_size=1).cuda()
    # m_23 = nn.Conv2d(2, 3, kernel_size=1).cuda()

    model_pre = MIMOUNetPlus()
    # model_pre = MHNet()
    checkpoint = torch.load('E:/Second_paper/Checkpoint/Pretrained_models/MIMO_UNetPlus/MIMO-UNetPlus.pkl')
    model_pre.load_state_dict(checkpoint["model"])
    model_pre.cuda()
    # model_pre.eval()

    optimizer_pre = torch.optim.Adam(model_pre.parameters(), lr=1e-4, weight_decay=0)
    # warmup_epochs = 3
    # scheduler_pre = torch.optim.lr_scheduler.MultiStepLR(optimizer_pre, [(x + 1) * 50 for x in range(1000 // 50)], 0.5)
    scheduler_pre = torch.optim.lr_scheduler.StepLR(optimizer_pre, step_size=opts.step_size, gamma=0.1)

    # model_pretrained = UNet(n_channels=3, n_classes=2, bilinear=False)
    # model_pretrained.load_state_dict((torch.load(opts.ckpt_pretrained, map_location=device)))
    # pretrained_dict = model_pretrained.state_dict()

    model = UNet_my_2(n_channels=3, n_classes=12, bilinear=True)
    # checkpoint = torch.load(
    #     'E:/Second_paper/Checkpoint/Camvid/Original_firstfold/main_10_UNetmy_2_Edge_perceptual_lr_0.0001_0,1,0_1000epo/best_deeplabv3plus_resnet50_camvid_sample_os16.pth')
    # model.load_state_dict(checkpoint["model_state"])
    model.cuda()

    # Set up metrics
    metrics = StreamSegMetrics(12)

    optimizer = optim.RMSprop(model.parameters(), lr=opts.lr, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score

    # optimizer = torch.optim.SGD(params=model.parameters(), lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    # torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.lr_decay_step, gamma=opts.lr_decay_factor)

    if opts.lr_policy == 'poly':
        scheduler = utils.PolyLR(optimizer, opts.total_itrs, power=0.9)
        # scheduler_pre = utils.PolyLR(optimizer_pre, opts.total_itrs, power=0.9)
    elif opts.lr_policy == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.1)
        # scheduler_pre = torch.optim.lr_scheduler.StepLR(optimizer_pre, step_size=opts.step_size, gamma=0.1)
    #

    if opts.loss_type_pre == 'focal_loss':
        criterion_pre = utils.FocalLoss(ignore_index=255, size_average=True)
    elif opts.loss_type_pre == 'cross_entropy':
        criterion_pre = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')

    # Set up criterion
    # criterion = utils.get_loss(opts.loss_type)
    if opts.loss_type == 'focal_loss':
        criterion = utils.FocalLoss(ignore_index=255, size_average=True)
    # elif opts.loss_type == 'cross_entropy':
    #     criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean', weight=train_weight)
    elif opts.loss_type == 'cross_entropy':
        criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')

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
            "best_score": best_score,
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
        checkpoint_pre = torch.load(opts.ckpt_pre, map_location=torch.device('cpu'))
        model_pre.load_state_dict(checkpoint_pre["model_state"])

        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])

        # PATH_1 = 'C:/checkpoint_void/pytorch/segmentation/camvid_original_model/model.pt'
        # model = torch.load(PATH_1)
        model = nn.DataParallel(model)
        model.to(device)
        print("Model restored from %s" % opts.ckpt)
        # del checkpoint  # free memory
    else:
        # pass
        print("[!] Retrain")
        # model = nn.DataParallel(model)
        model.to(device)

    # ==========   Train Loop   ==========#
    vis_sample_id = np.random.randint(0, len(val_loader_resto), opts.vis_num_samples,
                                      np.int32) if opts.enable_vis else None  # sample idxs for visualization
    denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # denormalization for ori images

    if opts.test_only:
        model.eval()
        model_pre.eval()
        val_score, ret_samples = validate_With_resto(
            opts=opts, model1=model_pre, model2=model, loader=val_loader_resto, device=device, metrics=metrics,
            ret_samples_ids=vis_sample_id)
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

        while True:  # cur_itrs < opts.total_itrs:
            # =====  Train  =====
            # PATH_1 = 'C:/checkpoint_void/pytorch/segmentation/camvid_original_model/model.pt'
            # model = torch.load(PATH_1)
            # print(model)

            model.train()
            model_pre.train()
            cur_epochs += 1

            for resto_target, blurred, semantic_target in train_loader_resto:
                cur_itrs += 1

                target = resto_target.cuda()
                input_ = blurred.cuda()

                target_2 = F.interpolate(target, scale_factor=0.5, mode='bilinear')
                target_3 = F.interpolate(target, scale_factor=0.25, mode='bilinear')
                # optimizer.zero_grad()
                optimizer_pre.zero_grad()

                outputs_pre = model_pre(input_)

                l1 = criterion_mimo(outputs_pre[0], target_3)
                l2 = criterion_mimo(outputs_pre[1], target_2)
                l3 = criterion_mimo(outputs_pre[2], target)
                l_total = l1+l2+l3

                label_fft1 = torch.fft.rfft(target_3, dim=2, norm='backward')
                pred_fft1 = torch.fft.rfft(outputs_pre[0], dim=2, norm='backward')
                label_fft2 = torch.fft.rfft(target_2, dim=2, norm='backward')
                pred_fft2 = torch.fft.rfft(outputs_pre[1], dim=2, norm='backward')
                label_fft3 = torch.fft.rfft(target, dim=2, norm='backward')
                pred_fft3 = torch.fft.rfft(outputs_pre[2], dim=2, norm='backward')

                f1 = criterion_mimo(pred_fft1, label_fft1)
                f2 = criterion_mimo(pred_fft2, label_fft2)
                f3 = criterion_mimo(pred_fft3, label_fft3)
                loss_fft = f1 + f2 + f3

                loss_resto = l_total + 0.1 * loss_fft
                loss_resto.backward()
                optimizer_pre.step()

                psnr_te = 0
                if (cur_itrs) % opts.val_interval == 0:
                    model_pre.eval()
                    psnr_val_rgb = []
                    for resto_target_val, blurred_val, semantic_target_val in val_loader_resto:
                        target_val_restored = resto_target_val.cuda()
                        input_val_restored = blurred_val.cuda()

                        with torch.no_grad():
                            restored_val = model_pre(input_val_restored)
                        restore_val = restored_val[2]
                        for res, tar in zip(restore_val, target_val_restored):
                            tssss = utils.torchPSNR(res, tar)
                            psnr_te = psnr_te + tssss
                            psnr_val_rgb.append(utils.torchPSNR(res, tar))

                    psnr_val_rgb = torch.stack(psnr_val_rgb).mean().item()
                    print("te", psnr_te)
                    print(
                        "[epoch %d PSNR: %.4f ]" % (cur_epochs, psnr_val_rgb))

                with torch.no_grad():
                    outputs_pre = model_pre(input_)[2]
                # tf(outputs_pre[0]).show()

                params = transforms.RandomCrop.get_params(outputs_pre, output_size=(opts.crop_size, opts.crop_size))

                outputs_pre = TF.crop(outputs_pre, *params)
                outputs_pre = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5)(outputs_pre)
                outputs_pre = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(outputs_pre)

                semantic_target = TF.crop(semantic_target, *params)

                semantic_target = train_dst.encode_target(semantic_target)
                # semantic_target = torch.from_numpy(semantic_target)
                semantic_target = torch.tensor(semantic_target, dtype=torch.uint8)

                outputs_pre = outputs_pre.to(device, dtype=torch.float32)
                semantic_target = semantic_target.to(device, dtype=torch.long)

                optimizer.zero_grad()
                outputs = model(outputs_pre.detach())
                loss = criterion(outputs, semantic_target)

                loss.backward()
                optimizer.step()

                np_loss = loss.detach().cpu().numpy()
                interval_loss += np_loss
                internval_loss_plt += np_loss

                if vis is not None:
                    vis.vis_scalar('Loss', cur_itrs, np_loss)

                if (cur_itrs) % 10 == 0:
                    interval_loss = interval_loss / 10
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
                    train_val_score, ret_samples = validate_With_resto(
                        opts=opts, model1=model_pre, model2=model, loader=train_loader_resto, device=device,
                        metrics=metrics,
                        ret_samples_ids=vis_sample_id)
                    print(train_val_score['Mean IoU'])
                    print('-----------------------2_stage_miou-------------------------')
                    print("---------Epoch %d, Itrs %d/%d, train_Miou=%f----------" %
                          (cur_epochs, cur_itrs, opts.total_itrs, train_val_score['Mean IoU']))

                    total_train_miou.append(train_val_score['Mean IoU'])

                if (cur_itrs) % opts.val_interval == 0:
                    # save_ckpt('checkpoints/latest_%s_%s_os%d.pth' %
                    #           (opts.model, opts.dataset, opts.output_stride))
                    save_ckpt(
                        'E:/Second_paper/Checkpoint/Kitti/Segmentation/Blurred/Firstfold_30_Withresto/With_mimoPlus_UNet2my_Withoutperceptual/latest_%s_%s_os%d.pth' %
                        (opts.model, opts.dataset, opts.output_stride))
                    print("validation...")
                    model.eval()
                    model_pre.eval()
                    val_score, ret_samples = validate_With_resto(
                        opts=opts, model1=model_pre, model2=model, loader=val_loader_resto, device=device,
                        metrics=metrics, ret_samples_ids=vis_sample_id)
                    print(metrics.to_str(val_score))

                    if val_score['Mean IoU'] > best_score:  # save best model
                        best_score = val_score['Mean IoU']
                        # save_ckpt('checkpoints/best_%s_%s_os%d.pth' %
                        #           (opts.model, opts.dataset,opts.output_stride))
                        save_ckpt(
                            'E:/Second_paper/Checkpoint/Kitti/Segmentation/Blurred/Firstfold_30_Withresto/With_mimoPlus_UNet2my_Withoutperceptual/best_%s_%s_os%d.pth' %
                            (opts.model, opts.dataset, opts.output_stride))
                        save_ckpt_pre(
                            'E:/Second_paper/Checkpoint/Kitti/Segmentation/Blurred/Firstfold_30_Withresto/With_mimoPlus_UNet2my_Withoutperceptual/model_pre_best_%s_%s_os%d.pth' %
                            (opts.model, opts.dataset, opts.output_stride))

                    save_ckpt(
                        'E:/Second_paper/Checkpoint/Kitti/Segmentation/Blurred/Firstfold_30_Withresto/With_mimoPlus_UNet2my_Withoutperceptual/_%s_%s_%s_os%d.pth' %
                        (cur_epochs, opts.model, opts.dataset, opts.output_stride))

                    model.train()
                    model_pre.train()

                scheduler.step()
                scheduler_pre.step()

                if cur_itrs >= opts.total_itrs:
                    df_train_loss = pd.DataFrame(total_train_loss)
                    df_train_miou = pd.DataFrame(total_train_miou)

                    df_train_miou.to_csv(
                        'E:/Second_paper/Checkpoint/Kitti/Segmentation/Blurred/Firstfold_30_Withresto/With_mimoPlus_UNet2my_Withoutperceptual/train_miou_2.csv',
                        index=False)
                    df_train_loss.to_csv(
                        'E:/Second_paper/Checkpoint/Kitti/Segmentation/Blurred/Firstfold_30_Withresto/With_mimoPlus_UNet2my_Withoutperceptual/train_loss_2.csv',
                        index=False)

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
