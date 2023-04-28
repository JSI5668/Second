from torch.utils.data import dataset
from tqdm import tqdm
import network
import utils
import os
import random
import argparse
import numpy as np

from torch.utils import data
from datasets import VOCSegmentation, Cityscapes, cityscapes, Camvid, Dacon_segmentation, Camvid_Edge, Camvid_sample, Kitti_Edge, Kitti_sample
from torchvision import transforms as T
from metrics import StreamSegMetrics
from main import validate
from main import get_dataset
from network.unet import UNet
from torch import optim
from network.unet import UNet_my_2, EdgeNet

import torch
import torch.nn as nn

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from glob import glob
import cv2


def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--input", type=str,
                        default='E:/REVIEW_EXPERIMENT/For_Segmentation/Comparison/MHNet/Camvid_Firstfold',
                        # required=True
                        help="path to a single image or image directory")
    # parser.add_argument("--dataset", type=str, default='camvid',
    #                     choices=['voc', 'cityscapes', 'camvid'], help='Name of training set')
    parser.add_argument("--dataset", type=str, default='camvid_sample',
                        choices=['voc', 'cityscapes, camvid', 'camvid_open', 'kitti_sample'
                                  'dacon', 'camvid_sample', 'camvid_edge', 'kitti_edge'], help='Name of dataset')
    # Deeplab Options
    parser.add_argument("--model", type=str, default='deeplabv3plus_resnet50',
                        choices=['deeplabv3_resnet50', 'deeplabv3plus_resnet50',
                                 'deeplabv3_resnet101', 'deeplabv3plus_resnet101',
                                 'deeplabv3_mobilenet', 'deeplabv3plus_mobilenet'], help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    # Train Options
    parser.add_argument("--save_val_results_to",
                        default='E:/REVIEW_EXPERIMENT/Output/Segmentation_output/MHNet/Camvid_Firstfold',
                        help="save segmentation results to the specified dir")

    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: False)')
    parser.add_argument("--val_batch_size", type=int, default=1,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=240)

    # parser.add_argument("--ckpt",
    #                     default='D:/checkpoint/Segmentation/camvid/torch_deeplabv3plus_secondfold/original/best_deeplabv3plus_resnet50_camvid_sample_os16.pth',
    #                     type=str,
    #                     help="resume from checkpoint")
    parser.add_argument("--ckpt",
        default='D:/checkpoint/Segmentation/camvid/torch_deeplabv3plus/camvid_original/best_deeplabv3plus_resnet50_camvid_os16.pth',
        type=str,
        help="resume from checkpoint")

    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    return parser




def main():
    # global decode_fn
    opts = get_argparser().parse_args()
    # mean = [0.485, 0.456, 0.406],
    # std = [0.229, 0.224, 0.225])
    denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # denormalization for ori images
    if opts.dataset.lower() == 'voc':
        opts.num_classes = 21
        decode_fn = VOCSegmentation.decode_target
    elif opts.dataset.lower() == 'cityscapes':
        opts.num_classes = 19
        decode_fn = Cityscapes.decode_target
    elif opts.dataset.lower() == 'camvid':
        opts.num_classes = 12
        decode_fn = Camvid.decode_target
    elif opts.dataset.lower() == 'camvid_open':
        opts.num_classes = 12
        decode_fn = Camvid.decode_target
    elif opts.dataset.lower() == 'camvid_45':
        opts.num_classes = 12
        decode_fn = Camvid.decode_target
    elif opts.dataset.lower() == 'camvid_deblur45_v2':
        opts.num_classes = 12
        decode_fn = Camvid.decode_target
    elif opts.dataset.lower() == 'camvid_60':
        opts.num_classes = 12
        decode_fn = Camvid.decode_target
    elif opts.dataset.lower() == 'camvid_deblur60_v2':
        opts.num_classes = 12
        decode_fn = Camvid.decode_target
    elif opts.dataset.lower() == 'camvid_sample':
        opts.num_classes = 12
        decode_fn = Camvid.decode_target
    elif opts.dataset.lower() == 'dacon':
        opts.num_classes = 2
        decode_fn = Dacon_segmentation.decode_target
    elif opts.dataset.lower() == 'camvid_edge':
        opts.num_classes = 2
        decode_fn = Camvid_Edge.decode_target
    elif opts.dataset.lower() == 'kitti_edge':
        opts.num_classes = 2
        decode_fn = Kitti_Edge.decode_target
    elif opts.dataset.lower() == 'kitti_sample':
        opts.num_classes = 12
        decode_fn = Kitti_sample.decode_target

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Setup dataloader
    image_files = []
    if os.path.isdir(opts.input):
        for ext in ['png', 'jpeg', 'jpg', 'JPEG']:
            files = glob(os.path.join(opts.input, '**/*.%s' % (ext)), recursive=True)
            if len(files) > 0:
                image_files.extend(files)
    elif os.path.isfile(opts.input):
        image_files.append(opts.input)

    # Set up model (all models are 'constructed at network.modeling)
    # model = network.modeling.__dict__[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
    model_map = {
        'deeplabv3_resnet50': network.deeplabv3_resnet50,
        'deeplabv3plus_resnet50': network.deeplabv3plus_resnet50,
        'deeplabv3_resnet101': network.deeplabv3_resnet101,
        'deeplabv3plus_resnet101': network.deeplabv3plus_resnet101,
        'deeplabv3_mobilenet': network.deeplabv3_mobilenet,
        'deeplabv3plus_mobilenet': network.deeplabv3plus_mobilenet
    }

    model = model_map[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
    # model = UNet_my_2(n_channels=3, n_classes=12, bilinear=True)
    # model = EdgeNet(n_channels=1, n_classes=2, bilinear=True)

    # if opts.separable_conv and 'plus' in opts.model:
    #     network.convert_to_separable_conv(model.classifier)
    # utils.set_bn_momentum(model.backbone, momentum=0.01)

    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        # https://github.com/VainF/DeepLabV3Plus-Pytorch/issues/8#issuecomment-605601402, @PytaichukBohdan
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model)
        model.to(device)
        print("Resume model from %s" % opts.ckpt)
        del checkpoint
    else:
        print("[!] Retrain")
        model = nn.DataParallel(model)
        model.to(device)

    # denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # denormalization for ori images

    if opts.crop_val:
        transform = T.Compose([
            T.Resize(opts.crop_size),
            T.CenterCrop(opts.crop_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])
    else:
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],     ## --> edge 아닐때는 주석 풀기
                        std=[0.229, 0.224, 0.225]),
        ])

        # if opts.crop_val:
        #     transform = T.Compose([
        #         T.Resize(opts.crop_size),
        #         T.CenterCrop(opts.crop_size),
        #         T.ToTensor(),
        #         T.Normalize(mean=[0.485, 0.456, 0.406],
        #                     std=[0.229, 0.224, 0.225]),
        #     ])
        # else:
        #     transform = T.Compose([
        #         T.ToTensor(),
        #         # T.Normalize(mean=[0.485, 0.456, 0.406],
        #         #             std=[0.229, 0.224, 0.225]),
        #     ])

    if opts.save_val_results_to is not None:
        os.makedirs(opts.save_val_results_to, exist_ok=True)
    with torch.no_grad():
        model = model.eval()
        for img_path in tqdm(image_files):
            img_name = os.path.basename(img_path).split('.')[0]
            img = Image.open(img_path).convert('RGB')
            # img = Image.open(img_path)
            img = transform(img).unsqueeze(0)  # To tensor of NCHW
            img = img.to(device)

            pred = model(img).max(1)[1].cpu().numpy()[0]  # HW
            colorized_preds = decode_fn(pred).astype('uint8')
            colorized_preds = Image.fromarray(colorized_preds)
            if opts.save_val_results_to:
                colorized_preds.save(os.path.join(opts.save_val_results_to, img_name + '.png'))


if __name__ == '__main__':
    main()
