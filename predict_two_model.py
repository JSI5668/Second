from torch.utils.data import dataset
from tqdm import tqdm
import network
import utils
import os
import random
import argparse
import numpy as np

from torch.utils import data
from datasets import VOCSegmentation, Cityscapes, cityscapes, Camvid, Camvid_sample, Kitti_sample
from torchvision import transforms as T
from metrics import StreamSegMetrics
from main import validate
from main import get_dataset
from network.unet import UNet
import torch
import torch.nn as nn
from network.unet import UNet_attnetion_encoder_concat

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from glob import glob
import cv2


def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--input", type=str, default='D:/Dataset/Camvid/camvid_original_1',  # required=True
                        help="path to a single image or image directory")
    # parser.add_argument("--dataset", type=str, default='camvid',
    #                     choices=['voc', 'cityscapes', 'camvid'], help='Name of training set')
    parser.add_argument("--dataset", type=str, default='camvid_sample',
                        choices=['voc', 'cityscapes, camvid', 'camvid_open', 'camvid_45', 'camvid_60',
                                 'camvid_deblur45_v2', 'camvid_deblur60_v2', 'camvid_sample', 'kitti'],
                        help='Name of dataset')
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
                        default='E:/Second_paper/Result_Image/Segmentation/Camvid/Original_Firstfold/main_6_attention_encoder_concat_inputnoattention',
                        help="save segmentation results to the specified dir")

    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: False)')
    parser.add_argument("--val_batch_size", type=int, default=1,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=240)
    # parser.add_argument("--crop_size", type=int, default=176)

    parser.add_argument("--ckpt",
                        default='D:/Second_Paper/Checkpoint/Camvid/Segmentation/Original_using_binary/Subcom_Firstfold_UNet_BCE_Focal_alpha0.25_r2_lr0.001_500epoch_BlackWhite_/best_deeplabv3plus_resnet50_camvid_proposed_os16.pth',
                        type=str,
                        help="resume from checkpoint")

    parser.add_argument("--ckpt_2",
                        default='E:/Second_paper/Checkpoint/Camvid/Original_firstfold/main_6_attention_encoder_1x1conv/best_deeplabv3plus_resnet50_camvid_sample_os16.pth',
                        type=str,
                        help="resume from checkpoint")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    return parser


m_32 = nn.Conv2d(3, 2, kernel_size=1).cuda()


def main():
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
        decode_fn = Camvid_sample.decode_target
    elif opts.dataset.lower() == 'camvid_sample':
        opts.num_classes = 12
        decode_fn = Camvid_sample.decode_target
    elif opts.dataset.lower() == 'kitti':
        opts.num_classes = 12
        decode_fn = Kitti_sample.decode_target

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Setup dataloader
    image_files = []
    if os.path.isdir(opts.input):
        for ext in ['png', 'jpeg', 'jpg', 'JPEG']:
            files = glob(os.path.join(opts.input, '**/*.png'), recursive=True)
            if len(files) > 0:
                image_files.extend(files)
    elif os.path.isfile(opts.input):
        image_files.append(opts.input)

    # Set up model
    model_map = {
        'deeplabv3_resnet50': network.deeplabv3_resnet50,
        'deeplabv3plus_resnet50': network.deeplabv3plus_resnet50,
        'deeplabv3_resnet101': network.deeplabv3_resnet101,
        'deeplabv3plus_resnet101': network.deeplabv3plus_resnet101,
        'deeplabv3_mobilenet': network.deeplabv3_mobilenet,
        'deeplabv3plus_mobilenet': network.deeplabv3plus_mobilenet
    }

    model_pre = UNet(n_channels=3, n_classes=2, bilinear=True)
    checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
    model_pre.load_state_dict(checkpoint["model_state"])
    model_pre.cuda()
    model_pre.eval()

    # model = model_map[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
    # if opts.separable_conv and 'plus' in opts.model:
    #     network.convert_to_separable_conv(model.classifier)
    # utils.set_bn_momentum(model.backbone, momentum=0.01)

    model = UNet_attnetion_encoder_concat(n_channels=3, n_classes=12, bilinear=True)

    if opts.ckpt_2 is not None and os.path.isfile(opts.ckpt_2):
        # https://github.com/VainF/DeepLabV3Plus-Pytorch/issues/8#issuecomment-605601402, @PytaichukBohdan
        checkpoint = torch.load(opts.ckpt_2, map_location=torch.device('cpu'))
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
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])
    if opts.save_val_results_to is not None:
        os.makedirs(opts.save_val_results_to, exist_ok=True)
    with torch.no_grad():
        model_pre = model_pre.eval()
        model = model.eval()
        for img_path in tqdm(image_files):
            img_name = os.path.basename(img_path).split('.')[0]
            img = Image.open(img_path).convert('RGB')
            img = transform(img).unsqueeze(0)  # To tensor of NCHW
            img = img.to(device)

            outputs_pre = model_pre(img)
            outputs_pre = nn.Sigmoid()(outputs_pre)

            # images_attention_map_3 = outputs_pre[:, 1, :, :].unsqueeze(dim=1) * img
            # images_attention_map_4 = outputs_pre[:, 0, :, :].unsqueeze(dim=1) * img
            #
            # img = images_attention_map_3 + 0.3 * images_attention_map_4

            pred = model(img, outputs_pre).max(1)[1].cpu().numpy()[0]  # HW

            colorized_preds = decode_fn(pred).astype('uint8')
            colorized_preds = Image.fromarray(colorized_preds)

            if opts.save_val_results_to:
                colorized_preds.save(os.path.join(opts.save_val_results_to, img_name + '.png'))


if __name__ == '__main__':
    main()
