from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50
from pytorch_grad_cam.utils.model_targets import SemanticSegmentationTarget



from tqdm import tqdm
import network
import utils
import os
import random
import argparse
import numpy as np

from torch.utils import data
from datasets import VOCSegmentation, Cityscapes, Camvid,  Camvid_sample, Camvid_focus, Kitti_sample, Camvid_sample_GradCAM
from utils import ext_transforms as et
from metrics import StreamSegMetrics
from torchsummary import summary
import torch
import torch.nn as nn
from utils.visualizer import Visualizer

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

#python -m visdom.server

#
# class ddddd(DataLoader):
#     def __init__(self, X):
#         self.X = X
#

#     def __len__(self):
#         return len(self.X)
#
#     def __getitem__(self, idx):
#         original = '' + self.X[idx]
#         noise = '' + self.X[idx]
#
#
#         return original, noise,

def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    # parser.add_argument("--data_root", type=str, default='./datasets/data',
    #                     help="path to Dataset")
    # parser.add_argument("--data_root", type=str, default='D:/Dataset/KITTI/segmentation/firstfold/original',
    #                     help="path to Dataset")
    # parser.add_argument("--data_root", type=str,
    #                     default='D:/Dataset/KITTI/segmentation/firstfold/psf_30_firstfold',
    #                     help="path to Dataset")
    # parser.add_argument("--data_root", type=str, default='D:/Dataset/KITTI/segmentation/firstfold/psf_30_deblur/SDAM_perceptual_last_2',
    #                     help="path to Dataset")
    parser.add_argument("--data_root", type=str, default='E:/REVIEW_EXPERIMENT/For_Segmentation/Comparison/Attentive/Camvid_Firstfold',
                        help="path to Dataset")

    # parser.add_argument("--data_root_blur", type=str,
    #                     default='C:/Users/JSIISPR/Desktop/github_deeplab/pytorch_deeplab/DeepLabV3Plus-Pytorch-master/camvid_60',
    #                      help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='camvid_sample',
                        choices=['voc', 'cityscapes, camvid', 'camvid_sample', 'camvid_focus', 'kitti_sample'], help='Name of dataset')
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
    parser.add_argument("--test_only", action='store_true', default=True)
    parser.add_argument("--save_val_results", action='store_true', default=False,
                        help="save segmentation results to \"./results\"")
    parser.add_argument("--total_itrs", type=int, default=27e3,
                        help="epoch number (default: 30k)")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--lr_policy", type=str, default='poly', choices=['poly', 'step'],
                        help="learning rate scheduler policy")
    parser.add_argument("--step_size", type=int, default=10000)
    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: False)')
    parser.add_argument("--batch_size", type=int, default=4,
                        help='batch size (default: 16)')
    parser.add_argument("--val_batch_size", type=int, default=1,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=240) ##513
    # parser.add_argument("--crop_size", type=int, default=176)  ##513

    parser.add_argument("--ckpt", default='D:/checkpoint/Segmentation/camvid/torch_deeplabv3plus/camvid_original/best_deeplabv3plus_resnet50_camvid_os16.pth', type=str,
                        help="restore from checkpoint")

    # parser.add_argument("--ckpt",
    #                     default='D:/checkpoint/Segmentation/kitti/original/firstfold/best_deeplabv3plus_resnet50_kitti_sample_os16.pth',
    #                     type=str,
    #                     help="restore from checkpoint")

    # parser.add_argument("--ckpt",default=None, type=str,
    #                     help="restore from checkpoint")
    parser.add_argument("--continue_training", action='store_true', default=False)

    parser.add_argument("--overlay",  default=True)

    parser.add_argument("--loss_type", type=str, default='cross_entropy',
                        choices=['cross_entropy', 'focal_loss'], help="loss type (default: False)")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--print_interval", type=int, default=10,
                        help="print interval of loss (default: 10)")
    parser.add_argument("--val_interval", type=int, default=80,
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

        train_dst = Camvid_sample_GradCAM(root=opts.data_root, split='train', transform=train_transform)

        val_dst = Camvid_sample_GradCAM(root=opts.data_root, split='val', transform=val_transform)


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

    return train_dst, val_dst





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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Setup random seed
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    # Setup dataloader
    if opts.dataset=='voc' and not opts.crop_val:
        opts.val_batch_size = 1

    train_dst, val_dst = get_dataset(opts)

    train_loader = data.DataLoader(
        train_dst, batch_size=opts.batch_size, shuffle=True, num_workers=2)

    val_loader = data.DataLoader(
        val_dst, batch_size=opts.val_batch_size, shuffle=True, num_workers=2)


    print("Dataset: %s, Train set: %d, Val set: %d" %
          (opts.dataset, len(train_dst), len(val_dst)))

    # Set up model
    model_map = {
        'deeplabv3_resnet50': network.deeplabv3_resnet50,
        'deeplabv3plus_resnet50': network.deeplabv3plus_resnet50,
        'deeplabv3_resnet101': network.deeplabv3_resnet101,
        'deeplabv3plus_resnet101': network.deeplabv3plus_resnet101,
        'deeplabv3_mobilenet': network.deeplabv3_mobilenet,
        'deeplabv3plus_mobilenet': network.deeplabv3plus_mobilenet
    }

    model = model_map[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)


    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)

    # Set up metrics
    metrics = StreamSegMetrics(opts.num_classes)

    # Set up optimizer
    optimizer = torch.optim.SGD(params=[
        {'params': model.backbone.parameters(), 'lr': 0.1*opts.lr},
        {'params': model.classifier.parameters(), 'lr': opts.lr},
    ], lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    #optimizer = torch.optim.SGD(params=model.parameters(), lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    #torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.lr_decay_step, gamma=opts.lr_decay_factor)
    if opts.lr_policy=='poly':
        scheduler = utils.PolyLR(optimizer, opts.total_itrs, power=0.9)
    elif opts.lr_policy=='step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.1)

    # Set up criterion
    #criterion = utils.get_loss(opts.loss_type)
    if opts.loss_type == 'focal_loss':
        criterion = utils.FocalLoss(ignore_index=255, size_average=True)
    elif opts.loss_type == 'cross_entropy':
        criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')

    def save_ckpt(path):
        """ save current model
        """
        torch.save({
            "cur_itrs": cur_itrs,
            "model_state": model.module.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_score": best_score,
        }, path)
        print("Model saved as %s" % path)

    utils.mkdir('checkpoints')
    # Restore
    best_score = 0.0
    cur_itrs = 0
    cur_epochs = 0

    # if opts.ckpt is not None and os.path.isfile(opts.ckpt):
    if opts.ckpt is not None and opts.test_only:
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
        print("[!] Retrain")
        model = nn.DataParallel(model)
        model.to(device)

    #==========   Train Loop   ==========#
    vis_sample_id = np.random.randint(0, len(val_loader), opts.vis_num_samples,
                                      np.int32) if opts.enable_vis else None  # sample idxs for visualization
    denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # denormalization for ori images

    if opts.test_only:
        model.eval()

        # print(model)
        # for idx, (images, labels, original_image) in enumerate(val_loader):
        for idx, images in enumerate(val_loader):
            images = images[1].to(device, dtype=torch.float32)
            # labels = labels.to(device, dtype=torch.long)
            original_image = images

            output = model(images)
            normalized_masks = torch.nn.functional.softmax(output, dim=1).cpu()
            ##kitti class
            # sem_classes = [
            #     'sky', 'building', 'road', 'sidewalk', 'fence', 'tree', 'pole',
            #     'car', 'sign', 'pedestrian', 'bicyclist', 'unlabelled'
            # ]

            ##camvid class
            sem_classes = [
                'sky', 'building', 'pole', 'road', 'sidewalk', 'tree', 'signsymbol',
                'fence', 'car', 'pedestrian', 'bicyclist', 'unlabelled'
            ]

##-> kitti class 에 대한 것임, camvid class 는 다름 (윗줄)
            sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(sem_classes)}
            car_category = sem_class_to_idx["fence"]
            car_mask = normalized_masks[0, :, :, :].argmax(axis=0).detach().cpu().numpy()
            car_mask_uint8 = 255 * np.uint8(car_mask == car_category)
            car_mask_float = np.float32(car_mask == car_category)

            # both_images = np.hstack((images.detach().cpu().numpy(), np.repeat(car_mask_uint8[:, :, None], 3, axis=-1)))
            # Image.fromarray(both_images)


            # target_layers = [model.module.classifier.aspp.project[2]]  ## aspp block 지나고 1x1
            # target_layers = [model.module.classifier.aspp.convs[1][2]]   ## aspp block 1
            # target_layers = [model.module.classifier.aspp.convs[2][2]]    ## aspp blcok 2
            target_layers = [model.module.classifier.aspp.convs[3][2]]    ## aspp block 3
            targets = [SemanticSegmentationTarget(car_category, car_mask_float)]

            with GradCAM(model=model,
                         target_layers=target_layers,
                         use_cuda=torch.cuda.is_available()) as cam:
                grayscale_cam = cam(input_tensor=images,
                                    targets=targets)[0, :]
                # cam_image = show_cam_on_image(original_image[0].cpu().numpy()/255., grayscale_cam, use_rgb=True)
                cam_image = show_cam_on_image(original_image[0].permute(1, 2, 0).cpu().numpy() / 255., grayscale_cam, use_rgb=True)

            save_img = Image.fromarray(cam_image)
            save_img.save(f'E:/REVIEW_EXPERIMENT/Output/GradCAM/fence/MHNet/Camvid_Firstfold/{idx}.png')

        return

    else:
        interval_loss = 0
        while True: #cur_itrs < opts.total_itrs:
            # =====  Train  =====
            # PATH_1 = 'C:/checkpoint_void/pytorch/segmentation/camvid_original_model/model.pt'
            # model = torch.load(PATH_1)
            # print(model)
            model.train()
            cur_epochs += 1


            return


if __name__ == '__main__':
    main()








