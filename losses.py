import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss

class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(),k).unsqueeze(0).repeat(3,1,1,1)
        if torch.cuda.is_available():
            self.kernel = self.kernel.cuda()
        self.loss = CharbonnierLoss()

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered    = self.conv_gauss(current)    # filter
        down        = filtered[:,:,::2,::2]               # downsample
        new_filter  = torch.zeros_like(filtered)
        new_filter[:,:,::2,::2] = down*4                  # upsample
        filtered    = self.conv_gauss(new_filter) # filter
        diff = current - filtered
        return diff

    def forward(self, x, y):
        loss = self.loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
        return loss




class PerceptualLoss():

    def initialize(self):
        self.activation = {}
        # PATH = 'D:/checkpoint/Segmentation/camvid/torch_deeplabv3plus/camvid_original_model/model.pt'
        # PATH = 'D:/checkpoint/Segmentation/kitti/original/firstfold_model/model.pt'
        PATH = 'D:/checkpoint/Segmentation/kitti/original/secondfold_model/model.pt'
        self.model = torch.load(PATH)
        print(self.model)
        with torch.no_grad():
            # self.criterion = loss
            self.contentFunc = self.contentFunc()
            self.transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # for name, param in self.contentFunc.named_parameters():
        #     print(f'param.requries_grad:{param.requires_grad}')
        # print(self.model)
        # for a in self.model['model_state']:
        #     print('param.requries_grad:'+str(self.model['model_state'][a].requires_grad))


    def contentFunc(self):

        # conv_3_3_layer = 14
        # cnn = models.vgg19(pretrained=True).features
        # cnn = cnn.cuda()
        # model = nn.Sequential()
        # model = model.cuda()
        # model = model.eval()
        # for i, layer in enumerate(list(cnn)):
        #     model.add_module(str(i), layer)
        #     if i == conv_3_3_layer:
        #         break

        return self.model
        # return model
## ----------------------------------------------------------------------------------
    # def contentFunc(self):
    #
    #     conv_3_3_layer = 14
    #     cnn = models.vgg19(pretrained=True).features
    #     cnn = cnn.cuda()
    #     model = nn.Sequential()
    #     model = model.cuda()
    #     model = model.eval()
    #     for i, layer in enumerate(list(cnn)):
    #         model.add_module(str(i), layer)
    #         if i == conv_3_3_layer:
    #             break
    #     return model


    # def initialize(self, loss):
    #     # self.activation = {}
    #     with torch.no_grad():
    #         self.criterion = loss
    #         self.contentFunc = self.contentFunc()
    #         self.transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

## ----------------------------------------------------------------------------------

    def get_activation(self,name):
        def hook(model, input, output):
            for nm in name:
                self.activation[nm] = output
        return hook

##------------------------------------------------------------------ 원래 perceptual loss
    # def get_loss(self, fakeIm, realIm):
        # fakeIm = (fakeIm + 1) / 2.0
        # realIm = (realIm + 1) / 2.0
        # fakeIm[0, :, :, :] = self.transform(fakeIm[0, :, :, :])
        # realIm[0, :, :, :] = self.transform(realIm[0, :, :, :])
        # f_fake = self.contentFunc.forward(fakeIm)
        # f_real = self.contentFunc.forward(realIm)
        # f_real_no_grad = f_real.detach()
        # loss = self.criterion(f_fake, f_real_no_grad)
        # return 0.006 * torch.mean(loss) + 0.5 * nn.MSELoss()(fakeIm, realIm)
# -----------------------------------------------------------------
    def get_loss(self, fakeIm, realIm):
        fakeIm = (fakeIm + 1) / 2.0    ## retored Image
        realIm = (realIm + 1) / 2.0
        fakeIm[0, :, :, :] = self.transform(fakeIm[0, :, :, :])
        realIm[0, :, :, :] = self.transform(realIm[0, :, :, :])
        # f_fake = self.contentFunc.forward(fakeIm)
        # f_real = self.contentFunc.forward(realIm)

        ## Feature 추출 하는 부분--------------------------------------------------------------------------------------------------------------
        # deeplab 에서 backbone lowlevel feature 와 aspp 지나고 1x1 conv 한 feature
        self.model.module.classifier.project[2].register_forward_hook(self.get_activation(['low_feature_fake']))
        self.model.module.classifier.aspp.project[2].register_forward_hook(self.get_activation(['high_feature_fake']))
        ##--------------------------------------------------------------------------------------------------------------
        # self.contentFunc.module.classifier.aspp.convs[1].register_forward_hook(self.get_activation(['low_feature_fake']))  ## new featuremap aspp conv단
        # self.contentFunc.module.classifier.aspp.convs[3].register_forward_hook(self.get_activation(['high_feature_fake']))
        ##--------------------------------------------------------------------------------------------------------------
        ## 1x1 conv 단에서 뽑은 feature
        # self.contentFunc.module.classifier.aspp.project[2].register_forward_hook(self.get_activation(['restored']))

        ##--------------------------------------------------------------------------------------------------------------

        ## no_grad 를 한 이유는 deeplab 을 update 를 안해주려 한 것이고, detach 를 안 한 이유는 mprnet 에서는 update 를 해주려 한 것이고,
        ## 밑에서 real 에 detach 를 한 것은 real 은 mprnet에서도 update 를 안해주려 한 것이다.
        self.contentFunc.eval()
        with torch.no_grad():
            f_fake = self.contentFunc.forward(fakeIm)

        # print(self.activation)



        ##이름 지어주는 부분--------------------------------------------------------------------------------------------------------------

        low_level_feature = self.activation['low_feature_fake']
        high_level_feature = self.activation['high_feature_fake']
        high_level_feature = F.interpolate(high_level_feature, size=low_level_feature.shape[2:], mode='bilinear',
                                           align_corners=False)
        fake_fusion_layer = torch.cat([low_level_feature, high_level_feature ], dim=1)

        ##--------------------------------------------------------------------------------------------------------------

        # low_level_feature_fake = self.activation['low_feature_fake']
        # high_level_feature_fake = self.activation['high_feature_fake']

        ##--------------------------------------------------------------------------------------------------------------

        # feature_restored = self.activation['restored']

        ##--------------------------------------------------------------------------------------------------------------
        ##--------------------------------------------------------------------------------------------------------------
        ##--------------------------------------------------------------------------------------------------------------





        ##------지금부턴 real image feature

        self.model.module.classifier.project[2].register_forward_hook(self.get_activation(['low_feature_live']))
        self.model.module.classifier.aspp.project[2].register_forward_hook(self.get_activation(['high_feature_live']))
        ##--------------------------------------------------------------------------------------------------------------
        # self.model.module.classifier.aspp.convs[1].register_forward_hook(self.get_activation(['low_feature_live']))
        # self.model.module.classifier.aspp.convs[3].register_forward_hook(self.get_activation(['high_feature_live']))
        ##--------------------------------------------------------------------------------------------------------------
        # self.contentFunc.module.classifier.aspp.project[2].register_forward_hook(self.get_activation(['real'])) ## 1x1 conv 단에서 뽑은 real image feature

        ##--------------------------------------------------------------------------------------------------------------
        self.contentFunc.eval()
        with torch.no_grad():
            f_real = self.contentFunc.forward(realIm)

        # print(f_real.requires_grad)
            # for name, param in f_real.named_parameters():
            #     print(f'param.requries_grad:{param.requires_grad}')

        ##--------------------------------------------------------------------------------------------------------------
        low_level_feature = self.activation['low_feature_live']
        high_level_feature = self.activation['high_feature_live']
        high_level_feature = F.interpolate(high_level_feature, size=low_level_feature.shape[2:], mode='bilinear',
                                       align_corners=False)
        live_fusion_layer = torch.cat([low_level_feature, high_level_feature], dim=1)

        # low_level_feature_real = self.activation['low_feature_live']
        # high_level_feature_real = self.activation['high_feature_live']

        # feature_real = self.activation['real']

        ##--------------------------------------------------------------------------------------------------------------
        # f_real_no_grad = f_real.detach()
        # loss = self.criterion(f_fake, f_real_no_grad)
        ##--------------------------------------------------------------------------------------------------------------
        f_real_no_grad = live_fusion_layer.detach()
        loss_perceptual = nn.MSELoss()(fake_fusion_layer, f_real_no_grad)  ## 이전 수정한것 ( conccat 된 부분 )
        # loss_perceptual = self.criterion(fake_fusion_layer, f_real_no_grad) ## 이전 수정한것 ( conccat 된 부분 )

        # f_real_no_grad_low = low_level_feature_real.detach()
        # f_real_no_grad_high = high_level_feature_real.detach()
        # loss_low = self.criterion(low_level_feature_fake, f_real_no_grad_low)
        # loss_high = self.criterion(high_level_feature_fake, f_real_no_grad_high)

        # feature_real_no_grad = feature_real.detach()
        # loss_perceptual = nn.MSELoss()(feature_restored, feature_real_no_grad)

        ##--------------------------------------------------------------------------------------------------------------
        # print("loss_perceptual:{}".format(loss_perceptual))

        # print(torch.mean(loss_low) + torch.mean(loss_high) + 0.5 * nn.MSELoss()(fakeIm, realIm))

        # return 0.006 * torch.mean(loss) + 0.5 * nn.MSELoss()(fakeIm, realIm)
        # return torch.mean(loss_low) + torch.mean(loss_high) + 0.5 * nn.MSELoss()(fakeIm, realIm)
        return 0.05 * torch.mean(loss_perceptual)

##------------------
    # def charbonnierloss(selfself, fakeIm, realIm):
## -------------------

    def __call__(self, fakeIm, realIm):
        return self.get_loss(fakeIm, realIm)

class PSNRLoss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            pass
        assert len(pred.size()) == 4

        return self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()
