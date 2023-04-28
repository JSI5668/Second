import os
import glob

import PIL.Image
from PIL import Image
import re
import numpy as np

path = 'E:/sujin_marindata/all'
# path = 'D:/Dataset/Camvid/sum_two_onefold'
# files = glob.glob('C:\Users\JSIISPR\Desktop\github_deeplab/amazing/Amazing-Semantic-Segmentation-master/camvid_blurred45_twofold_gray/train/images/*.png')


imgNames = os.listdir(path)
unique = 0
counts = 0
sum_count = np.zeros(shape=(255))
for name in imgNames:
    try:
        # name -> 이미지 이름
        img_path = f'{path}/{name}'
        img = Image.open(img_path)
        image = np.array(img)
        unique, counts = np.unique(image, return_counts=True)
        _count = np.zeros(shape=(255))

        idx_C=0
        for idx in unique:
            _count[idx] = counts[idx_C]
            idx_C+=1

        sum_count+=_count

        # print(dict(zip(unique, counts)))


    except:
        pass
print(sum_count)
a = np.sum(sum_count)
print(sum_count/a)

# print(a)
# print(sum_count[11])
# ## 전체 픽셀에서 void 픽셀 갯수 빼기
# residual = a - sum_count[11]
# print(residual)

## void 포함 ratio
# for i in range (12):
#     b = sum_count[i] / a
#     print(b)

## void 포함하지 않는 ratio
# for i in range (11):
#     b = sum_count[i] / residual
#     print(b)

# for i in range(350):
    # for f in files:
    #     # name = os.listdir(path)
    #     img = Image.open(f)
    #     img_resize = img.resize((320, 240))
    #     title, ext = os.path.splitext(f)
    #
    #     name = f.split('/', aixs=-1)
    #     img_resize.save(
    #         'C:/Users/JSIISPR/Desktop/github_deeplab/Deblur_dataset/fold_A/fold_A_1/{name}'.format(name=name[i]))
    #     print(name[i])
    #     a = name[i]