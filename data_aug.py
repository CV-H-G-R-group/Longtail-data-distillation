import torch
import torch.nn as nn
import numpy as np
import cv2
import torchvision.transforms.functional as F

from torch.utils.data import Subset


# 随机裁剪 todo
def random_crop(image, crop_size):
    height, width = image.shape[:2]
    crop_height, crop_width = crop_size

    if crop_width >= width or crop_height >= height:
        raise ValueError("Crop size should be smaller than the original image size.")

    start_x = np.random.randint(0, width - crop_width)
    start_y = np.random.randint(0, height - crop_height)

    cropped_image = image[start_y:start_y + crop_height, start_x:start_x + crop_width]

    return cropped_image

# 翻转
def flip(image):
    flipped_image = torch.flip(image, dims=[2])
    return flipped_image

# 旋转
def rotation(image, max_angle=30):
    angle = torch.empty(1).uniform_(-max_angle, max_angle).item()
    rotated_image = F.rotate(image, angle)

    return rotated_image

# 平移 todo
def random_translation(image, max_shift):
    height, width = image.shape[:2]
    shift_x = np.random.randint(-max_shift, max_shift)
    shift_y = np.random.randint(-max_shift, max_shift)
    translation_matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    translated_image = cv2.warpAffine(image, translation_matrix, (width, height))

    return translated_image

# 提取指定label对应的图片
def fliter(dataset_subset, label):
    res = []
    for item in dataset_subset:
        _, l = item
        if l == label:
            res.append(item)

    return res

# 按照label顺序打印数据集中各类的图片数量
def print_shape(res_num, dataset_subset):
    list = []
    for label in range(res_num, res_num*2):
        dataloaer = fliter(dataset_subset, label)
        list.append(len(dataloaer))

    print(list)

class Data_Aug(nn.Module):
    def __init__(self, dataset):
        super(Data_Aug, self).__init__()
        self.dataset = dataset

    def process(self, res_num, img_per_class):
        # print('flag 1')
        # print_shape(res_num, self.dataset)
        res_dataset_list = list(self.dataset)

        index = [i for i in range(res_num, 100)]

        for idx in index:
            dataloader = fliter(self.dataset, idx)
            label = idx
            cur_num = len(dataloader)
            
            if cur_num > img_per_class : continue

            for i in range(5):
                for image, _ in dataloader:
                    if cur_num > img_per_class : continue
                    aug_img = flip(image)
                    new_data = (aug_img, label)
                    res_dataset_list.append(new_data)
                    cur_num = cur_num + 1
                if cur_num > img_per_class : continue

                for image, _ in dataloader:
                    if cur_num > img_per_class : continue
                    aug_img = rotation(image)
                    new_data = (aug_img, label)
                    res_dataset_list.append(new_data)
                    cur_num = cur_num + 1
                if cur_num > img_per_class : continue

        res_dataset = Subset(res_dataset_list, range(len(res_dataset_list)))
        # print('flag 2')
        # print_shape(res_num, res_dataset)

        return res_dataset
        

    
