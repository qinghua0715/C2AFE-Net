import os
import imageio.v2 as imageio
import PIL
import torch
import numpy as np
# import nibabel as nib
import matplotlib.pyplot as plt

from os import listdir
from os.path import join
from PIL import Image
# from utils.transform import itensity_normalize
from torch.utils.data.dataset import Dataset


class ISIC2017_dataset(Dataset):
    def __init__(self, path_list, train_type='train', transform=None):
        # 根据传入的路径列表和索引生成文件地址列表
        self.image_list, self.label_list = path_list
        self.transform = transform
        self.train_type = train_type
        # 生成图像文件路径列表
        if self.train_type in ['train', 'val', 'test']:
            self.image_path_list = self.image_list
            self.label_path_list = self.label_list
        else:
            print("Choosing type error, You have to choose the loading data type including: train, validation, test")
        assert len(self.image_path_list) == len(self.label_path_list)

    def __getitem__(self, item: int):
        image_name = self.image_path_list[item]
        label_name = self.label_path_list[item]
        if image_name.split('.')[-1] == 'npy':
            # 加载.npy格式的图片
            image = np.load(image_name)
        elif image_name.split('.')[-1] == 'png':
            # 加载.png格式的数据
            image = imageio.imread(image_name)

        if label_name.split('.')[-1] == 'npy':
            # 加载.npy格式的图片
            label = np.load(label_name)
        elif label_name.split('.')[-1] == 'png':
            # 加载.png格式的数据
            label = imageio.imread(label_name)

        name = image_name.split('/')[-1]

        sample = {'image': image, 'label': label}

        if self.transform is not None:
            # TODO: transformation to argument datasets
            sample = self.transform(sample, self.train_type)

        if self.train_type in ['test']:
            return name, sample['image'], sample['label']
        else:
            return sample['image'], sample['label']

    def __len__(self):
        return len(self.image_path_list)


if __name__ == '__main__':
    print(os.listdir('../segment_result/gold_standard'))