import random
import numpy as np
import numbers
import torchvision.transforms as ts
import torchvision.transforms.functional as TF
from PIL import Image


def randomflip_rotate(img, lab, p=0.5, degrees=0):
    if random.random() < p:    # random.random() 生成一个0-1的随机数，比p小则进行水平旋转
        img = TF.hflip(img)
        lab = TF.hflip(lab)
    if random.random() < p:
        img = TF.vflip(img)    # # random.random() 生成一个0-1的随机数，比p小则进行垂直旋转
        lab = TF.vflip(lab)

    if isinstance(degrees, numbers.Number):
        if degrees < 0:
            raise ValueError("If degrees is a single number, it must be positive.")
        degrees = (-degrees, degrees)
    else:
        if len(degrees) != 2:
            raise ValueError("If degrees is a sequence, it must be of len 2.")
        degrees = degrees
    angle = random.uniform(degrees[0], degrees[1])    # 从-degrees到+degrees均匀分布取一个值
    img = TF.rotate(img, angle)
    lab = TF.rotate(lab, angle)

    return img, lab


class randomcrop(object):
    """Crop the given PIL Image and mask at a random location.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is 0, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively.
        pad_if_needed (boolean): It will pad the image if smaller than the
            desired size to avoid raising an exception.
    """

    def __init__(self, size, padding=0, pad_if_needed=False):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img, lab):
        """
        Args:
            img (PIL Image): Image to be cropped.
            lab (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped image and mask.
        """
        if self.padding > 0:
            img = TF.pad(img, self.padding)
            lab = TF.pad(lab, self.padding)

        # pad the width if needed
        if self.pad_if_needed and img.size[0] < self.size[1]:
            img = TF.pad(img, (int((1 + self.size[1] - img.size[0]) / 2), 0))
            lab = TF.pad(lab, (int((1 + self.size[1] - lab.size[0]) / 2), 0))
        # pad the height if needed
        if self.pad_if_needed and img.size[1] < self.size[0]:
            img = TF.pad(img, (0, int((1 + self.size[0] - img.size[1]) / 2)))
            lab = TF.pad(lab, (0, int((1 + self.size[0] - lab.size[1]) / 2)))

        i, j, h, w = self.get_params(img, self.size)

        return TF.crop(img, i, j, h, w), TF.crop(lab, i, j, h, w)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)


def dataset_transform(sample, train_type):
    image, label = Image.fromarray(np.uint8(sample['image']), mode='RGB'), \
                   Image.fromarray(np.uint8(sample['label']), mode='L')  # “L” (灰度图), “RGB” (彩色图)

    if train_type == 'train':
        image, label = randomcrop(size=(224, 224))(image, label)
        image, label = randomflip_rotate(image, label, p=0.5, degrees=30)
    else:
        # image, label = randomcrop(size=(224, 224))(image, label)
        pass

    image = ts.Compose([ts.ToTensor(),
                        ts.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])(image)  # 将数据归一化到[-1,1]
    label = ts.ToTensor()(label)

    return {'image': image, 'label': label}