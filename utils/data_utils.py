from os import listdir
from os.path import join
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, \
    RandomCrop, ToTensor, CenterCrop, Grayscale
import math
import torch
import numpy as np


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', 'bmp', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


def calculate_valid_crop_size(crop_size, blocksize):
    return crop_size - (crop_size % blocksize)


def train_hr_transform(crop_size):
    return Compose([
        RandomCrop(crop_size),
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.5),
        Grayscale(),
        ToTensor(),
    ])


def psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


class TrainDataset(Dataset):
    def __init__(self, dataset_dir, crop_size, blocksize):
        super(TrainDataset, self).__init__()
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]
        crop_size = calculate_valid_crop_size(crop_size, blocksize)
        self.hr_transform = train_hr_transform(crop_size)

    def __getitem__(self, index):
        index = index % 400
        try:
            hr_image = self.hr_transform(Image.open(self.image_filenames[index]))
            return hr_image, hr_image
        except:
            hr_image = self.hr_transform(Image.open(self.image_filenames[index + 1]))
            return hr_image, hr_image

    def __len__(self):
        return 89600


class TestDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, blocksize):
        super(TestDatasetFromFolder, self).__init__()
        self.blocksize = blocksize
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]

    def __getitem__(self, index):
        hr_image = Image.open(self.image_filenames[index])

        w, h = hr_image.size
        w = int(np.floor(w / self.blocksize) * self.blocksize)
        h = int(np.floor(h / self.blocksize) * self.blocksize)
        crop_size = (h, w)

        hr_image = CenterCrop(crop_size)(hr_image)
        hr_image = Grayscale()(hr_image)

        return ToTensor()(hr_image), ToTensor()(hr_image)

    def __len__(self):
        return len(self.image_filenames)


def numpy_to_img(data):
    data = data * 255.
    data[data < 0] = 0
    data[data > 255.] = 255.
    data = data[0, :, :]

    return data
