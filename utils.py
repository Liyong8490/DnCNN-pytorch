import numpy as np
import scipy.misc
import scipy.io as sio
import h5py
import os
import glob
import PIL
import pywt
import torch
import torch.utils.data as data
from PIL import Image
import skimage.measure as measure

class DatasetFromMat(data.Dataset):
    def __init__(self, file_path, sigma):
        super(DatasetFromMat, self).__init__()
        self.sigma = sigma
        odata = sio.loadmat(file_path)
        self.data = np.transpose(odata['inputs'], [3, 2, 0, 1])
        # data augmentation
        rnd_aug = np.random.randint(8,size=self.data.shape[0])
        shape = self.data.shape
        for i in range(shape[0]):
            self.data[i,:,:,:] = np.reshape(data_aug(
                np.reshape(self.data[i,:,:,:], [shape[2],shape[3]]),
                rnd_aug[i]),[1,shape[1],shape[2],shape[3]])
        self.label = sigma / 255.0 * np.random.normal(size=np.shape(self.data))
        self.input = self.data + self.label
    def __getitem__(self, index):
        return torch.from_numpy(self.input[index, :, :, :]).float(),\
                torch.from_numpy(self.label[index, :, :, :]).float()

    def __len__(self):
        # return self.data.Fileshape[0]
        return self.data.shape[0]

def data_aug(image, mode=0):

    if mode == 0:
        # original
        return image
    elif mode == 1:
        # flip up and down
        return np.flipud(image)
    elif mode == 2:
        # rotate counterwise 90 degree
        return np.rot90(image)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        return np.flipud(np.rot90(image))
    elif mode == 4:
        # rotate 180 degree
        return np.rot90(image, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        return np.flipud(np.rot90(image, k=2))
    elif mode == 6:
        # rotate 270 degree
        return np.rot90(image, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        return np.flipud(np.rot90(image, k=3))

def c_psnr(im1, im2):
    # mse = np.power(im1 - im2, 2).mean()
    # psnr = 20 * np.log10(255.0) - 10 * np.log10(mse)
    im1 = np.maximum(np.minimum(im1, 1.0), 0.0)
    im2 = np.maximum(np.minimum(im2, 1.0), 0.0)
    return measure.compare_psnr(im1, im2)

def c_ssim(im1, im2):

    im1 = np.maximum(np.minimum(im1, 1.0), 0.0)
    im2 = np.maximum(np.minimum(im2, 1.0), 0.0)
    return measure.compare_ssim(im1, im2, win_size=11, data_range=1, gaussian_weights=True)

