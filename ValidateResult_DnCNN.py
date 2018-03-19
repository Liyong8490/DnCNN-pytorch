# -*- coding: utf-8 -*-
import argparse
import torch
from torch.autograd import Variable
from scipy.ndimage import imread
from PIL import Image
import numpy as np
import time
import math
import matplotlib.pyplot as plt
import warnings
import os
import scipy.io as sio
import scipy.misc
from utils import c_psnr, c_ssim

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="PyTorch SANet Test")
parser.add_argument("--cuda", action="store_false", help="use cuda?")
parser.add_argument("--model", default=50, type=int, help="model path")
parser.add_argument("--model_s", default=1, type=int, help="model path")
# parser.add_argument("--model", default="model/model_ISSR_epoch_15.pth", type=str, help="model path")
parser.add_argument("--image", default="woman_GT", type=str, help="image name")
parser.add_argument("--scale", default=25, type=int, help="scale factor, Default: 4")
parser.add_argument('--dataset', default='../denoising_data/_noisyset/', type=str, help='path to general model')

set_name = 'Set12'
model_name = 'DnCNN_waug_wo_clipgrad_lossmod_yan'

def denoising(path, imgName, model):

    mat = scipy.io.loadmat(path + "/" + imgName + ".mat")
    original = mat['img']
    input_ = mat['noisyimg']
    original = original.astype(np.float32)
    input_ = input_.astype(np.float32)
    input_ = input_ / 255.0
    label_ = original / 255.0

    im_input = Variable(torch.from_numpy(input_).float().contiguous(), volatile=True).view(1, -1, input_.shape[0], input_.shape[1])

    if cuda:
        model = model.cuda()
        im_input = im_input.cuda()

    # start_time = time.time()
    out = model(im_input)
    # elapsed_time = time.time() - start_time

    out = out.cpu()

    result = out.data[0].numpy().astype(np.float32).reshape(input_.shape[0], input_.shape[1])
    result = input_ - result
    result[result < 0] = 0
    result[result > 1.] = 1.

    psnr_predicted = c_psnr(result, label_)
    ssim_predicted = c_ssim(result, label_)
    # im_h.save("./result/" + imgName + "_" + mName +".bmp")

    return psnr_predicted, ssim_predicted, result


opt = parser.parse_args()
cuda = opt.cuda
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

psnr_summary = []
ssim_summary = []

for index in range(opt.model_s, opt.model + 1):
# for index in range(1, 30):
    modelName = "model_"+model_name+"/model_"+model_name+"_epoch_" + str(index) + ".pth"
    model = torch.load(modelName)["model"]
    model.eval()

    fpath = opt.dataset
    path = fpath + str(opt.scale) + '/' + set_name  # file folder
    files = os.listdir(path)
    PSNRs_SANet = []
    SSIMs_SANet = []
    if not os.path.exists('./results_'+model_name+'/'+set_name+'/'+str(index)):
        os.makedirs('./results_'+model_name+'/'+set_name+'/'+str(index))
    for file in files:
        if not os.path.isdir(file):
             imageName = os.path.splitext(file)[0]
             # print("==========   SANet on " + imageName + " ==============")
             psnr, ssim, recon = denoising(path, imageName, model)
             PSNRs_SANet.append(psnr)
             SSIMs_SANet.append(ssim)
             scipy.misc.imsave('./results_'+model_name+'/'+set_name+'/'+str(index)+'/'+imageName+'.png', recon)

    # print("========== model {} Average results ==============".format(index))
    print("model {}   PSNR={}   SSIM={}".format(index, np.mean(PSNRs_SANet), np.mean(SSIMs_SANet)))
    psnr_summary.append(np.array(PSNRs_SANet))
    ssim_summary.append(np.array(SSIMs_SANet))
sio.savemat(model_name+'_'+set_name+'_resArray.mat', {'psnr': np.array(psnr_summary), 'ssim':np.array(ssim_summary)})
