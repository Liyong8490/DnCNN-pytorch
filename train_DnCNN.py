import os
import argparse
import scipy.io as sio
import numpy as np
import time
import random
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as func
import torch.nn as nn
from utils import DatasetFromMat
from torch.autograd import Variable
from models_DnCNN import DnCNN
from torch.utils.data import DataLoader

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Training settings
parser = argparse.ArgumentParser(description="PyTorch LHFreqDCNN")
parser.add_argument("--batchSize", type=int, default=128, help="Training batch size")
parser.add_argument("--nEpochs", type=int, default=50, help="Number of epochs to train for")
parser.add_argument("--lr", type=float, default=0.001, help="Learning Rate. Default=0.1")
parser.add_argument("--step", type=int, default=30,
                    help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=10")
parser.add_argument("--cuda", action="store_false", help="Use cuda?")
parser.add_argument("--resume", default="", type=str, help="Path to checkpoint (default: none)")
# parser.add_argument("--resume", default="model/model_ISSR_epoch_80.pth", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--clip", type=float, default=0.005, help="Clipping Gradients. Default=0.4")
parser.add_argument("--threads", type=int, default=1, help="Number of threads for data loader to use, Default: 4")
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")
parser.add_argument("--weight-decay", "--wd", default=1e-3, type=float, help="Weight decay, Default: 1e-4")
parser.add_argument('--dataset', default='data/imdb_40_128_V1.mat', type=str, help='path to general model')
method_name = 'DnCNN_datav1'
sigma = 25
def main():
    global opt, model
    opt = parser.parse_args()
    print(opt)

    cuda = opt.cuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    opt.seed = random.randint(1, 10000)
    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)

    cudnn.benchmark = True

    #print("===> Loading datasets")
    #train_set = DatasetFromMat(opt.dataset, sigma)

    print("===> Building model")
    model = DnCNN(input_chnl=1, groups=1)
    # criterion = nn.MSELoss()
    criterion = nn.MSELoss(size_average=False)

    print("===> Setting GPU")
    if cuda:
        # model = torch.nn.DataParallel(model).cuda()
        model = dataparallel(model, 1)  # set the number of parallel GPUs
        criterion = criterion.cuda()
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            opt.start_epoch = checkpoint["epoch"] + 1
            model.load_state_dict(checkpoint["model"].state_dict())
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    print("===> Setting Optimizer")
    # optimizer = optim.SGD([
    #     {'params': model.parameters()}
    # ], lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)
    optimizer = optim.Adam([
        {'params': model.parameters()}
        ], lr=opt.lr)

    print("===> Training")
    lossAarry = np.zeros(opt.nEpochs)
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        train_set = DatasetFromMat(opt.dataset, sigma)
        training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize,
                                          shuffle=True)
        lossAarry[epoch - 1] = train(training_data_loader, optimizer, model, criterion, epoch)
        save_checkpoint(model, epoch)

    sio.savemat(method_name+'_lossArray.mat', {'lossArray': lossAarry})

def train(training_data_loader, optimizer, model, criterion, epoch):
    lr = adjust_learning_rate(epoch - 1, opt.step)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    print("Epoch={}, low_lr={}".format(epoch, optimizer.param_groups[0]["lr"]))

    start_time = time.time()

    model.train()
    lossValue = 0

    for iteration, batch in enumerate(training_data_loader, 1):
        input, label = Variable(batch[0]), Variable(batch[1], requires_grad=False)
        if opt.cuda:
            input = input.cuda()
            label = label.cuda()
        res = model(input)

        # loss = criterion(res, label)

        # lossfunc = myloss(input.data.shape[0])
        # loss = lossfunc.forward(res, label)

        loss = criterion(res, label)/(input.data.shape[0]*2)

        optimizer.zero_grad()
        loss.backward()
        # nn.utils.clip_grad_norm(model.parameters(), opt.clip)
        optimizer.step()

        lossValue = lossValue + loss.data[0]
        if (iteration+1)%50 == 0:
            elapsed_time = time.time() - start_time
            # save_checkpoint(model, iteration)
            print("===> Epoch[{}]: iteration[{}]: Loss={:.5f}, time = {:.4f}".format(epoch, iteration+1,
                                            # criterion(lres + hres, target).data[0], loss_low.data[0], 0, elapsed_time))
                                            loss.data[0], elapsed_time))

    elapsed_time = time.time() - start_time
    lossValue = lossValue / (iteration + 1)
    print("===> Epoch[{}]: Loss={:.5f}, time = {:.4f}".format(epoch, lossValue, elapsed_time))
    return lossValue

class myloss(nn.Module):
    def __init__(self, N):
        super(myloss, self).__init__()
        self.N = N
        return

    def forward(self, res, label):
        mse = func.mse_loss(res, label, size_average=False)
        loss = mse/(self.N*2)
        return loss

def adjust_learning_rate(epoch, step):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    if epoch < step:
        lr = opt.lr #* (0.1 ** (epoch // opt.step))#0.2
    else:
        lr = opt.lr*0.1
    return lr

def save_checkpoint(model, epoch):
    fold = "model_"+method_name+"/"
    model_out_path = fold + "model_"+method_name+"_epoch_{}.pth".format(epoch)
    state = {"epoch": epoch, "model": model}
    if not os.path.exists(fold):
        os.makedirs(fold)

    torch.save(state, model_out_path)

    print("Checkpoint saved to {}".format(model_out_path))

def dataparallel(model, ngpus, gpu0=0):
    if ngpus==0:
        assert False, "only support gpu mode"
    gpu_list = list(range(gpu0, gpu0+ngpus))
    assert torch.cuda.device_count() >= gpu0 + ngpus
    if ngpus > 1:
        if not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model, gpu_list).cuda()
        else:
            model = model.cuda()
    elif ngpus == 1:
        model = model.cuda()
    return model

if __name__ == "__main__":
    main()
    exit(0)
