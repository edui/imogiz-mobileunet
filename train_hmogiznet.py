import argparse
import logging
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from tensorboardX import SummaryWriter
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, RandomResizedCrop, RandomRotation, RandomHorizontalFlip, ToTensor, \
    Resize, RandomAffine, ColorJitter, Normalize
from torch.autograd import Variable

from dataset import MogizDataset, get_img_files
from loss import dice_loss
from nets.MobileNetV2_unet import MobileNetV2_unet
from torch.utils.mobile_optimizer import optimize_for_mobile

np.random.seed(1)
torch.backends.cudnn.deterministic = True
torch.manual_seed(1)

# %%
N_CV = 5
BATCH_SIZE = 8
BATCH_SIZE = 16
LR = 1e-3

N_EPOCHS = 1000
IMG_SIZE = 224
#IMG_SIZE = 128
RANDOM_STATE = 1

EXPERIMENT = 'train_mogiznet'
OUT_DIR = 'outputs/{}'.format(EXPERIMENT)

ds_dir = '/content/drive/MyDrive/Research/mogiz/Dataset/resized_128/'
ds_dir = '/content/drive/MyDrive/Research/mogiz/Dataset/square_224/'
ds_dir = 'data/square_224/'

train_ds_name = 'TRAINING.csv'
val_ds_name = 'TRAINING.csv'

train_ds_name = 'TRAINING-90.csv'
val_ds_name = 'VALIDATION-90.csv'
# %%


def get_data_loaders(ds_dir, train_ds_name, val_ds_name, img_size=224):
    train_transform = Compose([
        ColorJitter(0.3, 0.3, 0.3, 0.3),
        Resize((img_size, img_size)),
        # RandomResizedCrop(img_size, scale=(1.0, 1.0)),
        RandomAffine(10.),
        RandomRotation(13.),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    # train_mask_transform = Compose([
    #     RandomResizedCrop(img_size, scale=(0.8, 1.2)),
    #     RandomAffine(10.),
    #     RandomRotation(13.),
    #     RandomHorizontalFlip(),
    #     ToTensor(),
    # ])
    val_transform = Compose([
        Resize((img_size, img_size)),
        ToTensor(),
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    mask_transform = Compose([
        Resize((img_size, img_size)),
        RandomAffine(10.),
        RandomRotation(13.),
        RandomHorizontalFlip(),
        ToTensor(),
    ])

    train_loader = DataLoader(MogizDataset(ds_dir, train_ds_name, train_transform, mask_transform=mask_transform),
                              batch_size=BATCH_SIZE,
                              shuffle=True,
                              pin_memory=True,
                              num_workers=3)
    val_loader = DataLoader(MogizDataset(ds_dir, val_ds_name, val_transform, mask_transform=mask_transform),
                            batch_size=BATCH_SIZE,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=3)

    return train_loader, val_loader


def save_best_model(cv, model, df_hist):
    format_name = '{}-{}-{}'
    if df_hist['val_loss'].tail(1).iloc[0] <= df_hist['val_loss'].min():
        best_name = format_name.format(0, str(BATCH_SIZE)+'b', 'lr'+str(LR))
        torch.save(model.state_dict(),
                   '{}/{}-best.pth'.format(OUT_DIR, best_name))
        # save_to_mobile(model)
    if(cv % 200 == 0):
        best_name = format_name.format(cv, str(BATCH_SIZE)+'b', 'lr'+str(LR))
        df_hist.to_csv(
            '{}/{}-hist.csv'.format(OUT_DIR, best_name), index=False)


def save_to_mobile(model):
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.to(device)
    torchscript_model = torch.jit.script(model)
    torchscript_model_optimized = optimize_for_mobile(torchscript_model)
    torch.jit.save(torchscript_model_optimized,
                   '{}/{}-best.pt'.format(OUT_DIR, 0))
    #torchscript_model_optimized._save_for_lite_interpreter('{}/{}-best.ptl'.format(OUT_DIR, 0))


def write_on_board(writer, df_hist):
    row = df_hist.tail(1).iloc[0]

    writer.add_scalars('{}/loss'.format(EXPERIMENT), {
        'train': row.train_loss,
        'val': row.val_loss,
    }, row.epoch)

    if(row.epoch % 10 == 0):
        print("epoch : ", str(row.epoch), " | train: ",
              row.train_loss, "val: ", row.val_loss)


def log_hist(df_hist):
    last = df_hist.tail(1)
    best = df_hist.sort_values('val_loss').head(1)
    summary = pd.concat((last, best)).reset_index(drop=True)
    summary['name'] = ['Last', 'Best']
    logger.debug(summary[['name', 'epoch', 'train_loss', 'val_loss']])
    logger.debug('')


def on_after_epoch(n, writer, m, df_hist):
    save_best_model(n, m, df_hist)
    write_on_board(writer, df_hist)
    log_hist(df_hist)


def run_training(img_size, pre_trained=None, loss='mse'):
    data_loaders = get_data_loaders(
        ds_dir, train_ds_name, val_ds_name, img_size)
    data_train = data_loaders[0]
    data_validation = data_loaders[1]
    train_num = len(data_train)*BATCH_SIZE
    epoch_num = N_EPOCHS

    print("---")
    print("train images: ", str(train_num))
    print("epoch : ", str(epoch_num))
    print("batch size : ", str(BATCH_SIZE))
    print("LR : ", str(LR))
    print("---")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = "cpu"
    model = MobileNetV2_unet(mode="train", pre_trained=pre_trained)
    model.to(device)
    optimizer = Adam(model.parameters(), lr=LR)
    criterion = dice_loss(scale=2)

    decayRate = 0.96
    my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer=optimizer, gamma=decayRate)

    if loss == 'mse':
        height_loss = nn.MSELoss()
        weight_loss = nn.MSELoss()
    elif loss == 'mae':
        height_loss = nn.L1Loss()
        weight_loss = nn.L1Loss()
    elif loss == 'huber':
        height_loss = nn.SmoothL1Loss()
        weight_loss = nn.SmoothL1Loss()

    writer = SummaryWriter()

    history = []

    for epoch in range(0, epoch_num):
        train_epoch_loss, train_epoch_acc = _train_on_epoch(
            data_train, model, optimizer, criterion, height_loss,  weight_loss, device)
        val_epoch_loss, val_epoch_acc = _val_on_epoch(
            data_validation, model, optimizer, criterion, height_loss,  weight_loss, device)
        my_lr_scheduler.step()

        hist = {
            'epoch': epoch,
            'train_loss': train_epoch_loss,
            'val_loss': val_epoch_loss,
            'train_acc': train_epoch_acc,
            'val_acc': val_epoch_acc,
        }
        history.append(hist)

        on_after_epoch(epoch, writer, model, pd.DataFrame(history))

    hist = pd.DataFrame(history)
    hist.to_csv('{}/{}-hist.csv'.format(OUT_DIR, epoch_num), index=False)
    writer.close()


def _train_on_epoch(data_loader, model, optimizer, criterion, height_loss, weight_loss, device):
    model.train()
    running_loss = 0.0
    correct_train = 0.0

    for _, data in enumerate(data_loader):
        inputs, labels, joints = data['i'], data['l'], data['j']
        y_height = data['h']

        inputs = inputs.to(device)
        labels = labels.to(device)
        joints = joints.to(device)
        y_height = y_height.to(device)

        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            outputs, joint_o, height_o = model(inputs)
            loss_m = criterion(outputs, labels)
            # print(joint_o.size())
            # print(joints.shape)
            #loss_j = nn.CrossEntropyLoss()(joint_o, joints)
            #loss_j = manual_loss(joint_o, joints)

            loss_h = height_loss(height_o, y_height)
            loss = loss_m + loss_h
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        correct_train += (y_height == height_o).sum().item()

    train_acc = (100 * correct_train) / len(data_loader.dataset)
    epoch_loss = running_loss / len(data_loader.dataset)

    return epoch_loss, train_acc


def _val_on_epoch(data_loader, model, optimizer, criterion, height_loss, weight_loss, device):
    model.eval()
    running_loss = 0.0
    correct_val = 0.0

    for _, data in enumerate(data_loader):

        inputs, labels, joints = data['i'], data['l'],  data['j']
        y_height = data['h']

        inputs = inputs.to(device)
        labels = labels.to(device)
        joints = joints.to(device)
        y_height = y_height.to(device)

        optimizer.zero_grad()

        with torch.set_grad_enabled(False):
            outputs, joint_o, height_o = model(inputs)
            loss_m = criterion(outputs, labels)
            #loss_j = nn.CrossEntropyLoss()(joint_o, joints)
            #loss_j = manual_loss(joint_o, joints)

            loss_h = height_loss(height_o, y_height)
            loss = loss_m + loss_h

        running_loss += loss.item() * inputs.size(0)
        correct_val += (y_height == height_o).sum().item()

    val_acc = (100 * correct_val) / len(data_loader.dataset)
    epoch_loss = running_loss / len(data_loader.dataset)

    return epoch_loss, val_acc


def manual_loss(probs1, target):
    loss_manual = -1 * torch.log(probs1).gather(1, target.unsqueeze(1))
    loss_manual = loss_manual.sum()
    return loss_manual


if __name__ == '__main__':
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    logger = logging.getLogger("logger")
    logger.setLevel(logging.DEBUG)
    if not logger.hasHandlers():
        logger.addHandler(logging.FileHandler(
            filename="outputs/{}.log".format(EXPERIMENT)))

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--img_size',
        type=int,
        default=IMG_SIZE,
        help='image size',
    )
    parser.add_argument(
        '--pre_trained',
        type=str,
        default=None,
        help='path of pre trained weight',
    )
    args, _ = parser.parse_known_args()
    print(args)
    run_training(**vars(args))
