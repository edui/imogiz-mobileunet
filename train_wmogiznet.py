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
from nets.MobileNetV2_wunet import MobileNetV2_wunet
from trainer import Trainer
from torch.utils.mobile_optimizer import optimize_for_mobile

np.random.seed(1)
torch.backends.cudnn.deterministic = True
torch.manual_seed(1)

# %%
N_CV = 5
BATCH_SIZE = 8
LR = 1e-3

N_EPOCHS = 1
IMG_SIZE = 224
RANDOM_STATE = 1

EXPERIMENT = 'train_mogiznet'
OUT_DIR = 'outputs/{}'.format(EXPERIMENT)

ds_dir = '/content/drive/MyDrive/Research/mogiz/Dataset/resized_128/'
ds_dir = '/content/drive/MyDrive/Research/mogiz/Dataset/square_224/'
train_ds_name = 'TRAINING.csv'
val_ds_name = 'TRAINING.csv'

# %%


def get_data_loaders(ds_dir, train_ds_name, val_ds_name, img_size=224):
    train_transform = Compose([
        # ColorJitter(0.3, 0.3, 0.3, 0.3),
        Resize((img_size, img_size)),
        # RandomResizedCrop(img_size, scale=(1.0, 1.0)),
        # RandomAffine(10.),
        # RandomRotation(13.),
        # RandomHorizontalFlip(),
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
        ToTensor(),
    ])

    train_loader = DataLoader(MogizDataset(ds_dir, train_ds_name, train_transform, mask_transform=mask_transform),
                              batch_size=BATCH_SIZE,
                              shuffle=True,
                              pin_memory=True,
                              num_workers=2)
    val_loader = DataLoader(MogizDataset(ds_dir, val_ds_name, val_transform, mask_transform=mask_transform),
                            batch_size=BATCH_SIZE,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=2)

    return train_loader, val_loader


def save_best_model(cv, model, df_hist):
    if df_hist['val_loss'].tail(1).iloc[0] <= df_hist['val_loss'].min():
        torch.save(model.state_dict(),
                   '{}/{}-best-wmogiznet.pth'.format(OUT_DIR, cv))
        save_to_mobile(model)


def save_to_mobile(model):
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.to(device)
    torchscript_model = torch.jit.script(model)
    torchscript_model_optimized = optimize_for_mobile(torchscript_model)
    torch.jit.save(torchscript_model_optimized,
                   '{}/{}-best-wmogiznet.pt'.format(OUT_DIR, 0))
    #torchscript_model_optimized._save_for_lite_interpreter('{}/{}-best-wmogiznet.ptl'.format(OUT_DIR, 0))


def write_on_board(writer, df_hist):
    row = df_hist.tail(1).iloc[0]

    writer.add_scalars('{}/loss-wmogiznet'.format(EXPERIMENT), {
        'train': row.train_loss,
        'val': row.val_loss,
    }, row.epoch)


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


def run_training(img_size, pre_trained, loss='mse'):
    data_loaders = get_data_loaders(
        ds_dir, train_ds_name, val_ds_name, img_size)
    data_train = data_loaders[0]
    data_validation = data_loaders[1]
    train_num = len(data_train)*BATCH_SIZE
    epoch_num = N_EPOCHS

    print("---")
    print("train images: ", str(train_num))
    print("epoch : ", str(epoch_num))
    print("---")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MobileNetV2_wunet(mode="train", pre_trained=pre_trained)
    model.to(device)
    optimizer = Adam(model.parameters(), lr=LR)
    criterion = dice_loss(scale=2)

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
        train_epoch_loss = _train_on_epoch(
            data_train, model, optimizer, criterion, height_loss,  weight_loss, device)
        val_epoch_loss = _val_on_epoch(
            data_validation, model, optimizer, criterion, height_loss,  weight_loss, device)
        hist = {
            'epoch': epoch,
            'train_loss': train_epoch_loss,
            'val_loss': val_epoch_loss,
        }
        history.append(hist)

        on_after_epoch(0, writer, model, pd.DataFrame(history))

    hist = pd.DataFrame(history)
    hist.to_csv('{}/{}-hist-wmogiznet.csv'.format(OUT_DIR,
                                                  epoch_num), index=False)
    writer.close()


def _train_on_epoch(data_loader, model, optimizer, criterion, height_loss, weight_loss, device):
    model.train()
    running_loss = 0.0

    for _, data in enumerate(data_loader):
        inputs, labels, joints = data['i'], data['l'], data['j']
        y_height = data['h']
        y_weight = data['w']

        inputs = inputs.to(device)
        labels = labels.to(device)
        joints = joints.to(device)
        y_height = y_height.to(device)
        y_weight = y_weight.to(device)

        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            outputs, joint_o, height_o, weight_o = model(inputs)
            loss_m = criterion(outputs, labels)
            #loss_j = nn.CrossEntropyLoss()(joint_o, joints)
            loss_h = height_loss(height_o, y_height)
            loss_w = weight_loss(weight_o, y_weight)
            loss = loss_m + loss_h + loss_w
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(data_loader.dataset)

    return epoch_loss


def _val_on_epoch(data_loader, model, optimizer, criterion, height_loss, weight_loss, device):
    model.eval()
    running_loss = 0.0

    for _, data in enumerate(data_loader):

        inputs, labels, joints = data['i'], data['l'], data['j']
        y_height = data['h']
        y_weight = data['w']

        inputs = inputs.to(device)
        labels = labels.to(device)
        joints = joints.to(device)
        y_height = y_height.to(device)
        y_weight = y_weight.to(device)

        optimizer.zero_grad()

        with torch.set_grad_enabled(False):
            outputs, joint_o, height_o, weight_o = model(inputs)
            loss_m = criterion(outputs, labels)
            #loss_j = nn.CrossEntropyLoss()(joint_o, joints)
            loss_h = height_loss(height_o, y_height)
            loss_w = weight_loss(weight_o, y_weight)
            loss = loss_m + loss_h + loss_w

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(data_loader.dataset)

    return epoch_loss


if __name__ == '__main__':
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    logger = logging.getLogger("logger")
    logger.setLevel(logging.DEBUG)
    if not logger.hasHandlers():
        logger.addHandler(logging.FileHandler(
            filename="outputs/{}-wmogiznet.log".format(EXPERIMENT)))

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
