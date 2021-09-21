import cv2
import os
import numpy as np
from skimage import io, transform
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from dataset import MaskDataset, get_img_files, get_img_files_eval
from nets.MobileNetV2_unet import MobileNetV2_unet

MEAN = torch.tensor([0.485, 0.456, 0.406])
STD = torch.tensor([0.229, 0.224, 0.225])


def save_output(image_name, pred, d_dir):
    #x = (pred * STD[:, None, None]) + MEAN[:, None, None]
    # print(x.shape)
    print(pred.shape)
    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    im = Image.fromarray(predict_np*255).convert('RGB')
    img_name = image_name.split(os.sep)[-1]
    image = io.imread(image_name)
    imo = im.resize((image.shape[1], image.shape[0]), resample=Image.BILINEAR)

    pb_np = np.array(imo)

    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1, len(bbb)):
        imidx = imidx + "." + bbb[i]

    imo.save(d_dir+imidx+'.png')


def save_output_j(image_name, pred, d_dir):
    print(save_output_j)


model_path = "/content/MobileUNET/outputs/train_mogiznet/0-best.pth"
image_path = "/content/drive/MyDrive/Research/mogiz/Dataset/resized_128/img/143.96_30.3_delvino_front.png"
IMG_SIZE = 224
output_shape = (720, 1280)

# Process the network
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"
# data_loader = get_data_loaders(frames)
model = MobileNetV2_unet(mode="eval", pre_trained=None)
model.load_state_dict(torch.load(model_path))
model.to(device)
model.eval()
transform = Compose([Resize((IMG_SIZE, IMG_SIZE)), ToTensor(), Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
),
])

with torch.no_grad():
    image = cv2.imread(image_path)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Apply transform to img
    img_trf = Image.fromarray(img)
    img_trf = transform(img_trf)
    img_trf = img_trf.unsqueeze(0)
    inputs = img_trf.to(device)
    # Apply model to get output
    outputs, scale, height, joints = model(inputs)
    print('Height : ', (height.item()*100), ' cm')
    save_output(image_path, outputs[0], "outputs/")
    #save_output_j(image_path, joints, "outputs/")
