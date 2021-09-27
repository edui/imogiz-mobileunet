import os
import cv2
import json
import numpy as np


def imcrop(img, bbox):
    x1, y1, x2, y2 = bbox
    if x1 < 0 or y1 < 0 or x2 > img.shape[1] or y2 > img.shape[0]:
        img, x1, x2, y1, y2 = pad_img_to_fit_bbox(img, x1, x2, y1, y2)
    return img[y1:y2, x1:x2, :]


def pad_img_to_fit_bbox(img, x1, x2, y1, y2):
    img = cv2.copyMakeBorder(img, - min(0, y1), max(y2 - img.shape[0], 0), -min(
        0, x1), max(x2 - img.shape[1], 0), cv2.BORDER_REPLICATE)
    y2 += -min(0, y1)
    y1 += -min(0, y1)
    x2 += -min(0, x1)
    x1 += -min(0, x1)
    return img, x1, x2, y1, y2


def resize_square_with_padding(im_pth, desired_size=224):
    im = cv2.imread(im_pth)
    old_size = im.shape[:2]  # old_size is in (height, width) format

    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    # new_size should be in (width, height) format

    im = cv2.resize(im, (new_size[1], new_size[0]))

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(
        im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return new_im, ratio
    #cv2.imshow("image", new_im)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def resize_joint(json_file, ratio):
    data = json.load(open(json_file))
    x = np.empty([0, 2])
    shapes = data["shapes"]
    for shape in shapes:
        points = shape["points"]
        xy = [tuple(point * ratio) for point in points]
        if (len(xy) == 1):
            # (16,2)
            x = np.append(x, xy, axis=0)
    return x


images_folder_path = "/Volumes/Part1/Users/dwimiyanto/Projects/mogiz-seameo/dataset/foto-mogiz/Bantul/"
suffix = '_b'
images_folder_path = "/Volumes/Part1/Users/dwimiyanto/Projects/mogiz-seameo/dataset/foto-mogiz/Godean/"
suffix = '_g'
images_folder_path = "/Volumes/Part1/Users/dwimiyanto/Projects/mogiz-seameo/dataset/foto-mogiz/IMOGIRI/"
suffix = '_i'
images_folder_path = "/Volumes/Part1/Users/dwimiyanto/Projects/mogiz-seameo/dataset/foto-mogiz/PRANCAK GLONDONG/"
suffix = '_p'
images_folder_path = "/Volumes/Part1/Users/dwimiyanto/Projects/mogiz-seameo/dataset/foto-mogiz/db/"
suffix = '_d'

joint_folder_path = "/Volumes/Part1/Users/dwimiyanto/Projects/mogiz-seameo/dataset/foto-mogiz/joint/"

output_folder_path = "/Volumes/Part1/Users/dwimiyanto/Projects/mogiz-seameo/dataset/foto-mogiz/square/"
IMG_SIZE = 224
for filename in os.listdir(images_folder_path):
    if os.path.isfile(images_folder_path + filename):
        print("is file ", filename)
        continue

    if 'depan' not in filename:
        if 'belakang' not in filename:
            if 'front' not in filename:
                if 'back' not in filename:
                    continue

    myfolder = images_folder_path + filename
    print("processing ", myfolder)
    myfile = myfolder + '/img.png'
    #image = cv2.imread(myfile)
    #image, x1, x2, y1, y2 = pad_img_to_fit_bbox(image, 1, image.shape[1], 1, image.shape[1])
    image, ratio = resize_square_with_padding(myfile, IMG_SIZE)
    myoutput = output_folder_path + filename + suffix+'.png'
    cv2.imwrite(myoutput, image)

    myfile = myfolder + '/label.png'
    #mask = cv2.imread(myfile)
    #mask, x1, x2, y1, y2 = pad_img_to_fit_bbox(mask, 1, mask.shape[1], 1 , mask.shape[1])
    mask, _ = resize_square_with_padding(myfile, IMG_SIZE)
    myoutput = output_folder_path + filename + suffix+'_mask.png'
    cv2.imwrite(myoutput, mask)

    json_file = joint_folder_path + filename + '_joint.json'
    if os.path.exists(json_file):
        x = resize_joint(json_file, ratio)
        outfile = output_folder_path + filename + suffix + '_joint.npy'
        np.save(outfile, x)

    # break
