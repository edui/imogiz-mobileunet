import os
from PIL import Image, ImageOps


def crop_to_square(image):
    size = min(image.size)
    left, upper = (image.width - size) // 2, (image.height - size) // 2
    right, bottom = (image.width + size) // 2, (image.height + size) // 2
    return image.crop((left, upper, right, bottom))


def padding(image, expected_size):
    d_width = expected_size - image.size[0]
    d_height = expected_size - image.size[1]
    pad_width = d_width // 2
    pad_height = d_height // 2
    padding = (pad_width, pad_height, d_width -
               pad_width, d_height - pad_height)
    return ImageOps.expand(image, padding)


# def imcrop(img, bbox):
#    x1, y1, x2, y2 = bbox
#    if x1 < 0 or y1 < 0 or x2 > img.shape[1] or y2 > img.shape[0]:
#        img, x1, x2, y1, y2 = pad_img_to_fit_bbox(img, x1, x2, y1, y2)
#    return img[y1:y2, x1:x2, :]


# def pad_img_to_fit_bbox(img, x1, x2, y1, y2):
#    img = cv2.copyMakeBorder(img, - min(0, y1), max(y2 - img.shape[0], 0), -min(
#        0, x1), max(x2 - img.shape[1], 0), cv2.BORDER_REPLICATE)
#    y2 += -min(0, y1)
#    y1 += -min(0, y1)
#    x2 += -min(0, x1)
#    x1 += -min(0, x1)
#    return img, x1, x2, y1, y2


images_folder_path = "/Volumes/Part1/Users/dwimiyanto/Projects/mogiz-seameo/dataset/foto-mogiz/Bantul/"
#images_folder_path = "/Volumes/Part1/Users/dwimiyanto/Projects/mogiz-seameo/dataset/foto-mogiz/Godean/"
#images_folder_path = "/Volumes/Part1/Users/dwimiyanto/Projects/mogiz-seameo/dataset/foto-mogiz/IMOGIRI (12 Agustus 2021)/"
#images_folder_path = "/Volumes/Part1/Users/dwimiyanto/Projects/mogiz-seameo/dataset/foto-mogiz/PRANCAK GLONDONG/"
output_folder_path = "/Volumes/Part1/Users/dwimiyanto/Projects/mogiz-seameo/dataset/foto-mogiz/square/"
for filename in os.listdir(images_folder_path):
    if os.path.isfile(images_folder_path + filename):
        print("is file ", filename)
        continue
    myfolder = images_folder_path + filename
    image = Image.open(myfolder + '/img.png')
    #image = crop_to_square(image)
    image = padding(image, image.size[1])
    myoutput = output_folder_path + filename +'.png'
    image.save(myoutput)

    mask = Image.open(myfolder + '/label.png')
    #mask = crop_to_square(image)
    mask = padding(mask, mask.size[1])
    myoutput = output_folder_path + filename + '_mask.png'
    mask.save(myoutput)
