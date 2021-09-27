import cv2
import os
# Size of test input images should be 128
# Please run the code to resize the test images
# before testing

images_folder_path = "/Volumes/Part1/Users/dwimiyanto/Projects/mogiz-seameo/dataset/foto-mogiz/db/"
save_resized_image = "/Volumes/Part1/Users/dwimiyanto/Projects/mogiz-seameo/dataset/foto-mogiz/resized_128/"
for filename in os.listdir(images_folder_path):

    if os.path.isfile(images_folder_path + filename):
        print("is file ", filename)
        continue
    myfile = images_folder_path+filename+"/img.png"
    print("img file ", myfile)
    image = cv2.imread(myfile)
    resized_image = cv2.resize(image, (128, 128))
    image_resized = save_resized_image + filename+".png"
    print("resizing")
    cv2.imwrite(image_resized, resized_image)

    myfile = images_folder_path+filename+"/label.png"
    print("mask file ", myfile)
    image = cv2.imread(myfile)
    resized_image = cv2.resize(image, (128, 128))
    image_resized = save_resized_image + filename+"_mask.png"
    print("resizing")
    cv2.imwrite(image_resized, resized_image)
