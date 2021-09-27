import os
import pandas as pd

images_folder_path = "/Volumes/Part1/Users/dwimiyanto/Projects/mogiz-seameo/dataset/foto-mogiz/resized_128/"
images_folder_path = "/Volumes/Part1/Users/dwimiyanto/Projects/mogiz-seameo/dataset/foto-mogiz/square_224/"

history = []

for filename in os.listdir(images_folder_path):
    if not os.path.isfile(images_folder_path + filename):
        print("is folder ", filename)
        continue

    if 'mask' in filename:
        continue
    if 'joint' in filename:
        continue
    if '.DS_Store' in filename:
        continue
    if '.csv' in filename:
        continue

    spl = filename.split('.png')
    myfile = spl[0]
    spl = filename.split('_')
    height = spl[0]
    weight = spl[1]

    hist = {
        'image': filename,
        'label': myfile+'_mask.png',
        'joint': myfile+'_joint.npy',
        'height': height,
        'weight': weight,
    }
    history.append(hist)

hist = pd.DataFrame(history)
hist.to_csv('{}/TRAINING.csv'.format(images_folder_path), index=False)
