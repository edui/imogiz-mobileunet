import os

source_folder_path = "/Volumes/Part1/Users/dwimiyanto/Projects/mogiz-seameo/dataset/foto-mogiz/joint/"
destination_folder_path = "/Volumes/Part1/Users/dwimiyanto/Projects/mogiz-seameo/dataset/foto-mogiz/resized_128/"

for filename in os.listdir(source_folder_path):

    if os.path.isfile(source_folder_path + filename):
        continue
    myfile = source_folder_path + filename + '/label.png'
    mydestfile = destination_folder_path + filename + '.png'
    comd = 'cp "' + myfile + '" "' + mydestfile+'"'
    os.system(comd)
