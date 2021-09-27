import os

images_folder_path = "/Volumes/Part1/Users/dwimiyanto/Projects/mogiz-seameo/dataset/foto-mogiz/Bantul/"
#images_folder_path = "/Volumes/Part1/Users/dwimiyanto/Projects/mogiz-seameo/dataset/foto-mogiz/Godean/"
#images_folder_path = "/Volumes/Part1/Users/dwimiyanto/Projects/mogiz-seameo/dataset/foto-mogiz/IMOGIRI (12 Agustus 2021)/"
#images_folder_path = "/Volumes/Part1/Users/dwimiyanto/Projects/mogiz-seameo/dataset/foto-mogiz/PRANCAK GLONDONG/"

images_folder_path = "/Volumes/Part1/Users/dwimiyanto/Projects/mogiz-seameo/dataset/foto-mogiz/joint/"

for filename in os.listdir(images_folder_path):

    if os.path.isfile(images_folder_path + filename):
        print("is file ", filename)
        # continue
        splt = filename.split('.json')
        myfile = images_folder_path + filename
        myoutput = images_folder_path + splt[0]
        comd = 'labelme_json_to_dataset "' + myfile + '" -o "' + myoutput+'"'
        os.system(comd)
