import os
import json
import numpy as np

source_folder_path = "/Volumes/Part1/Users/dwimiyanto/Projects/mogiz-seameo/dataset/foto-mogiz/joint/"
destination_folder_path = "/Volumes/Part1/Users/dwimiyanto/Projects/mogiz-seameo/dataset/foto-mogiz/resized_128/"

for filename in os.listdir(source_folder_path):

    if os.path.isfile(source_folder_path + filename):
        json_file = source_folder_path + filename
        data = json.load(open(json_file))
        x = np.empty([0, 2])
        shapes = data["shapes"]
        for shape in shapes:
            points = shape["points"]
            xy = [tuple(point) for point in points]
            if (len(xy) == 1):
                # (16,2)
                x = np.append(x, xy, axis=0)

                # (32,)
                #cx, cy = xy[0]
                #x = np.append(x, [cx], axis=0)
                #x = np.append(x, [cy], axis=0)
        # print(x.shape)
        splt = filename.split('.json')
        outfile = destination_folder_path + splt[0] + '.npy'
        np.save(outfile, x)
