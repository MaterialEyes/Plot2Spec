# script to collect raman/xanes graph images from nature

import json
import requests
import os
import shutil
from tqdm import tqdm
from PIL import Image
import numpy as np


def main(data_json, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    with open(data_json, "r") as fp:
        record = json.load(fp)
    
    num_images = 0
    for img_name in tqdm(record.keys()):
        try:
            image_url = record[img_name]["url"]
            response = requests.get(image_url, stream=True)
            figure_path = os.path.join(save_dir, img_name)
            with open(figure_path, 'wb') as out_file:
                shutil.copyfileobj(response.raw, out_file)
            del response 
            x1, y1 = record[img_name]["top_left"]
            x2, y2 = record[img_name]["bottom_right"]
            patch = Image.open(figure_path).crop((x1,y1,x2,y2))
            patch.save(figure_path)
            num_images += 1
        except:
            pass
    print("Done! {} images!".format(num_images))
        
        

if __name__ == "__main__":
    main("data.json", "./data")