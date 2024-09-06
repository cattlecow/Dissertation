#!/usr/bin/python3
# -*- coding: utf-8 -*-
import glob
import os

import json
import numpy as np
import cv2

json_names = glob.glob(r'H:\stoneseg\stone_seg\train\image\*.json')
save_path = r'H:\stoneseg\stone_seg\train\mask'
os.makedirs(save_path, exist_ok=True)
for name in json_names:
    # read json file
    with open(name, "r") as f:
        data = f.read()
    # convert str to json objs
    data = json.loads(data)
    # get the points
    image = cv2.imread(name.replace('json', 'png'))
    # create a blank image
    mask = np.zeros_like(image, dtype=np.uint8)

    nums = len(data['shapes'])
    for i in range(nums):
        points = data["shapes"][i]["points"]
        points = np.array(points, dtype=np.int32)  # tips: points location must be int32
        # read image to get shape

        # fill the contour with 255
        cv2.fillPoly(mask, [points], (255, 255, 255))
    # save the mask
    cv2.imwrite(name.replace('.json', '.png').replace('\\image\\', '\\mask\\'), mask)