# coding:utf-8
import cv2
import numpy as np


def img_resize_to_target_white(input_image):
    img = input_image
    h = img.shape[0]
    w = img.shape[1]
    target = np.ones((2 * h, 2 * w), dtype=np.uint8) * 255

    half_h = int(h / 2)
    half_w = int(w / 2)
    ret = cv2.cvtColor(target, cv2.COLOR_GRAY2BGR)
    for i in range(2 * h):
        for j in range(2 * w):
            if (half_h < i) and (i < h + half_h) and (half_w < j) and (j < w + half_w):
                ret[i, j, 0] = img[i - half_h, j - half_w, 0]
                ret[i, j, 1] = img[i - half_h, j - half_w, 1]
                ret[i, j, 2] = img[i - half_h, j - half_w, 2]
            else:
                ret[i, j, 0] = 255
                ret[i, j, 1] = 255
                ret[i, j, 2] = 255

    return ret
