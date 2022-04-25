
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from scipy import interpolate
import numpy as np
import random as rm
import torch
import cv2
import os

def draw_box(img, bbox, color, thick):
    cv2.rectangle(img, ((int)(bbox[1]), (int)(bbox[0])), ((int)(bbox[3]), (int)(bbox[2])), color, thick)
    return img

def resize(img, pixel=224, type='short_side'):
    h, w, c = img.shape
    factor = 1

    if type == 'short_side':
        factor = pixel / min(h, w)
    elif type == 'long_side':
        factor = pixel / max(h, w)
    else:
        print('Error in resizing image')

    return cv2.resize(img, (int(w * factor), int(h * factor)))

def res_down(response, dim):
    output = np.zeros(dim)
    xstep = response.shape[0] / dim[0]
    ystep = response.shape[1] / dim[1]

    for x in range(dim[0]):
        for y in range(dim[1]):
            output[x][y] = np.max(response[int(x * xstep) : int((x + 1) * xstep), int(y * ystep) : int((y + 1) * ystep)])

    return output

def visualize(img, response, name, definition=200, cbar=False, resize_img=True):
    if resize_img:
        large_rsp = cv2.resize(response, (img.shape[1], img.shape[0]), interpolation = cv2.INTER_NEAREST)