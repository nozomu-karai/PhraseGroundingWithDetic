import json
from tqdm import tqdm

import numpy as np

def load_json(file):
    with open(file, 'r') as f:
        json_file = json.loads(f.read())
    
    return json_file


def merge_bboxes(bboxes):
    x_min = [box[0] for box in bboxes]
    y_min = [box[1] for box in bboxes]
    x_max = [box[2] for box in bboxes]
    y_max = [box[3] for box in bboxes]

    return [min(x_min), min(y_min), max(x_max), max(y_max)]


def load_glove(path):
    glove = {}
    with open(path, 'r') as f:
        for line in tqdm(f):
            vals = line.rstrip().split(' ')
            glove[vals[0]] = np.array([float(x) for x in vals[1:]])
    
    return glove


def calcurate_iou(a, b):
    a_area = (a[2] - a[0] + 1) * (a[3] - a[1] + 1)
    b_area = (b[2] - b[0] + 1) * (b[3] - b[1] + 1)
    
    abx_mn = max(a[0], b[0]) 
    aby_mn = max(a[1], b[1]) 
    abx_mx = min(a[2], b[2]) 
    aby_mx = min(a[3], b[3]) 
    w = max(0, abx_mx - abx_mn + 1)
    h = max(0, aby_mx - aby_mn + 1)
    intersect = w * h
    
    iou = intersect / (a_area + b_area - intersect)

    return iou
