import json

from PIL import Image
import matplotlib.pyplot as plt
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


def calcurate_area(box):
    return (box[2] - box[0] + 1) * (box[3] - box[1] + 1)


def calcurate_iou(a, b):
    a_area = calcurate_area(a)
    b_area = calcurate_area(b)
    
    abx_mn = max(a[0], b[0]) 
    aby_mn = max(a[1], b[1]) 
    abx_mx = min(a[2], b[2]) 
    aby_mx = min(a[3], b[3]) 
    w = max(0, abx_mx - abx_mn + 1)
    h = max(0, aby_mx - aby_mn + 1)
    intersect = w * h
    
    iou = intersect / (a_area + b_area - intersect)

    return iou


# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

def plot_results(img_path, first, pred_cls, pred_box, gold_cls, gold_box, sentence, save_path):
    plt.figure(figsize=(16,10))
    img = Image.open(img_path)
    plt.imshow(img)
    ax = plt.gca()
    xmin, ymin, xmax, ymax = pred_box[0], pred_box[1], pred_box[2], pred_box[3]
    ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                fill=False, color=COLORS[0], linewidth=3))
    ax.text(xmin, ymin, pred_cls, fontsize=15,
            bbox=dict(facecolor='skyblue', alpha=0.5),
            verticalalignment='top')
    
    xmin, ymin, xmax, ymax = gold_box[0], gold_box[1], gold_box[2], gold_box[3]
    ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                fill=False, color=COLORS[1], linewidth=3))
    text = f'{first}: {gold_cls}'
    ax.text(xmin, ymin, text, fontsize=15,
            bbox=dict(facecolor='yellow', alpha=0.5))
    
    ax.set_title(sentence)
    
    plt.savefig(save_path, format='jpg')

    plt.close()

