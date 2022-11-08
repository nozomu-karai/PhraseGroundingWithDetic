from argparse import ArgumentParser
import os
import random
from logging import getLogger, FileHandler, StreamHandler, DEBUG
logger = getLogger(__name__)
logger.setLevel(DEBUG)
sh = StreamHandler()
logger.addHandler(sh)

import numpy as np
from tqdm import tqdm

from dataset import Dataset
from utils import load_glove, merge_bboxes, calcurate_iou, calcurate_area

def get_args():
    parser = ArgumentParser()
    parser.add_argument('--detector_output', type=str)
    parser.add_argument('--sentence_file', type=str)
    parser.add_argument('--bbox_file', type=str)
    parser.add_argument('--glove', type=str)
    parser.add_argument('--similarity', type=str, default='cosine', choices=['cosine', 'norm'])
    parser.add_argument('--strategy', type=str, default='union', choices=['union', 'random', 'largest'])

    parser.add_argument('--output', type=str, default='./result')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    fh = FileHandler(os.path.join(args.output, 'run.log'))
    logger.addHandler(fh)

    random.seed(2022)

    return args


def get_vector(word, glove, args):
    tokens = word.lower().split(' ')
    invocab = 0
    vector = np.zeros(args.glove_dim) 
    for token in tokens:
        if token in glove:
            vector += glove[token]
            invocab += 1
    
    return vector / invocab if invocab != 0 else vector


def distance(x, y, glove, args):
    x_vec = get_vector(x, glove, args)
    y_vec = get_vector(y, glove, args)

    if args.similarity == 'cosine':
        if np.linalg.norm(x_vec) * np.linalg.norm(y_vec) != 0:
            return np.dot(x_vec, y_vec) / (np.linalg.norm(x_vec) * np.linalg.norm(y_vec))
        else:
            return np.linalg.norm(x_vec - y_vec)
    elif args.similarity == 'norm':
        return np.linalg.norm(x_vec - y_vec)


def main():
    args = get_args()

    dataset = Dataset(args.detector_output, args.sentence_file, args.bbox_file)
    logger.info('---- Make dataset ----')
    logger.info(f'dataset size: {len(dataset)}')
 
    glove = load_glove(args.glove)
    num_vocab = len(glove)
    glove_dim = glove[list(glove.keys())[0]].shape[0]
    args.glove_dim = glove_dim
    args.glove_vocab = num_vocab
    logger.info(f'GloVe is loaded from {args.glove}')
    logger.info(f'num vocab: {num_vocab}, dim: {glove_dim}')
    
    logger.info('---- Start grounding ----')
    gold_boxes = []
    pred_boxes = []
    total = 0
    correct = 0
    for i in tqdm(range(len(dataset))):
        batch = dataset[i]
        for gold_cls, gold_box in batch['gold_pairs']:
            gold_boxes.append(gold_box)
            max_key = None
            max_val = -1
            for det_cls in batch['detected_pairs'].keys():
                dist = distance(gold_cls, det_cls, glove, args)
                if dist > max_val:
                    max_key = det_cls
                    max_val = dist
            
            if args.strategy == 'union':
                pred_box = merge_bboxes(batch['detected_pairs'][max_key]['boxes'])
            elif args.strategy == 'random':
                pred_box = random.choice(batch['detected_pairs'][max_key]['boxes'])
            elif args.strategy == 'largest':
                max_score = -1
                max_box = None
                for box in batch['detected_pairs'][max_key]['boxes']:
                    area = calcurate_area(box)
                    if area > max_score:
                        max_box = box
                        max_score = area
                    pred_box = max_box
            pred_boxes.append(pred_box)
            total += 1
            iou_score = calcurate_iou(pred_box, gold_box)
            if iou_score >= 0.5:
                correct += 1
    
    logger.info('grounding finished!')

    logger.info(f'ACC: {100*correct/total:.2f} (COR: {correct}, TOT: {total})')


if __name__ == '__main__':
    main()
