from argparse import ArgumentParser
import os
import random
from logging import getLogger, FileHandler, StreamHandler, DEBUG, Formatter
logger = getLogger(__name__)
logger.setLevel(DEBUG)
sh = StreamHandler()
logger.addHandler(sh)

from tqdm import tqdm

from dataset import Dataset
from embedding import Glove, FastText
from utils import merge_bboxes, calcurate_iou, calcurate_area, plot_results

def get_args():
    parser = ArgumentParser()
    parser.add_argument('--detector_output', type=str)
    parser.add_argument('--sentence_file', type=str)
    parser.add_argument('--bbox_file', type=str)
    parser.add_argument('--image_dir', type=str)
    parser.add_argument('--embedding', type=str, default='glove', choices=['glove', 'fasttext'])
    parser.add_argument('--embedding_path', type=str)
    parser.add_argument('--similarity', type=str, default='cosine', choices=['cosine', 'norm'])
    parser.add_argument('--strategy', type=str, default='union', choices=['union', 'random', 'largest'])

    parser.add_argument('--output', type=str, default='./result')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    os.makedirs(os.path.join(args.output, 'simple'), exist_ok=True)
    os.makedirs(os.path.join(args.output, 'complex'), exist_ok=True)

    fh = FileHandler(os.path.join(args.output, 'run.log'))
    fh.setFormatter(Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(fh)

    random.seed(2022)

    return args


def main():
    args = get_args()

    dataset = Dataset(args.detector_output, args.sentence_file, args.bbox_file)
    logger.info('---- Make dataset ----')
    logger.info(f'dataset size: {len(dataset)}')
    
    logger.info('Loading embedding')
    if args.embedding == 'glove':
        embed = Glove(args.embedding_path)
        logger.info(f'GloVe is loaded from {args.embedding_path}')
    elif args.embedding == 'fasttext':
        embed = FastText(args.embedding_path)
        logger.info(f'FastText is loaded from {args.embedding_path}')
    logger.info(f'num vocab: {embed.num_vocab}, dim: {embed.emb_dim}')
    
    logger.info('---- Start grounding ----')
    total = 0
    correct = 0
    cor_s, tot_s = 0, 0
    cor_c, tot_c = 0, 0
    for i in tqdm(range(len(dataset))):
        batch = dataset[i]
        for (gold_cls, gold_box, first), sentence in zip(batch['gold_pairs'], batch['sentences']):
            max_key = None
            max_val = -1
            for det_cls in batch['detected_pairs'].keys():
                dist = embed.distance(gold_cls, det_cls, args.similarity)
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
            total += 1
            iou_score = calcurate_iou(pred_box, gold_box)
            th = random.random()
            img_path = os.path.join(args.image_dir, batch['id'] + '.jpg')
            
            if len(batch['detected_pairs'][max_key]['boxes']) == 1:
                category = 'simple'
                tot_s += 1
                if th < 0.01:
                    save_path = os.path.join(args.output, 'simple/' + batch['id'] + '.jpg')
                    plot_results(img_path, first, max_key, pred_box, gold_cls, gold_box, sentence, save_path)
            else:
                category = 'complex'
                tot_c += 1
                if th < 0.01:
                    save_path = os.path.join(args.output, 'complex/' + batch['id'] + '.jpg')
                    plot_results(img_path, first, max_key, pred_box, gold_cls, gold_box, sentence, save_path)
            
            if iou_score >= 0.5:
                correct += 1
                if category == 'simple':
                    cor_s += 1
                elif category == 'complex':
                    cor_c += 1
    
    logger.info('grounding finished!')

    logger.info(f'ACC: {100*correct/total:.2f} (COR: {correct}, TOT: {total})')
    logger.info(f'ACC (simple): {100*cor_s/tot_s:.2f} (COR: {cor_s}, TOT: {tot_s})')
    logger.info(f'ACC (complex): {100*cor_c/tot_c:.2f} (COR: {cor_c}, TOT: {tot_c})')

if __name__ == '__main__':
    main()
