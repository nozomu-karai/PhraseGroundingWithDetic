from collections import defaultdict

from utils import load_json, merge_bboxes


class Dataset():
    def __init__(self, detector_file, sentence_file, bbox_file):
        self.detector_json = load_json(detector_file)
        self.sentence_json = load_json(sentence_file)
        self.bbox_json = load_json(bbox_file)

        self.ids = [k for k in self.detector_json.keys()]

    def __len__(self):
        return len(self.ids)

    def get_detected_pairs(self, id):
        pairs = defaultdict(lambda: defaultdict(list))
        info = self.detector_json[id]
        for cls, score, box in zip(info['classes'], info['scores'], info['boxes']):
            cls = cls.replace('_', ' ').replace("(","").replace(")","")
            box = list(map(int, box))
            pairs[cls]['scores'].append(score)
            pairs[cls]['boxes'].append(box)

        return pairs

    def get_gold_pairs(self, id):
        pairs = []
        sentences = []
        for sentence in self.sentence_json[id]:
            for phrase in sentence['phrases']:
                phrase_id = phrase['phrase_id']
                first = phrase['first_word_index']
                if phrase_id == 0 or phrase_id not in self.bbox_json[id]['boxes']:
                    continue
                cls = phrase['phrase']
                box = self.bbox_json[id]['boxes'][str(phrase_id)]
                box = merge_bboxes(box)
                pairs.append((cls, box, first))
                sentences.append(sentence['sentence'])

        return pairs, sentences

    def __getitem__(self, idx):
        id = self.ids[idx]
        detected_pairs = self.get_detected_pairs(id)
        gold_pairs, sentences = self.get_gold_pairs(id)
 
        to_return = {
            'id': id,
            'detected_pairs': detected_pairs, 
            'gold_pairs': gold_pairs,
            'sentences': sentences,
        }

        return to_return