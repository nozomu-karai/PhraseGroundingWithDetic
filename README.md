# Phrase Grounding With Detic

Unsupervised phrase grounding using Detic and GloVe.

Implementation is reffering [Phrase Localization Without Paired Training Example](https://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_Phrase_Localization_Without_Paired_Training_Examples_ICCV_2019_paper.pdf).

## News
- 2022-11-17 Update: Support FastText


## Installation

Follow Detiic installation. See [installation instructions](docs/INSTALL.md).

## How to use

### Preprocess

- Follow [info-ground](https://github.com/BigRedT/info-ground)
```bash
# clone Flickr30K Entities github repo and extract annotations and splits
bash data/flickr/download.sh
# process annotations into easy to read json files
bash data/flickr/process_annos.sh
```
- Download GloVe
```bash
wget https://huggingface.co/stanfordnlp/glove/resolve/main/glove.42B.300d.zip
unzip glove.42B.300d.zip
```

- Download FastText
```bash
git clone https://github.com/facebookresearch/fastText.git
cd fasttext
pip install .
```
```python
import fasttext.util
fasttext.util.download_model('en', if_exists='ignore')  # English
```

### Grounding
```bash
# cache information
python cache_info.py --config-file configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml --input ~/dataset/flickr30k_entities/test.txt --flickr_dir [Flickr30k images directory] --output [output directory] --vocabulary lvis --opts MODEL.WEIGHTS models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth
# grounding
python phrase_grounding/run.py --detector_output [output directory]/cache_data.json --sentence_file path/to/sentences_test.json --bbox_file path/to/bounding_boxes_test.json --embedding [glove, fasttext] --embedding_path path/to/embedding_file --image_dir path/to/image
```

## Benchmark evaluation
- Flickr 30k entities
    - Acc (IoU > 0.5) : 41.10
