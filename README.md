# cycle-image-gan revision for Colab

This is cycle-image-gan revised version for Google Colab.

Refer to [https://github.com/suetAndTie/cycle-image-gan](https://github.com/suetAndTie/cycle-image-gan) for original source version, <br>
it is implementation for the paper [Cycle Text-To-Image GAN with BERT](https://arxiv.org/abs/2003.12137).

Refer to troubleshooting [issues](https://github.com/rightlit/cycle-image-gan-rev/issues) while running with original source code. 

### Dependencies
python 3.6

Pytorch 1.7.0+cu101


## Data
1. Download AttnGAN preprocessed data and captions [birds](https://drive.google.com/open?id=1O_LtUP9sch09QH3s_EBAgLEctBQ5JBSJ)
2. Download the [birds](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) image data. Extract them to `data/birds/`

## Instructions
* pretrain STREAM
```
python pretrain_STREAM.py --cfg cfg/STREAM/bird.yaml --gpu 0
```
* train CycleGAN
```
python main.py --cfg cfg/bird_cycle.yaml --gpu 0
```
