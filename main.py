from __future__ import print_function

from miscc.config import cfg, cfg_from_file
from datasets import TextDataset, TextBertDataset
import trainer

import os
import sys
import time
import random
import pprint
import datetime
import dateutil.tz
import argparse
import numpy as np

import torch
import torchvision.transforms as transforms

from pytorch_pretrained_bert import BertTokenizer
import tokenization

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)


def parse_args():
    parser = argparse.ArgumentParser(description='Train a AttnGAN network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfg/bird_attn2.yaml', type=str)
    parser.add_argument('--gpu', dest='gpu_id', type=int, default=0)
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='data/birds')
    parser.add_argument('--manualSeed', type=int, default=0, help='manual seed')
    args = parser.parse_args()
    return args


def gen_example(wordtoix, algo):
    '''generate images from example sentences'''

    from nltk.tokenize import RegexpTokenizer
    #tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    if(cfg.LOCAL_PRETRAINED):
        tokenizer = tokenization.FullTokenizer(vocab_file=cfg.BERT_ENCODER.VOCAB, do_lower_case=True)
    else:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


    filepath = '%s/example_filenames.txt' % (cfg.DATA_DIR)
    data_dic = {}
    with open(filepath, "r") as f:
        #filenames = f.read().decode('utf8').split('\n')
        filenames = f.read().split('\n')
        for name in filenames:
            if len(name) == 0:
                continue
            #filepath = '%s/%s.txt' % (cfg.DATA_DIR, name)
            filepath = '%s/text/%s.txt' % (cfg.DATA_DIR, name)
            with open(filepath, "r") as f:
                print('Load from:', name)
                #sentences = f.read().decode('utf8').split('\n')
                sentences = f.read().split('\n')
                # a list of indices for a sentence
                captions = []
                cap_lens = []
                for sent in sentences:
                    if len(sent) == 0:
                        continue
                    sent = sent.replace("\ufffd\ufffd", " ")
                    #tokenizer = RegexpTokenizer(r'\w+')
                    #tokens = tokenizer.tokenize(sent.lower())
                    tokens = tokenizer.tokenize(sent.lower())
                    if len(tokens) == 0:
                        print('sent', sent)
                        continue

                    rev = []
                    for t in tokens:
                        t = t.encode('ascii', 'ignore').decode('ascii')
                        if len(t) > 0 and t in wordtoix:
                            rev.append(wordtoix[t])
                    #rev = tokenizer.convert_tokens_to_ids(tokens)
                    captions.append(rev)
                    cap_lens.append(len(rev))
            max_len = np.max(cap_lens)

            sorted_indices = np.argsort(cap_lens)[::-1]
            cap_lens = np.asarray(cap_lens)
            cap_lens = cap_lens[sorted_indices]
            cap_array = np.zeros((len(captions), max_len), dtype='int64')
            for i in range(len(captions)):
                idx = sorted_indices[i]
                cap = captions[idx]
                c_len = len(cap)
                cap_array[i, :c_len] = cap
            key = name[(name.rfind('/') + 1):]
            data_dic[key] = [cap_array, cap_lens, sorted_indices]
    algo.gen_example(data_dic)


if __name__ == "__main__":
    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    if args.gpu_id != -1:
        cfg.GPU_ID = args.gpu_id
    else:
        cfg.CUDA = False

    if args.data_dir != '':
        cfg.DATA_DIR = args.data_dir
    print('Using config:')
    pprint.pprint(cfg)

    if not cfg.TRAIN.FLAG:
        args.manualSeed = 100
    elif args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if cfg.CUDA:
        torch.cuda.manual_seed_all(args.manualSeed)

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = 'output/%s_%s_%s' % \
        (cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp)

    split_dir, bshuffle = 'train', True
    if not cfg.TRAIN.FLAG:
        # bshuffle = False
        split_dir = 'test'

    # Get data loader
    imsize = cfg.TREE.BASE_SIZE * (2 ** (cfg.TREE.BRANCH_NUM - 1))
    image_transform = transforms.Compose([
        transforms.Scale(int(imsize * 76 / 64)),
        transforms.RandomCrop(imsize),
        transforms.RandomHorizontalFlip()])

    if(cfg.LOCAL_PRETRAINED):
        tokenizer = tokenization.FullTokenizer(vocab_file=cfg.BERT_ENCODER.VOCAB, do_lower_case=True)
    else:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    vocab_size = len(tokenizer.vocab)

    #dataset = TextDataset(cfg.DATA_DIR, split_dir, base_size=cfg.TREE.BASE_SIZE, transform=image_transform)
    #dataset = TextBertDataset(cfg.DATA_DIR, split_dir, base_size=cfg.TREE.BASE_SIZE, transform=image_transform)
    dataset = TextBertDataset(cfg.DATA_DIR, split_dir, base_size=cfg.TREE.BASE_SIZE, transform=image_transform, tokenizer=tokenizer)

    assert dataset
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.TRAIN.BATCH_SIZE,
        drop_last=True, shuffle=bshuffle, num_workers=int(cfg.WORKERS))

    #rev = tokenizer.convert_tokens_to_ids(tokens)

    ixtoword = {}
    wordtoix = {}
    if(vocab_size > 0):
        for w, ix in tokenizer.vocab.items():
            wordtoix[w] = ix
            ixtoword[ix] = w

    # Define models and go to train/evaluate
    trainer_ = getattr(trainer, cfg.TRAIN.TRAINER)
    #algo = trainer_(output_dir, dataloader, dataset.n_words, dataset.ixtoword)
    algo = trainer_(output_dir, dataloader, vocab_size, ixtoword)

    start_t = time.time()
    if cfg.TRAIN.FLAG:
        algo.train()
    else:
        '''generate images from pre-extracted embeddings'''
        if cfg.B_VALIDATION:
            algo.sampling(split_dir)  # generate images for the whole valid dataset
        else:
            #gen_example(dataset.wordtoix, algo)  # generate images for customized captions
            gen_example(wordtoix, algo)  # generate images for customized captions
    end_t = time.time()
    print('Total time for training:', end_t - start_t)
