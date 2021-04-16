from __future__ import print_function

from miscc.utils import mkdir_p
from miscc.utils import build_super_images
from miscc.losses import sent_loss, words_loss, image_to_text_loss
from miscc.config import cfg, cfg_from_file

from datasets import TextDataset, TextBertDataset
from datasets import prepare_data, prepare_data_bert

from model import BERT_RNN_ENCODER, BERT_CNN_ENCODER_RNN_DECODER

import os
import sys
import time
import random
import pprint
import datetime
import dateutil.tz
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

from pytorch_pretrained_bert import BertTokenizer
import pickle

import tokenization
from datasets import DevTextBertDataset
from datasets import prepare_data_dev

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)

UPDATE_INTERVAL = 200
def parse_args():
    parser = argparse.ArgumentParser(description='Calculate Similarity')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfg/STREAM/bird.yaml', type=str)
    parser.add_argument('--gpu', dest='gpu_id', type=int, default=0)
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='data/birds')
    parser.add_argument('--manualSeed', type=int, default=0, help='manual seed')
    args = parser.parse_args()
    return args

def words_similarity(img_features, words_emb, labels, cap_lens, class_ids, batch_size):
    """
        words_emb(query): batch x nef x seq_len
        img_features(context): batch x nef x 17 x 17
    """
    #words_emb = torch.randn(1,768, 18)
    #img_features = torch.randn(1,768,17,17)

    masks = []
    att_maps = []
    similarities = []
    #cap_lens = cap_lens.data.tolist()
    #batch_size = 1
    words_num = 18
    GAMMA1= 4.0
    GAMMA2= 5.0
    
    for i in range(batch_size):

        # Get the i-th text description
        #words_num = cap_lens[i]
        # -> 1 x nef x words_num
        word = words_emb[i, :, :words_num].unsqueeze(0).contiguous()
        # -> batch_size x nef x words_num
        word = word.repeat(batch_size, 1, 1)
        # batch x nef x 17*17
        context = img_features
        """
            word(query): batch x nef x words_num
            context: batch x nef x 17 x 17
            weiContext: batch x nef x words_num
            attn: batch x words_num x 17 x 17
        """
        weiContext, attn = func_attention(word, context, GAMMA1)
        att_maps.append(attn[i].unsqueeze(0).contiguous())
        # --> batch_size x words_num x nef
        word = word.transpose(1, 2).contiguous()
        weiContext = weiContext.transpose(1, 2).contiguous()
        # --> batch_size*words_num x nef
        word = word.view(batch_size * words_num, -1)
        weiContext = weiContext.view(batch_size * words_num, -1)
        #
        # -->batch_size*words_num
        row_sim = cosine_similarity(word, weiContext)
        # --> batch_size x words_num
        row_sim = row_sim.view(batch_size, words_num)

        # Eq. (10)
        row_sim.mul_(GAMMA2).exp_()
        row_sim = row_sim.sum(dim=1, keepdim=True)
        row_sim = torch.log(row_sim)

        # --> 1 x batch_size
        # similarities(i, j): the similarity between the i-th image and the j-th text description
        similarities.append(row_sim)


def evaluate(dataloader, cnn_model, rnn_model, batch_size, labels):
    cnn_model.eval()
    rnn_model.eval()
    s_total_loss = 0
    w_total_loss = 0
    t_total_loss = 0
    debug_flag = True
    
    for step, data in enumerate(dataloader, 0):
        #imgs, captions, cap_lens, class_ids, keys = prepare_data(data)
        #imgs, captions, cap_lens, class_ids, keys = prepare_data_bert(data, tokenizer=None)
        imgs, captions, cap_lens, class_ids, keys = prepare_data_dev(data)
        if(debug_flag):
            with open('./debug1.pkl', 'wb') as f:
                pickle.dump({'imgs':imgs, 'captions':captions, 'cap_lens':cap_lens, 'class_ids':class_ids, 'keys':keys}, f)  

        #words_features, sent_code, word_logits = cnn_model(imgs[-1], captions)
        words_features, sent_code, word_logits = cnn_model(imgs[-1], captions, cap_lens)
        # nef = words_features.size(1)
        # words_features = words_features.view(batch_size, nef, -1)

        hidden = rnn_model.init_hidden(batch_size)
        words_emb, sent_emb = rnn_model(captions, cap_lens, hidden)

        w_loss0, w_loss1, attn = words_loss(words_features, words_emb, labels,
                                            cap_lens, class_ids, batch_size)
        w_total_loss += (w_loss0 + w_loss1).data

        # similarity score
        print('calculating similarity')
        similarities = words_similarity(words_features, words_emb, labels, cap_lens, class_ids, batch_size)
        print(similarities)

        '''
        s_loss0, s_loss1 = \
            sent_loss(sent_code, sent_emb, labels, class_ids, batch_size)
        s_total_loss += (s_loss0 + s_loss1).data

        t_loss = image_to_text_loss(word_logits, captions)
        t_total_loss += t_loss.data
        '''

        if step == 50:
            break

    s_cur_loss = s_total_loss.item() / step
    w_cur_loss = w_total_loss.item() / step
    t_cur_loss = t_total_loss.item() / step

    return s_cur_loss, w_cur_loss, t_cur_loss


def build_models():
  
    # build model ############################################################
    #cfg.LOCAL_PRETRAINED = False
    if(cfg.LOCAL_PRETRAINED):
        tokenizer = tokenization.FullTokenizer(vocab_file=cfg.BERT_ENCODER.VOCAB, do_lower_case=True)
        vocab_size = len(tokenizer.vocab)
        #vocab_size = 3770
        #vocab_size = 4000
    else:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        vocab_size = len(tokenizer.vocab)
        #vocab_size = 30522

    #text_encoder = BERT_RNN_ENCODER(dataset.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
    #image_encoder = BERT_CNN_ENCODER_RNN_DECODER(cfg.TEXT.EMBEDDING_DIM, cfg.CNN_RNN.HIDDEN_DIM, dataset.n_words, rec_unit=cfg.RNN_TYPE)
    text_encoder = BERT_RNN_ENCODER(vocab_size, nhidden=cfg.TEXT.EMBEDDING_DIM)
    image_encoder = BERT_CNN_ENCODER_RNN_DECODER(cfg.TEXT.EMBEDDING_DIM, cfg.CNN_RNN.HIDDEN_DIM, vocab_size, rec_unit=cfg.RNN_TYPE)

    labels = Variable(torch.LongTensor(range(batch_size)))
    start_epoch = 0
    if cfg.TRAIN.NET_E != '':
        state_dict = torch.load(cfg.TRAIN.NET_E)
        text_encoder.load_state_dict(state_dict)
        print('Load ', cfg.TRAIN.NET_E)
        #
        name = cfg.TRAIN.NET_E.replace('text_encoder', 'image_encoder')
        state_dict = torch.load(name)
        image_encoder.load_state_dict(state_dict)
        print('Load ', name)

        istart = cfg.TRAIN.NET_E.rfind('_') + 8
        iend = cfg.TRAIN.NET_E.rfind('.')
        start_epoch = cfg.TRAIN.NET_E[istart:iend]
        start_epoch = int(start_epoch) + 1
        print('start_epoch', start_epoch)
    if cfg.CUDA:
        text_encoder = text_encoder.cuda()
        image_encoder = image_encoder.cuda()
        labels = labels.cuda()

    return text_encoder, image_encoder, labels, start_epoch


if __name__ == "__main__":
    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    if args.gpu_id == -1:
        cfg.CUDA = False
    else:
        cfg.GPU_ID = args.gpu_id

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

    ##########################################################################
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = 'output/%s_%s_%s' % \
        (cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp)

    model_dir = os.path.join(output_dir, 'Model')
    image_dir = os.path.join(output_dir, 'Image')
    mkdir_p(model_dir)
    mkdir_p(image_dir)

    torch.cuda.set_device(cfg.GPU_ID)
    cudnn.benchmark = True

    # Get data loader ##################################################
    imsize = cfg.TREE.BASE_SIZE * (2 ** (cfg.TREE.BRANCH_NUM-1))
    #batch_size = cfg.TRAIN.BATCH_SIZE
    batch_size = 1
    image_transform = transforms.Compose([
        transforms.Scale(int(imsize * 76 / 64)),
        transforms.RandomCrop(imsize),
        transforms.RandomHorizontalFlip()])
    '''
    #dataset = TextDataset(cfg.DATA_DIR, 'train', base_size=cfg.TREE.BASE_SIZE, transform=image_transform)
    #dataset = TextBertDataset(cfg.DATA_DIR, 'train', base_size=cfg.TREE.BASE_SIZE, transform=image_transform)
    dataset = DevTextBertDataset(cfg.DATA_DIR, 'dev', base_size=cfg.TREE.BASE_SIZE, transform=image_transform)

    print(dataset.n_words, dataset.embeddings_num)
    assert dataset
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, drop_last=True,
        shuffle=True, num_workers=int(cfg.WORKERS))
    '''

    # # validation data #
    #dataset_val = TextDataset(cfg.DATA_DIR, 'test', base_size=cfg.TREE.BASE_SIZE, transform=image_transform)
    #dataset_val = TextBertDataset(cfg.DATA_DIR, 'test', base_size=cfg.TREE.BASE_SIZE, transform=image_transform)
    dataset_val = DevTextBertDataset(cfg.DATA_DIR, 'dev', base_size=cfg.TREE.BASE_SIZE, transform=image_transform)
    dataloader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=batch_size, drop_last=True,
        shuffle=True, num_workers=int(cfg.WORKERS))

    # Train ##############################################################
    text_encoder, image_encoder, labels, start_epoch = build_models()
    para = list(text_encoder.parameters())
    for v in image_encoder.parameters():
        if v.requires_grad:
            para.append(v)
    # optimizer = optim.Adam(para, lr=cfg.TRAIN.ENCODER_LR, betas=(0.5, 0.999))
    # At any point you can hit Ctrl + C to break out of training early.
    try:
        lr = cfg.TRAIN.ENCODER_LR
        if(True):
            print('dataloader_val : ', len(dataloader_val))
            if len(dataloader_val) > 0:
                s_loss, w_loss, t_loss = evaluate(dataloader_val, image_encoder,
                                          text_encoder, batch_size, labels)
                print('| end epoch {:3d} | valid loss '
                      '{:5.2f} {:5.2f} {:5.2f} | lr {:.5f}|'
                      .format(epoch, s_loss, w_loss, t_loss, lr))
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')
