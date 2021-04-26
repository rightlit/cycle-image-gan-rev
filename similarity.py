from __future__ import print_function

from miscc.utils import mkdir_p
from miscc.utils import build_super_images
from miscc.losses import sent_loss, words_loss, image_to_text_loss
from miscc.config import cfg, cfg_from_file

from datasets import TextDataset, TextBertDataset
from datasets import prepare_data, prepare_data_bert

from model import BERT_RNN_ENCODER, BERT_CNN_ENCODER_RNN_DECODER
from model import RNN_ENCODER, CNN_ENCODER
from model import CNN_ENCODER_RNN_DECODER

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
from datasets import DevTextDataset
from datasets import prepare_data_dev
from GlobalAttention import func_attention
from miscc.losses import cosine_similarity

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
    parser.add_argument('--model_type', dest='model_type', type=str, default='bert')
    parser.add_argument('--local_pretrained', dest='local_pretrained', type=int, default=0)
    parser.add_argument('--manualSeed', type=int, default=0, help='manual seed')
    args = parser.parse_args()
    return args

def sent_probability(cnn_code, rnn_code, labels, class_ids,
              batch_size, eps=1e-8):
    # ### Mask mis-match samples  ###
    # that come from the same class as the real sample ###
    masks = []
    if class_ids is not None:
        for i in range(batch_size):
            mask = (class_ids == class_ids[i]).astype(np.uint8)
            mask[i] = 0
            masks.append(mask.reshape((1, -1)))
        masks = np.concatenate(masks, 0)
        # masks: batch_size x batch_size
        masks = torch.ByteTensor(masks)
        if cfg.CUDA:
            masks = masks.cuda()

    # --> seq_len x batch_size x nef
    if cnn_code.dim() == 2:
        cnn_code = cnn_code.unsqueeze(0)
        rnn_code = rnn_code.unsqueeze(0)

    # cnn_code_norm / rnn_code_norm: seq_len x batch_size x 1
    cnn_code_norm = torch.norm(cnn_code, 2, dim=2, keepdim=True)
    rnn_code_norm = torch.norm(rnn_code, 2, dim=2, keepdim=True)
    # scores* / norm*: seq_len x batch_size x batch_size
    scores0 = torch.bmm(cnn_code, rnn_code.transpose(1, 2))
    norm0 = torch.bmm(cnn_code_norm, rnn_code_norm.transpose(1, 2))
    scores0 = scores0 / norm0.clamp(min=eps) * cfg.TRAIN.SMOOTH.GAMMA3

    scores0 = scores0.squeeze()
    print('scores0 = ', scores0)
    sent_prob = scores0.item()
    return sent_prob

def words_similarity(img_features, words_emb, labels, cap_lens, class_ids, batch_size):
    """
        words_emb(query): batch x nef x seq_len
        img_features(context): batch x nef x 17 x 17
    """
    masks = []
    att_maps = []
    similarities = []
    cap_lens = cap_lens.data.tolist()
    #print(cap_lens)
    #print(words_emb)

    for i in range(batch_size):

        # Get the i-th text description
        words_num = cap_lens[i]
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
        #weiContext, attn = func_attention(word, context, GAMMA1)
        weiContext, attn = func_attention(word, context, cfg.TRAIN.SMOOTH.GAMMA1)
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
        #row_sim.mul_(GAMMA2).exp_()
        row_sim.mul_(cfg.TRAIN.SMOOTH.GAMMA2).exp_()
        row_sim = row_sim.sum(dim=1, keepdim=True)
        row_sim = torch.log(row_sim)
        #print(row_sim)
        row_sim = row_sim.cpu().squeeze(0)
        #print(row_sim.item())

        # --> 1 x batch_size
        # similarities(i, j): the similarity between the i-th image and the j-th text description
        similarities.append(row_sim.item())

    #average
    print(similarities)
    avg_sim = np.mean(similarities,axis=0)
    print('average(batch): ', avg_sim)
    return avg_sim

def evaluate(dataloader, cnn_model, rnn_model, batch_size, labels):
    cnn_model.eval()
    rnn_model.eval()

    s_total_loss = 0
    w_total_loss = 0
    t_total_loss = 0

    s_total_loss0 = 0
    s_total_loss1 = 0
    w_total_loss0 = 0
    w_total_loss1 = 0
   
    #debug_flag = False
    debug_flag = True
    similarities = []
    probabilities = []

    for step, data in enumerate(dataloader, 0):
        print('dataloader step : ', step, batch_size)
        if(debug_flag):
            with open('./debug0.pkl', 'wb') as f:
                pickle.dump({'data':data, 'cnn_model':cnn_model, 'rnn_model':rnn_model, 'labels':labels}, f)  

        #imgs, captions, cap_lens, class_ids, keys = prepare_data_bert(data, tokenizer=None)
        imgs, captions, cap_lens, class_ids, keys = prepare_data_dev(data)
        if(debug_flag):
            with open('./debug1.pkl', 'wb') as f:
                pickle.dump({'imgs':imgs, 'captions':captions, 'cap_lens':cap_lens, 'class_ids':class_ids, 'keys':keys}, f)  

        #print(imgs[-1].shape)
        #print(captions.shape, cap_lens.shape)
        captions = captions.unsqueeze(0)
        print(captions, cap_lens)
        #print(captions.shape, cap_lens.shape)

        print('evaluate(), model_type: ', model_type)
        if(model_type == 'bert'):
            words_features, sent_code, word_logits = cnn_model(imgs[-1], captions, cap_lens)
        else:
            words_features, sent_code = cnn_model(imgs[-1])
        
        # nef = words_features.size(1)
        # words_features = words_features.view(batch_size, nef, -1)

        hidden = rnn_model.init_hidden(batch_size)
        words_emb, sent_emb = rnn_model(captions, cap_lens, hidden)

        '''
        w_loss0, w_loss1, attn = words_loss(words_features, words_emb, labels,
                                            cap_lens, class_ids, batch_size)
        w_total_loss += (w_loss0 + w_loss1).data
        '''

        # similarity score
        print('calculating similarity')
        #print(words_features.shape, words_emb.shape)
        words_sim = words_similarity(words_features, words_emb, labels, cap_lens, class_ids, batch_size)
        similarities.append(words_sim)

        if(debug_flag):
            with open('./debug3.pkl', 'wb') as f:
                pickle.dump({'sent_code':sent_code, 'sent_emb':sent_emb, 'labels':labels, 'class_ids':class_ids, 'batch_size':batch_size}, f)  

        sent_prob = sent_probability(sent_code, sent_emb, labels, class_ids, batch_size)
        probabilities.append(sent_prob)

    # average
    print(similarities)
    avg_sim = np.mean(similarities,axis=0)
    std_sim = np.std(similarities,axis=0)
    print('similarities average(total), std: ', avg_sim, std_sim)

    avg_prob = np.mean(probabilities,axis=0)
    std_prob = np.std(probabilities,axis=0)
    print('probabilities average(total), std: ', avg_prob, std_prob)
    
    #return s_cur_loss, w_cur_loss, t_cur_loss
    return avg_sim


def build_models():
  
    # build model ############################################################
    print('build_model(), model_type: ', model_type)
    if(model_type == 'bert'):
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
        text_encoder = BERT_RNN_ENCODER(vocab_size, nhidden=cfg.TEXT.EMBEDDING_DIM)
        image_encoder = BERT_CNN_ENCODER_RNN_DECODER(cfg.TEXT.EMBEDDING_DIM, cfg.CNN_RNN.HIDDEN_DIM, vocab_size, rec_unit=cfg.RNN_TYPE)
    else:
        vocab_size = dataset_val.n_words
        text_encoder = RNN_ENCODER(vocab_size, nhidden=cfg.TEXT.EMBEDDING_DIM)
        image_encoder = CNN_ENCODER(cfg.TEXT.EMBEDDING_DIM)

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

    model_type = args.model_type

    #cfg.LOCAL_PRETRAINED = False
    if(args.local_pretrained == 1):
        cfg.LOCAL_PRETRAINED = True

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

    if(cfg.LOCAL_PRETRAINED):
        tokenizer = tokenization.FullTokenizer(vocab_file=cfg.BERT_ENCODER.VOCAB, do_lower_case=True)
    else:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # # validation data #
    #dataset_val = TextDataset(cfg.DATA_DIR, 'test', base_size=cfg.TREE.BASE_SIZE, transform=image_transform)
    #dataset_val = TextBertDataset(cfg.DATA_DIR, 'test', base_size=cfg.TREE.BASE_SIZE, transform=image_transform)
    #model_type = 'attn'

    cap_indices = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    if(model_type == 'bert'):
        #dataset_val = DevTextBertDataset(cfg.DATA_DIR, 'dev', base_size=cfg.TREE.BASE_SIZE, transform=image_transform)
        #dataset_val = DevTextBertDataset(cfg.DATA_DIR, 'dev', base_size=cfg.TREE.BASE_SIZE, transform=image_transform, tokenizer=tokenizer)
        dataset_val = DevTextBertDataset(cfg.DATA_DIR, 'dev', base_size=cfg.TREE.BASE_SIZE, transform=image_transform, tokenizer=tokenizer, cap_indices=cap_indices)
    else:
        #dataset_val = DevTextDataset(cfg.DATA_DIR, 'dev', base_size=cfg.TREE.BASE_SIZE, transform=image_transform)
        #dataset_val = DevTextDataset(cfg.DATA_DIR, 'dev', base_size=cfg.TREE.BASE_SIZE, transform=image_transform, tokenizer=None)
        dataset_val = DevTextDataset(cfg.DATA_DIR, 'dev', base_size=cfg.TREE.BASE_SIZE, transform=image_transform, tokenizer=None, cap_indices=None)

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
        print('dataloader_val : ', len(dataloader_val))
        if len(dataloader_val) > 0:
            #s_loss, w_loss, t_loss = evaluate(dataloader_val, image_encoder, text_encoder, batch_size, labels)
            similarities = evaluate(dataloader_val, image_encoder, text_encoder, batch_size, labels)
        print('similarities : ', similarities)

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')
