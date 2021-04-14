'''
This script handles the training process.
'''

import argparse
import math
import os
import time
from tqdm import tqdm
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset, Dataset

from Vocab import Vocab
from data_functions import get_example_loader, covert_loader_to_dataset, from_batch_get_model_input

import transformer.Constants as Constants
from transformer.Models import Transformer
from transformer.Optim import ScheduledOptim

__author__ = "Yu-Hsiang Huang"

def cross_entropy(input, target, ignore_index=-100, reduction='mean'):
    return F.nll_loss(torch.log(input + 1e-08), target, None, None, ignore_index, None, reduction)



def cal_performance(pred, gold, classify_res, label, trg_pad_idx, smoothing=False, cover=None):
    # print(gold.shape)
    ''' Apply label smoothing if needed '''
    non_pad_mask = gold.ne(trg_pad_idx)
    n_word = non_pad_mask.sum().item()
    loss, loss_wo_cover, cover, bce_loss = cal_loss(pred, gold, classify_res, label, trg_pad_idx, smoothing=smoothing, cover=cover, n_word = n_word)
    pred = pred.view(-1, pred.size(2))
    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    non_pad_mask = non_pad_mask.view(-1)
    # print("pred: ",pred.shape)
    # print("gold: ",gold.shape)
    # print("non: ",non_pad_mask.shape)
    # exit()
    n_correct = pred.eq(gold).masked_select(non_pad_mask).sum().item()


    return loss, loss_wo_cover, cover, n_correct, n_word, bce_loss


def cal_loss(preds, golds, classify_res, label, trg_pad_idx, smoothing=False, cover=None, n_word = None):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''
    if smoothing:
        eps = 0.1
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = torch.log(pred + 1e-08)

        non_pad_mask = gold.ne(trg_pad_idx)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()  # average later
        loss += cover
    else:
        loss = torch.zeros([preds.size(0)], dtype=torch.float32).cuda()
        for idx, (pred, gold) in enumerate(zip(preds, golds)):
            loss[idx] += cross_entropy(pred, gold, ignore_index=trg_pad_idx, reduction='mean')

        if classify_res is not None:
            BCE = nn.CrossEntropyLoss(reduction='sum')
            # BCE = nn.CrossEntropyLoss()
            # bce_loss = torch.zeros([classify_res.shape[0]], dtype = torch.float32).cuda()
            # for idx, (pred_label, gold_label) in enumerate(zip(classify_res, label)):
            #     # bce_loss += BCE(pred_label, gold_label)
            #     bce_loss[idx] += BCE(pred_label, gold_label)
            bce_loss = BCE(classify_res, label)
        else:
            bce_loss = 0.


        #print("loss: ",loss)
        if cover is not None:
            final_loss = loss + 0.1 * cover
        else:
            final_loss = loss

        final_loss = torch.sum(final_loss) + 0.5 * bce_loss

        loss = torch.sum(loss)


        if cover is not None:
            cover = torch.sum(0.1*cover)



    return final_loss, loss, cover, 0.5 * bce_loss


def patch_src(src, pad_idx):
    src = src.transpose(0, 1)
    return src


def patch_trg(trg, pad_idx):
    trg = trg.transpose(0, 1)
    trg, gold = trg[:, :-1], trg[:, 1:]  #.contiguous().view(-1)
    return trg, gold


def train_epoch(model, training_data, optimizer, opt, smoothing):
    ''' Epoch operation in training phase'''

    model.train()
    total_loss, total_loss_wo_cover, total_cover, total_bce_loss, n_word_total, n_word_correct = 0, 0, 0, 0, 0, 0
    step = 0
    # vocab = Vocab(opt.vocab_path, opt.vocab_size)

    desc = '  - (Training)   '
    for batch in tqdm(training_data, mininterval=2, desc=desc, leave=False):

        # prepare data
        model_inputs = from_batch_get_model_input(batch, hidden_dim=opt.hidden_dim,
                                        use_pointer=opt.use_pointer, use_coverage=opt.use_coverage, 
                                        use_utter_trunc=opt.use_utter_trunc, use_user_mask=opt.use_user_mask, use_turns_mask=opt.use_turns_mask)
        #print(len(model_inputs))
        src_seq = model_inputs[0]
        trg_seq, gold = model_inputs[6], model_inputs[8]
        src_seq_with_oov = model_inputs[2]
        oov_zeros = model_inputs[3]
        attn_mask1 = None
        attn_mask2 = model_inputs[9]
        attn_mask3 = None
        init_coverage = model_inputs[5]
        
        if opt.use_cls_layers:
            article_lens = model_inputs[11]
            max_art_len = model_inputs[12]
            utt_num = model_inputs[13]
            label = model_inputs[10]
            total_utt_num = torch.sum(utt_num)
            sz = label.shape[0]
            tmp = []
            # label_per_batch = torch.zeros([total_utt_num], dtype=torch.long).cuda()
            for i in range(sz):
                tmp.append(label[i, :utt_num[i]])
            tmp = tuple(tmp)
            label_per_batch = torch.cat(tmp, dim = 0)
            label = label_per_batch
            # print(trg_seq)
            # print(gold)
            # exit()
        else:
            label = None
            article_lens = None
            max_art_len = None
            utt_num  = None

        # forward
        optimizer.zero_grad()

        pred, classify_res = model(src_seq, trg_seq, src_seq_with_oov, oov_zeros, attn_mask1, attn_mask2, attn_mask3, init_coverage, article_lens, utt_num)
        # print(init_coverage)

        coverage = init_coverage
        if coverage is not None:
            coverage = torch.mean(coverage, dim=-1)



        # predictions = pred.argmax(axis=-1).view(pred.size(0), -1)

        #pred = pred.view(-1, pred.size(2))
        #gold = gold.contiguous().view(-1)

        # backward and update parameters
        loss, loss_wo_cover, cover, n_correct, n_word, bce_loss = cal_performance(
            pred, gold, classify_res, label, opt.pad_idx, smoothing=smoothing, cover=coverage)
        loss.backward()
        # bce_loss.backward(retain_graph=True)
        #print(loss)
        # for name, parameters in model.named_parameters():
        #     if name == "encoder.src_word_emb.weight":
        #         print("encoder.src_word_emb.weight grad:", parameters.grad)
        # print('loss:', loss / n_word)
        # print('================================')
        # time.sleep(1)
        optimizer.step_and_update_lr()

        # note keeping
        step += 1
        n_word_total += n_word
        n_word_correct += n_correct
        total_loss += loss.item()
        total_loss_wo_cover += loss_wo_cover.item()
        # print("cover: ",cover)
        #if cover != 0.0:
        if cover is not None:
            total_cover += cover.item()
        if opt.use_cls_layers:
            total_bce_loss += bce_loss.item()
    #print("n_word_total: ",n_word_total)
    #exit()
    #print("step: ",step)
    loss_per_word = total_loss/step       #loss per batch
    loss_wo_cover_per_batch = total_loss_wo_cover/step     #loss per batch
    if cover is not None:
        cover_per_batch = total_cover/step
    else:
        cover_per_batch = None
    bce_loss_per_batch = total_bce_loss/step
    accuracy = n_word_correct/n_word_total
    return loss_per_word, loss_wo_cover_per_batch, cover_per_batch, bce_loss_per_batch, accuracy


def eval_epoch(model, validation_data, opt):
    ''' Epoch operation in evaluation phase '''

    model.eval()
    total_loss, total_loss_wo_cover, total_cover, total_bce_loss, n_word_total, n_word_correct = 0, 0, 0, 0, 0, 0
    step = 0

    desc = '  - (Validation) '
    with torch.no_grad():
        for batch in tqdm(validation_data, mininterval=2, desc=desc, leave=False):

            # prepare data
            model_inputs = from_batch_get_model_input(batch, hidden_dim=opt.hidden_dim,
                                        use_pointer=opt.use_pointer, use_coverage=opt.use_coverage, 
                                        use_utter_trunc=opt.use_utter_trunc, use_user_mask=opt.use_user_mask, use_turns_mask=opt.use_turns_mask)
            src_seq = model_inputs[0]
            trg_seq, gold = model_inputs[6], model_inputs[8]
            src_seq_with_oov = model_inputs[2]
            oov_zeros = model_inputs[3]
            attn_mask1 = None
            attn_mask2 = model_inputs[9]
            attn_mask3 = None
            init_coverage = model_inputs[5]
            article_lens = model_inputs[11]
            max_art_len = model_inputs[12]
            utt_num = model_inputs[13]

            if opt.use_cls_layers:
                label = model_inputs[10]
                total_utt_num = torch.sum(utt_num)
                sz = label.shape[0]
                tmp = []
                # label_per_batch = torch.zeros([total_utt_num], dtype=torch.long).cuda()
                for i in range(sz):
                    tmp.append(label[i, :utt_num[i]])
                tmp = tuple(tmp)
                label_per_batch = torch.cat(tmp, dim = 0)
                label = label_per_batch
            else:
                label = None

            # forward
            pred, classify_res = model(src_seq, trg_seq, src_seq_with_oov, oov_zeros, attn_mask1, attn_mask2, attn_mask3, init_coverage, article_lens, utt_num)

            #pred = pred.view(-1, pred.size(2))
            #gold = gold.contiguous().view(-1)

            coverage = init_coverage
            if coverage is not None:
                coverage = torch.mean(coverage, dim=-1)

            loss, loss_wo_cover, cover, n_correct, n_word, bce_loss = cal_performance(
                pred, gold, classify_res, label, opt.pad_idx, smoothing=False, cover = coverage)

            # note keeping
            step += 1
            n_word_total += n_word
            n_word_correct += n_correct
            total_loss += loss.item()
            total_loss_wo_cover += loss_wo_cover.item()
            if cover is not None:
                total_cover += cover.item()
            if opt.use_cls_layers:
                total_bce_loss += bce_loss.item()

    loss_per_word = total_loss/step   #loss per batch
    loss_wo_cover_per_batch = total_loss_wo_cover/step  #loss per batch
    if cover is not None:
        cover_per_batch = total_cover/step       #cover per batch
    else:
        cover_per_batch = None
    bce_loss_per_batch = total_bce_loss/step
    accuracy = n_word_correct/n_word_total
    return loss_per_word, loss_wo_cover_per_batch, cover_per_batch, bce_loss_per_batch, accuracy


def train(model, training_data, validation_data, optimizer, opt):
    ''' Start training '''

    log_train_file, log_valid_file = None, None

    if opt.log:
        if not os.path.exists(opt.log):
            os.makedirs(opt.log)
        log_train_file = os.path.join(opt.log, 'train.log')
        log_valid_file = os.path.join(opt.log, 'valid.log')

        print('[Info] Training performance will be written to file: {} and {}'.format(
            log_train_file, log_valid_file))

        with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
            log_tf.write('epoch,loss,loss_wo_cover,cover,ppl,accuracy\n')
            log_vf.write('epoch,loss,loss_wo_cover,cover,ppl,accuracy\n')

    def print_performances(header, loss, loss_wo_cover, cover, bce_loss, accu, start_time):
        #print("loss_wo_cover: ",loss_wo_cover)
        #print("cover: ",float(cover))
        print('  - {header:12} loss: {loss: 8.5f}, loss_wo_cover: {loss_wo_cover: 8.5f}, cover:{cover}, bce_loss:{bce_loss}, accuracy: {accu:3.3f} %, '\
              'elapse: {elapse:3.3f} min'.format(
                  header=f"({header})", loss=loss, loss_wo_cover=loss_wo_cover, cover=cover, bce_loss = bce_loss,
                  accu=100*accu, elapse=(time.time()-start_time)/60))

    # valid_accus = []
    valid_wo_cover_losses = []
    for epoch_i in range(opt.epoch):
        print('[ Epoch', epoch_i, ']')

        start = time.time()
        train_loss, train_loss_wo_cover, train_cover, train_bce_loss, train_accu = train_epoch(
            model, training_data, optimizer, opt, smoothing=opt.label_smoothing)
        print_performances('Training', train_loss, train_loss_wo_cover, train_cover,train_bce_loss, train_accu, start)

        start = time.time()
        valid_loss, valid_loss_wo_cover, valid_cover, valid_bce_loss, valid_accu = eval_epoch(model, validation_data, opt)
        print_performances('Validation', valid_loss, valid_loss_wo_cover, valid_cover, valid_bce_loss, valid_accu, start)

        valid_wo_cover_losses += [valid_loss_wo_cover]
        # valid_accus += [valid_accu]

        checkpoint = {'epoch': epoch_i, 'settings': opt, 'model': model.state_dict()}

        if opt.save_model:
            if opt.save_mode == 'all':
                model_name = opt.save_model + '_accu_{accu:3.3f}.chkpt'.format(accu=100*valid_accu)
                torch.save(checkpoint, model_name)
            elif opt.save_mode == 'best':
                model_name = opt.save_model + 'save_best.chkpt'
                # if valid_loss <= min(valid_losses):
                if valid_loss_wo_cover <= min(valid_wo_cover_losses):
                    torch.save(checkpoint, model_name)
                    print('    - [Info] The checkpoint file has been updated.')

        if log_train_file and log_valid_file:
            with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
                log_tf.write('{epoch},{loss: 8.5f},{loss_wo_cover: 8.5f},{cover},{bce_loss},{ppl: 8.5f},{accu:3.3f}\n'.format(
                    epoch=epoch_i, loss=train_loss, loss_wo_cover=train_loss_wo_cover, cover=train_cover, bce_loss=train_bce_loss,
                    ppl=math.exp(min(train_loss, 100)), accu=100*train_accu))
                log_vf.write('{epoch},{loss: 8.5f},{loss_wo_cover: 8.5f},{cover},{bce_loss},{ppl: 8.5f},{accu:3.3f}\n'.format(
                    epoch=epoch_i, loss=valid_loss,loss_wo_cover=valid_loss_wo_cover,cover=valid_cover,bce_loss=valid_bce_loss,
                    ppl=math.exp(min(valid_loss, 100)), accu=100*valid_accu))

def main():
    ''' 
    Usage:
    python train.py -data_pkl m30k_deen_shr.pkl -log m30k_deen_shr -embs_share_weight -proj_share_weight -label_smoothing -save_model trained -b 256 -warmup 128000
    '''

    parser = argparse.ArgumentParser()

    parser.add_argument('-train_path', type=str, default="/home/disk2/zfj2020/workspace/dataset/qichedashi/finished_csv_files/train.csv")
    parser.add_argument('-valid_path', type=str, default="/home/disk2/zfj2020/workspace/dataset/qichedashi/finished_csv_files/valid.csv")
    parser.add_argument('-label_path', type=str, default="./rouge_result_0.1.json")
    parser.add_argument("-vocab_path", type=str, default="/home/disk2/zfj2020/workspace/dataset/qichedashi/finished_csv_files/vocab")
    parser.add_argument("-vocab_size", type=int, default=50000)

    parser.add_argument('-epoch', type=int, default=100)
    parser.add_argument('-b', '--batch_size', type=int, default=15)

    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-d_inner_hid', type=int, default=2048)
    parser.add_argument('-d_k', type=int, default=64)
    parser.add_argument('-d_v', type=int, default=64)
    parser.add_argument('-hidden_dim', type=int, default=256)

    parser.add_argument('-n_head', type=int, default=8)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-warmup','--n_warmup_steps', type=int, default=500000)
    # parser.add_argument('-warmup','--n_warmup_steps', type=int, default=350000)

    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-embs_share_weight', action='store_true')
    parser.add_argument('-proj_share_weight', action='store_true')

    parser.add_argument('-log', default="./logs/dudu_test/")
    parser.add_argument('-save_model', default="./save_model/dudu_test/")
    parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')

    parser.add_argument('-pad_idx', type=int, default=0)
    parser.add_argument('-unk_idx', type=int, default=1)
    parser.add_argument('-bos_idx', type=int, default=2)
    parser.add_argument('-eos_idx', type=int, default=3)



    parser.add_argument('-use_pointer', type=bool, default=True)
    parser.add_argument("-use_coverage", action="store_true")
    parser.add_argument("-use_utter_trunc", action="store_true")
    parser.add_argument("-use_user_mask", action="store_true")
    parser.add_argument("-use_turns_mask", action="store_true")
    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-label_smoothing', action='store_true')
    parser.add_argument('-test_mode', action='store_true')
    parser.add_argument('-use_cls_layers', action='store_true')

    parser.add_argument('-max_article_len', type=int, default=300)
    parser.add_argument('-max_title_len', type=int, default=50)


    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda
    opt.d_word_vec = opt.d_model

    if not opt.log and not opt.save_model:
        print('No experiment result will be saved.')
        raise

    if opt.batch_size < 2048 and opt.n_warmup_steps <= 4000:
        print('[Warning] The warmup steps may be not enough.\n'\
              '(sz_b, warmup) = (2048, 4000) is the official setting.\n'\
              'Using smaller batch w/o longer warmup may cause '\
              'the warmup stage ends with only little data trained.')

    device = torch.device('cuda' if opt.cuda else 'cpu')

    #========= Loading Dataset =========#

    if all((opt.train_path, opt.valid_path)):
        training_data, validation_data = prepare_dataloaders(opt)
    else:
        raise

    print(opt)

    transformer = Transformer(
        opt.vocab_size,
        opt.vocab_size,
        src_pad_idx=opt.pad_idx,
        trg_pad_idx=opt.pad_idx,
        trg_emb_prj_weight_sharing=opt.proj_share_weight,
        emb_src_trg_weight_sharing=opt.embs_share_weight,
        d_k=opt.d_k,
        d_v=opt.d_v,
        d_model=opt.d_model,
        d_word_vec=opt.d_word_vec,
        d_inner=opt.d_inner_hid,
        n_layers=opt.n_layers,
        n_head=opt.n_head,
        dropout=opt.dropout,
        n_position=opt.max_article_len,
        use_pointer=opt.use_pointer,
        use_cls_layers=opt.use_cls_layers
        ).to(device)

    for name, parameters in transformer.named_parameters():
        print(name, parameters.requires_grad)


    optimizer = ScheduledOptim(
        optim.Adam(transformer.parameters(), betas=(0.9, 0.98), eps=1e-09),
        10.0, opt.d_model, opt.n_warmup_steps)

    train(transformer, training_data, validation_data, optimizer, opt)


# def prepare_dataloaders_from_bpe_files(opt, device):
#     batch_size = opt.batch_size
#     MIN_FREQ = 2
#     if not opt.embs_share_weight:
#         raise
#
#     data = pickle.load(open(opt.data_pkl, 'rb'))
#     MAX_LEN = data['settings'].max_len
#     field = data['vocab']
#     fields = (field, field)
#
#     def filter_examples_with_length(x):
#         return len(vars(x)['src']) <= MAX_LEN and len(vars(x)['trg']) <= MAX_LEN
#
#     train = TranslationDataset(
#         fields=fields,
#         path=opt.train_path,
#         exts=('.src', '.trg'),
#         filter_pred=filter_examples_with_length)
#     val = TranslationDataset(
#         fields=fields,
#         path=opt.val_path,
#         exts=('.src', '.trg'),
#         filter_pred=filter_examples_with_length)
#
#     opt.max_token_seq_len = MAX_LEN + 2
#     opt.src_pad_idx = opt.trg_pad_idx = field.vocab.stoi[Constants.PAD_WORD]
#     opt.src_vocab_size = opt.trg_vocab_size = len(field.vocab)
#
#     train_iterator = BucketIterator(train, batch_size=batch_size, device=device, train=True)
#     val_iterator = BucketIterator(val, batch_size=batch_size, device=device)
#     return train_iterator, val_iterator


def prepare_dataloaders(opt):
    batch_size = opt.batch_size
    #vocab = Vocab(opt.vocab_path, opt.vocab_size)

    train_examples = get_example_loader(opt.train_path, opt.label_path, opt.vocab_path, opt.vocab_size,
                                        opt.max_article_len, opt.max_title_len,
                                        use_pointer=opt.use_pointer,
                                        test_mode=opt.test_mode,
                                        use_utter_trunc=opt.use_utter_trunc,
                                        use_user_mask=opt.use_user_mask,
                                        use_turns_mask=opt.use_turns_mask)
    train_dataset = covert_loader_to_dataset(train_examples)
    train_sampler = RandomSampler(train_dataset) # for training random shuffle
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)

    valid_examples = get_example_loader(opt.valid_path, opt.label_path, opt.vocab_path, opt.vocab_size,
                                        opt.max_article_len, opt.max_title_len,
                                        use_pointer=opt.use_pointer,
                                        test_mode=opt.test_mode,
                                        use_utter_trunc=opt.use_utter_trunc,
                                        use_user_mask=opt.use_user_mask,
                                        use_turns_mask=opt.use_turns_mask)
    valid_dataset = covert_loader_to_dataset(valid_examples)
    valid_sampler = SequentialSampler(valid_dataset) # for training random shuffle
    valid_dataloader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=batch_size)



    return train_dataloader, valid_dataloader


if __name__ == '__main__':
    main()
