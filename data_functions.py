import os
import torch
import json
from torch.utils.data import DataLoader, RandomSampler, TensorDataset, Dataset
import pandas as pd
from tqdm import tqdm
from multiprocessing import Manager

from Vocab import SENTENCE_START, SENTENCE_END, PAD_TOKEN, UNKNOWN_TOKEN, START_DECODING, STOP_DECODING
from Vocab import article2ids, abstract2ids, Vocab


import numpy as np

class TestTensorDataset(Dataset):
    r"""Dataset wrapping tensors.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Arguments:
        *tensors (Tensor): tensors that have the same size of the first dimension.
    """

    def __init__(self, *tensors, titles, oovs, labels):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        manager = Manager()
        self.tensors = manager.list(tensors)
        self.titles = titles
        self.oovs = oovs
        self.labels = labels

    def __getitem__(self, index):
        tensor = tuple(tensor[index] for tensor in self.tensors)
        title = self.titles[index]
        oov = self.oovs[index]
        label = self.labels[index]
        return tensor, title, oov, label

    def __len__(self):
        return self.tensors[0].size(0)


class Example(object):
    '''
        Example:
            self.article
            self.title
            self.encoder_input
            self.encoder_mask
            self.encoder_input_with_oov
            self.decoder_input
            self.decoder_mask
            self.decoder_target
            self.decoder_target_with_oov
            self.article_oovs
            self.oov_len
    '''
    def __init__(self, article, title, label, utt_num, 
                 encoder_input, decoder_input, decoder_target,
                 encoder_input_with_oov, article_oovs, decoder_target_with_oov,
                 max_encoder_len, max_decoder_len, pad_idx=0, article_lens=None, mask_lens = None, use_utter_trunc=False,
                 use_user_mask=False, use_turns_mask = False, users=None):

        # articles & titles
        assert len(decoder_input) == len(decoder_target)
        self.article = article  # str
        self.article_lens = article_lens #list
        self.mask_lens = mask_lens
        self.title = title # str
        self.label = label  # list
        self.utt_num = utt_num # int

        self.encoder_input ,self.encoder_mask = \
            self._add_pad_and_gene_mask(encoder_input, max_encoder_len, pad_idx=pad_idx)
        self.encoder_input_with_oov = \
            self._add_pad_and_gene_mask(encoder_input_with_oov, max_encoder_len, pad_idx=pad_idx, return_mask=False)
        
        self.decoder_input, self.decoder_mask = \
            self._add_pad_and_gene_mask(decoder_input, max_decoder_len, pad_idx=pad_idx)
        self.decoder_target = \
            self._add_pad_and_gene_mask(decoder_target, max_decoder_len, pad_idx=pad_idx, return_mask=False)
        self.decoder_target_with_oov = \
            self._add_pad_and_gene_mask(decoder_target_with_oov, max_decoder_len, pad_idx=pad_idx, return_mask=False)


        self.article_oovs = article_oovs
        self.oov_len = len(article_oovs)

        self.use_utter_trunc = use_utter_trunc
        self.use_user_mask = use_user_mask
        self.use_turns_mask = use_turns_mask
        
        self.max_encoder_len = max_encoder_len
        
        self.user_mask_ = None
        if self.use_user_mask == True:
            self.users = users

    @classmethod
    def _add_pad_and_gene_mask(cls, x, max_len, pad_idx=0, return_mask=True):
        pad_len = max_len - len(x)
        assert pad_len >= 0
        if return_mask:
            mask = [1]*len(x)
            mask.extend([0] * pad_len)
            assert len(mask) == max_len

        x.extend([pad_idx] * pad_len)
        assert  len(x) == max_len

        if return_mask:
            return x, mask
        else:
            return x

def from_sample_covert_example(vocab, article, art_lens, mask_lens, title, label, max_article_len, max_title_len,
                               use_pointer=False, print_details=False, use_utter_trunc=False, use_user_mask=False, use_turns_mask=False):
    if 0 == len(title) or 0 == len(article):
        return None

    if len(article) <= len(title):
        return None

    #截断文章
    temp = 0
    len_list = []
    for len_ in art_lens:
        temp += len_
        if temp <= max_article_len:
            len_list.append(len_)
        else:
            break
    article_lens = len_list
    # print("article_lens:", article_lens)
    label = label[:len(article_lens)]
    utt_num = len(article_lens)
    # print(len(label), len(article_lens))
    # exit()

    #轮次mask，截断方式同句子mask，但是mask_lens列表要做一些处理
    # minus = sum(mask_lens)-sum(article_lens)
    # idx = len(mask_lens)-1
    # while minus > 0:
    #     if mask_lens[idx] >= minus:
    #         mask_lens[idx] = mask_lens[idx] - minus
    #         minus = 0
    #     else:
    #         minus = minus - mask_lens[idx]
    #         mask_lens = 0
    #         idx -= 1
    # mask_lens = [l for l in mask_lens if l != 0]
    # # print(sum(article_lens))
    # # print(sum(mask_lens))
    # assert sum(article_lens) == sum(mask_lens)
    # exit()
    
    users= None

    if use_utter_trunc or use_turns_mask:
        if len(article) > max_article_len:
            article = article[:sum(len_list)]
            assert len(article) == sum(len_list)
        else:
            article_lens = art_lens
            assert len(article) == sum(article_lens)
    else:
        if len(article) > max_article_len:
            article = article[:max_article_len]



    if use_user_mask:
        article = article[:sum(len_list)]
        assert len(article) == sum(article_lens)
        assert len(article) <= max_article_len
        user_mask_ = []
        start = 0
        for len_ in article_lens:
            dia = article[start: start + len_]
            dia = [dia[0] for _ in dia]
            user_mask_ += dia
            start += len_
        assert len(user_mask_) == len(article)
        all_users = list(set(user_mask_))
        user2id = {user: idx+1 for idx, user in enumerate(all_users)}
        users = [user2id[user] for user in user_mask_]



    if len(article) == 0:
        return None

    encoder_input = [vocab.word2id(word) for word in article]
    # 加上 start 和 end
    title = [START_DECODING] + title + [STOP_DECODING]
    # 截断，限制摘要的长度
    title = title[:max_title_len+1]
    title_idx = [vocab.word2id(word) for word in title]
    decoder_input = title_idx[:-1]
    decoder_target = title_idx[1:]
    assert len(decoder_target) == len(decoder_input) <= max_title_len

    encoder_input_with_oov = None
    decoder_target_with_oov = None
    article_oovs = None

    if use_pointer:
        encoder_input_with_oov, article_oovs = article2ids(article, vocab)
        decoder_target_with_oov = abstract2ids(title[1:], vocab, article_oovs)

    example = Example(
        article = article,
        article_lens = article_lens,
        mask_lens = mask_lens,
        title = title,
        label = label,
        utt_num = utt_num,
        encoder_input = encoder_input,
        decoder_input = decoder_input,
        decoder_target = decoder_target,
        encoder_input_with_oov = encoder_input_with_oov,
        decoder_target_with_oov = decoder_target_with_oov,
        article_oovs = article_oovs,
        max_encoder_len = max_article_len,
        max_decoder_len = max_title_len,
        pad_idx=0,
        use_utter_trunc=use_utter_trunc,
        use_user_mask=use_user_mask,
        use_turns_mask=use_turns_mask,
        users=users)

    if print_details:
        print("encoder_input :[{}]".format(" ".join([str(i) for i in example.encoder_input])))
        print("encoder_mask  :[{}]".format(" ".join([str(i) for i in example.encoder_mask])))
        print("encoder_input_with_oov :[{}]".format(" ".join([str(i) for i in example.encoder_input_with_oov])))
        print("decoder_input :[{}]".format(" ".join([str(i) for i in example.decoder_input])))
        print("decoder_mask  :[{}]".format(" ".join([str(i) for i in example.decoder_mask])))
        print("decoder_target  :[{}]".format(" ".join([str(i) for i in example.decoder_target])))
        print("decoder_target_with_oov  :[{}]".format(" ".join([str(i) for i in example.decoder_target_with_oov])))
        print("oovs          :[{}]".format(" ".join(example.article_oovs)))
        print("\n")

    return example

def get_example_loader(data_path, labels_path, vocab_path, vocab_size, max_article_len,
                       max_title_len, use_pointer, test_mode=False, test_num=100, use_utter_trunc=False, use_user_mask=False, use_turns_mask=False):
    assert os.path.exists(data_path)

    #TODO assume operating csv files
    # load datas and vocab
    print("[INFO] loading datas...")
    with open(labels_path, 'r', encoding='utf-8')as f:
        labels_data = json.load(f)
    df = pd.read_csv(data_path)
    qids = df['QID'].tolist()
    problems = df['Problem'].tolist()
    articles = df['Conversation'].tolist()
    titles = df['Report'].tolist()
    assert len(problems) == len(articles) == len(titles)
    print("[INFO] loading vocab...")
    vocab = Vocab(vocab_path, vocab_size)

    example_loader = []

    # print_details = True if test_mode else False
    print_details=False

    punctuation = ['。','，','？','！','、','；','：','“','”','‘','’','「','」',\
                    '『','』','（','）','[',']','【','】','——','~','《','》','<','>','/','\\','\\\\']


    for qid, prob, conv, tit in tqdm(zip(qids, problems, articles, titles)):
        try:
            # for qichedashi
            label = labels_data[qid]
            label = [1] + label
            art = "车主说：" + prob + "|" + conv
            #art_ = [jieba.lcut(line) for line in art_.split('|')] 

            # pre-processing
            art = list(art)
            for i in range(len(art)-1):
                if art[i] == '|' and art[i+1] not in ['车','技']:
                    art[i] = '^'
            art = ''.join(art)
            art = art.replace('^','')

            art_ = art.split('|')

            #tokenize(whitespace)
            art__ = ''
            for line in art_:
                line = list(line)
                for c in line:
                    if '\u4e00' <= c <= '\u9fa5': #中文范围
                        art__ += ' ' + c + ' '
                    elif c in punctuation:
                        art__ += ' ' + c + ' '
                    else:
                        art__ += c
                art__ += '|'
            
            art_ = art__
            art_ = [line.strip().split() for line in art_.split('|')]
            art_ = [line for line in art_ if line != []]

            mask_lens = []
            tmp_len = 0
            for i in range(len(art_)-1):
                if art_[i][0] == '技' and art_[i+1][0] == '车':
                    tmp_len += len(art_[i][4:]) + 2
                    mask_lens.append(tmp_len)
                    tmp_len = 0
                else:
                    tmp_len += len(art_[i][4:]) + 2
            tmp_len += len(art_[-1][4:]) + 2
            mask_lens.append(tmp_len)
            # print("mask_lens: ",mask_lens)
            

            art_ = [['<cls>'] + line[4:]+['<eou>'] for line in art_]
            # print(art_)


            art = []
            art_lens = []
            for line in art_:
                art += line
                art_lens.append(len(line))
            #print("art: ",art)
            # print("art_lens: ",art_lens)
            # break


            tit_ = ''
            tit = list(tit)
            for c in tit:
                if '\u4e00' <= c <= '\u9fa5': #中文范围
                    tit_ += ' ' + c + ' '
                elif c in punctuation:
                    tit_ += ' ' + c + ' '
                else:
                    tit_ += c
            tit = tit_.split(' ')
            tit = [word for word in tit if word != '']



        except:
            print('wrong line')
            # continue
            break
        example = from_sample_covert_example(
            vocab=vocab,
            article=art,
            art_lens=art_lens,
            mask_lens=mask_lens,
            title=tit,
            label = label,
            max_article_len=max_article_len,
            max_title_len=max_title_len,
            use_pointer=use_pointer,
            print_details=print_details,
            use_utter_trunc=use_utter_trunc,
            use_user_mask=use_user_mask,
            use_turns_mask=use_turns_mask
        )
        if example != None:
            example_loader.append(example)
        if test_mode:
            if len(example_loader) == test_num:
                break


    print("[INFO] all datas has been load...")
    print("[INFO] {} examples in total...".format(len(example_loader)))

    return example_loader

def get_example_loader_4_inference(prob, \
                                conv, \
                                tit, \
                                vocab, \
                                max_article_len, \
                                max_title_len, \
                                use_pointer, \
                                use_utter_trunc=False, \
                                use_user_mask=False, \
                                use_turns_mask=False):
    punctuation = ['。','，','？','！','、','；','：','“','”','‘','’','「','」',\
                    '『','』','（','）','[',']','【','】','——','~','《','》','<','>','/','\\','\\\\']
    # for qichedashi
    art = "车主说：" + prob + "|" + conv
    #art_ = [jieba.lcut(line) for line in art_.split('|')] 

    # pre-processing
    art = list(art)
    for i in range(len(art)-1):
        if art[i] == '|' and art[i+1] not in ['车','技']:
            art[i] = '^'
    art = ''.join(art)
    art = art.replace('^','')

    art_ = art.split('|')

    #tokenize(whitespace)
    art__ = ''
    for line in art_:
        line = list(line)
        for c in line:
            if '\u4e00' <= c <= '\u9fa5': #中文范围
                art__ += ' ' + c + ' '
            elif c in punctuation:
                art__ += ' ' + c + ' '
            else:
                art__ += c
        art__ += '|'
    
    art_ = art__
    art_ = [line.strip().split() for line in art_.split('|')]
    art_ = [line for line in art_ if line != []]

    mask_lens = []
    tmp_len = 0
    for i in range(len(art_)-1):
        if art_[i][0] == '技' and art_[i+1][0] == '车':
            tmp_len += len(art_[i][4:]) + 1
            mask_lens.append(tmp_len)
            tmp_len = 0
        else:
            tmp_len += len(art_[i][4:]) + 1
    tmp_len += len(art_[-1][4:]) + 1
    mask_lens.append(tmp_len)
    # print("mask_lens: ",mask_lens)
    

    art_ = [line[4:]+['<eou>'] for line in art_]

    art = []
    art_lens = []
    for line in art_:
        art += line
        art_lens.append(len(line))
    #print("art: ",art)
    # print("art_lens: ",art_lens)
    # break

    tit_ = ''
    tit = list(tit)
    for c in tit:
        if '\u4e00' <= c <= '\u9fa5': #中文范围
            tit_ += ' ' + c + ' '
        elif c in punctuation:
            tit_ += ' ' + c + ' '
        else:
            tit_ += c
    tit = tit_.split(' ')
    tit = [word for word in tit if word != '']


    example = from_sample_covert_example(
        vocab=vocab,
        article=art,
        art_lens=art_lens,
        mask_lens=mask_lens,
        title=tit,
        max_article_len=max_article_len,
        max_title_len=max_title_len,
        use_pointer=use_pointer,
        print_details=False,
        use_utter_trunc=use_utter_trunc,
        use_user_mask=use_user_mask,
        use_turns_mask=use_turns_mask
    )

    return example

def covert_loader_to_dataset(example_loader):
    def pad_to_same_length(lists):
        max_len = max(len(lst) for lst in lists)
        for i in range(len(lists)):
            if len(lists[i]) < max_len:
                lists[i] = lists[i] + [0] * (max_len - len(lists[i]))

    all_encoder_input = torch.tensor(np.array([ex.encoder_input for ex in example_loader]), dtype=torch.long)
    all_encoder_mask = torch.tensor(np.array([ex.encoder_mask for ex in example_loader]),dtype = torch.long)
    
    all_decoder_input = torch.tensor(np.array([ex.decoder_input for ex in example_loader]),dtype=torch.long)
    all_decoder_mask = torch.tensor(np.array([ex.decoder_mask for ex in example_loader]),dtype=torch.int)
    all_decoder_target = torch.tensor(np.array([ex.decoder_target for ex in example_loader]),dtype=torch.long)
    
    all_encoder_input_with_oov = torch.tensor(np.array([ex.encoder_input_with_oov for ex in example_loader]),dtype=torch.long )
    all_decoder_target_with_oov = torch.tensor(np.array([ex.decoder_target_with_oov for ex in example_loader]),dtype=torch.long )
    all_oov_len = torch.tensor(np.array([ex.oov_len for ex in example_loader]),dtype=torch.int)
    all_max_encoder_lens = torch.tensor([ex.max_encoder_len for ex in example_loader], dtype=torch.long)

    all_labels = [ex.label for ex in example_loader]
    pad_to_same_length(all_labels)
    all_labels = torch.tensor(all_labels, dtype=torch.long)

    all_utt_num = torch.tensor([ex.utt_num for ex in example_loader], dtype=torch.long)


    if example_loader[0].use_user_mask:
        all_users = [ex.users for ex in example_loader]
        # print(all_users)
        # exit()

        pad_to_same_length(all_users)
        all_users = torch.tensor(all_users, dtype=torch.long)

    # if example_loader[0].use_utter_trunc:

    
    all_article_lens = [ex.article_lens for ex in example_loader]
    max_art_len = max(len(art_lens) for art_lens in all_article_lens)
    pad_to_same_length(all_article_lens)
    all_article_lens = torch.tensor(all_article_lens, dtype=torch.long)
    size = all_article_lens.shape[0]
    max_art_len = torch.tensor([max_art_len]*size, dtype = torch.int)

    if example_loader[0].use_turns_mask:

        all_mask_lens = [ex.mask_lens for ex in example_loader]
        pad_to_same_length(all_mask_lens)    
        all_mask_lens = torch.tensor(all_mask_lens, dtype=torch.long)    

    if example_loader[0].use_utter_trunc and example_loader[0].use_user_mask and example_loader[0].use_turns_mask:  #utter + user + turns
        dataset = TensorDataset(all_encoder_input, all_encoder_mask, all_decoder_input, all_decoder_mask,
                                all_decoder_target, all_encoder_input_with_oov, all_decoder_target_with_oov,
                                all_oov_len, all_max_encoder_lens, all_article_lens, all_users, all_mask_lens)
    elif example_loader[0].use_utter_trunc and example_loader[0].use_user_mask and not example_loader[0].use_turns_mask:  # utter + user
        dataset = TensorDataset(all_encoder_input, all_encoder_mask, all_decoder_input, all_decoder_mask,
                                all_decoder_target, all_encoder_input_with_oov, all_decoder_target_with_oov,
                                all_oov_len,  all_max_encoder_lens, all_article_lens, all_users)
    elif example_loader[0].use_utter_trunc and not example_loader[0].use_user_mask and example_loader[0].use_turns_mask:   # utter + truns
        dataset = TensorDataset(all_encoder_input, all_encoder_mask, all_decoder_input, all_decoder_mask,
                                all_decoder_target, all_encoder_input_with_oov, all_decoder_target_with_oov,
                                all_oov_len, all_max_encoder_lens, all_article_lens, all_mask_lens)
    elif not example_loader[0].use_utter_trunc and example_loader[0].use_user_mask and example_loader[0].use_turns_mask:   # user + truns
        dataset = TensorDataset(all_encoder_input, all_encoder_mask, all_decoder_input, all_decoder_mask,
                                all_decoder_target, all_encoder_input_with_oov, all_decoder_target_with_oov,
                                all_oov_len, all_max_encoder_lens, all_users, all_mask_lens)
    elif example_loader[0].use_utter_trunc and not example_loader[0].use_user_mask and not example_loader[0].use_turns_mask:  # utter
        dataset = TensorDataset(all_encoder_input, all_encoder_mask, all_decoder_input, all_decoder_mask,
                                all_decoder_target, all_encoder_input_with_oov, all_decoder_target_with_oov,
                                all_oov_len, all_max_encoder_lens, all_article_lens)
    elif not example_loader[0].use_utter_trunc and example_loader[0].use_user_mask and not example_loader[0].use_turns_mask: # user
        dataset = TensorDataset(all_encoder_input, all_encoder_mask, all_decoder_input, all_decoder_mask,
                                all_decoder_target, all_encoder_input_with_oov, all_decoder_target_with_oov,
                                all_oov_len,  all_max_encoder_lens, all_users, all_labels, all_article_lens, max_art_len, all_utt_num)
    elif not example_loader[0].use_utter_trunc and not example_loader[0].use_user_mask and example_loader[0].use_turns_mask: # turns
        dataset = TensorDataset(all_encoder_input, all_encoder_mask, all_decoder_input, all_decoder_mask,
                                all_decoder_target, all_encoder_input_with_oov, all_decoder_target_with_oov,
                                all_oov_len, all_max_encoder_lens, all_mask_lens)
    else:
        dataset = TensorDataset(all_encoder_input,all_encoder_mask,all_decoder_input,all_decoder_mask, \
                                all_decoder_target,all_encoder_input_with_oov,all_decoder_target_with_oov,all_oov_len)
        '''无pointer版本'''
        # dataset = TensorDataset(all_encoder_input,all_encoder_mask,all_decoder_input,all_decoder_mask, all_decoder_target)


    return dataset

def covert_test_loader_to_dataset(example_loader):

    def pad_to_same_length(lists):
        max_len = max(len(lst) for lst in lists)
        for i in range(len(lists)):
            if len(lists[i]) < max_len:
                lists[i] = lists[i] + [0] * (max_len - len(lists[i]))
    all_encoder_input = torch.tensor(np.array([ex.encoder_input for ex in example_loader]), dtype=torch.long)
    all_encoder_mask = torch.tensor(np.array([ex.encoder_mask for ex in example_loader]),dtype = torch.long)
    
    all_decoder_input = torch.tensor(np.array([ex.decoder_input for ex in example_loader]),dtype=torch.long)
    all_decoder_mask = torch.tensor(np.array([ex.decoder_mask for ex in example_loader]),dtype=torch.int)
    
    all_decoder_target = torch.tensor(np.array([ex.decoder_target for ex in example_loader]),dtype=torch.long)
    
    all_encoder_input_with_oov = torch.tensor(np.array([ex.encoder_input_with_oov for ex in example_loader]),dtype=torch.long )
    all_decoder_target_with_oov = torch.tensor(np.array([ex.decoder_target_with_oov for ex in example_loader]),dtype=torch.long )
   
    
    all_oov_len = torch.tensor(np.array([ex.oov_len for ex in example_loader]),dtype=torch.int)
    all_max_encoder_lens = torch.tensor([ex.max_encoder_len for ex in example_loader], dtype=torch.long)

    # all_labels = [ex.label for ex in example_loader]
    # pad_to_same_length(all_labels)
    # all_labels = torch.tensor(all_labels, dtype=torch.long)

    all_article_lens = [ex.article_lens for ex in example_loader]
    pad_to_same_length(all_article_lens)
    all_article_lens = torch.tensor(all_article_lens, dtype=torch.long)


    all_utt_num = torch.tensor([ex.utt_num for ex in example_loader], dtype=torch.long)

    titles = np.array([f.title for f in example_loader])
    oovs = np.array([f.article_oovs for f in example_loader])
    labels = np.array([f.label for f in example_loader])


    if example_loader[0].use_user_mask:
        all_users = [ex.users for ex in example_loader]
        pad_to_same_length(all_users)
        all_users = torch.tensor(all_users, dtype=torch.long)

    if example_loader[0].use_utter_trunc:    
        all_article_lens = [ex.article_lens for ex in example_loader]
        pad_to_same_length(all_article_lens)
        all_article_lens = torch.tensor(all_article_lens, dtype=torch.long)
    
    if example_loader[0].use_turns_mask:
        all_mask_lens = [ex.mask_lens for ex in example_loader]
        pad_to_same_length(all_mask_lens)    
        all_mask_lens = torch.tensor(all_mask_lens, dtype=torch.long) 

    if example_loader[0].use_utter_trunc and example_loader[0].use_user_mask and example_loader[0].use_turns_mask:          #utter + user + turns
        dataset = TestTensorDataset(all_encoder_input, \
                                    all_encoder_mask, \
                                    all_decoder_input, \
                                    all_decoder_mask, \
                                    all_decoder_target, \
                                    all_encoder_input_with_oov, \
                                    all_decoder_target_with_oov, \
                                    all_oov_len, \
                                    all_max_encoder_lens, \
                                    all_article_lens, \
                                    all_users, \
                                    all_mask_lens, \
                                    titles=titles, \
                                    oovs=oovs)
    
    elif example_loader[0].use_utter_trunc and example_loader[0].use_user_mask and not example_loader[0].use_turns_mask:     #utter + user
        dataset = TestTensorDataset(all_encoder_input, all_encoder_mask, all_decoder_input, all_decoder_mask,
                                    all_decoder_target, all_encoder_input_with_oov, all_decoder_target_with_oov,
                                    all_oov_len, all_max_encoder_lens, all_article_lens, all_users, titles=titles, oovs=oovs)

    elif example_loader[0].use_utter_trunc and not example_loader[0].use_user_mask and example_loader[0].use_turns_mask:     #utter + turns
        dataset = TestTensorDataset(all_encoder_input, all_encoder_mask, all_decoder_input, all_decoder_mask,
                                    all_decoder_target, all_encoder_input_with_oov, all_decoder_target_with_oov,
                                    all_oov_len,all_max_encoder_lens, all_article_lens, all_mask_lens, titles=titles, oovs=oovs)

    elif not example_loader[0].use_utter_trunc and example_loader[0].use_user_mask and example_loader[0].use_turns_mask:      #user +turns
        dataset = TestTensorDataset(all_encoder_input, all_encoder_mask, all_decoder_input, all_decoder_mask,
                                    all_decoder_target, all_encoder_input_with_oov, all_decoder_target_with_oov,
                                    all_oov_len, all_max_encoder_lens, all_users, all_mask_lens, titles=titles, oovs=oovs)

    elif example_loader[0].use_utter_trunc and not example_loader[0].use_user_mask and not example_loader[0].use_turns_mask:    #utter
        dataset = TestTensorDataset(all_encoder_input, all_encoder_mask, all_decoder_input, all_decoder_mask,
                                    all_decoder_target, all_encoder_input_with_oov, all_decoder_target_with_oov,
                                    all_oov_len, all_max_encoder_lens, all_article_lens, titles=titles, oovs=oovs)

    elif not example_loader[0].use_utter_trunc and example_loader[0].use_user_mask and not example_loader[0].use_turns_mask:    #user
        dataset = TestTensorDataset(all_encoder_input, all_encoder_mask, all_decoder_input, all_decoder_mask,
                                    all_decoder_target, all_encoder_input_with_oov, all_decoder_target_with_oov,
                                    all_oov_len, all_max_encoder_lens, all_users, all_utt_num, all_article_lens, titles=titles, oovs=oovs, labels=labels)

    elif not example_loader[0].use_utter_trunc and not example_loader[0].use_user_mask and example_loader[0].use_turns_mask:    #turns
        dataset = TestTensorDataset(all_encoder_input, all_encoder_mask, all_decoder_input, all_decoder_mask,
                                    all_decoder_target, all_encoder_input_with_oov, all_decoder_target_with_oov,
                                    all_oov_len, all_max_encoder_lens, all_mask_lens, titles=titles, oovs=oovs)

    else:
        dataset = TestTensorDataset(all_encoder_input, all_encoder_mask, all_decoder_input, all_decoder_mask,
                              all_decoder_target, all_encoder_input_with_oov, all_decoder_target_with_oov, all_oov_len, titles=titles, oovs=oovs)
    return dataset

def from_batch_get_model_input(batch, hidden_dim, use_pointer=True, use_coverage=True, use_utter_trunc=False, use_user_mask=False, use_turns_mask=False):
    # common inputs
    all_encoder_input = batch[0]
    all_encoder_mask = batch[1]
    all_decoder_input = batch[2]
    all_decoder_mask = batch[3]
    all_decoder_target = batch[4]
    all_encoder_input_with_oov = batch[5]
    all_decoder_target_with_oov = batch[6]
    all_oov_len = batch[7]

    max_encoder_len = all_encoder_mask.sum(dim=-1).max()
    max_decoder_len = all_decoder_mask.sum(dim=-1).max()

    all_encoder_input = all_encoder_input[:, :max_encoder_len]
    all_encoder_mask = all_encoder_mask[:, :max_encoder_len]
    all_decoder_input = all_decoder_input[:, :max_decoder_len]
    all_decoder_mask = all_decoder_mask[:, :max_decoder_len]
    all_decoder_target = all_decoder_target[:, :max_decoder_len]
    all_encoder_input_with_oov = all_encoder_input_with_oov[:, :max_encoder_len]
    all_decoder_target_with_oov = all_decoder_target_with_oov[:, :max_decoder_len]

    batch_size = all_encoder_input.shape[0]
    max_oov_len = all_oov_len.max().item()

    oov_zeros = None
    if use_pointer:  # 当时用指针网络时，decoder_target应该要带上oovs
        all_decoder_target = all_decoder_target_with_oov
        if max_oov_len > 0:  # 使用指针时，并且在这个batch中存在oov的词汇，oov_zeros才不是None
            oov_zeros = torch.zeros((batch_size, max_oov_len), dtype=torch.float32)
    else:  # 当不使用指针时，带有oov的all_encoder_input_with_oov也不需要了
        all_encoder_input_with_oov = None

    init_coverage = None
    if use_coverage:
        init_coverage = torch.zeros(all_encoder_input.size(), dtype=torch.float32)  # 注意数据格式是float

    init_context_vec = torch.zeros((batch_size, 2 * hidden_dim), dtype=torch.float32)  # 注意数据格式是float

    if use_utter_trunc and use_user_mask and use_turns_mask:        # utter + user + turns 
        all_max_encoder_lens, all_article_lens, all_users, all_mask_lens = batch[8], batch[9], batch[10], batch[11]

        all_utter_mask = []
        for art_len, max_len in zip(all_article_lens, all_max_encoder_lens):
            all_utter_mask.append(_add_utter_pad(art_len, max_len))
        all_utter_mask = torch.cat(all_utter_mask, dim=0)
        all_utter_mask = all_utter_mask[:, :max_encoder_len, :max_encoder_len]

        all_user_mask = []
        for users, max_len in zip(all_users, all_max_encoder_lens):
            all_user_mask.append(_get_user_mask(users, max_len))
        all_user_mask = torch.cat(all_user_mask, dim=0)
        all_user_mask = all_user_mask[:, :max_encoder_len, :max_encoder_len]

        all_turns_mask = []
        for mask_len, max_len in zip(all_mask_lens, all_max_encoder_lens):
            all_turns_mask.append(_add_utter_pad(mask_len, max_len))
        all_turns_mask = torch.cat(all_turns_mask, dim=0)
        all_turns_mask = all_turns_mask[:, :max_encoder_len, :max_encoder_len]

        model_input = [all_encoder_input, all_encoder_mask, all_encoder_input_with_oov, oov_zeros, init_context_vec,
                        init_coverage, all_decoder_input, all_decoder_mask, all_decoder_target, all_utter_mask, all_user_mask, all_turns_mask]
        model_input = [t.cuda() if t is not None else None for t in model_input]

    elif use_utter_trunc and use_user_mask and not use_turns_mask:  # utter + user
        all_max_encoder_lens, all_article_lens, all_users = batch[8], batch[9], batch[10]

        all_utter_mask = []
        for art_len, max_len in zip(all_article_lens, all_max_encoder_lens):
            all_utter_mask.append(_add_utter_pad(art_len, max_len))
        all_utter_mask = torch.cat(all_utter_mask, dim=0)
        all_utter_mask = all_utter_mask[:, :max_encoder_len, :max_encoder_len]

        all_user_mask = []
        for users, max_len in zip(all_users, all_max_encoder_lens):
            all_user_mask.append(_get_user_mask(users, max_len))
        all_user_mask = torch.cat(all_user_mask, dim=0)
        all_user_mask = all_user_mask[:, :max_encoder_len, :max_encoder_len]


        model_input = [all_encoder_input, all_encoder_mask, all_encoder_input_with_oov, oov_zeros, init_context_vec,
                       init_coverage, all_decoder_input, all_decoder_mask, all_decoder_target, all_utter_mask, all_user_mask]
        model_input = [t.cuda() if t is not None else None for t in model_input]

    elif use_utter_trunc and not use_user_mask and use_turns_mask:  # utter + turns
        all_max_encoder_lens, all_article_lens, all_mask_lens = batch[8], batch[9], batch[10]

        all_utter_mask = []
        for art_len, max_len in zip(all_article_lens, all_max_encoder_lens):
            all_utter_mask.append(_add_utter_pad(art_len, max_len))
        all_utter_mask = torch.cat(all_utter_mask, dim=0)
        all_utter_mask = all_utter_mask[:, :max_encoder_len, :max_encoder_len]

        all_turns_mask = []
        for mask_len, max_len in zip(all_mask_lens, all_max_encoder_lens):
            all_turns_mask.append(_add_utter_pad(mask_len, max_len))
        all_turns_mask = torch.cat(all_turns_mask, dim=0)
        all_turns_mask = all_turns_mask[:, :max_encoder_len, :max_encoder_len]

        model_input = [all_encoder_input, all_encoder_mask, all_encoder_input_with_oov, oov_zeros, init_context_vec,
                        init_coverage, all_decoder_input, all_decoder_mask, all_decoder_target, all_utter_mask, all_turns_mask]
        model_input = [t.cuda() if t is not None else None for t in model_input]
    
    elif not use_utter_trunc and use_user_mask and use_turns_mask:  # user + turns
        all_max_encoder_lens, all_users, all_mask_lens = batch[8], batch[9], batch[10]

        all_user_mask = []
        for users, max_len in zip(all_users, all_max_encoder_lens):
            all_user_mask.append(_get_user_mask(users, max_len))
        all_user_mask = torch.cat(all_user_mask, dim=0)
        all_user_mask = all_user_mask[:, :max_encoder_len, :max_encoder_len]

        all_turns_mask = []
        for mask_len, max_len in zip(all_mask_lens, all_max_encoder_lens):
            all_turns_mask.append(_add_utter_pad(mask_len, max_len))
        all_turns_mask = torch.cat(all_turns_mask, dim=0)
        all_turns_mask = all_turns_mask[:, :max_encoder_len, :max_encoder_len]

        model_input = [all_encoder_input, all_encoder_mask, all_encoder_input_with_oov, oov_zeros, init_context_vec,
                        init_coverage, all_decoder_input, all_decoder_mask, all_decoder_target, all_user_mask, all_turns_mask]
        model_input = [t.cuda() if t is not None else None for t in model_input]
    
    elif use_utter_trunc and not use_user_mask and not use_turns_mask:  # utter
        all_max_encoder_lens, all_article_lens = batch[8], batch[9]

        all_utter_mask = []
        for mask_len, max_len in zip(all_article_lens, all_max_encoder_lens):
            all_utter_mask.append(_add_utter_pad(mask_len, max_len))
        all_utter_mask = torch.cat(all_utter_mask, dim=0)
        all_utter_mask = all_utter_mask[:, :max_encoder_len, :max_encoder_len]

        model_input = [all_encoder_input, all_encoder_mask, all_encoder_input_with_oov, oov_zeros, init_context_vec,
                       init_coverage, all_decoder_input, all_decoder_mask, all_decoder_target, all_utter_mask]
        model_input = [t.cuda() if t is not None else None for t in model_input]
    
    elif not use_utter_trunc and use_user_mask and not use_turns_mask:  # user
        all_max_encoder_lens, all_users, all_labels, all_article_lens, max_art_len, all_utt_num = batch[8], batch[9], batch[10], batch[11], batch[12], batch[13]

        all_user_mask = []
        for users, max_len in zip(all_users, all_max_encoder_lens):
            all_user_mask.append(_get_user_mask(users, max_len))
        all_user_mask = torch.cat(all_user_mask, dim=0)
        all_user_mask = all_user_mask[:, :max_encoder_len, :max_encoder_len]


        model_input = [all_encoder_input, all_encoder_mask, all_encoder_input_with_oov, oov_zeros, init_context_vec,
                       init_coverage, all_decoder_input, all_decoder_mask, all_decoder_target, all_user_mask, all_labels, all_article_lens, max_art_len, all_utt_num]
        model_input = [t.cuda() if t is not None else None for t in model_input]

    elif not use_utter_trunc and not use_user_mask and use_turns_mask:  #turns
        all_max_encoder_lens, all_mask_lens = batch[8], batch[9]
        
        all_turns_mask = []
        for mask_len, max_len in zip(all_mask_lens, all_max_encoder_lens):
            all_turns_mask.append(_add_utter_pad(mask_len, max_len))
        all_turns_mask = torch.cat(all_turns_mask, dim=0)
        all_turns_mask = all_turns_mask[:, :max_encoder_len, :max_encoder_len]

        model_input = [all_encoder_input, all_encoder_mask, all_encoder_input_with_oov, oov_zeros, init_context_vec,
                        init_coverage, all_decoder_input, all_decoder_mask, all_decoder_target, all_turns_mask]
        model_input = [t.cuda() if t is not None else None for t in model_input]

    else:
        model_input = [all_encoder_input,
                    all_encoder_mask,
                    all_encoder_input_with_oov,
                    oov_zeros,
                    init_context_vec,
                    init_coverage,
                    all_decoder_input,
                    all_decoder_mask,
                    all_decoder_target]

        # ''' 无pointer版本'''
        # model_input = [all_encoder_input, all_encoder_mask, all_decoder_input, all_decoder_mask, all_decoder_target]
        # model_input = [t.cuda() if t is not None else None for t in model_input]
    return model_input

def from_test_batch_get_model_input(batch,hidden_dim, use_pointer=True, use_coverage=True, use_utter_trunc=False, use_user_mask=False, use_turns_mask=False, inference=False):
    if inference:
        (encoder_input, \
        encoder_mask, \
        decoder_input, \
        decoder_mask, \
        decoder_target, \
        encoder_input_with_oov, \
        decoder_target_with_oov, \
        oov_len, \
        max_encoder_lens, \
        article_lens, \
        users, \
        mask_lens), title, oov = batch
        max_encoder_len = encoder_mask.sum(dim=-1).max()
        max_decoder_len = decoder_mask.sum(dim=-1).max()

        utter_mask = []
        utter_mask.append(_add_utter_pad(article_lens, max_encoder_lens))
        utter_mask = torch.cat(utter_mask, dim=0)
        # utter_mask = utter_mask[:, :max_encoder_len, :max_encoder_len]

        user_mask = []
        user_mask.append(_get_user_mask(users, max_encoder_lens))
        user_mask = torch.cat(user_mask, dim=0)
        # user_mask = user_mask[:, :max_encoder_len, :max_encoder_len]

        turns_mask = []
        turns_mask.append(_add_utter_pad(mask_lens, max_encoder_lens))
        turns_mask = torch.cat(turns_mask, dim=0)
        # turns_mask = turns_mask[:, :max_encoder_len, :max_encoder_len]
        # print(turns_mask.size())

        # print(max_encoder_len)
        # print(encoder_input.size())
        # exit()
        encoder_input = encoder_input.unsqueeze(0)
        encoder_mask = encoder_mask.unsqueeze(0)
        decoder_input = decoder_input.unsqueeze(0)
        decoder_mask = decoder_mask.unsqueeze(0)
        decoder_target = decoder_target.unsqueeze(0)
        encoder_input_with_oov = encoder_input_with_oov.unsqueeze(0)
        decoder_target_with_oov = decoder_target_with_oov.unsqueeze(0)

        batch_size = encoder_input.shape[0]
        max_oov_len = oov_len.max().item()

        oov_zeros = None
        if use_pointer:  # 当时用指针网络时，decoder_target应该要带上oovs
            decoder_target = decoder_target_with_oov
            if max_oov_len > 0:  # 使用指针时，并且在这个batch中存在oov的词汇，oov_zeros才不是None
                oov_zeros = torch.zeros((batch_size, max_oov_len), dtype=torch.float32)
        else:  # 当不使用指针时，带有oov的encoder_input_with_oov也不需要了
            encoder_input_with_oov = None

        init_coverage = None
        if use_coverage:
            init_coverage = torch.zeros(encoder_input.size(), dtype=torch.float32)  # 注意数据格式是float

        init_context_vec = torch.zeros((batch_size, 2 * hidden_dim), dtype=torch.float32)  # 注意数据格式是float

        model_input = [encoder_input, \
                        encoder_mask, \
                        encoder_input_with_oov, \
                        oov_zeros, \
                        init_context_vec, \
                       init_coverage, \
                       utter_mask, \
                       user_mask, \
                       turns_mask, \
                       decoder_input, \
                       decoder_mask, \
                       decoder_target, \
                       max_encoder_len]
        model_input = [t.cuda() if t is not None else None for t in model_input]

        return model_input, title, oov

    if use_utter_trunc and use_user_mask and use_turns_mask:   #utter + user + turns
        (all_encoder_input, \
        all_encoder_mask, \
        all_decoder_input, \
        all_decoder_mask, \
        all_decoder_target, \
        all_encoder_input_with_oov, \
        all_decoder_target_with_oov, \
        all_oov_len, \
        all_max_encoder_lens, \
        all_article_lens, \
        all_users, \
        all_mask_lens), title, oov = batch
        max_encoder_len = all_encoder_mask.sum(dim=-1).max()
        max_decoder_len = all_decoder_mask.sum(dim=-1).max()

        all_utter_mask = []
        for art_len, max_len in zip(all_article_lens, all_max_encoder_lens):
            all_utter_mask.append(_add_utter_pad(art_len, max_len))
        all_utter_mask = torch.cat(all_utter_mask, dim=0)
        all_utter_mask = all_utter_mask[:, :max_encoder_len, :max_encoder_len]

        all_user_mask = []
        for users, max_len in zip(all_users, all_max_encoder_lens):
            all_user_mask.append(_get_user_mask(users, max_len))
        all_user_mask = torch.cat(all_user_mask, dim=0)
        all_user_mask = all_user_mask[:, :max_encoder_len, :max_encoder_len]

        all_turns_mask = []
        for mask_len, max_len in zip(all_mask_lens, all_max_encoder_lens):
            all_turns_mask.append(_add_utter_pad(mask_len, max_len))
        all_turns_mask = torch.cat(all_turns_mask, dim=0)
        all_turns_mask = all_turns_mask[:, :max_encoder_len, :max_encoder_len]

        all_encoder_input = all_encoder_input[:, :max_encoder_len]
        all_encoder_mask = all_encoder_mask[:, :max_encoder_len]
        all_decoder_input = all_decoder_input[:, :max_decoder_len]
        all_decoder_mask = all_decoder_mask[:, :max_decoder_len]
        all_decoder_target = all_decoder_target[:, :max_decoder_len]
        all_encoder_input_with_oov = all_encoder_input_with_oov[:, :max_encoder_len]
        all_decoder_target_with_oov = all_decoder_target_with_oov[:, :max_decoder_len]

        batch_size = all_encoder_input.shape[0]
        max_oov_len = all_oov_len.max().item()

        oov_zeros = None
        if use_pointer:  # 当时用指针网络时，decoder_target应该要带上oovs
            all_decoder_target = all_decoder_target_with_oov
            if max_oov_len > 0:  # 使用指针时，并且在这个batch中存在oov的词汇，oov_zeros才不是None
                oov_zeros = torch.zeros((batch_size, max_oov_len), dtype=torch.float32)
        else:  # 当不使用指针时，带有oov的all_encoder_input_with_oov也不需要了
            all_encoder_input_with_oov = None

        init_coverage = None
        if use_coverage:
            init_coverage = torch.zeros(all_encoder_input.size(), dtype=torch.float32)  # 注意数据格式是float

        init_context_vec = torch.zeros((batch_size, 2 * hidden_dim), dtype=torch.float32)  # 注意数据格式是float

        model_input = [all_encoder_input, all_encoder_mask, all_encoder_input_with_oov, oov_zeros, init_context_vec,
                       init_coverage, all_utter_mask, all_user_mask, all_turns_mask, all_decoder_input, all_decoder_mask, all_decoder_target]
        model_input = [t.cuda() if t is not None else None for t in model_input]

    elif use_utter_trunc and use_user_mask and not use_turns_mask:   #utter + user
        (all_encoder_input, all_encoder_mask, all_decoder_input, all_decoder_mask, all_decoder_target, \
         all_encoder_input_with_oov, all_decoder_target_with_oov, all_oov_len, all_max_encoder_lens, \
         all_article_lens, all_users), title, oov = batch
        max_encoder_len = all_encoder_mask.sum(dim=-1).max()
        max_decoder_len = all_decoder_mask.sum(dim=-1).max()

        all_utter_mask = []
        for art_len, max_len in zip(all_article_lens, all_max_encoder_lens):
            all_utter_mask.append(_add_utter_pad(art_len, max_len))
        all_utter_mask = torch.cat(all_utter_mask, dim=0)
        all_utter_mask = all_utter_mask[:, :max_encoder_len, :max_encoder_len]

        all_user_mask = []
        for users, max_len in zip(all_users, all_max_encoder_lens):
            all_user_mask.append(_get_user_mask(users, max_len))
        all_user_mask = torch.cat(all_user_mask, dim=0)
        all_user_mask = all_user_mask[:, :max_encoder_len, :max_encoder_len]

        all_encoder_input = all_encoder_input[:, :max_encoder_len]
        all_encoder_mask = all_encoder_mask[:, :max_encoder_len]
        all_decoder_input = all_decoder_input[:, :max_decoder_len]
        all_decoder_mask = all_decoder_mask[:, :max_decoder_len]
        all_decoder_target = all_decoder_target[:, :max_decoder_len]
        all_encoder_input_with_oov = all_encoder_input_with_oov[:, :max_encoder_len]
        all_decoder_target_with_oov = all_decoder_target_with_oov[:, :max_decoder_len]

        batch_size = all_encoder_input.shape[0]
        max_oov_len = all_oov_len.max().item()

        oov_zeros = None
        if use_pointer:  # 当时用指针网络时，decoder_target应该要带上oovs
            all_decoder_target = all_decoder_target_with_oov
            if max_oov_len > 0:  # 使用指针时，并且在这个batch中存在oov的词汇，oov_zeros才不是None
                oov_zeros = torch.zeros((batch_size, max_oov_len), dtype=torch.float32)
        else:  # 当不使用指针时，带有oov的all_encoder_input_with_oov也不需要了
            all_encoder_input_with_oov = None

        init_coverage = None
        if use_coverage:
            init_coverage = torch.zeros(all_encoder_input.size(), dtype=torch.float32)  # 注意数据格式是float

        init_context_vec = torch.zeros((batch_size, 2 * hidden_dim), dtype=torch.float32)  # 注意数据格式是float

        model_input = [all_encoder_input, all_encoder_mask, all_encoder_input_with_oov, oov_zeros, init_context_vec,
                       init_coverage, all_utter_mask, all_user_mask, all_decoder_input, all_decoder_mask, all_decoder_target]
        model_input = [t.cuda() if t is not None else None for t in model_input]

    elif use_utter_trunc and not use_user_mask and use_turns_mask:   #utter + turns
        (all_encoder_input, all_encoder_mask, all_decoder_input, all_decoder_mask, all_decoder_target, \
         all_encoder_input_with_oov, all_decoder_target_with_oov, all_oov_len, all_max_encoder_lens, \
         all_article_lens, all_mask_lens), title, oov = batch
        max_encoder_len = all_encoder_mask.sum(dim=-1).max()
        max_decoder_len = all_decoder_mask.sum(dim=-1).max()

        all_utter_mask = []
        for mask_len, max_len in zip(all_article_lens, all_max_encoder_lens):
            all_utter_mask.append(_add_utter_pad(mask_len, max_len))
        all_utter_mask = torch.cat(all_utter_mask, dim=0)
        all_utter_mask = all_utter_mask[:, :max_encoder_len, :max_encoder_len]
        # print(all_utter_mask.size())
        # exit()

        all_turns_mask = []
        for mask_len, max_len in zip(all_mask_lens, all_max_encoder_lens):
            all_turns_mask.append(_add_utter_pad(mask_len, max_len))
        all_turns_mask = torch.cat(all_turns_mask, dim=0)
        all_turns_mask = all_turns_mask[:, :max_encoder_len, :max_encoder_len]

        all_encoder_input = all_encoder_input[:, :max_encoder_len]
        all_encoder_mask = all_encoder_mask[:, :max_encoder_len]
        all_decoder_input = all_decoder_input[:, :max_decoder_len]
        all_decoder_mask = all_decoder_mask[:, :max_decoder_len]
        all_decoder_target = all_decoder_target[:, :max_decoder_len]
        all_encoder_input_with_oov = all_encoder_input_with_oov[:, :max_encoder_len]
        all_decoder_target_with_oov = all_decoder_target_with_oov[:, :max_decoder_len]

        batch_size = all_encoder_input.shape[0]
        max_oov_len = all_oov_len.max().item()

        oov_zeros = None
        if use_pointer:  # 当时用指针网络时，decoder_target应该要带上oovs
            all_decoder_target = all_decoder_target_with_oov
            if max_oov_len > 0:  # 使用指针时，并且在这个batch中存在oov的词汇，oov_zeros才不是None
                oov_zeros = torch.zeros((batch_size, max_oov_len), dtype=torch.float32)
        else:  # 当不使用指针时，带有oov的all_encoder_input_with_oov也不需要了
            all_encoder_input_with_oov = None

        init_coverage = None
        if use_coverage:
            #init_coverage = torch.zeros(all_encoder_input.size(), dtype=torch.float32)  # 注意数据格式是float
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            init_coverage = torch.tensor(0,dtype=torch.float).to(device)

        init_context_vec = torch.zeros((batch_size, 2 * hidden_dim), dtype=torch.float32)  # 注意数据格式是float

        model_input = [all_encoder_input, all_encoder_mask, all_encoder_input_with_oov, oov_zeros, init_context_vec,
                       init_coverage, all_utter_mask, all_turns_mask, all_decoder_input, all_decoder_mask, all_decoder_target]
        model_input = [t.cuda() if t is not None else None for t in model_input]

    elif not use_utter_trunc and use_user_mask and use_turns_mask:   #user + turns
        (all_encoder_input, all_encoder_mask, all_decoder_input, all_decoder_mask, all_decoder_target, \
         all_encoder_input_with_oov, all_decoder_target_with_oov, all_oov_len, all_max_encoder_lens, \
         all_users, all_mask_lens), title, oov = batch
        max_encoder_len = all_encoder_mask.sum(dim=-1).max()
        max_decoder_len = all_decoder_mask.sum(dim=-1).max()

        all_user_mask = []
        for users, max_len in zip(all_users, all_max_encoder_lens):
            all_user_mask.append(_get_user_mask(users, max_len))
        all_user_mask = torch.cat(all_user_mask, dim=0)
        all_user_mask = all_user_mask[:, :max_encoder_len, :max_encoder_len]

        all_turns_mask = []
        for mask_len, max_len in zip(all_mask_lens, all_max_encoder_lens):
            all_turns_mask.append(_add_utter_pad(mask_len, max_len))
        all_turns_mask = torch.cat(all_turns_mask, dim=0)
        all_turns_mask = all_turns_mask[:, :max_encoder_len, :max_encoder_len]


        all_encoder_input = all_encoder_input[:, :max_encoder_len]
        all_encoder_mask = all_encoder_mask[:, :max_encoder_len]
        all_decoder_input = all_decoder_input[:, :max_decoder_len]
        all_decoder_mask = all_decoder_mask[:, :max_decoder_len]
        all_decoder_target = all_decoder_target[:, :max_decoder_len]
        all_encoder_input_with_oov = all_encoder_input_with_oov[:, :max_encoder_len]
        all_decoder_target_with_oov = all_decoder_target_with_oov[:, :max_decoder_len]

        batch_size = all_encoder_input.shape[0]
        max_oov_len = all_oov_len.max().item()

        oov_zeros = None
        if use_pointer:  # 当时用指针网络时，decoder_target应该要带上oovs
            all_decoder_target = all_decoder_target_with_oov
            if max_oov_len > 0:  # 使用指针时，并且在这个batch中存在oov的词汇，oov_zeros才不是None
                oov_zeros = torch.zeros((batch_size, max_oov_len), dtype=torch.float32)
        else:  # 当不使用指针时，带有oov的all_encoder_input_with_oov也不需要了
            all_encoder_input_with_oov = None

        init_coverage = None
        if use_coverage:
            init_coverage = torch.zeros(all_encoder_input.size(), dtype=torch.float32)  # 注意数据格式是float

        init_context_vec = torch.zeros((batch_size, 2 * hidden_dim), dtype=torch.float32)  # 注意数据格式是float

        model_input = [all_encoder_input, all_encoder_mask, all_encoder_input_with_oov, oov_zeros, init_context_vec,
                       init_coverage, all_user_mask, all_turns_mask, all_decoder_input, all_decoder_mask, all_decoder_target]
        model_input = [t.cuda() if t is not None else None for t in model_input]

    elif use_utter_trunc and not use_user_mask and not use_turns_mask:  #utter
        (all_encoder_input, all_encoder_mask, all_decoder_input, all_decoder_mask, all_decoder_target, \
         all_encoder_input_with_oov, all_decoder_target_with_oov, all_oov_len, all_max_encoder_lens, \
         all_article_lens), title, oov = batch
        max_encoder_len = all_encoder_mask.sum(dim=-1).max()
        max_decoder_len = all_decoder_mask.sum(dim=-1).max()

        all_utter_mask = []
        for mask_len, max_len in zip(all_article_lens, all_max_encoder_lens):
            all_utter_mask.append(_add_utter_pad(mask_len, max_len))
        all_utter_mask = torch.cat(all_utter_mask, dim=0)
        all_utter_mask = all_utter_mask[:, :max_encoder_len, :max_encoder_len]

        all_encoder_input = all_encoder_input[:, :max_encoder_len]
        all_encoder_mask = all_encoder_mask[:, :max_encoder_len]
        all_decoder_input = all_decoder_input[:, :max_decoder_len]
        all_decoder_mask = all_decoder_mask[:, :max_decoder_len]
        all_decoder_target = all_decoder_target[:, :max_decoder_len]
        all_encoder_input_with_oov = all_encoder_input_with_oov[:, :max_encoder_len]
        all_decoder_target_with_oov = all_decoder_target_with_oov[:, :max_decoder_len]

        batch_size = all_encoder_input.shape[0]
        max_oov_len = all_oov_len.max().item()

        oov_zeros = None
        if use_pointer:  # 当时用指针网络时，decoder_target应该要带上oovs
            all_decoder_target = all_decoder_target_with_oov
            if max_oov_len > 0:  # 使用指针时，并且在这个batch中存在oov的词汇，oov_zeros才不是None
                oov_zeros = torch.zeros((batch_size, max_oov_len), dtype=torch.float32)
        else:  # 当不使用指针时，带有oov的all_encoder_input_with_oov也不需要了
            all_encoder_input_with_oov = None

        init_coverage = None
        if use_coverage:
            init_coverage = torch.zeros(all_encoder_input.size(), dtype=torch.float32)  # 注意数据格式是float

        init_context_vec = torch.zeros((batch_size, 2 * hidden_dim), dtype=torch.float32)  # 注意数据格式是float

        model_input = [all_encoder_input, all_encoder_mask, all_encoder_input_with_oov, oov_zeros, init_context_vec,
                       init_coverage, all_utter_mask, all_decoder_input, all_decoder_mask, all_decoder_target]
        model_input = [t.cuda() if t is not None else None for t in model_input]

    elif not use_utter_trunc and use_user_mask and not use_turns_mask:  #user
        (all_encoder_input, all_encoder_mask, all_decoder_input, all_decoder_mask, all_decoder_target, \
         all_encoder_input_with_oov, all_decoder_target_with_oov, all_oov_len, all_max_encoder_lens, \
         all_users, all_utt_num, all_article_lens), title, oov, labels = batch
        max_encoder_len = all_encoder_mask.sum(dim=-1).max()
        max_decoder_len = all_decoder_mask.sum(dim=-1).max()

        all_user_mask = []
        for users, max_len in zip(all_users, all_max_encoder_lens):
            all_user_mask.append(_get_user_mask(users, max_len))
        all_user_mask = torch.cat(all_user_mask, dim=0)
        all_user_mask = all_user_mask[:, :max_encoder_len, :max_encoder_len]

        all_encoder_input = all_encoder_input[:, :max_encoder_len]
        all_encoder_mask = all_encoder_mask[:, :max_encoder_len]
        all_decoder_input = all_decoder_input[:, :max_decoder_len]
        all_decoder_mask = all_decoder_mask[:, :max_decoder_len]
        all_decoder_target = all_decoder_target[:, :max_decoder_len]
        all_encoder_input_with_oov = all_encoder_input_with_oov[:, :max_encoder_len]
        all_decoder_target_with_oov = all_decoder_target_with_oov[:, :max_decoder_len]

        batch_size = all_encoder_input.shape[0]
        max_oov_len = all_oov_len.max().item()

        oov_zeros = None
        if use_pointer:  # 当时用指针网络时，decoder_target应该要带上oovs
            all_decoder_target = all_decoder_target_with_oov
            if max_oov_len > 0:  # 使用指针时，并且在这个batch中存在oov的词汇，oov_zeros才不是None
                oov_zeros = torch.zeros((batch_size, max_oov_len), dtype=torch.float32)
        else:  # 当不使用指针时，带有oov的all_encoder_input_with_oov也不需要了
            all_encoder_input_with_oov = None

        init_coverage = None
        if use_coverage:
            init_coverage = torch.zeros(all_encoder_input.size(), dtype=torch.float32)  # 注意数据格式是float

        init_context_vec = torch.zeros((batch_size, 2 * hidden_dim), dtype=torch.float32)  # 注意数据格式是float

        model_input = [all_encoder_input, all_encoder_mask, all_encoder_input_with_oov, oov_zeros, init_context_vec,
                       init_coverage, all_user_mask, all_decoder_input, all_decoder_mask, all_decoder_target, all_utt_num, all_article_lens]
        model_input = [t.cuda() if t is not None else None for t in model_input]

    elif not use_utter_trunc and not use_user_mask and use_turns_mask:  #turns
        (all_encoder_input, all_encoder_mask, all_decoder_input, all_decoder_mask, all_decoder_target, \
         all_encoder_input_with_oov, all_decoder_target_with_oov, all_oov_len, all_max_encoder_lens, \
         all_mask_lens), title, oov = batch
        max_encoder_len = all_encoder_mask.sum(dim=-1).max()
        max_decoder_len = all_decoder_mask.sum(dim=-1).max()

        all_turns_mask = []
        for mask_len, max_len in zip(all_mask_lens, all_max_encoder_lens):
            all_turns_mask.append(_add_utter_pad(mask_len, max_len))
        all_turns_mask = torch.cat(all_turns_mask, dim=0)
        all_turns_mask = all_turns_mask[:, :max_encoder_len, :max_encoder_len]

        all_encoder_input = all_encoder_input[:, :max_encoder_len]
        all_encoder_mask = all_encoder_mask[:, :max_encoder_len]
        all_decoder_input = all_decoder_input[:, :max_decoder_len]
        all_decoder_mask = all_decoder_mask[:, :max_decoder_len]
        all_decoder_target = all_decoder_target[:, :max_decoder_len]
        all_encoder_input_with_oov = all_encoder_input_with_oov[:, :max_encoder_len]
        all_decoder_target_with_oov = all_decoder_target_with_oov[:, :max_decoder_len]

        batch_size = all_encoder_input.shape[0]
        max_oov_len = all_oov_len.max().item()

        oov_zeros = None
        if use_pointer:  # 当时用指针网络时，decoder_target应该要带上oovs
            all_decoder_target = all_decoder_target_with_oov
            if max_oov_len > 0:  # 使用指针时，并且在这个batch中存在oov的词汇，oov_zeros才不是None
                oov_zeros = torch.zeros((batch_size, max_oov_len), dtype=torch.float32)
        else:  # 当不使用指针时，带有oov的all_encoder_input_with_oov也不需要了
            all_encoder_input_with_oov = None

        init_coverage = None
        if use_coverage:
            init_coverage = torch.zeros(all_encoder_input.size(), dtype=torch.float32)  # 注意数据格式是float

        init_context_vec = torch.zeros((batch_size, 2 * hidden_dim), dtype=torch.float32)  # 注意数据格式是float

        model_input = [all_encoder_input, all_encoder_mask, all_encoder_input_with_oov, oov_zeros, init_context_vec,
                       init_coverage, all_turns_mask, all_decoder_input, all_decoder_mask, all_decoder_target]
        model_input = [t.cuda() if t is not None else None for t in model_input]

    else:
        (all_encoder_input, all_encoder_mask, all_decoder_input, all_decoder_mask,all_decoder_target,\
        all_encoder_input_with_oov, all_decoder_target_with_oov, all_oov_len), title, oov = batch


        max_encoder_len = all_encoder_mask.sum(dim=-1).max()
        max_decoder_len = all_decoder_mask.sum(dim=-1).max()

        all_encoder_input = all_encoder_input[:,:max_encoder_len]
        all_encoder_mask = all_encoder_mask[:,:max_encoder_len]
        all_decoder_input = all_decoder_input[:,:max_decoder_len]
        all_decoder_mask = all_decoder_mask[:,:max_decoder_len]
        all_decoder_target = all_decoder_target[:,:max_decoder_len]
        all_encoder_input_with_oov = all_encoder_input_with_oov[:,:max_encoder_len]
        all_decoder_target_with_oov = all_decoder_target_with_oov[:,:max_decoder_len]

        batch_size = all_encoder_input.shape[0]
        max_oov_len = all_oov_len.max().item()

        init_context_vec = torch.zeros((batch_size, 2 * hidden_dim), dtype=torch.float32)  # 注意数据格式是float


        oov_zeros = None
        if use_pointer:                # 当时用指针网络时，decoder_target应该要带上oovs
            all_decoder_target = all_decoder_target_with_oov
            if max_oov_len > 0:                # 使用指针时，并且在这个batch中存在oov的词汇，oov_zeros才不是None
                oov_zeros = torch.zeros((batch_size, max_oov_len),dtype= torch.float32)
        else:                                  # 当不使用指针时，带有oov的all_encoder_input_with_oov也不需要了
            all_encoder_input_with_oov = None
        init_coverage = None
        if use_coverage:
            init_coverage = torch.zeros(all_encoder_input.size(), dtype=torch.float32)  # 注意数据格式是float


        model_input = [all_encoder_input,all_encoder_mask,all_encoder_input_with_oov,oov_zeros,init_context_vec,\
                        init_coverage,all_decoder_input,all_decoder_mask,all_decoder_target]
        model_input = [t.cuda() if t is not None else None for t in model_input]

    return model_input, title, oov, labels



def _add_utter_pad(article_lens, max_len):
    start = 0
    utter_mask = torch.zeros((max_len, max_len), dtype=torch.int)
    for len_ in article_lens:
        end = start + len_
        utter_mask[start: end, start: end] = 1
        start += len_
    return utter_mask.unsqueeze(0)

def _get_user_mask(users, max_len):
    pad_nums = (users == 0).int().sum().item()
    n = len(users) - pad_nums
    users = users[:n]
    # users = torch.tensor(np.array(users), dtype=torch.int)
    users = ((users.unsqueeze(dim=0).expand(n, n)) == (users.unsqueeze(dim=1).expand(n, n))).int()
    user_mask = torch.zeros((max_len, max_len), dtype=torch.int)
    user_mask[:len(users), :len(users)] = users
    return user_mask.unsqueeze(0)


if __name__ == "__main__":
    data_path = '/home/disk1/lyj2019/zfj2020/finished_csv_files/train.csv'
    vocab_path = '/home/disk1/lyj2019/zfj2020/finished_csv_files/vocab'
    labels_path = './rouge_result.json'
    max_article_len = 300
    max_title_len = 100
    vocab_size = 20000
    batch_size = 2
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #vocab = Vocab(vocab_path, vocab_size)

    example_loader=get_example_loader(data_path, labels_path, vocab_path, vocab_size,
                                      max_article_len, max_title_len,
                                      use_pointer=True, test_mode=True, test_num=1000, use_user_mask=True)

    example_dataset = covert_loader_to_dataset(example_loader)
    sampler = RandomSampler(example_dataset) # for training random shuffle
    #sampler = SequentialSampler(example_dataset) # for evaluating sequential loading
    train_dataloader = DataLoader(example_dataset, sampler=sampler, batch_size=batch_size)

    for batch in train_dataloader:
        # print(batch)
        model_input = from_batch_get_model_input(batch, 256, use_pointer=True, use_coverage=True, use_user_mask=True)
        print(model_input)
        break


