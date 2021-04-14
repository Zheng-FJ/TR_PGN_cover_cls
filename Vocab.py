import collections
import pandas as pd
from tqdm import tqdm

SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

PAD_TOKEN = '<pad>'
UNKNOWN_TOKEN = '<unk>'  # OOV符号
START_DECODING = '<bos>'  # 解码开始符号
STOP_DECODING = '<eos>'  # 解码结束符号
'''for SAMsum'''
UTTERENCE_END = '<eou>'
UTTERANCE_REPRE = '<cls>'


class Vocab(object):
    def __init__(self, vocab_file=None, vocab_size=50000):
        '''
            根据构建好的vocab文件构造Vocab对象
            vocab_file content(word count):
                to 5751035
                a 5100555
                and 4892247
        '''
        self._word_to_id = {}
        self._id_to_word = {}
        self._count = 0  # 记录词表里的总词数

        # <pad>, <unk>, <bos> and <eos> <eou> 取得词表的对应ID：0，1，2，3，4
        for w in [PAD_TOKEN, UNKNOWN_TOKEN, START_DECODING, STOP_DECODING, UTTERENCE_END, UTTERANCE_REPRE]:
            self._word_to_id[w] = self._count
            self._id_to_word[self._count] = w
            self._count += 1


        # 读取词表文件，并向Vocab对象中添加词汇直到上限词
        with open(vocab_file, 'r') as vocab_f:
            for line in vocab_f:
                parts = line.split("\t")

                # 检测词表每行结构是否正确
                if len(parts) != 2:
                    print('Warning: incorrectly formatted line in vocabulary file: {}'.format(line))
                    continue

                w = parts[0]
                # 检测词表里是否有特殊符号
                if w in [SENTENCE_START, SENTENCE_END, UNKNOWN_TOKEN, PAD_TOKEN, START_DECODING, STOP_DECODING]:
                    raise Exception(
                        '<s>, </s>, [UNK], [PAD], [START] and [STOP] shouldn\'t be in the vocab file, but %s is' % w)
                # 检测词表是否重复记录
                if w in self._word_to_id:
                    raise Exception('Duplicated word in vocabulary file: %s' % w)

                # 检测没有问题就可以写入词表，直到写满词表
                self._word_to_id[w] = self._count
                self._id_to_word[self._count] = w
                self._count += 1
                if vocab_size != 0 and self._count >= vocab_size:
                    print("max_size of vocab was specified as {}; we now have {} words. Stopping reading.".format(
                        vocab_size, self._count))
                    break

        print("Finished constructing vocabulary of {} total words. Last word added: {}".format(self._count,
                                                                                               self._id_to_word[
                                                                                                    self._count - 1]))
        print("Vocab:")
        for idx in [0, 1, 2, 3, 4, 5, -1, self._count -3, self._count -2, self._count-1]:
            if idx == -1:
                print("    ......")
                continue
            print("   ", idx, self._id_to_word[idx])

    def word2id(self, word):
        '''获取单个词语的id'''
        if word not in self._word_to_id:
            return self._word_to_id[UNKNOWN_TOKEN]
        return self._word_to_id[word]

    def id2word(self, word_id):
        '''根据词语id解析对应的词语'''
        if word_id not in self._id_to_word:
            raise ValueError('Id not found in vocab: %d' % word_id)
        return self._id_to_word[word_id]

    def get_vocab_size(self):
        '''获取加上特殊符号后，词汇表大小'''
        return self._count


def article2ids(article_words, vocab):
    '''
    接收：
        list类别的article词汇列标；
        Vocab类别的实例对象；

    返回两个列表:
        ids：list类别的对应词汇的id；
        oovs：list类别的用于记录oov的列标。
    '''
    article_ids = []
    article_oovs = []
    unk_id = vocab.word2id(UNKNOWN_TOKEN)
    for w in article_words:
        i = vocab.word2id(w)
        if i == unk_id:  # 判断该词汇是否为oov词汇
            if w not in article_oovs:  # 添加到oov词汇列标里
                article_oovs.append(w)
            oov_num = article_oovs.index(w)  # 令该文章中第1个oov的索引为0；第二次oov的索引为1
            article_ids.append(
                vocab.get_vocab_size() + oov_num)  # 这么做的目的是拓展原有的词汇表，如上述第一个oov为词表id为50000的词，第二个oov则为50001的词
        else:
            article_ids.append(i)

    return article_ids, article_oovs

def abstract2ids(abstract_words, vocab, article_oovs):
    '''
    :param abstract_words: list类型的摘要词汇列表
    :param vocab: 实例化Vocab类别的对象
    :param article_oovs: 该摘要文本对应文章所记录的oov列标
    :return: 对应摘要词汇的id列表
    '''

    abstract_ids = []
    unk_id = vocab.word2id(UNKNOWN_TOKEN)
    for w in abstract_words:
        i = vocab.word2id(w)
        if i == unk_id:
            if w in article_oovs:
                vocab_idx = vocab.get_vocab_size() + article_oovs.index(w)
                abstract_ids.append(vocab_idx)
            else:
                abstract_ids.append(unk_id)

        else:
            abstract_ids.append(i)
    return abstract_ids


def makeVocabdict(data_dir, vocab_dir, vocab_size):
    vocab_counter = collections.Counter()
    # TODO create vocab file assume load csv file
    df = pd.read_csv(data_dir)
    problems = df['Problem'].tolist()
    articles = df['Conversation'].tolist()
    titles = df['Report'].tolist()

    punctuation = ['。','，','？','！','、','；','：','“','”','‘','’','「','」',\
                    '『','』','（','）','[',']','【','】','——','~','《','》','<','>','/','\\','\\\\']


    for ques, art, tit, in tqdm(zip(problems, articles, titles)):
        try:
            # for qichedashi
            art = ques+"|"+art
            art = list(art)
            for i in range(len(art)-1):
                if art[i] == '|' and art[i+1] not in ['车','技']:
                    art[i] = '^'
            art = ''.join(art)
            art = art.replace('^','')

            art_ = art.split('|')

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

            art_tokens = art__.split(' ')

            tit_ = ''
            tit = list(tit)
            for c in tit:
                if '\u4e00' <= c <= '\u9fa5': #中文范围
                    tit_ += ' ' + c + ' '
                elif c in punctuation:
                    tit_ += ' ' + c + ' '
                else:
                    tit_ += c
            tit_tokens = tit_.split(' ')

            tokens = art_tokens + tit_tokens

            tokens = [t for t in tokens]  # 去掉句子开头结尾的空字符
            tokens = [t for t in tokens if t != ""]  # 删除空行


            for token in tokens:
                if token not in vocab_counter:
                    vocab_counter[token] = 1
                else:
                    vocab_counter[token] += 1
        except:
            print(art)
            print(tit)

    with open(vocab_dir, 'w', encoding='utf-8') as writer:
        for word, count in vocab_counter.most_common(vocab_size):
            if word in [SENTENCE_START, SENTENCE_END, UNKNOWN_TOKEN, PAD_TOKEN, START_DECODING, STOP_DECODING]:
                continue
            else:
                writer.write(word + '\t' + str(count) + '\n')

    print("Finished writing vocab file")



if __name__ == '__main__':
    data_path = '/home/disk1/lyj2019/zfj2020/finished_csv_files/train.csv'
    vocab_path = '/home/disk1/lyj2019/zfj2020/finished_csv_files/vocab'
    makeVocabdict(data_path, vocab_path, 50000)
    vocab = Vocab(vocab_path, 20000)
    print(vocab.id2word(193))
    print(vocab.get_vocab_size())
    #
