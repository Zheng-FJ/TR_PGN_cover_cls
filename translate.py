''' Translate input text with trained model. '''

import torch
import argparse
from tqdm import tqdm
import random
import numpy as np

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset, Dataset

from Vocab import Vocab
from data_functions import get_example_loader, covert_test_loader_to_dataset, from_test_batch_get_model_input


import transformer.Constants as Constants
#from torchtext.data import Dataset
from transformer.Models import Transformer
from transformer.Translator1 import Translator


def load_model(opt, device):

    checkpoint = torch.load(opt.model, map_location=device)
    model_opt = checkpoint['settings']

    model = Transformer(
        model_opt.vocab_size,
        model_opt.vocab_size,

        model_opt.pad_idx,
        model_opt.pad_idx,

        trg_emb_prj_weight_sharing=model_opt.proj_share_weight,
        emb_src_trg_weight_sharing=model_opt.embs_share_weight,
        d_k=model_opt.d_k,
        d_v=model_opt.d_v,
        d_model=model_opt.d_model,
        d_word_vec=model_opt.d_word_vec,
        d_inner=model_opt.d_inner_hid,
        n_layers=model_opt.n_layers,
        n_head=model_opt.n_head,
        dropout=model_opt.dropout,
        n_position=model_opt.max_article_len,
        use_pointer = model_opt.use_pointer, 
        use_cls_layers=model_opt.use_cls_layers,
        use_score_matrix=model_opt.use_score_matrix,
        q_based=model_opt.q_based).to(device)

    model.load_state_dict(checkpoint['model'])
    print('[Info] Trained model state loaded.')
    return model 


def main():
    '''Main Function'''

    parser = argparse.ArgumentParser(description='translate.py')

    parser.add_argument('-model', type=str, default="/home/disk2/zfj2020/workspace/emnlp2021/modified/TR_PGN_cover_cls/save_model/dudu_test_sche_score_q_topk/save_best.chkpt")
    parser.add_argument('-test_path', type=str, default="/home/disk2/zfj2020/workspace/dataset/qichedashi/finished_csv_files/test.csv")
    parser.add_argument('-label_path', type=str, default="./rouge_result_topk.json")
    parser.add_argument("-vocab_path", type=str, default="/home/disk2/zfj2020/workspace/dataset/qichedashi/finished_csv_files/vocab")
    parser.add_argument("-vocab_size", type=int, default=50000)

    parser.add_argument('-output', default='./results/dudu_test_sche_score_q_topk/pred.txt',
                        help="""Path to output the predictions (each line will
                        be the decoded sequence""")
    parser.add_argument('-beam_size', type=int, default=5)
    parser.add_argument('-hidden_dim', type=int, default=256)
    parser.add_argument('-max_article_len', type=int, default=300)
    parser.add_argument('-max_title_len', type=int, default=50)
    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-test_mode', action='store_true')
    parser.add_argument('-use_pointer', type=bool, default=True)
    parser.add_argument("-use_utter_trunc", action="store_true")
    parser.add_argument("-use_user_mask", action="store_true")
    parser.add_argument("-use_turns_mask", action="store_true")
    parser.add_argument("-use_cls_layers", action="store_true")
    # parser.add_argument("-use_score_matrix", action="store_true")
    # parser.add_argument("-q_based", action="store_true")


    parser.add_argument('-pad_idx', type=int, default=0)
    parser.add_argument('-unk_idx', type=int, default=1)
    parser.add_argument('-bos_idx', type=int, default=2)
    parser.add_argument('-eos_idx', type=int, default=3)

    # TODO: Translate bpe encoded files 
    #parser.add_argument('-src', required=True,
    #                    help='Source sequence to decode (one line per sequence)')
    #parser.add_argument('-vocab', required=True,
    #                    help='Source sequence to decode (one line per sequence)')
    # TODO: Batch translation
    #parser.add_argument('-batch_size', type=int, default=30,
    #                    help='Batch size')
    #parser.add_argument('-n_best', type=int, default=1,
    #                    help="""If verbose is set, will output the n_best
    #                    decoded sentences""")

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda

    batch_size = 1
    vocab = Vocab(opt.vocab_path, opt.vocab_size)
    test_examples = get_example_loader(opt.test_path, opt.label_path, opt.vocab_path, opt.vocab_size,
                                        opt.max_article_len, opt.max_title_len,
                                        use_pointer=opt.use_pointer,
                                        test_mode=opt.test_mode,
                                        use_utter_trunc=opt.use_utter_trunc, use_user_mask=opt.use_user_mask, use_turns_mask=opt.use_turns_mask)
    test_dataset = covert_test_loader_to_dataset(test_examples)
    test_sampler = SequentialSampler(test_dataset) # for training random shuffle
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=batch_size)

    
    device = torch.device('cuda' if opt.cuda else 'cpu')
    translator = Translator(
        model=load_model(opt, device),
        beam_size=opt.beam_size,
        max_seq_len=opt.max_title_len,
        src_pad_idx=opt.pad_idx,
        trg_pad_idx=opt.pad_idx,
        trg_bos_idx=opt.bos_idx,
        trg_eos_idx=opt.eos_idx,
        unk_idx=opt.unk_idx,
        vocab_size=opt.vocab_size,
        use_pointer=opt.use_pointer
        ).cuda()

    with open(opt.output, 'w') as f:
        hyps = []
        refs = []
        cls_score = []
        for batch in tqdm(test_dataloader, mininterval=2, desc='  - (Test)', leave=False):
            #print(' '.join(example.src))
            model_inputs, titles, oovs, labels = from_test_batch_get_model_input(batch, opt.hidden_dim, use_pointer=opt.use_pointer,use_utter_trunc=opt.use_utter_trunc, use_user_mask=opt.use_user_mask, use_turns_mask=opt.use_turns_mask)
            # print(titles)
            # print(oovs)
            src_seq = model_inputs[0].cuda()
            src_seq_with_oovs = model_inputs[2].cuda()
            oov_zeros = model_inputs[3]
            attn_mask1 = None
            attn_mask2 = model_inputs[6]
            # attn_mask3 = model_inputs[8]
            attn_mask3 = None

            if opt.use_cls_layers:
                article_lens = model_inputs[11]
                utt_num = model_inputs[10]
            else:
                article_lens = None
                utt_num = None


            #print("src: ",src_seq)
            pred_seq, predicted_label = translator.translate_sentence(src_seq, src_seq_with_oovs, oov_zeros, attn_mask1, attn_mask2, attn_mask3, article_lens, utt_num)
            if predicted_label is not None:
                predicted_label =  predicted_label.cpu().numpy().tolist()
                gold_label = []
                for x in labels:
                    gold_label += x.numpy().tolist()

                length = len(predicted_label)
                corr = 0
                for i in range(length):
                    if predicted_label[i] == gold_label[i]:
                        corr += 1
                cls_score.append(corr / length)
            else:
                cls_score = [0.0]

            result = []
            # print(titles)
            # print(oovs)
            # exit()
            for idx in pred_seq:
                if idx < vocab.get_vocab_size():
                    result.append(vocab.id2word(idx))
                else:
                    result.append(oovs[idx-vocab.get_vocab_size()][0])
            pred_line = ' '.join(result)
            pred_line = pred_line.replace('<bos>', '').replace('<eos>', '')
            gold_line = ' '.join([tk[0] for tk in titles])
            gold_line = gold_line.replace('<bos>', '').replace('<eos>', '')
            hyps.append(pred_line)
            refs.append(gold_line)
            #print(pred_line)
            #print(' '.join([tk[0] for tk in titles]))
            f.write(pred_line.strip() + '\t' + '|'+ '\t' + gold_line +'\n')
    from rouge import Rouge
    rouge = Rouge()
    print(rouge.get_scores(hyps, refs, avg=True))
    print("classify correctness: ", np.mean(cls_score))
    print('[Info] Finished.')

if __name__ == "__main__":
    '''
    Usage: python translate.py -model trained.chkpt -data multi30k.pt -no_cuda
    '''
    # Problem0 = "安全带锁死了拽不出来怎么办"
    # Conversation0 = "技师说：你好，安全带一但是锁死一次，这个只能更换。你可以去修理厂拆下来确认一下|车主说：能往里送但是拽不出来|技师说：急刹车后出现的吗|车主说：后排|车主说：根本没用过|车主说：我也不知道怎么弄得|技师说：那个不应该，你可以去修理厂拆开看一下是否是卡住了。|技师说：拆开边上有小齿轮，润滑一下|车主说：现在我送到头了|技师说：慢一点拉一下也不行吗|车主说：不行|技师说：那需要拆开看一下了|车主说：好拆吗？|技师说：好像是内六花螺丝固定的，好拆。去修理厂看一下花不了几个钱|车主说：好的|车主说：麻烦了"
    # Report0 = "拆开看一下是否是里面卡住了，也不排除安全带出现自动锁死，这个情况只能更换"


    main()
