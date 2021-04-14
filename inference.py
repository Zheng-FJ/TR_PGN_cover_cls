''' Translate input text with trained model. '''

import torch
import argparse
from tqdm import tqdm
import random

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset, Dataset

from Vocab import Vocab
from data_functions import get_example_loader, \
                        get_example_loader_4_inference, \
                        covert_test_loader_to_dataset, \
                        from_test_batch_get_model_input


from position import position_collection


import transformer.Constants as Constants
#from torchtext.data import Dataset
from transformer.Models import Transformer
from transformer.Translator1 import Translator

import time


from flask import Flask,request,render_template,redirect,jsonify

app = Flask(__name__)

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
        use_pointer = model_opt.use_pointer).to(device)

    model.load_state_dict(checkpoint['model'])
    print('[Info] Trained model state loaded.')
    return model 


def main(article):
    '''Main Function'''
    parser = argparse.ArgumentParser(description='translate.py')

    parser.add_argument('-model', type=str, default="/home/disk1/lyj2019/zfj2020/TR_PGN_qichedashi_cover/save_model/use_utter_user_turns_cover/save_best.chkpt")
    parser.add_argument('-test_path', type=str, default="/home/disk1/lyj2019/zfj2020/finished_csv_files/test.csv")
    parser.add_argument("-vocab_path", type=str, default="/home/disk1/lyj2019/zfj2020/finished_csv_files/vocab")
    parser.add_argument("-vocab_size", type=int, default=50000)

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

    parser.add_argument('-pad_idx', type=int, default=0)
    parser.add_argument('-unk_idx', type=int, default=1)
    parser.add_argument('-bos_idx', type=int, default=2)
    parser.add_argument('-eos_idx', type=int, default=3)

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda

    batch_size = 1
    vocab = Vocab(opt.vocab_path, opt.vocab_size)

    device = torch.device('cuda' if opt.cuda else 'cpu')

    start = time.time()
    Problem0, Conversation0 = article.strip().split("<sep>")
    Report0 = Problem0
    example = get_example_loader_4_inference(Problem0, \
                                            Conversation0, \
                                            Report0, \
                                            vocab, \
                                            opt.max_article_len, \
                                            opt.max_title_len,  \
                                            opt.use_pointer, \
                                            opt.use_utter_trunc, \
                                            opt.use_user_mask, \
                                            opt.use_turns_mask)

    test_dataset = covert_test_loader_to_dataset([example])
    # print(test_dataset[0])
    # print(len(test_dataset[0][0]))
    # exit()
    
    device = torch.device('cuda' if opt.cuda else 'cpu')
    model_inputs, titles, oovs = from_test_batch_get_model_input(test_dataset[0], opt.hidden_dim, use_pointer=opt.use_pointer,use_utter_trunc=opt.use_utter_trunc, use_user_mask=opt.use_user_mask, use_turns_mask=opt.use_turns_mask, inference=True)
    # print(titles)
    # print(oovs)
    src_seq = model_inputs[0].cuda()
    src_seq_with_oovs = model_inputs[2].cuda()
    oov_zeros = model_inputs[3]
    attn_mask1 = model_inputs[6]
    attn_mask2 = model_inputs[7]
    # attn_mask3 = model_inputs[8]
    attn_mask3 = None
    # max_enc_len = model_input[-1]
    #print("src: ",src_seq)
    pred_seq = translator.translate_sentence(src_seq, src_seq_with_oovs, oov_zeros, attn_mask1, attn_mask2, attn_mask3)
    # print(pred_seq)
    # exit()
    result = []
    # print(titles)
    # print(oovs)
    # exit()
    for idx in pred_seq:
        if idx < vocab.get_vocab_size():
            result.append(vocab.id2word(idx))
        else:
            result.append(oovs[idx-vocab.get_vocab_size()][0])
    pred_line = ''.join(result)
    pred_line = pred_line.replace('<bos>', '').replace('<eos>', '')
    gold_line = ''.join([tk[0] for tk in titles])
    gold_line = gold_line.replace('<bos>', '').replace('<eos>', '')
    print('prediction line: ', pred_line)
    print('reference line: ', gold_line)
    from rouge import Rouge
    rouge = Rouge()
    print(rouge.get_scores(pred_line, gold_line, avg=True))
    print('[Info] Finished.')
    print("time costing: ", time.time()-start)
    return pred_line

@app.route("/",methods=['GET','POST'])
def login():
    return render_template('index.html',name=0)

@app.route("/summarization",methods=['POST'])
def login1():
    if request.method =='POST':
        original_text = request.json['original_text']
        summarized_text = main(original_text)

        parts = original_text.strip().split('<sep>')
        problem = parts[0]
        prefiex_len = len(problem)+4
        conversation = parts[1]
        conv_list_ = parts[1].strip().split('|')
        conv_list = ['【Utterance '+str(i+1)+'】'+utter for i, utter in enumerate(conv_list_)]
        problem = ['【Question】' + problem]
        original_text = '<br>'.join(conv_list)
        
        position = position_collection(key=summarized_text, value=original_text)
        counter = 0
        for i, part in enumerate(position):
            part.append(counter)
            counter += len(part[0])

        conv_len_list = [len(conv) for conv in conv_list_]
        utter_index = [prefiex_len+sum(conv_len_list[:i]) for i in range(len(conv_len_list))]
        # print(conv_len_list)
        # print(utter_index)
        # print(position)
        highlight_rule = []

        for rule in position:
            if rule[1] == 'Generated':
                continue
            else:
                temp_dict = {}
                temp_dict["dialog_idx"] = rule[1]
                temp_dict["analysis_idx"] = rule[3]
                temp_dict["length"] = rule[2]
                highlight_rule.append(temp_dict)
        print(highlight_rule)


        res = {
            "status": 0,
            "message": "successfully",
            "dialog_text": original_text,
            "question_text": problem,
            "highlight_analysis": summarized_text,
            "position": position,
            "highlight_rule": highlight_rule
        }


    return jsonify({"response": res})
    # return res

if __name__ == "__main__":
    '''Main Function'''
    parser = argparse.ArgumentParser(description='translate.py')

    parser.add_argument('-model', type=str, default="/home/disk1/lyj2019/zfj2020/TR_PGN_qichedashi_cover/save_model/use_utter_user_turns_cover/save_best.chkpt")
    parser.add_argument('-test_path', type=str, default="/home/disk1/lyj2019/zfj2020/finished_csv_files/test.csv")
    parser.add_argument("-vocab_path", type=str, default="/home/disk1/lyj2019/zfj2020/finished_csv_files/vocab")
    parser.add_argument("-vocab_size", type=int, default=50000)

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

    parser.add_argument('-pad_idx', type=int, default=0)
    parser.add_argument('-unk_idx', type=int, default=1)
    parser.add_argument('-bos_idx', type=int, default=2)
    parser.add_argument('-eos_idx', type=int, default=3)

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda

    batch_size = 1
    vocab = Vocab(opt.vocab_path, opt.vocab_size)

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
    use_pointer=opt.use_pointer).cuda()

    app.run(host='0.0.0.0', port=9001, debug=True, threaded=False, processes=False)
    
    '''
    Usage: python translate.py -model trained.chkpt -data multi30k.pt -no_cuda
    '''
    # Problem0 = "安全带锁死了拽不出来怎么办<sep>技师说：你好，安全带一但是锁死一次，这个只能更换。你可以去修理厂拆下来确认一下|车主说：能往里送但是拽不出来|技师说：急刹车后出现的吗|车主说：后排|车主说：根本没用过|车主说：我也不知道怎么弄得|技师说：那个不应该，你可以去修理厂拆开看一下是否是卡住了。|技师说：拆开边上有小齿轮，润滑一下|车主说：现在我送到头了|技师说：慢一点拉一下也不行吗|车主说：不行|技师说：那需要拆开看一下了|车主说：好拆吗？|技师说：好像是内六花螺丝固定的，好拆。去修理厂看一下花不了几个钱|车主说：好的|车主说：麻烦了"
    # Conversation0 = "技师说：你好，安全带一但是锁死一次，这个只能更换。你可以去修理厂拆下来确认一下|车主说：能往里送但是拽不出来|技师说：急刹车后出现的吗|车主说：后排|车主说：根本没用过|车主说：我也不知道怎么弄得|技师说：那个不应该，你可以去修理厂拆开看一下是否是卡住了。|技师说：拆开边上有小齿轮，润滑一下|车主说：现在我送到头了|技师说：慢一点拉一下也不行吗|车主说：不行|技师说：那需要拆开看一下了|车主说：好拆吗？|技师说：好像是内六花螺丝固定的，好拆。去修理厂看一下花不了几个钱|车主说：好的|车主说：麻烦了"
    # Report0 = "拆开看一下是否是里面卡住了，也不排除安全带出现自动锁死，这个情况只能更换"

    # python inference.py -use_utter_trunc -use_user_mask -use_turns_mask                            
    # article = "安全带锁死了拽不出来怎么办<sep>技师说：你好，安全带一但是锁死一次，这个只能更换。你可以去修理厂拆下来确认一下|车主说：能往里送但是拽不出来|技师说：急刹车后出现的吗|车主说：后排|车主说：根本没用过|车主说：我也不知道怎么弄得|技师说：那个不应该，你可以去修理厂拆开看一下是否是卡住了。|技师说：拆开边上有小齿轮，润滑一下|车主说：现在我送到头了|技师说：慢一点拉一下也不行吗|车主说：不行|技师说：那需要拆开看一下了|车主说：好拆吗？|技师说：好像是内六花螺丝固定的，好拆。去修理厂看一下花不了几个钱|车主说：好的|车主说：麻烦了"
    # article = '宝马5系熄火了会有电流声正常吗<sep>技师说：你好，这个如果是驾驶室里面发出来的，因为这款车熄火后有延迟断电的功能，所以会有时候会有电流的声音响，如果是别的位置发出来的，你需要检查一下是不是排气管热胀冷缩发出来的声响？这两种情况都是属于正常的|车主说：还有一个问题就是 刚进入车里 人坐在驾驶室里 一坐下去地盘会有滋拉一声 不是常有 是偶尔 这个是什么原因？|车主说：电流这个问题没什么大问题咯？|技师说：你说的这种情况是车辆在拉手刹的状态下，由于坐进车里以后车辆底盘会有一定的缓冲，车辆后部的刹车盘和刹车片处于抱死的状态，由于轻微的晃动，会就会发出这种声响。 没有问题，你大可放心|车主说：好的谢谢！'
    
    # main(article)
    # example:
    # 昊锐1.4t加油机油灯报警是怎么回事能解决问题，付费是应该的<sep>技师说：你好，这个情况你首先检查一下机油是否缺少？|车主说：我把机油机滤，机油泵，机油感塞都换了，加油还是亮，把机感塞线拔了，加油还是亮|车主说：还能是哪控制机油灯|技师说：你，你有没有查看一下机油压力感应塞的线路？|技师说：你测量一下机油压力够不够？|车主说：线没动过，关键是拔了线以后，开钥匙门，机油灯正常亮，着车正常灭，着会车，加几脚油就报警，总是闪，灭车在着就不报警，给几脚油又报|车主说：线都拔了我觉得就跟压力没关系了，|车主说：是不是哪个继电器控制灯呢，|车主说：我就是不知道哪能控制灯，线路没动过，线有问题的基律不大，请大师费费心|技师说：这个是没有继电器控制的。|车主说：明天交车|车主说：能是哪|技师说：你主要检查一下仪表盘下面的线路，包括保险丝盒。主要还是机油压力传感器上面那根线。|车主说：线拔了，灯还工作着呢，这就不对了|技师说：你直接打铁看一下灯亮不亮。|车主说：不亮|技师说：这就说明还是线路有问题。|车主说：我怎么找|车主说：这灯的线经过哪是直接进仪表吗|技师说：[图片]|技师说：你看一下这个图片。|车主说：您说这个是通过电脑了，油底壳没油温|车主说：他这车就一个传感器是吧|车主说：您说这个传感器给电脑信号，电脑在控制灯是吧|车主说：我应该怎么做|技师说：你就先测量一下。传感器插头。到电脑板线路是否正常？|技师说：测量一下电阻|车主说：我就不解的是，|技师说：再测量一下电脑到仪表盘的线路。如果线路没有问题，这个就可能是电脑控制的问题。|车主说：从哪测|车主说：您说电脑到仪表那有问题|技师说：从传感器你找一下电脑那儿有相同颜色的线。|车主说：仪表有可能出问题吗|技师说：仪表盘有问题，不可能一直只亮这一个故障灯|车主说：传感器到电脑的，然后电脑到议表的|车主说：气囊灯长亮|技师说：是的。|技师说：电脑有没有故障码。|车主说：会不会议表有电阻坏了，比如，但加油才亮呢|技师说：这个车子是没有电阻的|车主说：他这车电脑在哪|技师说：在发动机舱电瓶的旁边。|车主说：昊锐1.4t|技师说：是的，你看一下那个位置有没有？|车主说：明天看下|车主说：还有别的可能吗|车主说：车跑着没问题|技师说：好的|技师说：如果方便的话，你就测量一下机油压力。|车主说：好的|技师说：好的，你明天检查一下吧。|车主说：还有别的可能吗|车主说：他这车就一个传感器吗，没有别的有关机油的传感器吗|技师说：我只有这一个传感器。也没有其他的原因了|车主说：好吧明天查查，再联系|技师说：好的。|车主说：电脑没在电瓶边上，|车主说：还可能在哪|车主说：现在油门加大了就报警|技师说：不好意思刚看到。你看一下方向盘下面刹车踏板上面的位置。

