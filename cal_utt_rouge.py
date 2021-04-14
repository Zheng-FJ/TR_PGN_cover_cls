import os
import json
import collections
import pandas as pd 
import numpy as np
from tqdm import tqdm 
from rouge import Rouge 
import sys   
sys.setrecursionlimit(100000)

file_path = '/home/disk2/zfj2020/workspace/dataset/qichedashi/finished_csv_files/'
train_file = os.path.join(file_path, 'train.csv')
test_file = os.path.join(file_path, 'test.csv')
valid_file = os.path.join(file_path, 'valid.csv')
rouge = Rouge()

output_file = './rouge_result.json'

labels = collections.defaultdict(list)

for file_name in [valid_file, test_file, train_file]:
# for file_name in [valid_file]:
    print('processing %s ...'%(file_name))
    csv_data = pd.read_csv(file_name, encoding='utf-8')
    qids = csv_data['QID'].tolist()
    covs = csv_data['Conversation'].tolist()
    reps = csv_data['Report'].tolist()
    

    for qid, cov, rep in tqdm(zip(qids, covs, reps)):
        if len(labels[qid]) != 0:
            continue
        refs = []
        utts = []
        cov = list(cov)
        for i in range(len(cov)-1):
            if cov[i] == '|' and cov[i+1] not in ['车', '技']:
                cov[i] = '^'
        cov = ' '.join(cov)
        cov = cov.replace('^', '')
        cov = cov.split('|')

        rep = list(rep)
        rep = ' '.join(rep)

        utts += cov
        
        for utt in cov:
            refs.append(rep)
        r = rouge.get_scores(utts, refs)
        for line in r:
            f_score = line['rouge-l']['f']
            if f_score < 0.12:
                labels[qid].append(0)
            else:
                labels[qid].append(1)
print('writing to json ...')
with open(output_file, 'w', encoding='utf-8')as f:
    json.dump(labels, f)       
print('Done!')