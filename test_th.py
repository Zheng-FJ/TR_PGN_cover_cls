import json
import numpy as np
from tqdm import tqdm

input1 = './rouge_result_0.1.json'
input2 = './rouge_result_topk.json'
input3 = './rouge_result_0.12.json'
input4 = './rouge_result_0.08.json'

with open(input1, 'r')as f1:
    data1 = json.load(f1)

with open(input2,'r')as f2:
    data2 = json.load(f2)

with open(input3,'r')as f3:
    data3 = json.load(f3)

with open(input4,'r')as f4:
    data4 = json.load(f4)

res = []
for k in tqdm(data4.keys()):
    num = data4[k].count(1) / len(data4[k])
    res.append(num)
avg_res = np.mean(res)
print(avg_res)