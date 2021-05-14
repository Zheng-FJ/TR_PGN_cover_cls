import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np 
import itertools
import os
import json

def plot_confusion_matrix(output_dir, cm, classes, normalize=False,
                          title='Confusion matrix', figsize=(12, 10),
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    cm: sklearn 得到的混淆矩阵
    classes: 指定的类别名列表
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    cm[cm>0.15] = 0.15

    # Compute confusion matrix
    np.set_printoptions(precision=2)
    plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    # tick_marks = np.arange(len(classes))
    # plt.xticks(tick_marks, classes, rotation=45)
    # plt.yticks(tick_marks, classes)

    # fmt = '.2f' if normalize else 'd'
    # thresh = cm.max() / 2.
    # for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    #     plt.text(j, i, 
    #             #  format(cm[i, j], fmt),
    #              ' ',
    #              horizontalalignment="center",
    #              color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Q117476_8.png"))

def main():
    input_file = './results/test_attn_vanilla/tokenattn8.json'
    output_dir = './output_figure/vanilla/'
    with open(input_file, 'r') as f_attn:
        attn_datas = json.load(f_attn)
    cm = attn_datas['Q117476']
    cm = np.array(cm)

    classes = [1,2,3,4]
    plot_confusion_matrix(output_dir, cm, classes, normalize=True,\
                          title='Confusion matrix', figsize=(12, 10),\
                          cmap=plt.cm.Blues)

if __name__ == "__main__":
    main()