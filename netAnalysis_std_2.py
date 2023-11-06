"""
网络性能分析2
Args:
    1 输入y_true,y_pred；
    2 输出混淆矩阵 getCMatrix；
    3 输出根据混淆矩阵绘制的图像 draw_pic_from_CMatrix；
    4 输出分类报告
-----------------------------------------------------------
Std标准文件-网络性能评价的标准文件2
    网络性能评价指标：accuracy, precision, recall and F1-score;
"""
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from matplotlib import rcParams
# import csv
# import pandas
# import codecs


# @获取混淆矩阵
def getCMatrix(y_true, y_pred):
    # @获取混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    # plot_confusion_matrix(cm, 'confusion_matrix.png', title='confusion matrix')
    return cm


# @根据混淆矩阵绘制图像
def draw_pic_from_CMatrix(classes, CMatrix):
    classes = classes
    confusion_matrix = CMatrix

    # @计算百分比
    proportion = []
    length = len(confusion_matrix)  # 矩阵行数
    print(length)
    for i in confusion_matrix:
        for j in i:
            temp = j / (np.sum(i))
            proportion.append(temp)
    # print(np.sum(confusion_matrix[0]))
    # print(proportion)

    # @转化成百分比形式
    pshow = []
    for i in proportion:
        pt = "%.2f%%" % (i * 100)
        pshow.append(pt)

    # @转化成8*8形状
    proportion = np.array(proportion).reshape(length, length)  # reshape(列的长度，行的长度)
    pshow = np.array(pshow).reshape(length, length)
    # print(pshow)

    # @画图：
    # @设置字体
    config = {
        "font.family": 'Times New Roman',  # 设置字体类型
    }
    rcParams.update(config)

    # @显示出矩阵
    # plt.imshow(proportion, interpolation='nearest', cmap=plt.cm.Blues)
    plt.imshow(proportion, interpolation='nearest', cmap=plt.cm.YlOrBr)
    # (改变颜色：'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds','YlOrBr', 'YlOrRd',
    # 'OrRd', 'PuRd', 'RdPu', 'BuPu','GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn')
    # plt.title('confusion_matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=12)
    plt.yticks(tick_marks, classes, fontsize=12)

    # thresh = confusion_matrix.max() / 2.
    # # iters = [[i,j] for i in range(len(classes)) for j in range((classes))]

    iters = np.reshape([[[i, j] for j in range(length)] for i in range(length)], (confusion_matrix.size, 2))
    for i, j in iters:
        if (i == j):
            plt.text(j, i - 0.12, format(confusion_matrix[i, j]), va='center', ha='center', fontsize=10, color='white',
                     weight=5)  # 显示对应的数字
            plt.text(j, i + 0.12, pshow[i, j], va='center', ha='center', fontsize=10, color='white')
        else:
            plt.text(j, i - 0.12, format(confusion_matrix[i, j]), va='center', ha='center', fontsize=10)  # 显示对应的数字
            plt.text(j, i + 0.12, pshow[i, j], va='center', ha='center', fontsize=10)

    plt.ylabel('True label', fontsize=16)
    plt.xlabel('Predict label', fontsize=16)
    plt.tight_layout()
    plt.show()
    # plt.savefig('混淆矩阵.png')


if __name__ == '__main__':
    # @从npy日志读取保存的数据
    path_loss_tra_glo_log = './logs/SubAll_loss_tra_glo_log.npy'
    path_loss_val_glo_log = './logs/SubAll_loss_val_glo_log.npy'
    path_loss_tes_glo_log = './logs/SubAll_loss_tes_glo_log.npy'
    path_acc_val_glo_log = './logs/SubAll_acc_val_glo_log.npy'
    path_acc_tes_glo_log = './logs/SubAll_acc_tes_glo_log.npy'
    path_pdt_tes_glo_log = './logs/SubAll_pdt_tes_glo_log.npy'
    path_tgt_tes_glo_log = './logs/SubAll_tgt_tes_glo_log.npy'
    loss_tra_glo_log = np.load(path_loss_tra_glo_log, allow_pickle=True)
    loss_val_glo_log = np.load(path_loss_val_glo_log, allow_pickle=True)
    loss_tes_glo_log = np.load(path_loss_tes_glo_log, allow_pickle=True)
    acc_val_glo_log = np.load(path_acc_val_glo_log, allow_pickle=True)
    acc_tes_glo_log = np.load(path_acc_tes_glo_log, allow_pickle=True)
    pdt_tes_glo_log = np.load(path_pdt_tes_glo_log, allow_pickle=True)
    tgt_tes_glo_log = np.load(path_tgt_tes_glo_log, allow_pickle=True)

    # @解码tgt-pdt数据、计算混淆矩阵并画图
    classes = ['1', '2', '3', '4']  # 类别
    y_tgt = tgt_tes_glo_log[0][999][1]  # 0 for idx_sub, 9 for epo, 1/0 for data/idx_epo
    y_pdt = pdt_tes_glo_log[0][999][1]
    CMatrix = getCMatrix(y_tgt, y_pdt)  # 混淆矩阵
    draw_pic_from_CMatrix(classes, CMatrix)  # 画图

    # @分类报告
    print("每个类别的精确率和召回率：", classification_report(y_tgt, y_pdt, target_names=classes))
    cf_rpt = classification_report(y_tgt, y_pdt, target_names=classes, output_dict=True)

    print('netAnalysis_std_2 is OVER!')










