"""
网络性能分析3
Args:
    1 输入网络性能分析1；
    2 输入网络性能分析2；
    3 输出所有性能指标；
-----------------------------------------------------------
Std标准文件-网络性能评价的标准文件3
"""
import numpy as np
from netAnalysis_std_2 import getCMatrix, draw_pic_from_CMatrix
from sklearn.metrics import classification_report

# @必要参数
num_sub = 2
num_epo = 10
classes = ['1', '2', '3', '4']  # 类别

path_loss_tra_glo_log = './logs/SubAll_loss_tra_glo_log.npy'
path_acc_tes_glo_log = './logs/SubAll_acc_tes_glo_log.npy'
path_pdt_tes_glo_log = './logs/SubAll_pdt_tes_glo_log.npy'
path_tgt_tes_glo_log = './logs/SubAll_tgt_tes_glo_log.npy'

if __name__ == '__main__':
    # @数据加载
    loss_tra_glo_log = np.load(path_loss_tra_glo_log, allow_pickle=True)
    acc_tes_glo_log = np.load(path_acc_tes_glo_log, allow_pickle=True)
    pdt_tes_glo_log = np.load(path_pdt_tes_glo_log, allow_pickle=True)
    tgt_tes_glo_log = np.load(path_tgt_tes_glo_log, allow_pickle=True)

    # @读取测试acc历史数据并求均值
    Allsub_maxacc_idx = []
    Allsub_maxacc_value = []
    for sub_idx in range(num_sub):
        acc_sub_idx_x = acc_tes_glo_log[sub_idx][:, 1]  # 0 for subject, : for all epo, 1 for data
        max_index = np.argmax(acc_sub_idx_x)
        max_value = np.max(acc_sub_idx_x)
        Allsub_maxacc_idx.append(max_index)
        Allsub_maxacc_value.append(max_value)
    acc_mean = np.mean(Allsub_maxacc_value)
    acc_std = np.std(Allsub_maxacc_value)
    print('acc_mean is:', acc_mean)
    print('acc_std is:', acc_std)

    # @3根据@2中的Allsub_maxacc_idx，输出所有受试者拼接后的最优预测值/真实值List或Array
    y_pdt = []
    idx_pdt = 0
    for i in Allsub_maxacc_idx:
        y_pdt_subx = pdt_tes_glo_log[idx_pdt][i][1]
        y_pdt.extend(y_pdt_subx)
        idx_pdt += 1

    y_true = []
    idx_tgt = 0
    for i in Allsub_maxacc_idx:
        y_tgt_subx = tgt_tes_glo_log[idx_tgt][i][1]
        y_true.extend(y_tgt_subx)
        idx_tgt += 1
    np.save('./logs/y_pdt.npy', y_pdt)
    np.save('./logs/y_true.npy', y_true)

    # @4根据@3中的y_pdt和y_true输出混淆矩阵和报告表
    CMatrix = getCMatrix(y_true, y_pdt)  # 混淆矩阵
    draw_pic_from_CMatrix(classes, CMatrix)  # 画图

    # @5根据@3中的y_pdt和y_true输出混淆矩阵和报告表
    print("每个类别的精确率和召回率：", classification_report(y_true, y_pdt, target_names=classes))
    report = classification_report(y_true, y_pdt, target_names=classes, output_dict=True)
    np.save('./logs/report.npz', report)

    # @6输出平均训练损失历史数据
    loss_sum = np.zeros(num_epo)
    for i in range(num_sub):
        loss_one_sub = np.array(loss_tra_glo_log[i][:, 1])
        loss_sum = loss_sum + loss_one_sub
    c = np.float(1 / num_sub)
    loss_avg = loss_sum * c
    np.save('./logs/loss_avg.npz', loss_avg)

    print('netAnalysis_std_3 is OVER!')


































