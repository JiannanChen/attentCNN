"""
网络性能分析1
Args:
    1 输入acc_log和loss_tra;
    2 输出若干人的平均准确率和损失；
    3 输出y_true,y_pred；
-----------------------------
Std标准文件-网络性能评价的标准文件1
"""
import csv
import numpy as np


# @读取csv格式数据-注意此部分有待完善！
def readcsv(files):
    csvfile = open(files, 'r')
    plots = csv.reader(csvfile, delimiter=' ')  # 按行读取CSV文件中的数据,每一行以空格作为分隔符，再将内容保存成列表的形式
    next(plots)  # 读取首行
    x = []
    y = []
    for row in plots:
        x.append(float(row[0]))  # 从csv读取的数据是str类型，转成float类型
        y.append(float(row[1]))
    return x, y


if __name__ == '__main__':
    # @必要参数
    num_sub = 2
    num_epo = 10

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

    # @读取测试acc历史数据并求均值
    """
    计算方法：先求出54个受试者的最好的acc,再取平均
    """
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

    # @3输出所有受试者拼接后的最优预测值/真实值List或Array
    """
    计算方法：将最优的Epoch对应的预测和真实值直接拼接
    """
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

    # @4输出训练损失演示图
    """
    计算方法：将所有受试者对应的Epoch损失求和再去平均
    """
    loss_sum = np.zeros(num_epo)
    for i in range(num_sub):
        loss_one_sub = np.array(loss_tra_glo_log[i][:, 1])
        loss_sum = loss_sum + loss_one_sub
    c = np.float(1 / num_sub)
    loss_avg = loss_sum * c

    print('netAnalysis_std_1 is OVER!')













