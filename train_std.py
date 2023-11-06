"""
AggtCNN网络训练主逻辑
Args:
    1 时间窗口选择为0.2s;
    2 网络性能评价指标：accuracy, precision, recall and F1-score;
------------------------------------------------------------------------------
Std标准文件-网络训练的标准文件
    1 网络训练有3个阶段：@训练阶段-tra，@验证阶段-val，@测试阶段-tes;
    2 网络训练有三个循环：@受试者循环-sub，@世代循环-epo，@批次循环-bth, @全局global-glo;
    3 需要保存的文件用_log结尾; 中继变量以mid_开头;
    4 Rules:
        #1 变量命名示例1: data_tra_bth;
        #2 变量命名示例1: loss_tra_sub_log;
    5 常见缩写:
        #1 time->tim; #2 window->win; #3 number->num;
        #4 idx->index; #5 predict->pdt; #6 target->tgt;
        #7 middle->mid; #8 iteration->iter;
"""
import os
import csv
import codecs
import torch
import numpy as np
from net_std import tCNN
import scipy.io as scio
from scipy import signal
from random import sample
import torch.optim as optim
from torch.utils.data import DataLoader
from dataloader_std import train_BCIDataset, val_BCIDataset, test_BCIDataset

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


# @获取滤波后的训练数据，标签和起始时间
def get_train_data(wn11, wn21, wn12, wn22, wn13, wn23, wn14, wn24, path, down_sample):
    data = scio.loadmat(path)  # 读取原始数据

    # 下采样与通道选择
    x_data = data['EEG_SSVEP_train']['x'][0][0][::down_sample]
    c = [24, 28, 29, 30, 41, 42, 43, 60, 61]
    train_data = x_data[:, c]
    train_label = data['EEG_SSVEP_train']['y_dec'][0][0][0]
    train_start_time = data['EEG_SSVEP_train']['t'][0][0][0]

    # @ 滤波1
    channel_data_list1 = []
    for i in range(train_data.shape[1]):
        b, a = signal.butter(6, [wn11, wn21], 'bandpass')
        filtedData = signal.filtfilt(b, a, train_data[:, i])
        channel_data_list1.append(filtedData)
    channel_data_list1 = np.array(channel_data_list1)

    # @ 滤波2
    channel_data_list2 = []
    for i in range(train_data.shape[1]):
        b, a = signal.butter(6, [wn12, wn22], 'bandpass')
        filtedData = signal.filtfilt(b, a, train_data[:, i])
        channel_data_list2.append(filtedData)
    channel_data_list2 = np.array(channel_data_list2)

    # @ 滤波3
    channel_data_list3 = []
    for i in range(train_data.shape[1]):
        b, a = signal.butter(6, [wn13, wn23], 'bandpass')
        filtedData = signal.filtfilt(b, a, train_data[:, i])
        channel_data_list3.append(filtedData)
    channel_data_list3 = np.array(channel_data_list3)

    # @ 滤波4
    channel_data_list4 = []
    for i in range(train_data.shape[1]):
        b, a = signal.butter(6, [wn14, wn24], 'bandpass')
        filtedData = signal.filtfilt(b, a, train_data[:, i])
        channel_data_list4.append(filtedData)
    channel_data_list4 = np.array(channel_data_list4)

    return channel_data_list1, channel_data_list2, channel_data_list3, channel_data_list4, train_label, train_start_time


# @获取滤波后的测试数据、标签和起始时间
def get_test_data(wn11, wn21, wn12, wn22, wn13, wn23, wn14, wn24, path, down_sample):
    data = scio.loadmat(path)

    # 下采样与通道选择
    x_data = data['EEG_SSVEP_test']['x'][0][0][::down_sample]
    c = [24, 28, 29, 30, 41, 42, 43, 60, 61]
    test_data = x_data[:, c]
    test_label = data['EEG_SSVEP_test']['y_dec'][0][0][0]
    test_start_time = data['EEG_SSVEP_test']['t'][0][0][0]

    # @ 滤波1
    channel_data_list1 = []
    for i in range(test_data.shape[1]):
        b, a = signal.butter(6, [wn11, wn21], 'bandpass')
        filtedData = signal.filtfilt(b, a, test_data[:, i])
        channel_data_list1.append(filtedData)
    channel_data_list1 = np.array(channel_data_list1)

    # @ 滤波2
    channel_data_list2 = []
    for i in range(test_data.shape[1]):
        b, a = signal.butter(6, [wn12, wn22], 'bandpass')
        filtedData = signal.filtfilt(b, a, test_data[:, i])
        channel_data_list2.append(filtedData)
    channel_data_list2 = np.array(channel_data_list2)

    # @ 滤波3
    channel_data_list3 = []
    for i in range(test_data.shape[1]):
        b, a = signal.butter(6, [wn13, wn23], 'bandpass')
        filtedData = signal.filtfilt(b, a, test_data[:, i])
        channel_data_list3.append(filtedData)
    channel_data_list3 = np.array(channel_data_list3)

    # @ 滤波4
    channel_data_list4 = []
    for i in range(test_data.shape[1]):
        b, a = signal.butter(6, [wn14, wn24], 'bandpass')
        filtedData = signal.filtfilt(b, a, test_data[:, i])
        channel_data_list4.append(filtedData)
    channel_data_list4 = np.array(channel_data_list4)

    return channel_data_list1, channel_data_list2, channel_data_list3, channel_data_list4, test_label, test_start_time


# @保存csv格式数据-注意此部分有待完善！
def data_write_csv(file_name, datas):  # file_name为写入csv文件的路径，datas为要写入数据列表
    file_csv = codecs.open(file_name, 'w+', 'utf-8')  # 追加
    writer = csv.writer(file_csv, delimiter=' ', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
    for data in datas:
        writer.writerow(data)
    print("文件保存成功！")


if __name__ == '__main__':
    # @GPU加速
    # print(torch.cuda.device_count())  # 打印当前设备GPU数量，此笔记本只有1个GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('device is', device)

    # @训练参数与滤波器设置
    win_tim = 0.2  # 时间窗口0.2s
    num_sub = 2

    num_epo_tra = 10
    num_data = 1000
    bth_size = 250
    lr = 1e-3
    # lr = 0.8e-3

    down_sample = 4  # 下采样设置
    fs = 1000 / down_sample  # fs为float类型
    channel = 9  # 选取的通道数

    f_down1 = 3  # 第一个滤波器
    f_up1 = 14
    wn11 = 2 * f_down1 / fs
    wn21 = 2 * f_up1 / fs

    f_down2 = 9  # 第二个滤波器
    f_up2 = 26
    wn12 = 2 * f_down2 / fs
    wn22 = 2 * f_up2 / fs

    f_down3 = 14  # 第三个滤波器
    f_up3 = 38
    wn13 = 2 * f_down3 / fs
    wn23 = 2 * f_up3 / fs

    f_down4 = 19  # 第四个滤波器
    f_up4 = 50
    wn14 = 2 * f_down4 / fs
    wn24 = 2 * f_up4 / fs

    # @网络训练主逻辑
    loss_tra_glo_log, loss_val_glo_log, loss_tes_glo_log = [], [], []
    pdt_tra_glo_log, pdt_val_glo_log, pdt_tes_glo_log = [], [], []
    tgt_tra_glo_log, tgt_val_glo_log, tgt_tes_glo_log = [], [], []
    acc_tra_glo_log, acc_val_glo_log, acc_tes_glo_log = [], [], []
    # @subject循环
    for idx_sub in range(1, num_sub+1):
        # @数据集设置
        if idx_sub < 10:
            path_sub = './sess01/sess01_subj0%d_EEG_SSVEP.mat' % idx_sub
        else:
            path_sub = './sess01/sess01_subj%d_EEG_SSVEP.mat' % idx_sub

        win_data = int(fs * win_tim)  # 时间窗口对应帧数
        list_tra_val = list(range(100))  # 对100次实验
        list_val = sample(list_tra_val, 10)  # 随机划分10个用于验证，其余用于训练
        list_tra = [list_tra_val[i] for i in range(len(list_tra_val)) if (i not in list_val)]

        # @获取滤波后的训练数据、标签和起始时间
        mid_data_tra_1, mid_data_tra_2, mid_data_tra_3, mid_data_tra_4, label_tra, start_time_tra \
            = get_train_data(wn11, wn21, wn12, wn22, wn13, wn23, wn14, wn24, path_sub, down_sample)
        data_tra = [mid_data_tra_1, mid_data_tra_2, mid_data_tra_3, mid_data_tra_4]  # 数据聚合, 形状变为4*9*(?)
        data_tra = np.array(data_tra)

        # @获取滤波后的测试数据、标签和起始时间
        mid_data_tes_1, mid_data_tes_2, mid_data_tes_3, mid_data_tes_4, label_tes, start_time_tes \
            = get_test_data(wn11, wn21, wn12, wn22, wn13, wn23, wn14, wn24, path_sub, down_sample)
        data_tes = [mid_data_tes_1, mid_data_tes_2, mid_data_tes_3, mid_data_tes_4]
        data_tes = np.array(data_tes)

        # @数据集及产生器
        dataset_tra = train_BCIDataset(num_data, data_tra, win_data, label_tra, start_time_tra,
                                       down_sample, list_tra, channel)
        gen_tra = DataLoader(dataset_tra, shuffle=True, batch_size=bth_size, num_workers=1,
                             pin_memory=True, drop_last=True)

        dataset_val = val_BCIDataset(num_data, data_tra, win_data, label_tra, start_time_tra,
                                     down_sample, list_val, channel)
        gen_val = DataLoader(dataset_val, shuffle=True, batch_size=bth_size, num_workers=1,
                             pin_memory=True, drop_last=True)

        dataset_tes = test_BCIDataset(num_data, data_tes, win_data, label_tes, start_time_tes,
                                      down_sample, channel)
        gen_tes = DataLoader(dataset_tes, shuffle=True, batch_size=bth_size, num_workers=1,
                             pin_memory=True, drop_last=True)

        # @网络设置
        net = tCNN(win_data)
        net.to(device)
        loss_f = torch.nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr, weight_decay=0.01)  # 对参数进行正则化weight_decay, 防止过拟合

        loss_tra_sub_log, loss_val_sub_log, loss_tes_sub_log = [], [], []  # 保存的参数
        pdt_tra_sub_log, pdt_val_sub_log, pdt_tes_sub_log = [], [], []
        tgt_tra_sub_log, tgt_val_sub_log, tgt_tes_sub_log = [], [], []
        acc_tra_sub_log, acc_val_sub_log, acc_tes_sub_log = [], [], []

        # @epoch循环
        for epoch in range(num_epo_tra):

            loss_tra_epo_log, loss_val_epo_log, loss_tes_epo_log = [], [], []  # 保存的参数
            pdt_tra_epo_log, pdt_val_epo_log, pdt_tes_epo_log = [], [], []
            tgt_tra_epo_log, tgt_val_epo_log, tgt_tes_epo_log = [], [], []
            acc_tra_epo_log, acc_val_epo_log, acc_tes_epo_log = [], [], []

            epo_size = int(num_data / bth_size)  # 单个世代循环的次数

            # @网络训练及Batch循环
            net.train()
            # print('Start train:')
            for iter_tra, bth_tra in enumerate(gen_tra):  # iteration是批次，batch是每批次的数据
                if iter_tra >= epo_size:  # 单世代循环训练退出条件
                    break

                data_tra_bth, tgt_tra_bth = bth_tra[0], bth_tra[1]
                data_tra_bth, tgt_tra_bth = data_tra_bth.to(device), tgt_tra_bth.to(device)

                optimizer.zero_grad()  # 优化器梯度清零
                output_tra_bth = net(data_tra_bth)
                loss_tra_bth = loss_f(output_tra_bth, tgt_tra_bth.long())
                loss_tra_bth.backward()
                optimizer.step()

                mid_pdt_tra_bth = np.argmax(output_tra_bth.data.cpu().numpy(), axis=1)
                mid_tgt_tra_bth = tgt_tra_bth.data.cpu().numpy()  # 转化成numpy

                loss_tra_epo_log.append(loss_tra_bth)
                pdt_tra_epo_log.extend(list(mid_pdt_tra_bth))
                tgt_tra_epo_log.extend(list(mid_tgt_tra_bth))

            mid_loss_tra_epo = sum(loss_tra_epo_log) / epo_size
            loss_tra_sub_log.append([epoch, mid_loss_tra_epo.data.cpu().numpy()])  # 先脱离.data,再压到cpu中，再转化成numpy
            pdt_tra_sub_log.append([epoch, pdt_tra_epo_log])
            tgt_tra_sub_log.append([epoch, tgt_tra_epo_log])
            # print('End train!')

            # @网络验证及Batch循环
            net.eval()
            # print('Start validation:')
            for iter_val, bth_val in enumerate(gen_val):
                if iter_val >= epo_size:
                    break

                data_val_bth, tgt_val_bth = bth_val[0], bth_val[1]
                data_val_bth, tgt_val_bth = data_val_bth.to(device), tgt_val_bth.to(device)

                with torch.no_grad():
                    optimizer.zero_grad()
                    output_val_bth = net(data_val_bth)
                    loss_val_bth = loss_f(output_val_bth, tgt_val_bth.long())

                    mid_pdt_val_bth = np.argmax(output_val_bth.data.cpu().numpy(), axis=1)
                    mid_tgt_val_bth = tgt_val_bth.data.cpu().numpy()  # 转化成numpy

                    loss_val_epo_log.append(loss_val_bth)
                    pdt_val_epo_log.extend(list(mid_pdt_val_bth))
                    tgt_val_epo_log.extend(list(mid_tgt_val_bth))

                    # @计算acc
                    y_true_val_bth = tgt_val_bth.data.cpu().numpy()
                    a_val_bth, b_val_bth = 0, 0
                    for i in range(bth_size):
                        y_pdt_val_bth = np.argmax(output_val_bth.data.cpu().numpy()[i])
                        if y_true_val_bth[i] == y_pdt_val_bth:
                            a_val_bth += 1
                        else:
                            b_val_bth += 1
                    acc_val_bth = a_val_bth / (a_val_bth + b_val_bth)
                    acc_val_epo_log.append(acc_val_bth)

            mid_loss_val_epo = sum(loss_val_epo_log) / epo_size
            mid_acc_val_epo_log = np.mean(acc_val_epo_log)

            loss_val_sub_log.append([epoch, mid_loss_val_epo.data.cpu().numpy()])  # 先脱离.data,再压到cpu中，再转化成numpy
            pdt_val_sub_log.append([epoch, pdt_val_epo_log])
            tgt_val_sub_log.append([epoch, tgt_val_epo_log])
            acc_val_sub_log.append([epoch, mid_acc_val_epo_log])
            # print('Finish validation!')

            # @网络测试及Batch循环
            net.eval()
            # print('Start test:')
            for iter_tes, bth_tes in enumerate(gen_tes):
                if iter_tes >= epo_size:
                    break

                data_tes_bth, tgt_tes_bth = bth_tes[0], bth_tes[1]
                data_tes_bth, tgt_tes_bth = data_tes_bth.to(device), tgt_tes_bth.to(device)

                with torch.no_grad():
                    optimizer.zero_grad()
                    output_tes_bth = net(data_tes_bth)
                    loss_tes_bth = loss_f(output_tes_bth, tgt_tes_bth.long())

                    mid_pdt_tes_bth = np.argmax(output_tes_bth.data.cpu().numpy(), axis=1)
                    mid_tgt_tes_bth = tgt_tes_bth.data.cpu().numpy()  # 转化成numpy

                    loss_tes_epo_log.append(loss_tes_bth)
                    pdt_tes_epo_log.extend(list(mid_pdt_tes_bth))
                    tgt_tes_epo_log.extend(list(mid_tgt_tes_bth))

                    # @计算acc
                    y_true_tes_bth = tgt_tes_bth.data.cpu().numpy()
                    a_tes_bth, b_tes_bth = 0, 0
                    for i in range(bth_size):
                        y_pdt_tes_bth = np.argmax(output_tes_bth.data.cpu().numpy()[i])
                        if y_true_tes_bth[i] == y_pdt_tes_bth:
                            a_tes_bth += 1
                        else:
                            b_tes_bth += 1
                    acc_tes_bth = a_tes_bth / (a_tes_bth + b_tes_bth)
                    acc_tes_epo_log.append(acc_tes_bth)

            mid_loss_tes_epo = sum(loss_tes_epo_log) / epo_size
            mid_acc_tes_epo_log = np.mean(acc_tes_epo_log)

            loss_tes_sub_log.append([epoch, mid_loss_tes_epo.data.cpu().numpy()])  # 先脱离.data,再压到cpu中，再转化成numpy
            pdt_tes_sub_log.append([epoch, pdt_tes_epo_log])
            tgt_tes_sub_log.append([epoch, tgt_tes_epo_log])
            acc_tes_sub_log.append([epoch, mid_acc_tes_epo_log])
            # print('Finish test!')

            # @每次世代信息反馈
            print('Sub %d, Epo %d, loss_tra %.3f, loss_val is %.3f, loss_tes is %.3f, Acc_val %.3f, Acc_tes %.3f'
                  % (idx_sub, epoch, np.float(mid_loss_tra_epo.data.cpu().numpy()),
                     np.float(mid_loss_val_epo.data.cpu().numpy()), np.float(mid_loss_tes_epo.data.cpu().numpy()),
                     np.float(mid_acc_val_epo_log), np.float(mid_acc_tes_epo_log)))

        # @训练环节最终保存的数据
        loss_tra_glo_log.append(loss_tra_sub_log)
        pdt_tra_glo_log.append(pdt_tra_sub_log)
        tgt_tra_glo_log.append(tgt_tra_sub_log)

        # @验证环节最终保存的数据
        loss_val_glo_log.append(loss_val_sub_log)
        pdt_val_glo_log.append(pdt_val_sub_log)
        tgt_val_glo_log.append(tgt_val_sub_log)
        acc_val_glo_log.append(acc_val_sub_log)

        # @测试环节最终保存的数据
        loss_tes_glo_log.append(loss_tes_sub_log)
        pdt_tes_glo_log.append(pdt_tes_sub_log)
        tgt_tes_glo_log.append(tgt_tes_sub_log)
        acc_tes_glo_log.append(acc_tes_sub_log)

    # @训练-验证-测试后的文件保存
    path_loss_tra_glo_log = './logs/SubAll_loss_tra_glo_log.npy'
    path_loss_val_glo_log = './logs/SubAll_loss_val_glo_log.npy'
    path_loss_tes_glo_log = './logs/SubAll_loss_tes_glo_log.npy'
    path_acc_val_glo_log = './logs/SubAll_acc_val_glo_log.npy'
    path_acc_tes_glo_log = './logs/SubAll_acc_tes_glo_log.npy'
    path_pdt_tes_glo_log = './logs/SubAll_pdt_tes_glo_log.npy'
    path_tgt_tes_glo_log = './logs/SubAll_tgt_tes_glo_log.npy'
    np.save(path_loss_tra_glo_log, loss_tra_glo_log)
    np.save(path_loss_val_glo_log, loss_val_glo_log)
    np.save(path_loss_tes_glo_log, loss_tes_glo_log)
    np.save(path_acc_val_glo_log, acc_val_glo_log)
    np.save(path_acc_tes_glo_log, acc_tes_glo_log)
    np.save(path_pdt_tes_glo_log, pdt_tes_glo_log)
    np.save(path_tgt_tes_glo_log, tgt_tes_glo_log)

    print('Train_std is OVER!')



