#!/user/bin/env python3
# -*- coding: utf-8 -*-
import torch
import numpy as np
import csv
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
plt.switch_backend('Agg')


# print(torch.tensor([torch.tensor(10)])/torch.tensor([torch.tensor(10)]).sum(dim=0))
# p_matrix=np.array([1,2,3])
#
# test=torch.tensor(p_matrix)
# print(test,type(p_matrix))
# x = torch.tensor([[1, 2, 3, 4, 5]])
# print( x[0][1],x[0,1])
# complex_tensor = torch.tensor([1 + 2j, 3 + 4j], dtype=torch.complex64)
# real_tensor = complex_tensor.real
# print(real_tensor)
# print(x.shape)
#
# a=torch.zeros(2,3)
# a[0]=torch.tensor(0.2)*torch.tensor([[1],[2],[3]])
# print(a[0])

# p=torch.tensor([[1,2,3],[3,4,5]])
# t=0*p[0]
# print(t)
# for i in p:
#     t+=i
#     print(t)
# print(t)
# a=[]
# for i in range(10):
#     a.append(torch.tensor(600))
#
# print(a)
# a=torch.stack(a)
# print(a)
# torch.tensor(a)
# print(a)
filename = './/out//FedPAC_20_cifar10_results.csv'
# args = args_parser()
# 初始化空列表存储准确率和损失值
accuracy_list = []
loss_list = []

# 读取 CSV 文件并将数据存储到对应列表中
with open(filename, mode='r') as file:
    reader = csv.reader(file)
    header = next(reader)  # 读取表头
    for row in reader:
        accuracy_list.append(float(row[0]))  # 假设第一列是准确率数据，将其转换为浮点数并添加到准确率列表
        loss_list.append(float(row[1]))      # 假设第二列是损失值数据，将其转换为浮点数并添加到损失值列表

    plt.figure()
    plt.title('cifar10')
    # plt.plot(train_loss,'b',label='train_loss')
    plt.plot(loss_list,'r',label='train_loss')
    plt.ylabel('Train Loss')
    plt.xlabel('Communication Rounds')
    plt.legend()  # 使plot中label其作用
    plt.savefig(os.path.join('./out/{}_{}_{}_result_loss.jpg'.format('FedPAC',10,'cifar10')))

    plt.figure()
    plt.title('cifar10')
    plt.plot(accuracy_list, 'b', label='test_acc')
    # plt.plot(local_accs2, 'r',label='test_acc')
    plt.ylabel('Test Accuracy(%)')
    plt.xlabel('Communication Rounds')
    plt.legend()  # 使plot中label其作用
    plt.savefig(os.path.join('./out/{}_{}_{}_result_acc.jpg'.format('FedPAC',10,'cifar10')))