import os.path
import csv
import numpy as np
import torch
import torch.nn as nn
import math
from data_loader import get_dataset
from running import one_round_training
from methods import local_update
from models import CifarCNN, CNN_FMNIST
from options import args_parser
import tools
import copy
import time
import matplotlib.pyplot as plt
plt.switch_backend('Agg')

if __name__ == '__main__':
    args = args_parser()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = device
    print(device)
    # load dataset and user groups
    train_loader, test_loader, global_test_loader = get_dataset(args)
    seed = 520
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # construct model
    if args.dataset in ['cifar', 'cifar10', 'cinic', 'cinic_sep']:
        global_model = CifarCNN(num_classes=args.num_classes).to(device)
        args.lr = 0.02
    elif args.dataset == 'fmnist':
        global_model = CNN_FMNIST().to(device)
    elif args.dataset == 'emnist':
        args.num_classes = 62
        global_model = CNN_FMNIST(num_classes=args.num_classes).to(device)
    else:
        raise NotImplementedError()

    # Training Rule
    LocalUpdate = local_update(args.train_rule)
    # One Round Training Function
    train_round_parallel = one_round_training(args.train_rule)

    # Training
    train_loss, train_acc = [], []
    train_loss2,test_acc = [],[]
    local_accs1, local_accs2 = [], []
    # ======================================================================================================#
    local_clients = []
    for idx in range(args.num_users):
        local_clients.append(LocalUpdate(idx=idx, args=args, train_set=train_loader[idx], test_set=test_loader[idx],
                                         model=copy.deepcopy(global_model)))

    for round in range(args.epochs):
        loss1, loss2, local_acc1, local_acc2 = train_round_parallel(args, global_model, local_clients, round)
        train_loss.append(loss1)
        train_loss2.append(loss2)
        print("Train Loss: {}, {}".format(loss1, loss2))
        print("Local Accuracy on Local Data: {}%, {}%".format(local_acc1, local_acc2))
        # 聚合后权重
        local_accs1.append(local_acc1)
        # 最后一轮更新
        local_accs2.append(local_acc2)

    # # 结果写入csv
    # result = list(zip(local_accs1, train_loss2))
    # filename = './out/{}_{}_{}_results_{}_{}_{}.csv'.format(args.train_rule, args.num_users, args.dataset,args.agg_g,args.lam,'my_reg')
    # with open(filename, mode='w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(['Accuracy', 'Loss'])  # 写入表头
    #     writer.writerows(result)  # 写入数据
    # file.close()
    #
    # plt.figure()
    # plt.title(args.dataset)
    # # plt.plot(train_loss,'b',label='train_loss')
    # plt.plot(train_loss2,'r',label='train_loss')
    # plt.ylabel('Train Loss')
    # plt.xlabel('Communication Rounds')
    # plt.legend()  # 使plot中label其作用
    # plt.savefig(os.path.join('./out/{}_{}_{}_result_loss_{}_{}.jpg'.format(args.train_rule,args.num_users,args.dataset,args.agg_g,args.lam)))
    #
    # plt.figure()
    # plt.title(args.dataset)
    # plt.plot(local_accs1, 'b', label='test_acc')
    # # plt.plot(local_accs2, 'r',label='test_acc')
    # plt.ylabel('Test Accuracy(%)')
    # plt.xlabel('Communication Rounds')
    # plt.legend()  # 使plot中label其作用
    # plt.savefig(os.path.join('./out/{}_{}_{}_result_acc_{}_{}.jpg'.format(args.train_rule,args.num_users,args.dataset,args.agg_g,args.lam)))

