# coding: utf-8

from copy import deepcopy
import copy
import numpy as np
from numpy import random
from numpy.core.shape_base import stack
import tools
import math
import torch
from torch import nn
import time
# ---------------------------------------------------------------------------- #

class LocalUpdate_FedPAC(object):
    def __init__(self, idx, args, train_set, test_set, model):
        self.idx = idx
        self.args = args
        self.num_classes = args.num_classes
        self.train_data = train_set
        self.test_data = test_set
        self.device = args.device
        self.criterion = nn.CrossEntropyLoss()
        self.local_model = model
        self.last_model = deepcopy(model)
        self.w_local_keys = self.local_model.classifier_weight_keys  # ['fc2.weight', 'fc2.bias',]
        self.local_ep_rep = 1
        self.probs_label = self.prior_label(self.train_data).to(self.device)  # 类别占比
        self.sizes_label = self.size_label(self.train_data).to(self.device)  # 类别数量
        self.datasize = torch.tensor(len(self.train_data.dataset)).to(self.device)
        self.agg_weight = self.aggregate_weight()
        self.global_protos = {}  # 全局质心
        self.g_protos = None  # 全局质心(1,10)
        self.mse_loss = nn.MSELoss()
        self.lam = args.lam  # 1.0 for mse_loss
        self.personal_dict = None  # 第一轮权重
        # self.my_reg = args.my_reg

    def reg_term(self):
        new_weight=self.personal_dict
        local_weight=self.local_model.state_dict()
        teacher_model=copy.deepcopy(self.local_model)
        w_local_keys = self.w_local_keys
        for k in local_weight.keys():
            if k in w_local_keys:
                local_weight[k] = new_weight[k]
        teacher_model.load_state_dict(local_weight)
        for name, param in teacher_model.named_parameters():
            if name in self.w_local_keys:
                param.requires_grad = True
            else:
                param.requires_grad = False
        return teacher_model


    # 返回每种类别在整个客户端数据集中占比返回tensor数组 shape为行向量包含10个元素
    def prior_label(self, dataset):
        py = torch.zeros(self.args.num_classes)
        total = len(dataset.dataset)
        data_loader = iter(dataset)
        # 迭代次数
        iter_num = len(data_loader)
        for it in range(iter_num):
            images, labels = next(data_loader)
            for i in range(self.args.num_classes):
                # 标签格式为整数形式记录
                py[i] = py[i] + (i == labels).sum()
        py = py/(total)
        return py

    # 返回每种类别在整个客户端数据集中的数量 shape为行向量包含10个元素
    def size_label(self, dataset):
        py = torch.zeros(self.args.num_classes)
        total = len(dataset.dataset)
        data_loader = iter(dataset)
        iter_num = len(data_loader)
        for it in range(iter_num):
            images, labels = next(data_loader)
            for i in range(self.args.num_classes):
                py[i] = py[i] + (i == labels).sum()
        py = py/(total)
        size_label = py*total
        return size_label

    def aggregate_weight(self):
        data_size = len(self.train_data.dataset)
        w = torch.tensor(data_size).to(self.device)
        return w

    # 模型训练
    def local_test(self, test_loader):
        model = self.local_model
        model.eval()
        device = self.device
        correct = 0
        total = len(test_loader.dataset)
        loss_test = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                # outputs size为(batch_size,num_classes)
                _, outputs = model(inputs)
                loss = self.criterion(outputs, labels)
                loss_test.append(loss.item())
                # torch.max返回指定维度最大值及其索引，第一个返回最大值，第二返回索引，这里1表示沿着列维度寻找最大值
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
        # acc以百分比表示
        acc = 100.0*correct/total
        return acc, sum(loss_test)/len(loss_test)

    # 更新特征提取的权重
    def update_base_model(self, global_weight):
        local_weight = self.local_model.state_dict()
        w_local_keys = self.w_local_keys
        for k in local_weight.keys():
            if k not in w_local_keys:
                local_weight[k] = global_weight[k]
        self.local_model.load_state_dict(local_weight)

    def update_local_classifier(self, new_weight):
        local_weight = self.local_model.state_dict()
        w_local_keys = self.w_local_keys
        for k in local_weight.keys():
            if k in w_local_keys:
                local_weight[k] = new_weight[k]
        self.local_model.load_state_dict(local_weight)

    def update_global_protos(self, global_protos):
        self.global_protos = global_protos
        global_protos = self.global_protos
        g_classes, g_protos = [], []
        for i in range(self.num_classes):
            g_classes.append(torch.tensor(i))
            g_protos.append(global_protos[i])
        self.g_classes = torch.stack(g_classes).to(self.device)
        self.g_protos = torch.stack(g_protos)

    # 获得数据集每个类别，128个，特征的均值{类别:特征均值(1,128)}
    def get_local_protos(self):
        model = self.local_model
        local_protos_list = {}
        for inputs, labels in self.train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            features, outputs = model(inputs)
            protos = features.clone().detach()
            for i in range(len(labels)):
                if labels[i].item() in local_protos_list.keys():
                    local_protos_list[labels[i].item()].append(protos[i,:])
                else:
                    local_protos_list[labels[i].item()] = [protos[i,:]]
        # 获得数据集每个类别，128个，特征的均值
        local_protos = tools.get_protos(local_protos_list)
        return local_protos

    # 局部数据集的单遍提取局部特征统计量方差V用于估计每个客户端的最优分类器组合权重
    def statistics_extraction(self):
        model = self.local_model
        cls_keys = self.w_local_keys  # ['fc2.weight', 'fc2.bias]
        # cls_keys是否为列表数据类型,取参数矩阵中fc2的权重参数
        g_params = model.state_dict()[cls_keys[0]] if isinstance(cls_keys, list) else model.state_dict()[cls_keys]
        d = g_params[0].shape[0]
        feature_dict = {}
        with torch.no_grad():
            for inputs, labels in self.train_data:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                # features为特征提取器提取的特征矩阵，经过了fc1,features包含inputs行和每个图片提取的特征(列)
                features, outputs = model(inputs)
                feat_batch = features.clone().detach()
                # 相同label提取出的特征矩阵保存在同一个key下（feature_dict）
                for i in range(len(labels)):
                    yi = labels[i].item()
                    if yi in feature_dict.keys():
                        # [i,:]表示第i个条数据的特征值
                        feature_dict[yi].append(feat_batch[i,:])
                    else:
                        feature_dict[yi] = [feat_batch[i,:]]
        for k in feature_dict.keys():
            # 堆叠行向量，将相同key的特征进行堆叠，一维堆叠成二维
            feature_dict[k] = torch.stack(feature_dict[k])

        # 类别占比tensor数组
        py = self.probs_label
        # 矩阵对应元素相乘 和tensor*tensor一样 .mm或@是矩阵相乘
        py2 = py.mul(py)
        v = 0
        h_ref = torch.zeros((self.num_classes, d), device=self.device)
        for k in range(self.num_classes):
            if k in feature_dict.keys():
                feat_k = feature_dict[k]
                # shape[0]获取行数即该类别的数量
                num_k = feat_k.shape[0]
                feat_k_mu = feat_k.mean(dim=0)
                h_ref[k] = py[k]*feat_k_mu
                v += (py[k]*torch.trace((torch.mm(torch.t(feat_k), feat_k)/num_k))).item()
                v -= (py2[k]*(torch.mul(feat_k_mu, feat_k_mu))).sum().item()
        v = v/self.datasize.item()

        return v, h_ref

    def local_training(self, local_epoch, round=0):
        # Set mode to train model
        model = self.local_model
        model.train()
        round_loss = []
        iter_loss = []
        model.zero_grad()
        grad_accum = []
        global_protos = self.global_protos
        g_protos = self.g_protos

        # 返回测试的准确率和损失
        acc0, _ = self.local_test(self.test_data)
        self.last_model = deepcopy(model)

        # get local prototypes before training, dict:={label: list of sample features}
        local_protos1 = self.get_local_protos()

        # Set optimizer for the local updates, default sgd
        optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5, weight_decay=0.0005)

        local_ep_rep = local_epoch
        epoch_classifier = 5
        local_epoch = int(epoch_classifier + local_ep_rep)

        if local_epoch>0:
            # 返回可学习参数的可迭代器，这里fc2的参数以计算梯度，即更新分类器
            for name, param in model.named_parameters():
                if name in self.w_local_keys:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            lr_g = 0.1
            # lambda p: p.requires_grad 是一个匿名函数，它接收一个参数 p，并返回 p 的 requires_grad 属性的值
            # 结合 filter() 函数使用这样的 lambda 函数来从一组参数中筛选出需要梯度计算的参数
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr_g,
                                                   momentum=0.5, weight_decay=0.0005)
            # 只更新一轮分类器
            for ep in range(epoch_classifier):
                # local training for 1 epoch
                data_loader = iter(self.train_data)
                iter_num = len(data_loader)
                for it in range(iter_num):
                    images, labels = next(data_loader)
                    images, labels = images.to(self.device), labels.to(self.device)
                    model.zero_grad()
                    protos, output = model(images)
                    re_loss = 0
                    if self.personal_dict is not None:
                        teacher_model = self.reg_term()
                        _, teacher_output = teacher_model(images)
                        re_loss = self.criterion(teacher_output, output)
                    loss = self.criterion(output, labels)+0.001*re_loss
                    loss.backward()
                    optimizer.step()
                    iter_loss.append(loss.item())
                round_loss.append(sum(iter_loss)/len(iter_loss))
                iter_loss = []
            # ---------------------------------------------------------------------------

            acc1, _ = self.local_test(self.test_data)

            for name, param in model.named_parameters():
                    if name in self.w_local_keys:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=self.args.lr,
                                                   momentum=0.5, weight_decay=0.0005)
            # 更新特征提取器
            for ep in range(local_ep_rep):
                data_loader = iter(self.train_data)
                iter_num = len(data_loader)
                for it in range(iter_num):
                    images, labels = next(data_loader)
                    images, labels = images.to(self.device), labels.to(self.device)
                    model.zero_grad()
                    protos, output = model(images)
                    loss0 = self.criterion(output, labels)
                    loss1 = 0
                    if round > 0:
                        loss1 = 0
                        protos_new = protos.clone().detach()
                        for i in range(len(labels)):
                            yi = labels[i].item()
                            if yi in global_protos:
                                protos_new[i] = global_protos[yi].detach()
                            else:
                                protos_new[i] = local_protos1[yi].detach()
                        loss1 = self.mse_loss(protos_new, protos)
                    loss = loss0 + self.lam * loss1
                    loss.backward()
                    optimizer.step()
                    iter_loss.append(loss.item())
                round_loss.append(sum(iter_loss)/len(iter_loss))
                iter_loss = []

        # ------------------------------------------------------------------------
        # 获取局部质心
        local_protos2 = self.get_local_protos()
        round_loss1 = round_loss[0] # 更新分类器损失
        round_loss2 = round_loss[-1] # 更新提取器损失
        acc2, _ = self.local_test(self.test_data)

        return model.state_dict(), round_loss1, round_loss2, acc0, acc2, local_protos2



