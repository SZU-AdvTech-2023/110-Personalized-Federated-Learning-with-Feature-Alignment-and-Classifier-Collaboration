# FedPAC
Simplified Implementation of FedPAC for PFL

Personalized Federated Learning with Feature Alignment and Classifier Collaboration (ICLR 2023)

#### 环境要求

安装requirements.txt中所需要的包:

```shell
pip install -r requirements.txt
```

#### 运行

使用以下命令进行训练

```shel
python ./federated_main.py --local_epoch=5 --train_rule='FedPAC' --iid=0 --noniid_s=20 --num_users=20 --dataset='cifar10'
```

对于CINIC-10数据集使用以下命令训练

```shel
 python ./federated_main.py --local_epoch=5 --train_rule='FedPAC' --iid=0 --noniid_s=20 --num_users=20 --dataset='cinic_sep'
```

#### 数据集

对于EMNIST、Fashion-MNIST和Cifar-10数据集在代码中已经内置下载。对于CINIC-10数据集需要自行下载数据放置在data目录下并运行以下命令对数据集处理

```shel
python ./proc_cinic.py
```
