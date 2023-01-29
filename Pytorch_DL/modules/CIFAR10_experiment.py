"""
- Classification performance test using deep learning architectures
- Data set : CIFAR10
- Models : MLP, CNN, VGG, ResNet, MobileNetV2
- Main function
    - Fitting various models
    - Setting args containing model parameters
"""
import numpy as np
import torch
import argparse
from experimental_utils import experiment, save_exp_result
from copy import deepcopy # Add Deepcopy for args

def setting_args(model_name):
    parser = argparse.ArgumentParser()
    args = parser.parse_args("")
    

    # ====== MLP ====== #
    if model_name == "MLP":
        args.exp_name = "MLP"
        args.in_dim = 3072
        args.out_dim = 10
        args.hid_dim = 512
        args.n_layer = 3
        args.act = 'leakyrelu'
        args.use_bn = True
        args.dropout = 0.25

    # ====== VGG ====== #
    elif model_name == "VGG":
        args.exp_name = "VGG"
        args.model_code = 'VGG11'
        args.in_channels = 3
        args.out_dim = 10
        args.act = 'leakyrelu'
        args.fc_hid = 256
        args.fc_layers = 3

    # ====== Regularization ======= #
    args.l2 = 0.0005

    # ====== Optimizer & Training ====== #
    args.optim = 'Adam' #'RMSprop' #SGD, RMSprop, ADAM...
    args.lr = 0.00015
    args.epoch = 30
    args.train_batch_size = 512
    args.test_batch_size = 1024

    return args


def experiment_task(args, partition):
    # ====== Random Seed Initialization ====== #
    seed = 123
    np.random.seed(seed)
    torch.manual_seed(seed)                
    setting, result = experiment(partition, deepcopy(args))
    save_exp_result(setting, result)


MLP_args = setting_args(model_name="MLP")
