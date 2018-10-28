import copy
import argparse

from train_base import *

from module import BroadGuidedFilter

parser = argparse.ArgumentParser(description='Train Deep Guided Filtering Networks')
parser.add_argument('--task',  type=str, default='non_local_dehazing',          help='TASK')
parser.add_argument('--name',  type=str, default='LR',                 help='NAME')
parser.add_argument('--model', type=str, default='deep_guided_filter', help='model')
args = parser.parse_args()

config = copy.deepcopy(default_config)

config.TASK = args.task
config.NAME = args.name
config.N_EPOCH = 150
config.DATA_SET = 512


# model
config.model = BroadGuidedFilter(1*3*96*64, 100, 11000, 1*3*96*64)


def forward(imgs, config):
    x_hr, gt_hr, x_lr = imgs[:3]
    if config.GPU >= 0:
        with torch.cuda.device(config.GPU):
            x_hr, gt_hr, x_lr = x_hr.cuda(), gt_hr.cuda(), x_lr.cuda()

    return config.model(x_lr, x_hr), gt_hr


config.forward = forward


run(config)
