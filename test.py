import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.optim
import json
import torch.utils.data.sampler
import os
import glob
import random
import time

import configs
import backbone
import data.feature_loader as feat_loader
from data.datamgr import SetDataManager
from methods.baselinetrain import BaselineTrain
from methods.baselinefinetune import BaselineFinetune
from methods.protonet import ProtoNet
from methods.matchingnet import MatchingNet
from methods.relationnet import RelationNet
from methods.maml import MAML
from methods.apnet import APNet_w_attrLoc, APNet_wo_attrLoc
from io_utils import model_dict, parse_args, get_resume_file, get_best_file, get_assigned_file

seed = 1
np.random.seed(seed)
torch.random.manual_seed(seed)

def feature_evaluation(cl_data_file, model, n_way=5, n_support=5, n_query=15, adaptation=False):
    class_list = cl_data_file.keys()

    select_class = random.sample(class_list, n_way)
    z_all = []
    for cl in select_class:
        img_feat = cl_data_file[cl]
        perm_ids = np.random.permutation(len(img_feat)).tolist()
        z_all.append([np.squeeze(img_feat[perm_ids[i]]) for i in range(n_support + n_query)])  # stack each batch

    z_all = torch.from_numpy(np.array(z_all))

    model.n_query = n_query
    if adaptation:
        scores = model.set_forward_adaptation(z_all, is_feature=True)
    else:
        scores = model.set_forward(z_all, is_feature=True)
    pred = scores.data.cpu().numpy().argmax(axis=1)
    y = np.repeat(range(n_way), n_query)
    acc = np.mean(pred == y) * 100
    return acc


if __name__ == '__main__':
    params = parse_args('test')

    acc_all = []
    iter_num = 600
    attr_loc = False
    few_shot_params = dict(n_way=params.test_n_way, n_support=params.n_shot)

    split = params.split
    if params.save_iter != -1:
        split_str = split + "_" + str(params.save_iter)
    else:
        split_str = split
    if 'Conv' in params.model:
        image_size = 84
    else:
        image_size = 224
    datamgr = SetDataManager(image_size, n_eposide=iter_num, n_query=15, **few_shot_params)
    loadfile = configs.data_dir[params.dataset] + split + '.json'

    if params.method == 'baseline':
        model = BaselineFinetune(model_dict[params.model], **few_shot_params)
    elif params.method == 'baseline++':
        model = BaselineFinetune(model_dict[params.model], loss_type='dist', **few_shot_params)
    elif params.method == 'protonet':
        model = ProtoNet(model_dict[params.model], **few_shot_params)
    elif params.method == 'comet':
        assert params.dataset == 'CUB'
        model = COMET(model_dict[params.model], **few_shot_params)
    elif params.method == 'matchingnet':
        model = MatchingNet(model_dict[params.model], **few_shot_params)
    elif params.method in ['relationnet', 'relationnet_softmax']:
        if params.model == 'Conv4':
            feature_model = backbone.Conv4NP
        elif params.model == 'Conv6':
            feature_model = backbone.Conv6NP
        elif params.model == 'Conv4S':
            feature_model = backbone.Conv4SNP
        else:
            feature_model = lambda: model_dict[params.model](flatten=False)
        loss_type = 'mse' if params.method == 'relationnet' else 'softmax'
        model = RelationNet(feature_model, loss_type=loss_type, **few_shot_params)
    elif params.method in ['maml', 'maml_approx']:
        backbone.ConvBlock.maml = True
        backbone.SimpleBlock.maml = True
        backbone.BottleneckBlock.maml = True
        backbone.ResNet.maml = True
        model = MAML(model_dict[params.model], approx=(params.method == 'maml_approx'), **few_shot_params)
        if params.dataset in ['omniglot', 'cross_char']:  # maml use different parameter in omniglot
            model.n_task = 32
            model.task_update_num = 1
            model.train_lr = 0.1
    elif params.method == 'apnet':
        if params.dataset == 'CUB':
            attr_loc = True
            attr_num = 109
        elif params.dataset == 'SUN':
            attr_num = 102
        elif params.dataset == 'AWA2':
            attr_num = 85
        else:
            AssertionError("not implement!")

        few_shot_params = dict(n_way=params.test_n_way, n_support=params.n_shot, attr_num=attr_num, attr_loc=attr_loc, dataset=params.dataset)
        if attr_loc:
            model = APNet_w_attrLoc(model_dict[params.model], **few_shot_params)
        else:
            model = APNet_wo_attrLoc(model_dict[params.model], **few_shot_params)
    else:
        raise ValueError('Unknown method')

    model = model.cuda()

    checkpoint_dir = '%s/checkpoints/%s/%s_%s_%s' % (
    configs.save_dir, params.dataset, params.model, params.method, params.exp_str)
    if params.train_aug:
        checkpoint_dir += '_aug'
    if not params.method in ['baseline', 'baseline++']:
        checkpoint_dir += '_%dway_%dshot' % (params.train_n_way, params.n_shot)

    if not params.method in ['baseline', 'baseline++']:
        if params.save_iter != -1:
            modelfile = get_assigned_file(checkpoint_dir, params.save_iter)
        else:
            modelfile = get_best_file(checkpoint_dir)
        if modelfile is not None:
            tmp = torch.load(modelfile)
            model.load_state_dict(tmp['state'])

    if params.method in ['maml', 'maml_approx']:  # maml do not support testing with feature
        novel_loader = datamgr.get_data_loader(loadfile, aug=False, is_train=False)
        if params.dataset == 'SUN' and params.model == 'Conv4':
            model.train_lr = 0.1
        if params.adaptation:
            model.task_update_num = 100  # We perform adaptation on MAML simply by updating more times.
        model.eval()
        acc_mean, acc_std = model.test_loop(novel_loader, return_std=True)
    else:
        novel_file = os.path.join(checkpoint_dir.replace("checkpoints", "features"),
                                  split_str + ".hdf5")  # defaut split = novel, but you can also test base or val classes
        cl_data_file = feat_loader.init_loader(novel_file)

        from tqdm import tqdm
        for i in tqdm(range(iter_num)):
            acc = feature_evaluation(cl_data_file, model, n_query=15, adaptation=params.adaptation, **few_shot_params)
            acc_all.append(acc)

        acc_all = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std = np.std(acc_all)
        print('%d Test Acc = %4.2f%% +- %4.2f%%' % (iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num)))