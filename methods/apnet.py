import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import time
import json
from sklearn.metrics import confusion_matrix
from methods.meta_template import MetaTemplate
from torch.autograd import Variable

class APNetTemplate(MetaTemplate):
    def __init__(self, model_func, n_way, n_support, attr_num, attr_loc=False, dataset='CUB'):
        super(APNetTemplate, self).__init__(model_func,  n_way, n_support)
        self.attr_loc = attr_loc

        final_feat_dim = self.feature.final_feat_dim
        if isinstance(final_feat_dim, list):
            if self.attr_loc:
                self.input_dim = final_feat_dim[0] * 16
            else:
                self.input_dim = 1
                for dim in final_feat_dim:
                    self.input_dim *= dim
        else:
            self.input_dim = final_feat_dim
            if self.attr_loc:
                self.input_dim *= 16
        self.attr_num = attr_num
        self.beta = 0.6

        self.classifier = nn.Linear(self.input_dim, self.attr_num*2) # only consider binary attribute
        self.loss_fn = nn.CrossEntropyLoss()
        self.read_attr_labels(dataset)

    def read_attr_labels(self, dataset):
        # TODO: change the filenames with your own path, and add files to filter attribute labels in CUB
        if dataset == 'CUB':
            filename = '/home/huminyang/Code/comet/CUB/filelists/CUB/CUB_200_2011/masked_class_attribute_labels.txt'
        elif dataset == 'AWA2':
            filename = '/home/huminyang/Code/APNet/filelists/AWA2/class_attribute_label.txt'
        else:
            AssertionError('not implement!')

        attr_labels_binary = []
        with open(filename, 'r') as f:
            for line in f.readlines():
                line_split = line.strip().split(' ')
                float_line = []
                for str_num in line_split:
                    float_line.append(int(str_num))
                attr_labels_binary.append(float_line)
        self.attr_labels_split = torch.from_numpy(np.array(attr_labels_binary)).cuda()

    def correct(self, x, joints):
        # correct for image classification
        scores = self.set_forward(x, joints)
        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()

        y_query = np.repeat(range(self.n_way), self.n_query)
        top1_correct = np.sum(topk_ind[:, 0] == y_query)
        return float(top1_correct), len(y_query)

    def forward(self, x, joints):
        if joints is None:
            z_support, z_query = super().parse_feature(x, is_feature=False)
        else:
            z_support, z_query = self.parse_feature(x, joints, is_feature=False)
        feature = torch.cat([z_support, z_query], dim=1) # (n_way, n_support+n_query, dim)
        feature = feature.view(-1, self.input_dim)
        logits = self.classifier(feature)
        return torch.split(logits, 2, -1)

    def set_forward(self, x, joints):
        logits = self.forward(x, joints)
        logits = torch.cat(logits, dim=-1)
        logits = logits.view(self.n_way, self.n_support + self.n_query, -1)

        z_support = logits[:, :self.n_support]
        z_query = logits[:, self.n_support:]

        z_support = z_support.contiguous()
        z_proto = z_support.view(self.n_way, self.n_support, -1).mean(1)  # the shape of z is [n_data, n_dim]
        z_query = z_query.contiguous().view(self.n_way * self.n_query, -1)

        scores = F.cosine_similarity(z_query.unsqueeze(1), z_proto, dim=-1)
        return scores / 0.2

    def set_forward_loss1(self, x, joints):
        y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
        y_query = y_query.cuda()

        scores = self.set_forward(x, joints)
        return self.loss_fn(scores, y_query)

    def set_forward_loss2(self, x, ys, joints):
        ys = ys.view(-1, self.attr_num)
        logits = self.forward(x, joints)
        logits = torch.cat(logits, dim=0)
        return self.loss_fn(logits, ys.transpose(1, 0).reshape(-1))

class APNet_w_attrLoc(APNetTemplate):
    def __init__(self, model_func, n_way, n_support, attr_num, attr_loc=False, dataset='CUB'):
        super(APNet_w_attrLoc, self).__init__(model_func,  n_way, n_support, attr_num, attr_loc, dataset)
        self.globalpool = nn.AdaptiveAvgPool2d((1, 1))

    # this function originates from comet
    def parse_feature(self, x, joints, is_feature):
        x = Variable(x.cuda())
        if is_feature:
            z_all = x
        else:
            x = x.contiguous().view(self.n_way * (self.n_support + self.n_query), *x.size()[2:])
            z_all = self.feature.forward(x)
            z_avg = self.globalpool(z_all).view(z_all.size(0), z_all.size(1))

            joints = joints.contiguous().view(self.n_way * (self.n_support + self.n_query), *joints.size()[2:])
            img_len = x.size()[-1]
            feat_len = z_all.size()[-1]
            joints[:, :, :2] = joints[:, :, :2] / img_len * feat_len
            joints = joints.round().int()
            joints_num = joints.size(1)

            avg_mask = (joints[:, :, 2] == 0) + (joints[:, :, 0] < 0) + (joints[:, :, 1] < 0) + (
                        joints[:, :, 0] >= feat_len) + (joints[:, :, 1] >= feat_len)
            avg_mask = (avg_mask > 0).long().unsqueeze(-1).cuda()  # (85, 15, 1)
            mask_joints = joints.cuda() * (1 - avg_mask)
            mask_joints = (mask_joints[:, :, 0] * 7 + mask_joints[:, :, 1]).unsqueeze(1).repeat(1, 64, 1)
            z_all_2D = z_all.view(z_all.size(0), z_all.size(1), -1)
            mask_z = torch.gather(z_all_2D, dim=-1, index=mask_joints)
            mask_z = mask_z.permute(0, 2, 1)  # (85, 15, 64)
            mask_z_avg = z_avg.unsqueeze(1).repeat(1, joints_num, 1) * avg_mask
            z_all_tmp = mask_z * (1 - avg_mask) + mask_z_avg
            z_all = torch.cat([z_all_tmp, z_avg.unsqueeze(1)], dim=1).view(self.n_way, self.n_support + self.n_query, -1)
        z_support = z_all[:, :self.n_support]
        z_query = z_all[:, self.n_support:]
        return z_support, z_query

    def train_loop(self, epoch, train_loader, optimizer, tf_writer):
        print_freq = 10

        avg_loss, avg_loss1, avg_loss2 = 0, 0, 0
        start_time = time.time()
        for i, (x, y, joints) in enumerate(train_loader):
            self.n_query = x.size(1) - self.n_support
            x, y = x.cuda(), y.cuda()
            attr_labels = self.attr_labels_split[y]
            optimizer.zero_grad()
            loss1 = self.set_forward_loss1(x, joints)
            loss2 = self.set_forward_loss2(x, attr_labels, joints)
            loss = loss1 + self.beta * loss2
            loss.backward()
            optimizer.step()
            avg_loss = avg_loss + loss.item()
            avg_loss1 = avg_loss1 + loss1.item()
            avg_loss2 = avg_loss2 + loss2.item()

            if i % print_freq == 0:
                # print(optimizer.state_dict()['param_groups'][0]['lr'])
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f} | Loss1 {:f} | Loss2 {:f}'.format(epoch, i, len(train_loader),
                        avg_loss / float(i + 1), avg_loss1 / float(i + 1), avg_loss2 / float(i + 1)))
                tf_writer.add_scalar('loss/train', avg_loss / float(i + 1), epoch)
                tf_writer.add_scalar('loss1/train', avg_loss1 / float(i + 1), epoch)
                tf_writer.add_scalar('loss2/train', avg_loss2 / float(i + 1), epoch)
        print("Epoch (train) uses %.2f s!" % (time.time() - start_time))

    def test_loop(self, test_loader, return_std=False):
        acc_all = []

        iter_num = len(test_loader)
        start_time = time.time()
        for i, (x, _, joints) in enumerate(test_loader):
            x = x.cuda()
            self.n_query = x.size(1) - self.n_support
            correct_this, count_this = self.correct(x, joints)

            acc_all.append(correct_this / count_this * 100)
        print("Epoch (test) uses %.2f s!" % (time.time() - start_time))

        acc_all = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std = np.std(acc_all)
        print('%d Test Acc = %4.2f%% +- %4.2f%%' % (iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num)))

        if return_std:
            return acc_mean, acc_std
        else:
            return acc_mean

    def test_loop_with_dists(self, test_loader, all_cls_dists, base_dists, base_cls_dists, attr_num):
        acc_all, dist_all = [], []
        attr_num = all_cls_dists.shape[1]

        iter_num = len(test_loader)
        from tqdm import tqdm
        for i, (x, y, joints) in enumerate(tqdm(test_loader)):
            x, y = x.cuda(), y.cuda()
            self.n_query = x.size(1) - self.n_support
            if self.change_way:
                self.n_way = x.size(0)
            correct_this, count_this = self.correct(x, joints)
            acc_all.append(correct_this / count_this * 100)

            sc_cls = y.unique()
            # original mean-task (down trend)
            task_dists = all_cls_dists[sc_cls, :].mean(0).unsqueeze(0)
            dist_all.append(torch.abs(base_dists - task_dists).sum(-1).mean().item() / attr_num)

        acc_all = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std = np.std(acc_all)
        print('%d Test Acc = %4.2f%% +- %4.2f%%' % (iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num)))

        return acc_all, dist_all

class APNet_wo_attrLoc(APNetTemplate):
    def __init__(self, model_func, n_way, n_support, attr_num, attr_loc=False, dataset='CUB'):
        super(APNet_wo_attrLoc, self).__init__(model_func,  n_way, n_support, attr_num, attr_loc, dataset)

    def train_loop(self, epoch, train_loader, optimizer, tf_writer):
        print_freq = 10

        avg_loss, avg_loss1, avg_loss2 = 0, 0, 0
        start_time = time.time()
        for i, (x, y) in enumerate(train_loader):
            self.n_query = x.size(1) - self.n_support
            x, y = x.cuda(), y.cuda()
            attr_labels = self.attr_labels_split[y]
            optimizer.zero_grad()
            loss1 = self.set_forward_loss1(x, None)
            loss2 = self.set_forward_loss2(x, attr_labels, None)
            loss = loss1 + self.beta * loss2
            loss.backward()
            optimizer.step()
            avg_loss = avg_loss + loss.item()
            avg_loss1 = avg_loss1 + loss1.item()
            avg_loss2 = avg_loss2 + loss2.item()

            if i % print_freq == 0:
                # print(optimizer.state_dict()['param_groups'][0]['lr'])
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f} | Loss1 {:f} | Loss2 {:f}'.format(epoch, i, len(train_loader),
                        avg_loss / float(i + 1), avg_loss1 / float(i + 1), avg_loss2 / float(i + 1)))
                tf_writer.add_scalar('loss/train', avg_loss / float(i + 1), epoch)
                tf_writer.add_scalar('loss1/train', avg_loss1 / float(i + 1), epoch)
                tf_writer.add_scalar('loss2/train', avg_loss2 / float(i + 1), epoch)
        print("Epoch (train) uses %.2f s!" % (time.time() - start_time))

    def test_loop(self, test_loader, return_std=False):
        acc_all = []

        iter_num = len(test_loader)
        start_time = time.time()
        for i, (x, _) in enumerate(test_loader):
            x = x.cuda()
            self.n_query = x.size(1) - self.n_support
            correct_this, count_this = self.correct(x, None)

            acc_all.append(correct_this / count_this * 100)
        print("Epoch (test) uses %.2f s!" % (time.time() - start_time))

        acc_all = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std = np.std(acc_all)
        print('%d Test Acc = %4.2f%% +- %4.2f%%' % (iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num)))

        if return_std:
            return acc_mean, acc_std
        else:
            return acc_mean

    def test_loop_with_dists(self, test_loader, all_cls_dists, base_dists, base_cls_dists, attr_num):
        acc_all, dist_all = [], []

        iter_num = len(test_loader)
        from tqdm import tqdm
        for i, (x, y) in enumerate(tqdm(test_loader)):
            x, y = x.cuda(), y.cuda()
            self.n_query = x.size(1) - self.n_support
            if self.change_way:
                self.n_way = x.size(0)
            correct_this, count_this = self.correct(x, None)
            acc_all.append(correct_this / count_this * 100)

            sc_cls = y.unique()
            # original mean-task (down trend)
            task_dists = all_cls_dists[sc_cls, :].mean(0).unsqueeze(0)
            dist_all.append(torch.abs(base_dists - task_dists).sum(-1).mean().item() / attr_num)

        acc_all = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std = np.std(acc_all)
        print('%d Test Acc = %4.2f%% +- %4.2f%%' % (iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num)))

        return acc_all, dist_all