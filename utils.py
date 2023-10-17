import torch
import numpy as np
import torch
import matplotlib.pyplot as plt

def one_hot(y, num_class):
    return torch.zeros((len(y), num_class)).scatter_(1, y.unsqueeze(1), 1)


def DBindex(cl_data_file):
    class_list = cl_data_file.keys()
    cl_num = len(class_list)
    cl_means = []
    stds = []
    DBs = []
    for cl in class_list:
        cl_means.append(np.mean(cl_data_file[cl], axis=0))
        stds.append(np.sqrt(np.mean(np.sum(np.square(cl_data_file[cl] - cl_means[-1]), axis=1))))

    mu_i = np.tile(np.expand_dims(np.array(cl_means), axis=0), (len(class_list), 1, 1))
    mu_j = np.transpose(mu_i, (1, 0, 2))
    mdists = np.sqrt(np.sum(np.square(mu_i - mu_j), axis=2))

    for i in range(cl_num):
        DBs.append(np.max([(stds[i] + stds[j]) / mdists[i, j] for j in range(cl_num) if j != i]))
    return np.mean(DBs)


def sparsity(cl_data_file):
    class_list = cl_data_file.keys()
    cl_sparsity = []
    for cl in class_list:
        cl_sparsity.append(np.mean([np.sum(x != 0) for x in cl_data_file[cl]]))

    return np.mean(cl_sparsity)

"""
Files for plot figs of adaptation difficulty
"""

def read_attr_dists(trainloader, dataset):
    if dataset == 'SUN':
        print("attribute distance for SUN!")
        attr_dists = trainloader.dataset.meta['attr_labels']
        attr_dists_array = np.array(attr_dists).astype('float32')
        attr_dists = torch.from_numpy(attr_dists_array)

        base_labels = trainloader.dataset.cl_list
        base_ind = np.unique(base_labels).tolist()
    elif dataset == 'CUB':
        print("attribute distance for CUB!")
        filename = 'filelists/CUB/CUB_200_2011/masked_class_attribute_labels.txt'
        attr_dists = []
        with open(filename, 'r') as f:
            for line in f.readlines():
                line_split = line.strip().split(' ')
                float_line = []
                for str_num in line_split:
                    float_line.append(float(str_num))
                attr_dists.append(float_line)

        attr_dists_array = np.array(attr_dists)
        attr_dists = torch.from_numpy(attr_dists_array)

        base_ind = []
        for i in range(200):
            if i % 2 == 0:
                base_ind.append(i)
    elif dataset == 'AWA2':
        print("attribute distance for AWA2!")
        filename = 'filelists/AWA2/class_attribute_label.txt'
        attr_dists = []
        with open(filename, 'r') as f:
            for line in f.readlines():
                line_split = line.strip().split(' ')
                float_line = []
                for str_num in line_split:
                    float_line.append(float(str_num))
                attr_dists.append(float_line)

        attr_dists_array = np.array(attr_dists)
        attr_dists = torch.from_numpy(attr_dists_array)

        base_labels = trainloader.dataset.cl_list
        base_ind = np.unique(base_labels).tolist()
    else:
        AssertionError("not implement!")
    return attr_dists, base_ind

def get_attr_distance(trainloader, dataset):
    attr_dists, base_ind = read_attr_dists(trainloader, dataset)

    # class-agnostic or task-agnostic
    # part_dists = _dists_check(part_dists)
    # base_dists = part_dists[base_ind, :].mean(0)  # (102,)
    #
    # all_cls_dists = part_dists
    # base_cls_dists = part_dists[base_ind, :]  # (100, 102)

    # original
    import random
    base_cls_dists = []
    sc_cls_lists = [random.sample(base_ind, 5) for _ in range(10000)]
    for sc_cls in sc_cls_lists:
        sc_dists = attr_dists[sc_cls, :]
        base_cls_dists.append(sc_dists)
    base_cls_dists = torch.stack(base_cls_dists, dim=0)
    all_cls_dists = attr_dists
    base_dists = base_cls_dists.mean(1)  # (task_num, 102)

    return all_cls_dists, base_dists, base_cls_dists

def interval_avg(acc_all, dist_all):
    min_d = np.min(dist_all)
    max_d = np.max(dist_all)
    inr = (max_d - min_d) / 9
    acc_inr = [0 for _ in range(9)]
    dis_inr = [0 for _ in range(9)]
    cout_inr = [0 for _ in range(9)]
    for dis, acc in zip(dist_all, acc_all):
        for i in range(9):
            min_i = min_d + i * inr
            max_i = min_d + (i + 1) * inr

            if dis >= min_i and dis <= max_i:
                acc_inr[i] += acc
                dis_inr[i] += dis
                cout_inr[i] += 1

    acc_avg_inr, dis_avg_inr = [], []
    for acc, dis, num in zip(acc_inr, dis_inr, cout_inr):
        if num != 0:
            acc_avg_inr.append(1.0 * acc / num)
            dis_avg_inr.append(1.0 * dis / num)
    return acc_avg_inr, dis_avg_inr


def plot_fig(acc_all, dist_all):
    acc_avg_inr, dis_avg_inr = interval_avg(acc_all, dist_all)
    print("acc_avg_inr:", acc_avg_inr)
    print("dis_avg_inr:", dis_avg_inr)

    plt.scatter(dist_all, acc_all)
    plt.scatter(dis_avg_inr, acc_avg_inr, s=40, marker='x', c='red')

    for x, y in zip(dis_avg_inr, acc_avg_inr):
        plt.annotate("%.1f" % (y), xy=(x, y), xytext=(x - 0.005, y + 1.5), color='r', weight='heavy')

    plt.plot(dis_avg_inr, acc_avg_inr, c='red')
    # plt.show()
    plt.savefig('dist_acc.pdf')
    plt.close()