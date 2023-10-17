import numpy as np
from os import listdir
from os.path import isfile, isdir, join
import os
import json
import random
import re
import scipy.io as sio

def read_imgid_label_pair(filename):
    cwd = os.getcwd()
    prefix = os.path.join(cwd, 'materials/sun/images')

    image_dict = sio.loadmat(filename)
    image_path = image_dict['images']
    label_to_imgid = {}
    label_to_path = {}
    for i in range(image_path.shape[0]):
        path = image_path[i]
        path_list = path[0][0].split('/')
        label = ''
        for j in range(len(path_list)):
            if j > 0 and j < len(path_list) - 1:
                label += path_list[j]
        if label not in label_to_imgid.keys():
            label_to_imgid[label] = []
        if label not in label_to_path.keys():
            label_to_path[label] = []
        label_to_imgid[label].append(i)
        label_to_path[label].append(prefix + '/' + path[0][0])
    return label_to_imgid, label_to_path

def read_img_attr_label(filename):
    attr_dict = sio.loadmat(filename)
    img_attr_labels = attr_dict['labels_cv'] # (14340, 102)
    return img_attr_labels

def get_cl_attr_label(img_attr_labels, label_to_imgid, idx_to_label, cl_num):
    cl_attr_probs = []
    for i in range(cl_num):
        label = idx_to_label[i]
        imgid_list = label_to_imgid[label]
        attr_probs = img_attr_labels[imgid_list].mean(0)
        cl_attr_probs.append(attr_probs)
    cl_attr_probs = np.array(cl_attr_probs)
    avg_prob = cl_attr_probs.mean()
    cl_attr_labels = (cl_attr_probs > avg_prob).astype('float32')
    return cl_attr_labels

if __name__ == '__main__':

    prefix = './materials/sun/SUNAttributeDB'

    label_to_imgid, label_to_path = read_imgid_label_pair(os.path.join(prefix, 'images.mat'))
    img_attr_labels = read_img_attr_label(os.path.join(prefix, 'attributeLabels_continuous.mat'))

    all_label_list = list(label_to_imgid.keys())
    cl_num = len(all_label_list)
    all_label_list.sort()
    label_to_idx = {}
    idx_to_label = {}
    for i in range(len(all_label_list)):
        idx_to_label[i] = all_label_list[i]
        label_to_idx[all_label_list[i]] = i

    cl_attr_labels = get_cl_attr_label(img_attr_labels, label_to_imgid, idx_to_label, cl_num)

    savedir = './materials/sun/'
    dataset_list = ['base','val','novel']

    rs_label_list = list(label_to_imgid.keys())
    random.shuffle(rs_label_list)
    for dataset in dataset_list:
        file_list = []
        label_list = []
        for i, label in enumerate(rs_label_list):
            label_id = label_to_idx[label]
            if 'base' in dataset:
                if (i >= 0 and i < 430):
                    file_list = file_list + label_to_path[label]
                    label_list = label_list + np.repeat(label_id, len(label_to_path[label])).tolist()
            if 'val' in dataset:
                if (i >= 430 and i < 645):
                    file_list = file_list + label_to_path[label]
                    label_list = label_list + np.repeat(label_id, len(label_to_path[label])).tolist()
            if 'novel' in dataset:
                if (i >= 645):
                    file_list = file_list + label_to_path[label]
                    label_list = label_list + np.repeat(label_id, len(label_to_path[label])).tolist()
        with open(savedir + dataset + '.json', 'w') as outfile:
            json.dump({'label_names':all_label_list, 'image_names':file_list, 'image_labels':label_list,
                       'attr_labels': cl_attr_labels.tolist()}, outfile)

        print("%s -OK" %dataset)
