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

    imgid_to_label = {}
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

        imgid_to_label[i] = label
    return label_to_imgid, label_to_path, imgid_to_label

def read_img_attr_label(filename):
    attr_dict = sio.loadmat(filename)
    img_attr_labels = attr_dict['labels_cv'] # (14340, 102)
    return img_attr_labels

# def get_cl_attr_label(img_attr_labels, label_to_imgid, idx_to_label, cl_num):
#     cl_attr_probs = []
#     for i in range(cl_num):
#         label = idx_to_label[i]
#         imgid_list = label_to_imgid[label]
#         attr_probs = img_attr_labels[imgid_list].mean(0)
#         cl_attr_probs.append(attr_probs)
#     cl_attr_probs = np.array(cl_attr_probs)
#     return cl_attr_probs

def get_cl_attr_label(img_attr_labels, imgid_to_label, label_to_idx, cl_num):
    count = 0
    class_attr_count = np.zeros((cl_num, 102, 2))

    for i in range(img_attr_labels.shape[0]):
        count += 1
        class_label = imgid_to_label[i]
        class_label_id = label_to_idx[class_label]

        attr_labels = img_attr_labels[i, :]
        for j in range(102):
            attr_label_prob = attr_labels[j]
            if attr_label_prob >= 0.5:
                class_attr_count[class_label_id][j][1] += 1
            else:
                class_attr_count[class_label_id][j][0] += 1
    print("count:", count)

    class_attr_min_label = np.argmin(class_attr_count, axis=2)
    class_attr_max_label = np.argmax(class_attr_count, axis=2)
    equal_count = np.where(
        class_attr_min_label == class_attr_max_label)  # check where 0 count = 1 count, set the corresponding class attribute label to be 1
    class_attr_max_label[equal_count] = 1

    min_class_count = 10
    attr_class_count = np.sum(class_attr_max_label, axis=0)
    mask = np.where(attr_class_count >= min_class_count)[
        0]  # select attributes that are present (on a class level) in at least [min_class_count] classes
    class_attr_label_masked = class_attr_max_label[:, mask]
    return class_attr_label_masked


if __name__ == '__main__':

    prefix = './materials/sun/SUNAttributeDB'

    label_to_imgid, label_to_path, imgid_to_label = read_imgid_label_pair(os.path.join(prefix, 'images.mat'))
    img_attr_labels = read_img_attr_label(os.path.join(prefix, 'attributeLabels_continuous.mat'))

    all_label_list = list(label_to_imgid.keys())
    cl_num = len(all_label_list)
    all_label_list.sort()
    label_to_idx = {}
    idx_to_label = {}
    for i in range(len(all_label_list)):
        idx_to_label[i] = all_label_list[i]
        label_to_idx[all_label_list[i]] = i

    #cl_attr_probs = get_cl_attr_label(img_attr_labels, label_to_imgid, idx_to_label, cl_num)
    masked_cl_attr_probs = get_cl_attr_label(img_attr_labels, imgid_to_label, label_to_idx, cl_num)

    savedir = './materials/sun/'

    with open(savedir + 'masked_attr_dist.json', 'w') as outfile:
        json.dump({'attr_dist': masked_cl_attr_probs.tolist()}, outfile)
    print("%s -OK")
