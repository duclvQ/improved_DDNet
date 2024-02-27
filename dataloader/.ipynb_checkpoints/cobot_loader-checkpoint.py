#! /usr/bin/env python
#! coding:utf-8:w

from tqdm import tqdm
import numpy as np
from sklearn import preprocessing
import pickle5 as pickle
from pathlib import Path
import sys
sys.path.insert(0, '..')
from utils import *  # noqa
current_file_dirpath = Path(__file__).parent.absolute()
_dataset ="IPN"
_list = "_list"
def load_cobot_data(
        train_path=current_file_dirpath / Path(f"../data/{_dataset}/train{_list}.pickle"),
        test_path=current_file_dirpath / Path(f"../data/{_dataset}/test{_list}.pickle")):
    Train = pickle.load(open(train_path, "rb"))
    Test = pickle.load(open(test_path, "rb"))

    new_Train = {
        "pose": list(),
        "label": list()
    }
    new_Test = {
        "pose": list(),
        "label": list()
    }
    chosen_label_list = [2, 3 , 4, 5, 13, 14]
    for i in range(len(Train['pose'])):
        if Train["label"][i] not in chosen_label_list: continue
        new_Train["pose"].append(Train["pose"][i])
        new_Train["label"].append(chosen_label_list.index(Train["label"][i]))
    for i in range(len(Test['pose'])):
        if Test["label"][i] not in chosen_label_list: continue
        new_Test["pose"].append(Test["pose"][i])
        new_Test["label"].append(chosen_label_list.index(Test["label"][i]))
    #le = preprocessing.LabelEncoder()
    #le.fit(Train['label'])
    #print("Loading JHMDB Dataset")
    le = None
    print("Done")
    return new_Train, new_Test, le


class CConfig():
    def __init__(self):
        self.frame_l = 60  # the length of frames
        self.joint_n = 21  # the number of joints
        self.joint_d = 3  # the dimension of joints
        self.clc_num = 21  # the number of class
        self.feat_d = 210
        self.filters = 64

# Genrate dataset
# T: Dataset  C:config le:labelEncoder


def Cdata_generator(T, C, le):
    X_0 = []
    X_1 = []
    Y = []
    #labels = le.transform(T['label'])
    for i in tqdm(range(len(T['pose']))):
        p = np.copy(T['pose'][i])
        # p.shape (frame,joint_num,joint_coords_dims)
        p = zoom(p, target_l=C.frame_l,
                 joints_num=C.joint_n, joints_dim=C.joint_d)
        # p.shape (target_frame,joint_num,joint_coords_dims)
        # label = np.zeros(C.clc_num)
        # label[labels[i]] = 1
        label = T['label'][i]
        # M.shape (target_frame,(joint_num - 1) * joint_num / 2)
        M = get_CG(p, C)

        X_0.append(M)
        X_1.append(p)
        Y.append(label)

    X_0 = np.stack(X_0)
    X_1 = np.stack(X_1)
    Y = np.stack(Y)
    X_0 = np.nan_to_num(X_0)
    X_1 = np.nan_to_num(X_1)
    Y = np.nan_to_num(Y)
    return X_0, X_1, Y
