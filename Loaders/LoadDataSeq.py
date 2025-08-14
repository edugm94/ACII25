import re
import os
import glob
import pickle
import numpy as np
import pandas as pd
import scipy.io as sio
from utils.constants import *
from sklearn.preprocessing import scale
from sklearn.preprocessing import LabelEncoder

class LoadDataSequence:
    def __init__(self, dataset):
        self.dataset = dataset

        if dataset == "seed":
            self.path = PATH_DE_FEATURES
        elif dataset == "seed4":
            self.path = PATH_DE_FEATURES_SEED4
        elif dataset == "seed5":
            self.path = PATH_DE_FEATURES_SEED5
        elif dataset == "seed7":
            self.path = PATH_DE_FEATURES_SEED7

        else:
            raise ValueError("Invalid dataset")


    def _load_seed_data(self, pt, sess):
        path2labels = os.path.join(self.path, "label.mat")
        path2data = sorted(glob.glob(self.path + "{}_*".format(pt)))[sess - 1]

        data = sio.loadmat(path2data)
        labels = (sio.loadmat(path2labels)["label"] + 1).reshape(-1)
        keys_to_filter = ["de_LDS{}".format(i) for i in range(1, 16)]
        features = dict(filter(lambda item: item[0] in keys_to_filter, data.items()))

        seq_data = []
        seq_labels = []
        seq_trials = []

        for id, key in enumerate(features.keys()):
            # Get trial's features
            feat_ = np.swapaxes(features[key], 0, 1)
            n_samples, _, _ = feat_.shape
            feat_transpose = np.transpose(feat_, (0, 2, 1))
            feat_ = feat_transpose.reshape(n_samples, -1)

            # Create labels according to the number of samples
            labels_ = np.ones(n_samples) * labels[id]

            # Create trials according to the number of samples
            trials_ = [id + 1 for _ in range(n_samples)]

            seq_data.append(feat_)
            seq_labels.append(labels_)
            seq_trials.append(trials_)

        return seq_data, seq_labels, seq_trials

    def _load_seed4_data(self, pt, sess):
        path = os.path.join(self.path, str(sess))
        filename = sorted(glob.glob(f"{path}/{pt}_*"))[0]

        data = sio.loadmat(filename[0])
        labels = ...
        keys_to_filter = ["de_LDS{}".format(i) for i in range(1, 25)]
        features = dict(filter(lambda item: item[0] in keys_to_filter, data.items()))
        print("STOP")


    def _load_seed5_data(self, pt, sess):
        trials_id = {
            1: [id for id in range(0, 15)],
            2: [id for id in range(15, 30)],
            3: [id for id in range(30, 45)]
        }
        labels = {
            1: [4, 1, 3, 2, 0, 4, 1, 3, 2, 0, 4, 1, 3, 2, 0],
            2: [2, 1, 3, 0, 4, 4, 0, 3, 2, 1, 3, 4, 1, 2, 0],
            3: [2, 1, 3, 0, 4, 4, 0, 3, 2, 1, 3, 4, 1, 2, 0]
        }

        path2data = sorted(glob.glob(self.path + "{}_*".format(pt)))[0]
        data = np.load(path2data)
        features = pickle.loads(data['data'])

        seq_data = []
        seq_labels = []
        seq_trials = []

        for id, key in enumerate(trials_id[sess]):
            feat_ = features[key]
            n_samples = feat_.shape[0]

            # Create labels according to the number of samples
            labels_ = np.ones(n_samples) * labels[sess][id]

            # Create trials according to the number of samples
            trials_ = [id + 1 for _ in range(n_samples)]

            seq_data.append(feat_)
            seq_labels.append(labels_)
            seq_trials.append(trials_)

        return seq_data, seq_labels, seq_trials

    def _load_seed7_data(self, pt, sess):
        trials_id_dict = {
            1: [el for el in range(1, 21)],
            2: [el for el in range(21, 41)],
            3: [el for el in range(41, 61)],
            4: [el for el in range(61, 81)],
        }

        labels_dict = {
            1: [0, 2, 3, 5, 6, 6, 5, 3, 2, 0, 0, 2, 3, 5, 6, 6, 5, 3, 2, 0],
            2: [6, 5, 4, 2, 1, 1, 2, 4, 5, 6, 6, 5, 4, 2, 1, 1, 2, 4, 5, 6],
            3: [0, 1, 3, 4, 6, 6, 4, 3, 1, 0, 0, 1, 3, 4, 6, 6, 4, 3, 1, 0],
            4: [3, 5, 4, 1, 0, 0, 1, 4, 5, 3, 3, 5, 4, 1, 0, 0, 1, 4, 5, 3]
        }

        path2data = os.path.join(self.path, f"{pt}.mat")
        data = sio.loadmat(path2data)
        labels = labels_dict[sess]
        trials_id = trials_id_dict[sess]

        keys_to_filter = ["de_LDS_{}".format(i) for i in trials_id]
        # keys_to_filter = ["de_{}".format(i) for i in trials_id]
        features = dict(filter(lambda item: item[0] in keys_to_filter, data.items()))

        seq_data = []
        seq_labels = []
        seq_trials = []

        for id, key in enumerate(features.keys()):
            # Get trial's features
            feat_ = features[key]
            n_samples, _, _ = feat_.shape
            feat_transpose = np.transpose(feat_, (0, 2, 1))
            feat_ = feat_transpose.reshape(n_samples, -1)

            # Create labels according to the number of samples
            labels_ = np.ones(n_samples) * labels[id]

            # Create trials according to the number of samples
            trials_ = [id + 1 for _ in range(n_samples)]

            seq_data.append(feat_)
            seq_labels.append(labels_)
            seq_trials.append(trials_)

        return seq_data, seq_labels, seq_trials


    def _load_data(self, pt, sess):
        if self.dataset == "seed":
            sets = self._load_seed_data(pt, sess)
        elif self.dataset == "seed4":
            sets = self._load_seed4_data(pt, sess)
        elif self.dataset == "seed5":
            sets = self._load_seed5_data(pt, sess)
        elif self.dataset == "seed7":
            sets = self._load_seed7_data(pt, sess)
        else:
            raise ValueError("Invalid dataset")

        x = np.concatenate(sets[0], axis=0)
        x_norm = scale(x)
        y = np.concatenate(sets[1], axis=0)

        if self.dataset == "seed7":
            enc = LabelEncoder()
            y_enc = enc.fit_transform(y)

        trial = np.concatenate(sets[2], axis=0)

        assert x.shape[0] == y.shape[0] == trial.shape[0], "Invalid data shape"
        if self.dataset == "seed7":
            return pd.DataFrame({
                "x": list(x),
                "x_norm": list(x_norm),
                "y": list(y),
                "y_enc": list(y_enc),
                "trial": list(trial)
            })
        else:
            return pd.DataFrame({
                "x": list(x),
                "x_norm": list(x_norm),
                "y": list(y),
                "trial": list(trial)
            })


    def __call__(self, *args, **kwargs):
        pt = kwargs['pt']
        sess = kwargs['sess']

        if self.dataset == "seed":
            assert sess in range(1, 4), "Invalid session number"
            assert pt in range(1, 16), "Invalid participant number"
        else:
            assert sess in range(1, 5), "Invalid session number"
            assert pt in range(1, 21), "Invalid participant number"
        df = self._load_data(pt, sess)
        return df

