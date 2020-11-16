import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import OrdinalEncoder
import os
from torch import from_numpy

def BuildData(**inp):
    if len(inp) == 0:
        inp ={'task': 'classification', 'yardage_cutoff': 5, 'train_or_val': 'train'}
    task = inp.get('task', 'classification')
    tov = inp.get('train_or_val', 'train')
    cutoff = inp.get('yardage_cutoff', 5)
    if task == 'classification':
        return ClassificationData(cutoff, tov)
    elif task == 'approximation':
        return ApproximationData(tov)
    raise ValueError("`task` must be in {approximation, classification}")

TR_PERC = 0.25  # I would like to train on 75-80% of data but it is too large

''' base class & class to represent data for approximation task '''
class ApproximationData:

    def __init__(self, train_or_val='train'):
        self.train_or_val = train_or_val
        self.fpath = os.path.join('.', 'data', self.train_or_val) + '.csv'
        self.raw_data = pd.read_csv(self.fpath , low_memory=False)
        self._fillna()
        column_info = json.loads(open('columns.json').read())
        self.feature_encoders = {}
        self.features = column_info['features']
        self.label = column_info['label']
        self.processed_data = self.process()
        self.N, self.D = self.processed_data.shape

    # call helper processing methods
    def process(self):
        proc = self._drop_cols()
        proc = self._encode_feature(proc, 'OffensePersonnel')
        proc = self._encode_feature(proc, 'OffenseFormation')
        proc = self._encode_feature(proc, 'FieldPosition')
        proc = self._height_to_int(proc)
        proc = proc.astype(np.float64)
        return proc

    # returns 2 tuples (train_x, train_y), (test_x, test_y)
    def train_test_split(self, tr_perc=TR_PERC):
        tr_mask = np.array([False] * self.N)
        K = int(self.N * tr_perc)
        tr_mask[:K] = True
        tr_mask = np.random.permutation(tr_mask)
        train = self.processed_data[tr_mask]
        test  = self.processed_data[~tr_mask]
        return self._xy(train), self._xy(test)

    def _xy(self, mat):
        x = mat[self.features].to_numpy().astype(np.float32)
        y = mat[self.label].to_numpy().astype(np.float32)
        return from_numpy(x), from_numpy(y)

    def _fillna(self):
        for c in self.raw_data.columns:
            nmask = self.raw_data[c].isnull()
            if not nmask.values.any():
                continue
            dtype = self.raw_data.dtypes[c].type
            val = 0
            if np.int64 == dtype:
                val = np.int64(0)
            elif np.float64 == dtype:
                val = np.float64(0.)
            elif np.object == dtype or np.object_ == dtype:
                val = ""
            self.raw_data[c][nmask] = val
        return

    def _drop_cols(self):
        mat = self.raw_data[self.features + [self.label]]
        return mat

    def _encode_feature(self, mat, feature_column):
        feat = mat[feature_column].to_numpy().reshape(-1, 1)
        enc = OrdinalEncoder()
        enc.fit(feat)
        self.feature_encoders[feature_column] = enc
        mat.loc[:,feature_column] = enc.transform(feat)
        return mat

    def _height_to_int(self, mat):
        heights = mat['PlayerHeight'].to_numpy()
        for idx, _ in enumerate(heights):
            f, i = heights[idx].split('-')
            heights[idx] = 12*int(f) + int(i)
        mat.loc[:,'PlayerHeight'] = heights
        return mat

    def _inverse_transform_features(self):
        mat = self.processed_data.copy()
        for k, v in self.feature_encoders.items():
            mat.loc[:,k] = v.inverse_transform(mat[k].to_numpy().reshape(-1, 1))
        return mat

''' class to represent data for classification task when we want to
predict if yards >= yardage_cutoff '''
class ClassificationData(ApproximationData):

    def __init__(self, yardage_cutoff, train_or_val='train'):
        super().__init__(train_or_val=train_or_val)
        self.yardage_cutoff = yardage_cutoff
        self._cutoff()

    def balanced_train_test_split(self, tr_perc=TR_PERC):
        tr_mask = np.array([False] * self.N)
        y = self.processed_data[self.label]
        vc = y.value_counts()
        K = int(min(vc) * tr_perc)
        for val, _ in vc.iteritems():
            idxs = np.where(y == val)[0]
            idxs = np.random.choice(idxs, size=K, replace=False)
            tr_mask[idxs] = True
        train = self.processed_data[tr_mask]
        test  = self.processed_data[~tr_mask]
        return self._xy(train), self._xy(test)

    def _cutoff(self):
        yards = self.processed_data[self.label]
        yards = (yards >= self.yardage_cutoff).astype(np.int)
        yards[yards == 0] = -1
        self.processed_data[self.label] = yards
        return