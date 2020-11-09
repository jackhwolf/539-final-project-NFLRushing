from src.data import ClassificationData, ApproximationData
from src.data import BuildData
import models
import numpy as np
import os
import json
import time

def run_experiment(params):
    reports = []
    L = len(params)
    for i, p in enumerate(params):
        print(p)
        report = Experiment(*p[0], **p[1]).run()
        reports.append(report)
        print(f"========{i+1}/{L}========")
    out = {}
    for i in range(len(params)):
        out[i] = {'params': params[i], 'report': reports[i]}
    print(os.getcwd())
    fpath = os.path.join('Results', str(int(time.time()))) + '.json'
    with open(fpath, 'a') as fp:
        fp.write(json.dumps(out, indent=4))
    return fpath

class Experiment:

    def __init__(self, *modelargs, **dataparams):
        self.modelargs = modelargs
        self.dataparams = dataparams

    def run(self):
        data = BuildData(**self.dataparams)
        indims = data.D-1
        outdims = 1
        module = __import__(f'models.{self.modelargs[0]}')
        model = getattr(module, self.modelargs[0])
        model = getattr(model, self.modelargs[0])(indims, outdims, *self.modelargs[1:])
        if isinstance(data, ClassificationData):
            (train_x, train_y), (test_x, test_y) = data.balanced_train_test_split(tr_perc=0.05)
        else:
            (train_x, train_y), (test_x, test_y) = data.train_test_split(tr_perc=0.05)
        final_loss = model.learn(train_x, train_y)
        preds = model.predict(train_x)
        return self._report(final_loss, self._isclose(preds, train_x)) 

    def _report(self, final_loss, acc):
        out = {
            'final_loss': str(final_loss), 
            'accuracy': str(acc),
        }
        return out

    def _isclose(self, pred, true, tol=0.01):
        out = np.isclose(pred, true, tol)
        out = np.sum(out) / out.size
        return out