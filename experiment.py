from data import ClassificationData, ApproximationData
from data import BuildData
import models
import numpy as np


def experiment(*modelargs, **dataparams):
    return Experiment(*modelargs, **dataparams).run()

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
        return self._report(data, model, final_loss, self._isclose(preds, train_x)) 

    def _report(self, data, model, final_loss, acc):
        out = {
            'data': data,
            'model': model,
            'final_loss': final_loss, 
            'accuracy': acc,
        }
        return out

    def _isclose(self, pred, true, tol=0.01):
        out = np.isclose(pred, true, tol)
        out = np.sum(out) / out.size
        return out