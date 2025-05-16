import unittest
import torch
import os
import numpy
import time
from torchsvm.cvksvm import cvksvm
from torchsvm.functions import *

class Testcvksvm(unittest.TestCase):
    def test_fit_predict(self):
        # Simple training set: 2D
        nn = 200 # Number of samples
        nm = 5    # Number of clusters per class
        pp = 10   # Number of features
        p1 = p2 = pp // 2    # Number of positive/negative centers
        mu = 2.0  # Mean shift
        ro = 3  # Standard deviation for normal distribution
        sdn = 42  # Seed for reproducibility

        nlam = 10
        torch.manual_seed(sdn)
        ulam = torch.logspace(3, -3, steps=nlam)

        X_train, y_train, means_train = data_gen(nn, nm, pp, p1, p2, mu, ro, sdn)
        X_test, y_test, means_test = data_gen(nn // 10, nm, pp, p1, p2, mu, ro, sdn)
        X_train = standardize(X_train)
        X_test = standardize(X_test)

        sig = sigest(X_train)
        Kmat = rbf_kernel(X_train, sig)

        torch.manual_seed(sdn)
        nfolds = 5
        if nfolds == nn:
            foldid = torch.arange(nn) # Each row gets its own fold ID
        else:
            # Randomly assign fold IDs across the rows
            # foldid = torch.tensor(np.random.permutation(np.repeat(np.arange(1, nfolds + 1), nn // nfolds + 1)[:nn]))
            foldid = torch.randperm(nn) % nfolds + 1

        model1 = cvksvm(Kmat=Kmat, y=y_train, nlam=nlam, ulam=ulam, foldid=foldid, nfolds=nfolds, eps=1e-5, maxit=1000, gamma=1e-8, is_exact=0, device='cuda')
        model1.fit()

        cv_mis = model1.cv(model1.pred, y_train).numpy()
        best_ind = numpy.argmin(cv_mis)

        Kmat = Kmat.double()
        alpmat = model1.alpmat.to('cpu')
        intcpt = alpmat[0,best_ind]
        alp = alpmat[1:,best_ind]
        ka = torch.mv(Kmat, alp)
        aka = torch.dot(alp, ka)
        obj_magic = model1.objfun(intcpt, aka, ka, y_train, ulam[best_ind], nn)

        obj1 = model1.obj_value(alpmat[:,best_ind], ulam[best_ind])
        Kmat_new = kernelMult(X_test, X_train, sig)
        Kmat_new = Kmat_new.double()

        result = torch.mv(Kmat_new, alpmat[1:,best_ind]) + alpmat[0, best_ind]

        ypred = torch.where(result > 0, torch.tensor(1), torch.tensor(-1))
        model1.predict(Kmat_new, y_test, alpmat[:,best_ind])
        # # Check that alpha is set
        # self.assertIsNotNone(model.alpha, "Alpha should not be None after fit")

        # X_test = np.array([[0.5, 0.5]])
        # y_pred = model.predict(X_test)
        
        # # This is a naive test checking the shape of predictions
        # self.assertEqual(len(y_pred), 1, "Prediction shape mismatch")

        # # Optionally, check sign or magnitude
        # # For a real test, you'd compare against known correct labels
        # self.assertIn(y_pred[0], [-1, 1], "Predicted label should be -1 or 1")

if __name__ == '__main__':
    unittest.main()
