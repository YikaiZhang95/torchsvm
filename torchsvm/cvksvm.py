import torch
import os
import numpy
import time

from .functions import *

class cvksvm:
    """
    Kernel SVM with Regularization and Acceleration.

    This function initializes the optimization process for a kernel SVM model,
    supporting advanced features like GPU acceleration and iterative projection methods
    for large-scale data.

    Parameters
    ----------
    Kmat : ndarray or tensor
        The kernel matrix of shape (n_samples, n_samples).

    y : ndarray or tensor
        Target labels for each sample, of shape (n_samples,). Typically, -1 or 1.

    nlam : int
        The number of regularization parameters to consider in the optimization.

    ulam : ndarray or tensor
        User-specified regularization parameters, of shape (nlam,).

    foldid : ndarray
        Array indicating the fold assignment for cross-validation. Each element is an 
        integer corresponding to a fold.

    nfolds : int, default=5
        The number of cross-validation folds to use.

    eps : float, default=1e-5
        Tolerance for convergence in the optimization.

    maxit : int, default=1000
        Maximum number of iterations allowed for the optimization process.

    gamma : float, default=1.0
        Regularization parameter for kernel methods, controlling the trade-off between 
        margin width and misclassification.

    is_exact : int, default=0
        Indicates whether projection step is used (1 for exact, 0 for approximate).

    delta_len : int, default=8
        Length of delta vector used in projection steps.

    mproj : int, default=10
        Number of projection steps to perform for iterative optimization.

    KKTeps : float, default=1e-3
        Tolerance for KKT conditions in the primary optimization problem.

    KKTeps2 : float, default=1e-3
        Tolerance for KKT conditions in secondary checks.

    device : {'cuda', 'cpu'}, default='cuda'
        Device to perform computations on. Default is GPU ('cuda') for improved performance.

    Attributes
    ----------
    self.alpmat : ndarray or tensor
        Matrix of optimized alpha values after fitting the data, of shape (n_samples, nlam).

    self.npass : int
        Number of passes made over the data during the optimization.

    self.cvnpass : int
        Number of passes made during cross-validation.

    self.jerr : int
        Error flag to indicate any issues during computation (0 for success, non-zero for errors).

    self.pred : ndarray or tensor
        Predicted values based on the optimization, of shape (n_samples,).

    Notes
    -----
    This implementation is designed for large-scale data problems and leverages GPU
    acceleration for improved computational efficiency. Regularization is controlled
    through multiple hyperparameters, allowing fine-tuned trade-offs between accuracy
    and computational cost.

    Examples
    --------
    >>> from torchksvm import cvksvm
    >>> import numpy as np
    >>> Kmat = np.random.rand(100, 100)
    >>> y = np.random.choice([-1, 1], size=100)
    >>> model = cvksvm(Kmat, y, nlam=50, ulam=np.logspace(-4, 4, 50), foldid=np.random.randint(1, 6, size=100))
    >>> model.fit()
    """

    def __init__(self, Kmat, y, nlam, ulam, foldid, nfolds = 5, eps=1e-5, maxit=1000, gamma=1.0, is_exact=0, delta_len=8, mproj=10, KKTeps=1e-3, KKTeps2=1e-3, device='cuda'):
        self.device = device
        self.Kmat = Kmat.double().to(self.device)
        self.y = y.double().to(self.device)
        # self.Kmat = None
        # self.y = None
        self.nobs = Kmat.shape[0]
        self.nlam = nlam
        self.ulam = ulam.double()
        self.eps = eps
        self.maxit = maxit
        self.gamma = gamma
        self.is_exact = is_exact
        self.delta_len = delta_len
        self.mproj = mproj
        self.KKTeps = KKTeps
        self.KKTeps2 = KKTeps2
        self.nfolds = nfolds
        self.nmaxit = self.nlam * self.maxit
        self.foldid = foldid

        # Initialize outputs
        self.alpmat = torch.zeros((self.nobs + 1, self.nlam), dtype=torch.double).to(self.device)
        self.anlam = 0
        self.npass = torch.zeros(self.nlam, dtype=torch.int32).to(self.device)
        self.cvnpass = torch.zeros(self.nlam, dtype=torch.int32).to(self.device)
        self.pred = torch.zeros((self.nobs, self.nlam), dtype=torch.double).to(self.device)
        self.jerr = 0

    def fit(self):
        nobs = self.nobs
        nlam = self.nlam
        y = self.y
        Kmat = self.Kmat
        nfolds = self.nfolds

        r = torch.zeros(nobs, dtype=torch.double).to(self.device)
        alpmat = torch.zeros((nobs + 1, nlam), dtype=torch.double).to(self.device)
        npass = torch.zeros(nlam, dtype=torch.int32).to(self.device)
        cvnpass = torch.zeros(nlam, dtype=torch.int32).to(self.device)
        alpvec = torch.zeros(nobs + 1, dtype=torch.double).to(self.device)
        pred = torch.zeros((self.nobs, self.nlam), dtype=torch.double).to(self.device)
        jerr = 0
        eps2 = 1.0e-5

        # Precompute sum of Kmat along rows
        Ksum = torch.sum(Kmat, dim=1)
        # Kinv = torch.linalg.inv(Kmat)

        eigens, Umat = torch.linalg.eigh(Kmat)
        eigens = eigens.double().to(self.device)
        Umat = Umat.double().to(self.device)
        Kmat = Kmat.double().to(self.device)
        eigens += self.gamma
        Usum = torch.sum(Umat, dim = 0)
        einv = 1 / eigens
        # eU = torch.mm(torch.diag(einv), Umat.T)
        eU = (einv * Umat).T
        # Kinv1 = torch.mm(Umat, eU)

        vareps = 1.0e-8

        lpUsum = torch.zeros((nobs, self.delta_len), dtype=torch.double, device=self.device)
        lpinv = torch.zeros((nobs, self.delta_len), dtype=torch.double, device=self.device)
        svec = torch.zeros((nobs, self.delta_len), dtype=torch.double, device=self.device)
        vvec = torch.zeros((nobs, self.delta_len), dtype=torch.double, device=self.device)
        gval = torch.zeros((self.delta_len), dtype=torch.double, device=self.device)


        for l in range(nlam):
            # start = time.time()
            al = self.ulam[l].item()
            delta = 1.0
            delta_id = 0
            delta_save = 0
            oldalpvec = torch.zeros(nobs + 1, dtype=torch.double).to(self.device)

            while delta_id < self.delta_len:
                delta_id += 1
                opdelta = 1.0 + delta
                omdelta = 1.0 - delta
                oddelta = 1.0 / delta

                if delta_id > delta_save:
                    lpinv[:, delta_id - 1] = 1.0 / (eigens + 4.0 * float(nobs) * delta * al)
                    lpUsum[:, delta_id - 1] = lpinv[:, delta_id - 1] * Usum
                    vvec[:, delta_id - 1] = torch.mv(Umat, eigens * lpUsum[:, delta_id - 1])
                    svec[:, delta_id - 1] = torch.mv(Umat, lpUsum[:, delta_id - 1])
                    gval[delta_id - 1] = 1.0 / (nobs + 4.0 * nobs * delta * vareps - vvec[:, delta_id - 1].sum())
                    delta_save = delta_id

                # Compute residual r
                told = 1.0
                ka = torch.mv(Kmat, alpvec[1:])
                r = y * (alpvec[0] + ka)
                # Update alpha
                #alpha loop
                for iteration in range(self.maxit):
                    zvec = torch.where(r < omdelta, -y, torch.where(r > opdelta, torch.zeros(1, device=self.device), 0.5 * y * oddelta * (r - opdelta)))
                    gamvec = zvec + 2.0 * float(nobs) * al * alpvec[1:]##
                    rds = zvec.sum() + 2.0 * nobs * vareps * alpvec[0]
                    hval = rds - torch.dot(vvec[:, delta_id - 1], gamvec)

                    tnew = 0.5 + 0.5 * torch.sqrt(torch.tensor(1.0, device=self.device) + 4.0 * told * told)
                    mul = 1.0 + (told - 1.0) / tnew
                    told = tnew.item()
                    
                    # Update step using Pinv
                    if delta_id > self.delta_len:
                        print("Exceeded maximum delta_id")
                        break
                    
                    # Compute dif vector
                    
                    dif_step = torch.zeros((nobs + 1), dtype=torch.double, device=self.device)
                    dif_step[0] = -2.0 * mul * delta * gval[delta_id - 1] * hval
                    dif_step[1:] = -dif_step[0] * svec[:, delta_id - 1] - 2.0 * mul * delta * torch.mv(Umat, gamvec @ Umat * lpinv[:, delta_id - 1])
                    alpvec += dif_step

                    # Update residual
                    ka = torch.mv(Kmat, alpvec[1:])
                    r = y * (alpvec[0] + ka)
                    npass[l] += 1

                    # Check convergence
                    if torch.max(dif_step ** 2) < (self.eps * mul * mul):
                        break


                    if torch.sum(npass) > self.maxit:
                        jerr = -l - 1
                        break

                # Check KKT conditions
                dif_step = oldalpvec - alpvec
                ka = torch.mv(Kmat, alpvec[1:])
                aka = torch.dot(ka, alpvec[1:])
                obj_value = self.objfun(alpvec[0], aka, ka, y, al, nobs)
                # eps_float64 = np.finfo(np.float64).eps
                # optimal_intercept = minimize_scalar(self.objfun, args=(aka, ka, y, al, nobs), bracket=(-100.0, 100.0), method="brent")
                # obj_value_new = self.objfun(optimal_intercept.x, aka, ka, y, al, nobs)
                golden_s = self.golden_section_search(-100.0, 100.0, nobs, ka, aka, y, al)
                int_new = golden_s[0]
                obj_value_new = golden_s[1]
                if obj_value_new < obj_value:
                    dif_step[0] = dif_step[0] + int_new - alpvec[0]
                    r = r + y * (int_new - alpvec[0])
                    alpvec[0] = int_new

                oldalpvec = alpvec.clone()

                zvec = torch.where(r < 1.0, -y, torch.where(r > 1.0, torch.zeros(1).to(self.device), -0.5 * y))
                KKT = zvec / float(nobs) + 2.0 * al * alpvec[1:]
                uo = max(al, 1.0)
                KKT_norm = torch.sum(KKT ** 2) / (uo ** 2)
                if KKT_norm < self.KKTeps:
                    # Check convergence
                    dif_norm = torch.max(dif_step ** 2)
                    if dif_norm < float(nobs) * (self.eps * mul * mul):
                        if self.is_exact == 0:
                            break
                        else:
                            is_exit = False
                            alptmp = alpvec.clone()
                            for nn in range(self.mproj):
                                elbowid = torch.zeros(nobs, dtype=torch.bool)
                                elbchk = True
                                # Compute rmg and check elbow condition
                                rmg = torch.abs(1.0 - r)
                                elbowid = rmg < delta
                                elbchk = torch.all(rmg[elbowid] <= 1e-3).item()

                                if elbchk:
                                    break

                                # Projection update
                                told = 1.0
                                for _ in range(self.maxit):
                                    ka = torch.mv(Kmat, alptmp[1:])
                                    aKa = torch.dot(ka, alptmp[1:])
                                    obj_value = self.objfun(alptmp[0], aka, ka, y, al, nobs)

                                    # Optimize intercept
                                    # optimal_intercept = minimize_scalar(self.objfun, args=(aka, ka, y, al, nobs), bracket=(-100.0, 100.0), method = 'brent')
                                    # obj_value_new = self.objfun(optimal_intercept.x, aka, ka, y, al, nobs)
                                    golden_s = self.golden_section_search(-100.0, 100.0, nobs, ka, aka, y, al)
                                    int_new = golden_s[0]
                                    obj_value_new = golden_s[1]
                                    if obj_value_new < obj_value:
                                        dif_step[0] = dif_step[0] + int_new - alptmp[0]
                                        alptmp[0] = int_new

                                    r = y * (alptmp[0] + ka)
                                    zvec = torch.where(r < omdelta, -y, torch.where(r > opdelta, torch.zeros(1, device=self.device), 0.5 * y * oddelta * (r - opdelta)))
                                    gamvec = zvec + 2.0 * float(nobs) * al * alptmp[1:]##
                                    rds = zvec.sum() + 2.0 * nobs * vareps * alptmp[0]
                                    hval = rds - torch.dot(vvec[:, delta_id - 1], gamvec)

                                    tnew = 0.5 + 0.5 * torch.sqrt(torch.tensor(1.0, device=self.device) + 4.0 * told * told)
                                    mul = 1.0 + (told - 1.0) / tnew
                                    told = tnew.item()
                                        
                                    # Compute dif vector
                                    
                                    # dif_step = torch.zeros((nobs + 1), dtype=torch.double, device=self.device)
                                    dif_step[0] = -2.0 * mul * delta * gval[delta_id - 1] * hval
                                    dif_step[1:] = -dif_step[0] * svec[:, delta_id - 1] - 2.0 * mul * delta * torch.mv(Umat, gamvec @ Umat * lpinv[:, delta_id - 1])
                                    alptmp += dif_step

                                    ka = torch.mv(Kmat, alptmp[1:])
                                    r = y * (alptmp[0] + ka)
                                    npass[l] += 1
                                    alp_old = alptmp.clone()

                                    if torch.sum(elbowid).item() > 1:
                                        theta = torch.mv(Kmat, alptmp[1:])
                                        theta[elbowid] += y[elbowid] * (1.0 - r[elbowid])
                                        alptmp[1:] = torch.mv(Umat, torch.mv(eU, theta))

                                    dif_step = dif_step + alptmp - alp_old
                                    r = y * (alptmp[0] + torch.mv(Kmat, alptmp[1:]))
                                    mdd = torch.max(dif_step ** 2)
                                    # Check convergence
                                    if mdd < self.eps * mul ** 2:
                                        break
                                    elif mdd > nobs and npass[l] > 2:
                                        is_exit = True
                                        break
                                    if torch.sum(npass) > self.maxit:
                                        is_exit = True
                                        break

                            # Check KKT condition
                            if is_exit:
                                break
                            zvec = torch.where(r < 1.0, -y, torch.where(r > 1.0, torch.zeros(1).to(self.device), -0.5 * y))
                            KKT = zvec / nobs + 2.0 * al * alptmp[1:]
                            uo = max(al, 1.0)

                            if torch.sum(KKT ** 2) / (uo ** 2) < self.KKTeps:
                                alpvec = alptmp.clone()
                                break
                # else:
                #     # Reduce delta
                #     delta *= 0.125
                if delta_id >= self.delta_len:
                    print(f"Exceeded maximum delta iterations for lambda {l}")
                    break
                delta *= 0.125
            # Save the alpha vector for current lambda
            alpmat[:, l] = alpvec
            # Update anlam
            self.anlam = l

            # Check if maximum iterations exceeded
            if torch.sum(npass) > self.maxit:
                self.jerr = -l - 1
                break
            # print(f'Single fitting:{time.time() - start}')
            
            ######### cross-validation
            for nf in range(nfolds):
                # start = time.time()
                yn = y.clone()

                # Set the current fold's labels to zero
                yn[self.foldid == (nf + 1)] = 0.0

                loor = r.clone() # Initial residuals
                looalp = alpvec.clone() # Initial alphas

                delta = 1.0
                delta_id = 0

                # while delta_id < self.delta_len:
                while True:
                    delta_id += 1
                    opdelta = 1.0 + delta
                    omdelta = 1.0 - delta
                    oddelta = 1.0 / delta

                    if delta_id > delta_save:
                        lpinv[:, delta_id - 1] = 1.0 / (eigens + 4.0 * float(nobs) * delta * al)
                        lpUsum[:, delta_id - 1] = lpinv[:, delta_id - 1] * Usum
                        vvec[:, delta_id - 1] = torch.mv(Umat, eigens * lpUsum[:, delta_id - 1])
                        svec[:, delta_id - 1] = torch.mv(Umat, lpUsum[:, delta_id - 1])
                        gval[delta_id - 1] = 1.0 / (nobs + 4.0 * nobs * delta * vareps - vvec[:, delta_id - 1].sum())
                        delta_save = delta_id

                    # Compute residual r
                    told = 1.0
                    dif_step = torch.zeros_like(alpvec)
                    ka = torch.mv(Kmat, looalp[1:])
                    loor = yn * (looalp[0] + ka)

                    while torch.sum(cvnpass) <= self.nmaxit:
                        zvec = torch.where(loor < omdelta, -yn, torch.where(loor > opdelta, torch.zeros(1).to(self.device), yn * torch.tensor(0.5) * oddelta * (loor - opdelta)))                        
                        gamvec = zvec + 2.0 * float(nobs) * al * looalp[1:]##
                        rds = zvec.sum() + 2.0 * nobs * vareps * looalp[0]
                        hval = rds - torch.dot(vvec[:, delta_id - 1], gamvec)

                        tnew = 0.5 + 0.5 * torch.sqrt(torch.tensor(1.0, device=self.device) + 4.0 * told * told)
                        mul = 1.0 + (told - 1.0) / tnew
                        told = tnew.item()
                    
                        # Compute dif vector
                        
                        dif_step = torch.zeros((nobs + 1), dtype=torch.double, device=self.device)
                        dif_step[0] = -2.0 * mul * delta * gval[delta_id - 1] * hval
                        dif_step[1:] = -dif_step[0] * svec[:, delta_id - 1] - 2.0 * mul * delta * torch.mv(Umat, gamvec @ Umat * lpinv[:, delta_id - 1])
                        looalp += dif_step

                        # zvec = torch.where(loor < omdelta, -yn, torch.where(loor > opdelta, torch.zeros(1).to(self.device), yn * torch.tensor(0.5) * oddelta * (loor - opdelta)))

                        # rds = torch.zeros(nobs + 1, dtype=torch.double).to(self.device)
                        # rds[0] = torch.sum(zvec) + 2.0 * nobs * vareps * looalp[0]
                        # rds[1:] = torch.mv(Kmat, zvec + 2.0 * float(nobs) * al * looalp[1:])

                        # tnew = 0.5 + 0.5 * torch.sqrt(torch.tensor(1.0).to(self.device) + 4.0 * told ** 2)
                        # mul = 1.0 + (told - 1.0) / tnew
                        # told = tnew.item()

                        # dif_step = -2.0 * delta * mul * torch.mv(Pinv[:, :, delta_id - 1], rds)
                        # looalp += dif_step

                        loor = yn * (looalp[0] + torch.mv(Kmat, looalp[1:]))

                        cvnpass[l] += 1

                        # Check convergence
                        if torch.max(dif_step ** 2) < eps2 * (mul ** 2):
                            break
                    if torch.sum(cvnpass) > self.nmaxit:
                        break
                    # dif_step = oldalpvec - alpvec
                    # print(f'Fitting alp time:{time.time() - start}')

                    ka = torch.mv(Kmat, looalp[1:])
                    aka = torch.dot(ka, looalp[1:])
                    obj_value = self.objfun(looalp[0], aka, ka, yn, al, nobs)
                    # optimal_intercept = minimize_scalar(self.objfun, args=(aka, ka, yn, al, nobs), bracket=(-100.0, 100.0), method="brent")
                    # obj_value_new = self.objfun(optimal_intercept.x, aka, ka, yn, al, nobs)
                    golden_s = self.golden_section_search(-100.0, 100.0, nobs, ka, aka, yn, al)
                    int_new = golden_s[0]
                    obj_value_new = golden_s[1]
                    if obj_value_new < obj_value:
                        dif_step[0] = dif_step[0] + int_new - looalp[0]
                        loor = loor + y * (int_new - looalp[0])
                        looalp[0] = int_new

                    # print(f'Fitting intercpt time:{time.time() - start}')
                    oldalpvec = looalp.clone()

                    zvec = torch.where(loor < 1.0, -yn, torch.where(loor > 1.0, torch.zeros(1).to(self.device), -torch.tensor(0.5) * yn))
                    KKT = zvec / float(nobs) + 2.0 * al * looalp[1:]
                    uo = max(al, 1.0)
                    KKT_norm = torch.sum(KKT ** 2) / (uo ** 2)

                    if KKT_norm < self.KKTeps2:
                        # Check convergence
                        # print(f'dif_step{dif_step}')
                        # dif_norm = torch.max(dif_step ** 2)
                        # print(f'dif:{dif_norm}')
                        # print(f'mul:{mul}')
                        # print(f'dif_cont:{float(nobs) * self.eps * mul * mul}')
                        # if dif_norm < float(nobs) * (self.eps * mul * mul):
                        if self.is_exact == 0:
                            break
                        else:
                            is_exit = False
                            alptmp = looalp.clone()
                            for nn in range(self.mproj):
                                elbowid = torch.zeros(nobs, dtype=torch.bool)
                                elbchk = True
                                # Compute rmg and check elbow condition
                                rmg = torch.abs(1.0 - loor)
                                elbowid = rmg < delta
                                elbchk = torch.all(rmg[elbowid] <= 1e-2).item()

                                if elbchk:
                                    break

                                # Projection update
                                told = 1.0
                                for _ in range(self.maxit):
                                    ka = torch.mv(Kmat, alptmp[1:])
                                    aKa = torch.dot(ka, alptmp[1:])
                                    obj_value = self.objfun(alptmp[0], aka, ka, yn, al, nobs)

                                    # Optimize intercept
                                    golden_s = self.golden_section_search(-100.0, 100.0, nobs, ka, aka, yn, al)
                                    int_new = golden_s[0]
                                    obj_value_new = golden_s[1]
                                    if obj_value_new < obj_value:
                                        dif_step[0] = dif_step[0] + int_new - alptmp[0]
                                        alptmp[0] = int_new

                                    loor = yn * (alptmp[0] + ka)
                                    zvec = torch.where(loor < omdelta, -yn, torch.where(loor > opdelta, torch.zeros(1).to(self.device), 0.5 * yn * oddelta* (loor - opdelta)))

                                    # rds = torch.zeros(nobs + 1, dtype=torch.double).to(self.device)
                                    # rds[0] = torch.sum(zvec) + 2.0 * float(nobs) * vareps * alptmp[0]
                                    # rds[1:] = torch.mv(Kmat, zvec + 2.0 * float(nobs) * al * alptmp[1:])

                                    # tnew = 0.5 + 0.5 * torch.sqrt(torch.tensor(1.0).to(self.device) + 4.0 * told ** 2)
                                    # mul = 1.0 + (told - 1.0) / tnew
                                    # told = tnew.item()

                                    # dif_step = - 2.0 * delta * mul * torch.mv(Pinv[:, :, delta_id - 1], rds)
                                    # alptmp += dif_step

                                    gamvec = zvec + 2.0 * float(nobs) * al * alptmp[1:]##
                                    rds = zvec.sum() + 2.0 * nobs * vareps * alptmp[0]
                                    hval = rds - torch.dot(vvec[:, delta_id - 1], gamvec)

                                    tnew = 0.5 + 0.5 * torch.sqrt(torch.tensor(1.0, device=self.device) + 4.0 * told * told)
                                    mul = 1.0 + (told - 1.0) / tnew
                                    told = tnew.item()
                                        
                                    # Compute dif vector
                                    
                                    # dif_step = torch.zeros((nobs + 1), dtype=torch.double, device=self.device)
                                    dif_step[0] = -2.0 * mul * delta * gval[delta_id - 1] * hval
                                    dif_step[1:] = -dif_step[0] * svec[:, delta_id - 1] - 2.0 * mul * delta * torch.mv(Umat, gamvec @ Umat * lpinv[:, delta_id - 1])
                                    alptmp += dif_step

                                    ka = torch.mv(Kmat, alptmp[1:])
                                    loor = yn * (alptmp[0] + ka)
                                    alp_old = alptmp.clone()

                                    if torch.sum(elbowid).item() > 1:
                                        theta = torch.mv(Kmat, alptmp[1:])
                                        theta[elbowid] += yn[elbowid] * (1.0 - loor[elbowid])
                                        alptmp[1:] = torch.mv(Umat, torch.mv(eU, theta))

                                    dif_step = dif_step + alptmp - alp_old
                                    loor = yn * (alptmp[0] + torch.mv(Kmat, alptmp[1:]))
                                    cvnpass[l] += 1
                                    mdd = torch.max(dif_step ** 2)
                                    # Check convergence
                                    if mdd < nobs * eps2 * mul ** 2:
                                        break
                                    elif mdd > nobs and cvnpass[l] > 2:
                                        is_exit = True
                                        break
                                    if torch.sum(cvnpass) > self.nmaxit:
                                        is_exit = True
                                        break
                                if is_exit:
                                    break
                            if is_exit:
                                break
                            looalp = alptmp.clone()
                            break
                    if delta_id >= self.delta_len:
                        print(f"Exceeded maximum delta iterations for lambda {l}")
                        break
                    delta *= 0.125

                # for j in range(nobs):
                #     if self.foldid[j] == (nf + 1):
                #         looalp[j + 1] = 0.0
                loo_ind = (self.foldid == (nf + 1))
                looalp[1:][loo_ind] = 0.0
                pred[loo_ind, l] = looalp[1:] @ Kmat[:, loo_ind]  + looalp[0]
                # print(pred[loo_ind, l][:10])
                # for j in range(nobs):
                #     if self.foldid[j] == (nf + 1):
                #         pred[j, l] = torch.sum(Kmat[:, j] * looalp[1:]) + looalp[0]
                # print(pred[loo_ind, l][:10])
                # print(f'{nf}-fold: {time.time() - start}')
            self.anlam = l


        self.alpmat = alpmat
        self.npass = npass
        self.cvnpass = cvnpass
        self.jerr = jerr
        self.pred = pred


    def cv(self, pred, y):
        pred_label = torch.where(pred > 0, 1, -1).to(device = 'cpu')
        y_expanded = y[:, None]
        misclass_matrix = (pred_label != y_expanded).float()
        misclass_rate = misclass_matrix.mean(dim=0)
        return misclass_rate
        

        
    def objfun(self, intcpt, aka, ka, y, lam, nobs):
        """
        Compute the objective function value for SVM.

        Parameters:
        - intcpt (float): Intercept term.
        - aka (torch.Tensor): Regularization term (alpha * K * alpha).
        - ka (torch.Tensor): Kernel matrix dot alpha vector (K * alpha).
        - y (torch.Tensor): Labels vector of shape (nobs,).
        - lam (float): Regularization parameter.
        - nobs (int): Number of observations.

        Returns:
        - objval (float): Objective function value.
        """
        # Initialize xi (hinge loss terms)
        xi = torch.zeros(nobs, dtype=torch.double)

        # Compute f_hat (fh) and the hinge loss xi
        fh = ka + intcpt
        xi_tmp = 1.0 - y * fh
        xi = torch.where(xi_tmp > 0, xi_tmp, torch.zeros_like(xi_tmp))

        # Compute the objective value
        objval = lam * aka + torch.sum(xi) / nobs

        return objval

    def golden_section_search(self, lmin, lmax, nobs, ka, aka, y, lam):
        """
        Optimize the intercept using golden section search (Brent's method).

        Parameters:
        - lmin (float): Lower bound for the search interval.
        - lmax (float): Upper bound for the search interval.
        - nobs (int): Number of observations.
        - ka (torch.Tensor): Kernel matrix dot alpha vector (K * alpha).
        - aka (float): Regularization term (alpha * K * alpha).
        - y (torch.Tensor): Labels vector of shape (nobs,).
        - lam (float): Regularization parameter.

        Returns:
        - lhat (float): Optimized intercept value.
        - fx (float): Objective function value at the optimized intercept.
        """
        eps = torch.tensor(torch.finfo(torch.float64).eps)
        tol = eps ** 0.25
        tol1 = eps + 1.0
        eps = torch.sqrt(eps)

        # Golden ratio constant
        gold = (3.0 - torch.sqrt(torch.tensor(5.0))) * 0.5

        # Initialize variables
        a = lmin
        b = lmax
        v = a + gold * (b - a)
        w = v
        x = v
        d = 0.0
        e = 0.0

        # Evaluate the objective function at the initial x value
        fx = self.objfun(x, aka, ka, y, lam, nobs)
        fv = fx
        fw = fx
        tol3 = tol / 3.0
        # Main optimization loop
        while True:
            xm = (a + b) * 0.5
            tol1 = eps * abs(x) + tol3
            t2 = 2.0 * tol1

            # Check if the interval is small enough to exit
            if abs(x - xm) <= t2 - (b - a) * 0.5:
                break

            p = 0.0
            q = 0.0
            r = 0.0
            if abs(e) > tol1:
                r = (x - w) * (fx - fv)
                q = (x - v) * (fx - fw)
                p = (x - v) * q - (x - w) * r
                q = 2.0 * (q - r)
                if q > 0.0:
                    p = -p
                else:
                    q = -q
                r = e
                e = d
            # Conditions to use golden section step
            if (abs(p) >= abs(0.5 * q * r)) or (p <= q * (a - x)) or (p >= q * (b - x)):
                if x < xm:
                    e = b - x
                else:
                    e = a - x
                d = gold * e
            else:
                # Parabolic interpolation step
                d = p / q
                u = x + d
                if (u - a < t2) or (b - u < t2):
                    d = tol1
                    if x >= xm:
                        d = - d

            # Set the new point u
            u = x + d if abs(d) >= tol1 else (x + tol1 if d > 0 else x - tol1)
            # Evaluate the objective function at u
            fu = self.objfun(u, aka, ka, y, lam, nobs)
            # Update the search bounds and objective values
            if fu <= fx:
                if u < x:
                    b = x
                else:
                    a = x
                v = w
                fv = fw
                w = x
                fw = fx
                x = u
                fx = fu
            else:
                if u < x:
                    a = u
                else:
                    b = u
                if fu <= fw or w == x:
                    v = w
                    fv = fw
                    w = u
                    fw = fu
                elif fu <= fv or v == x or v == w:
                    v = u
                    fv = fu
        # Return the optimal intercept and the objective value
        lhat = x
        res = self.objfun(x, aka, ka, y, lam, nobs)


        return lhat, res
