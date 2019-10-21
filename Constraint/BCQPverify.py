import scipy.optimize as so
import numpy as np
import scipy as sp
import matplotlib as mpl
import scipy.io as sio
import matplotlib.pyplot as plt
import copy

Amat = sio.mmread('Amat_TCMAT.mtx')
bvec = sio.mmread('bvec_TV.mtx').flatten()
lb = sio.mmread('lbvec_TV.mtx').flatten()
ub = sio.mmread('ubvec_TV.mtx').flatten()
xsolBBPGD = sio.mmread('xsolBBPGD_TV.mtx').flatten()
xsolAPGD = sio.mmread('xsolAPGD_TV.mtx').flatten()
xguess = np.zeros(len(xsolBBPGD))


def func(x):
    return 0.5*x.dot((Amat.dot(x)))+bvec.dot(x)


def grad(x):
    return Amat.dot(x)+bvec


# minimize f = x^T A x + b^T x, subject to lb <= x <= ub
bound = so.Bounds(lb, ub)
res = so.minimize(func, xguess, jac=grad, bounds=bound,
                  tol=1e-6, method='L-BFGS-B')
print(res)

print(func(xsolBBPGD))
print(func(xsolAPGD))
