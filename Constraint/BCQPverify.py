import scipy.optimize as so
import numpy as np
import scipy as sp
import scipy.io as sio
import os
import sys

localSize = 500
diagAdd = 0.5

if len(sys.argv) > 1:
    localSize = sys.argv[1]
if len(sys.argv) > 2:
    diagAdd = sys.argv[2]

print('mpirun -n 2 --map-by numa ./BCQPSolver_test ' +
      str(localSize)+' '+str(diagAdd)+' > ./testLog')
os.system('mpirun -n 2 --map-by numa ./BCQPSolver_test ' +
          str(localSize)+' '+str(diagAdd)+' > ./testLog')


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
                  tol=1e-8, method='TNC')

if res.success:
    print('reference min:', res.fun)
    print('BBPGD min:', func(xsolBBPGD))
    print('APGD min:', func(xsolAPGD))
    errorBBPGD = xsolBBPGD-res.x
    errorAPGD = xsolAPGD-res.x
    print('BBPGD error norm: ', np.linalg.norm(errorBBPGD))
    print(' APGD error norm: ', np.linalg.norm(errorAPGD))
else:
    print('scipy optimization failed')
