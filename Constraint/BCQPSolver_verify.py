import scipy.optimize as so
import numpy as np
import scipy as sp
import scipy.io as sio
import os
import sys
import matplotlib.pyplot as plt

localSize = 200
diagAdd = 0
maxIte = localSize

if len(sys.argv) > 1:
    localSize = sys.argv[1]
if len(sys.argv) > 2:
    diagAdd = sys.argv[2]
if len(sys.argv) > 3:
    maxIte = sys.argv[3]

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
                  tol=1e-8, method='L-BFGS-B')

print('scipy optimization func eval:', res.nfev)

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

# plot BBPGD and APGD history
os.system('grep BBPGD_HISTORY ./testLog > ./BBPGD.log')
os.system('grep APGD_HISTORY ./testLog > ./APGD.log')

bbhistory = np.genfromtxt('BBPGD.log', usecols=(5, 6), delimiter=',')
ahistory = np.genfromtxt('APGD.log', usecols=(5, 6), delimiter=',')

plt.semilogy(bbhistory[:, 1], bbhistory[:, 0], label='BBPGD')
plt.semilogy(ahistory[:, 1], ahistory[:, 0], label='APGD')
plt.ylabel('residual')
plt.xlabel('MV Count')
plt.legend()
plt.show()
plt.savefig('testBCQP.png', dpi=150)

plt.loglog(bbhistory[:, 1], bbhistory[:, 0], label='BBPGD')
plt.loglog(ahistory[:, 1], ahistory[:, 0], label='APGD')
plt.ylabel('residual')
plt.xlabel('MV Count')
plt.legend()
plt.show()
plt.savefig('testBCQP-log.png', dpi=150)
