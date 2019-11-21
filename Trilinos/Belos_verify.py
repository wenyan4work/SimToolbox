import numpy as np
import scipy as sp
import scipy.io as sio
import matplotlib as mpl
import matplotlib.pyplot as plt

A = sio.mmread('A_TCMAT.mtx').toarray()
b = sio.mmread('b_TV.mtx')
xTrue = sio.mmread('xTrue_TV.mtx')
xGuess = sio.mmread('xGuess_TV.mtx')
xsolGMRES=sio.mmread('xsol_GMRES_TV.mtx')
xsolBICGSTAB=sio.mmread('xsol_BICGSTAB_TV.mtx')

print(np.linalg.cond(A))
fig, ax = plt.subplots()
from matplotlib.colors import LogNorm
cs = ax.imshow(A,norm=LogNorm())
cbar = fig.colorbar(cs)

xErrorInit = xGuess - xTrue
xErrorGMRES=xsolGMRES-xTrue
bErrorGMRES = np.dot(A,xsolGMRES) - b
xErrorBICGSTAB=xsolBICGSTAB-xTrue
bErrorBICGSTAB = np.dot(A,xsolBICGSTAB) - b

xErrorInitNorm = np.linalg.norm(xErrorInit)
xErrorGMRESNorm=np.linalg.norm(xErrorGMRES)
bErrorGMRESNorm = np.linalg.norm(bErrorGMRES) 
xErrorBICGSTABNorm=np.linalg.norm(xErrorBICGSTAB)
bErrorBICGSTABNorm = np.linalg.norm(bErrorBICGSTAB) 

print("xErrorInitNorm",xErrorInitNorm)
print("xErrorGMRES",xErrorGMRESNorm,"bErrorGMRES",bErrorGMRESNorm)
print("xErrorBICGSTAB",xErrorBICGSTABNorm,"bErrorBICGSTAB",bErrorBICGSTABNorm)