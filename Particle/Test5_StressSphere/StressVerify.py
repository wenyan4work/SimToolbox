import numpy as np
import os
import argparse

parser = argparse.ArgumentParser(description='Process XF from log file.')
parser.add_argument('-file', type=str,
                    help='log file name', default='StressSphere.log')
args = parser.parse_args()
file = args.file

print('log file: ', file)

grepString = 'ColXF'
recordFile = 'record_'+grepString+'.csv'
cmd = 'grep '+grepString+' '+file+' > '+recordFile
print(cmd)
os.system(cmd)
XFHistory = np.genfromtxt(recordFile, usecols=(
    1, 2, 3, 4, 5, 6, 7, 8, 9), delimiter=',')

# reshape to 3x3 tensor history
# use last 2000 frames
SigmaHistory = np.reshape(XFHistory, (-1, 3, 3))[-2000:]
PHistory = np.mean(XFHistory[:, [0, 4, 8]], axis=1)

Sigma = np.mean(SigmaHistory, axis=0)
Pcol = np.mean(PHistory)

# pressure accuracy
Pcol_CS = 3.037

if np.abs(Pcol - Pcol_CS) > 0.3:
    print("Fail: pressure error: \n", Pcol)

# isotropy accuracy
if np.abs(Sigma[0, 0] - Pcol_CS) > 0.3:
    print("Fail: sigma_xx error\n", Sigma[0, 0])
if np.abs(Sigma[1, 1] - Pcol_CS) > 0.3:
    print("Fail: sigma_yy error\n", Sigma[1, 1])
if np.abs(Sigma[2, 2] - Pcol_CS) > 0.3:
    print("Fail: sigma_zz error\n", Sigma[2, 2])
