import numpy as np
import os
import argparse

parser = argparse.ArgumentParser(description='Process XF from log file.')
parser.add_argument('-file', type=str,
                    help='log file name', default='StressLC.log')
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
Pcol_Ref = 8.2


def verify(p, message):
    if np.abs(p - Pcol_Ref)/Pcol_Ref > 0.1:
        print(message, p)
    return


verify(Pcol, "Error Pcol")
verify(Sigma[0, 0], "Error P_xx")
verify(Sigma[1, 1], "Error P_yy")
verify(Sigma[2, 2], "Error P_zz")
