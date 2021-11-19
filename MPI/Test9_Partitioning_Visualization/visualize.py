import os
import shutil

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import matplotlib.cm as cmx

# set up save path
save_path = 'figs'
if os.path.exists(save_path):
    shutil.rmtree(save_path)
os.mkdir(save_path)

cm = plt.get_cmap('jet')
cNorm = mc.Normalize(vmin=0, vmax=4)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)

fig_name = 'NonUniformMPI_A_'
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
pars = np.loadtxt('NonUniformMPI_A.csv', usecols=(0, 1, 2, 4), dtype=np.double)
ax.scatter(pars[:, 0], pars[:, 1], pars[:, 2], c=scalarMap.to_rgba(pars[:, 3]))
for i in range(0, 361, 60):
    ax.view_init(elev=10., azim=i)
    plt.savefig(os.path.join(save_path, fig_name + str(i) + ".png"))
