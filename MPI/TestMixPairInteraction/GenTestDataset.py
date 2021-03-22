import numpy as np
import scipy as sp
import scipy.spatial as ss
import scipy.sparse as sp
import scipy.io as sio

import matplotlib as mpl
import matplotlib.pyplot as plt


n_pts = 10000  # points
n_q = 100000  # queries

box_edge = 10
box_size = np.array([box_edge, box_edge, box_edge])


def impose_pbc(coords, boxsize):
    dim = len(boxsize)
    for p in coords:
        for i in range(dim):
            while p[i] < 0:
                p[i] = p[i]+boxsize[i]
            while p[i] >= boxsize[i]:
                p[i] = p[i]-boxsize[i]

    return


pts = np.zeros(shape=[n_pts, 4])
query = np.zeros(shape=[n_q, 4])

# generate coordinates
pts[:, :3] = np.random.lognormal(mean=1, sigma=1, size=[n_pts, 3])
query[:, :3] = np.random.uniform(low=0, high=box_edge, size=[n_q, 3])

# generate search radius
pts[:, 3] = np.random.uniform(low=0, high=1, size=n_pts)
query[:, 3] = np.random.uniform(low=0, high=1, size=n_q)

impose_pbc(pts[:, :3], box_size)
impose_pbc(query[:, :3], box_size)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pts[::10, 0], pts[::10, 1], pts[::10, 2])
ax.scatter(query[::100, 0], query[::100, 1], query[::100, 2])
plt.savefig('Distribution.png')

np.savetxt('Pts.txt', pts, delimiter=' ', header='x,y,z,r')
np.savetxt('Query.txt', query, delimiter=' ', header='x,y,z,r')

# build trees for neighbor detection
tree_pts = ss.cKDTree(pts[:, :3], boxsize=box_size)
tree_query = ss.cKDTree(pts[:, :3], boxsize=box_size)

pairs = []

# query neighbors for each query point
for i in range(n_q):
    q = query[i]
    rad = q[3]
    nbs = tree_pts.query_ball_point(x=q[:3], r=q[3])
    for nb in nbs:
        pairs.append([i, nb])


# query neighbors for each pts point
for i in range(n_pts):
    q = pts[i]
    rad = q[3]
    nbs = tree_query.query_ball_point(x=q[:3], r=q[3])
    for nb in nbs:
        pairs.append([nb, i])

data = np.ones(len(pairs))
ii = np.array([p[0] for p in pairs])
jj = np.array([p[1] for p in pairs])
# convert pairs to sparse adjacency matrix
# non zero M_[ij] -> query i, pts j pair
nb_mat = sp.coo_matrix((data, (ii, jj)), shape=[n_q, n_pts], dtype=np.int)
sio.mmwrite("nb_mat.mtx", nb_mat)
