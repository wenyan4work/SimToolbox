import numpy as np
import scipy as sp
import scipy.spatial as ss
import scipy.sparse as sp
import scipy.io as sio

n_pts = 100000  # points
n_query = 100000  # queries

box_edge = 10
box_size = np.array([box_edge, box_edge, box_edge])

pts = np.zeros(shape=[n_pts, 4])
query = np.zeros(shape=[n_query, 4])

# generate coordinates
# pts is an unbound set of points
pts[:, :3] = np.random.lognormal(mean=1, sigma=1, size=[n_pts, 3])
query[:, :3] = np.random.uniform(low=0, high=box_edge, size=[n_query, 3])


# generate search radius
pts[:, 3] = np.random.uniform(low=0, high=0.2, size=n_pts)
query[:, 3] = np.random.uniform(low=0, high=0.2, size=n_query)

np.savetxt('Pts.txt', pts, delimiter=' ', header='x,y,z,r')
np.savetxt('Query.txt', query, delimiter=' ', header='x,y,z,r')

# build trees for neighbor detection
tree_pts = ss.KDTree(pts[:, :3])
tree_query = ss.KDTree(query[:, :3])

query_search_pts = tree_pts.query_ball_point(
    x=query[:, :3], r=query[:, 3], workers=-1)
pts_search_query = tree_query.query_ball_point(
    x=pts[:, :3], r=pts[:, 3], workers=-1)


# list of [i,j], i is index of query, j is index of pts
pairs = []

for i in range(len(query_search_pts)):
    for j in query_search_pts[i]:
        pairs.append([i, j])

for i in range(len(pts_search_query)):
    for j in pts_search_query[i]:
        pairs.append([j, i])


# convert pairs to sparse adjacency matrix
# non zero M_[ij] -> query i, pts j pair
data = np.ones(len(pairs))
ii = np.array([p[0] for p in pairs])
jj = np.array([p[1] for p in pairs])
nb_mat = sp.coo_matrix((data, (ii, jj)), shape=[
                       n_query, n_pts], dtype=int).tocsr()
sio.mmwrite("nb_mat.mtx", nb_mat)  # the saved file has base index=1
