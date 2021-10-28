import numpy as np

# ref_data = np.loadtxt('bnb_mat.txt', usecols=(0, 1), dtype=int)
ref_data = np.loadtxt('nb_mat.mtx', usecols=(0, 1), dtype=int, skiprows=3)
mix_data = np.loadtxt('test.txt', usecols=(0, 1), dtype=int)


print(ref_data[:10])
print(mix_data[:10])


assert np.array_equal(ref_data, mix_data)
