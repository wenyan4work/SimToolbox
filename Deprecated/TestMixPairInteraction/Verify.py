import numpy as np
import scipy as sp

ref_data = np.loadtxt('nb_mat.mtx', skiprows=3, usecols=(0, 1), dtype=int)
mix_data = np.loadtxt('Test.log', usecols=(0, 1), dtype=int)

# ref_data has base index=1
# mix_data has based index=0
ref_data = ref_data - 1

print(ref_data[:10])
print(mix_data[:10])


assert np.array_equal(ref_data, mix_data)
