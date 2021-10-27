import numpy as np
import scipy as sp
import scipy.io as sio
import msgpack as mp
from glob import glob
import os


def load_data():
    files = glob('conBlockPool_r*.msgpack')
    files.sort()
    data = dict()
    for f in files:
        fh = open(f, 'rb')
        unp = mp.Unpacker(fh, raw=False)
        for unpacker in unp:
            for k, v in unpacker.items():
                if k in data.keys():
                    data[k].extend(v)
                else:
                    data[k] = v
        fh.close()

    return data


def is_close(a, b, eps=1e-7):
    return True if np.abs(a-b) < 1e-7 else False


def test(nprocs, nthreads):
    os.system('rm ./*.msgpack')
    os.system(
        'OMP_NUM_THREADS={} && mpirun -n {} ConstraintCollector_test'.format(nthreads, nprocs))

    # verify vector generation

    data = load_data()
    delta0 = sio.mmread('delta0_TV.mtx').flatten()
    invKappa = sio.mmread('invKappa_TV.mtx').flatten()
    gammaGuess = sio.mmread('gammaGuess_TV.mtx').flatten()

    assert(np.linalg.norm(delta0-np.array(data['delta0'])) < 1e-7)
    assert(np.linalg.norm(gammaGuess-np.array(data['gamma'])) < 1e-7)

    data_invKappa = [1/kappa if bilat else 0 for kappa,
                     bilat in zip(data['kappa'], data['bilateral'])]
    assert(np.linalg.norm(invKappa-np.array(data_invKappa)) < 1e-7)

    biFlag = sio.mmread('biFlag_TV.mtx').flatten()
    biBool = [True if biFlag[i] >
              0 else False for i in range(biFlag.shape[0])]
    assert(np.array_equal(np.array(biBool), np.array(data['bilateral'])))

    # verify matrix generation
    DMatTrans = sio.mmread('DMatTrans_TCMAT.mtx')
    nBlk = len(data['gamma'])
    assert(nBlk == DMatTrans.shape[0])

    for i in range(nBlk):
        row = DMatTrans.getrow(i).todense()
        print(i)
        # check entry locations at [6*I,6*I+5)
        globalIndexI = data['globalIndexI'][i]
        norm = data['normI'][i]
        pos = data['posI'][i]
        torque = np.cross(pos, norm)
        assert is_close(row[0, 6*globalIndexI], norm[0])
        assert is_close(row[0, 6*globalIndexI+1], norm[1])
        assert is_close(row[0, 6*globalIndexI+2], norm[2])
        assert is_close(row[0, 6*globalIndexI+3], torque[0])
        assert is_close(row[0, 6*globalIndexI+4], torque[1])
        assert is_close(row[0, 6*globalIndexI+5], torque[2])
        if data['oneSide'][i]:
            assert DMatTrans.getrow(i).getnnz() == 6
        else:
            assert DMatTrans.getrow(i).getnnz() == 12
            globalIndexJ = data['globalIndexJ'][i]
            norm = data['normJ'][i]
            pos = data['posJ'][i]
            torque = np.cross(pos, norm)
            assert is_close(row[0, 6*globalIndexJ], norm[0])
            assert is_close(row[0, 6*globalIndexJ+1], norm[1])
            assert is_close(row[0, 6*globalIndexJ+2], norm[2])
            assert is_close(row[0, 6*globalIndexJ+3], torque[0])
            assert is_close(row[0, 6*globalIndexJ+4], torque[1])
            assert is_close(row[0, 6*globalIndexJ+5], torque[2])


if __name__ == "__main__":
    test(nprocs=1, nthreads=1)
    test(nprocs=1, nthreads=4)
    test(nprocs=4, nthreads=1)
    test(nprocs=4, nthreads=3)
