"""Pure permutation operations in the preregistered original-token coordinate system."""
import hashlib

import numpy as np


def permutation_arrays(row):
    p=np.asarray(row['permutation'],dtype=np.int64)
    inverse=np.asarray(row['inverse_permutation'],dtype=np.int64)
    assert p.shape==inverse.shape==(256,)
    assert np.array_equal(np.sort(p),np.arange(256))
    assert np.array_equal(p[inverse],np.arange(256)) and np.array_equal(inverse[p],np.arange(256))
    generated=np.random.Generator(np.random.PCG64(row['seed'])).permutation(256)
    assert np.array_equal(p,generated)
    assert hashlib.sha256(p.astype('<i8').tobytes()).hexdigest()==row['int64_little_endian_sha256']
    return p,inverse


def restore_reception(permuted_received,oracle_packets,crc_packets,inverse):
    rx=np.asarray(permuted_received).reshape(256)
    oracle=np.asarray(oracle_packets,dtype=bool).reshape(64)
    crc=np.asarray(crc_packets,dtype=bool).reshape(64)
    inverse=np.asarray(inverse,dtype=np.int64)
    assert np.array_equal(np.sort(inverse),np.arange(256))
    return rx[inverse],np.tile(oracle,4)[inverse],np.tile(crc,4)[inverse]
