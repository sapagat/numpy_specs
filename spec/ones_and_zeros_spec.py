from mamba import *
from expects import *
import numpy as np
from .matchers import equal_np_array

with description('Ones & Zeros array creation routines') as self:
    with it('creates an array of ones (floats)'):
        shape = (2, 1)

        ones = np.ones(shape)

        expect(ones).to(equal_np_array([
            [1.],
            [1.]
        ]))
        expect(ones[0,0]).to(be_a(float))

    with it('can create an array of ones as integers'):
        shape = (2, 1)

        ones = np.ones(shape, dtype=np.int8)

        expect(ones).to(equal_np_array([
            [1],
            [1]
        ]))
        expect(ones[0,0]).to(be_a(np.int8))

    with it('can create an array of ones copying the shape of another array'):
        x = np.array([
            [1, 2],
            [3, 4]
        ])

        ones = np.ones_like(x)

        expect(ones.shape).to(equal(x.shape))

    with it('can create an array of zeros (floats)'):
        shape = (2, 1)

        zeros = np.zeros(shape)

        expect(zeros).to(equal_np_array([
            [0.],
            [0.]
        ]))
        expect(zeros[0,0]).to(be_a(float))

    with it('can create an array of zeros as integers'):
        shape = (2, 1)

        zeros = np.zeros(shape, dtype=np.int8)

        expect(zeros).to(equal_np_array([
            [0],
            [0]
        ]))
        expect(zeros[0,0]).to(be_a(np.int8))

    with it('can create an array of zeros copying the shape of another array'):
        x = np.array([
            [1, 2],
            [3, 4]
        ])

        zeros = np.zeros_like(x)

        expect(zeros.shape).to(equal(x.shape))

    with it('can create the Identity array of a given size'):
        size = 2

        identity = np.identity(size, dtype=np.int8)

        expect(identity).to(equal_np_array([
            [1, 0],
            [0, 1]
        ]))
