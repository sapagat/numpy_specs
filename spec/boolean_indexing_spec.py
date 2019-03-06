from mamba import *
from expects import *
import numpy as np
from .matchers import equal_np_array

with description('Boolean array indexing'):
    with it('allows to select elements with booleans'):
        x = np.array([10, 20, 30, 40])
        boolean_indices = [True, False, False, True]

        result = x[boolean_indices]

        expect(result).to(equal_np_array([10, 40]))

    with it('does not modify the original array when changing the copy'):
        x = np.array([10, 20, 30, 40])
        boolean_indices = [True, False, False, True]

        result = x[boolean_indices]
        result[1] = 2

        expect(x).to(equal_np_array([10, 20, 30, 40]))

    with it('is usefull for selecting with a condition'):
        x = np.array([10, 20, 30, 40])
        multiples_of_20 = x % 20 == 0

        result = x[multiples_of_20]

        expect(result).to(equal_np_array([20, 40]))

    with it('can update the value when evaluating a condition'):
        x = np.array([10, 20, 30, 40])
        multiples_of_20 = x % 20 == 0

        x[multiples_of_20] += 5

        expect(x).to(equal_np_array([10, 25, 30, 45]))

    with it('fails when trying to index out-of-bounds'):
        x = np.array([10, 20, 30, 40])
        out_of_bounds_indices = [True, False, False, True, False]

        expect(lambda :
            x[out_of_bounds_indices]
        ).to(raise_error(IndexError))

    with it('allows to use "ix_" function to combine it with integer indexing'):
        x = np.array([
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
            [9, 10, 11]
        ])
        rows = [False, True, False, True]
        columns = [0, 2]

        result = x[np.ix_(rows, columns)]

        expect(result).to(equal_np_array([
            [3, 5],
            [9, 11]
        ]))
