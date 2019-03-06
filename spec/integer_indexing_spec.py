from mamba import *
from expects import *
import numpy as np
from .matchers import equal_np_array

with description('Integer array indexing'):
    with it('uses a row index and a column index'):
        x = np.array([
            [10, 20],
            [30, 40],
            [50, 60]
        ])
        rows = [0, 1, 2]
        columns = [0, 1, 0]

        result = x[rows, columns]

        expect(result).to(equal_np_array([10, 40, 50]))

    with it('does not modify the original array when changing the copy'):
        x = np.array([
            [10, 20],
            [30, 40],
            [50, 60]
        ])
        rows = [0, 1, 2]
        columns = [0, 1, 0]

        result = x[rows, columns]
        result[1] = 2

        expect(x).to(equal_np_array([
            [10, 20],
            [30, 40],
            [50, 60]
        ]))

    with it('builds a different output depending on the indices shape'):
        x = np.array([
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
            [9, 10, 11]
        ])

        expect(x[[[0, 0], [3, 3]], [[0, 2], [0, 2]]]).to(equal_np_array([
            [0, 2],
            [9, 11]
        ]))
        expect(x[[0, 3], [0, 2]]).to(equal_np_array([0, 11]))
        expect(x[[0, 0, 3, 3], [0, 2, 0, 2]]).to(equal_np_array([0, 2, 9, 11]))
        expect(x[[[0], [0], [3],[3]], [[0], [2], [0], [2]]]).to(equal_np_array([
            [0],
            [2],
            [9],
            [11]
        ]))

    with it('allows to combine itself with broadcasting'):
        x = np.array([
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
            [9, 10, 11]
        ])
        rows = np.array([0, 3], dtype=np.intp)
        columns = np.array([0, 2], dtype=np.intp)
        result = x[rows[:, np.newaxis], columns]

        expect(result).to(equal_np_array([
            [0, 2],
            [9, 11]
        ]))

    with it('allows to combine itself with "ix_" function'):
        x = np.array([
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
            [9, 10, 11]
        ])
        rows = [0, 3]
        columns = [0, 2]

        result = x[np.ix_(rows, columns)]

        expect(result).to(equal_np_array([
            [0, 2],
            [9, 11]
        ]))
