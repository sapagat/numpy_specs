from mamba import *
from expects import *
import numpy as np
from ..matchers import equal_np_array

with description('Item selection and manipulation'):
    with description('working with "put"'):
        with it('replaces the elements of an array with the given values'):
            x = np.array([10, 20, 30, 40])
            indices = [2, 3]
            values = [50, 60]

            x.put(indices, values)

            expect(x).to(equal_np_array([10, 20, 50, 60]))

        with it('is equivalent to use "numpy.put"'):
            x = np.array([10, 20, 30, 40])
            indices = [2, 3]
            values = [50, 60]

            np.put(x, indices, values)

            expect(x).to(equal_np_array([10, 20, 50, 60]))

        with it('repeats the values if shorter than the indices'):
            x = np.array([10, 20, 30, 40])
            indices = [2, 3]
            values = [50]

            x.put(indices, values)

            expect(x).to(equal_np_array([10, 20, 50, 50]))

        with it('does not affect to provide more values than necessary'):
            x = np.array([10, 20, 30, 40])
            indices = [2]
            values = [50, 60]

            x.put(indices, values)

            expect(x).to(equal_np_array([10, 20, 50, 40]))

        with it('allows to pass parameters as sclalars'):
            x = np.array([10, 20, 30, 40])
            indices = 2
            values = 50

            x.put(indices, values)

            expect(x).to(equal_np_array([10, 20, 50, 40]))

        with it('raises an error by default when the indices are out-of-bounds'):
            x = np.array([10, 20, 30, 40])
            out_of_bounds_indices = [4]
            any_values = [33]

            expect(lambda :
                x.put(out_of_bounds_indices, any_values)
            ).to(raise_error(IndexError))

        with it('allows to wrap around when indices are out-of-bounds'):
            x = np.array([10, 20, 30, 40])
            indices = [3, 4, 5, 6]
            values = [3, 4, 5, 6]

            x.put(indices, values, mode='wrap')

            expect(x).to(equal_np_array([4, 5, 6, 3]))

        with it('allows to clip to the last element when indices are out-of-bounds'):
            x = np.array([10, 20, 30, 40])
            indices = [3, 4, 5, 6]
            values = [3, 4, 5, 6]

            x.put(indices, values, mode='clip')

            expect(x).to(equal_np_array([10, 20, 30, 6]))

        with it('allows negative indexing'):
            x = np.array([10, 20, 30, 40])
            indices = [-1, -2]
            values= [50, 60]

            x.put(indices, values)

            expect(x).to(equal_np_array([10, 20, 60, 50]))

        with it('is not reliable with multi-dimensional indexing'):
            x = np.array([
                [10, 20],
                [30, 40]
            ])
            indices = [[0,0], [1, 1]]
            values= [50, 60]

            x.put(indices, values)

            expect(x).to(equal_np_array([
                [60, 60],
                [30, 40]
            ]))
