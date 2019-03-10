from mamba import *
from expects import *
import numpy as np
from .matchers import equal_np_array

with description('Item selection and manipulation'):
    with description('working with "take" with an array, x'):
        with it('returns the items given some indices'):
            x = np.array([10, 20, 30, 40])
            indices = [1, 2]

            result = x.take(indices)

            expect(result).to(equal_np_array([20, 30]))

        with it('is equivalent to calling "np.take"'):
            x = np.array([10, 20, 30, 40])
            indices = [1, 2]

            expect(x.take(indices)).to(equal_np_array(np.take(x, indices)))

        with it('does not modify the original array'):
            x = np.array([10, 20, 30, 40])
            indices = [1, 2]

            x.take(indices)

            expect(x).to(equal_np_array([10, 20, 30, 40]))

        with it('provides a copy, so changes do not mutate original array'):
            x = np.array([10, 20, 30, 40])
            indices = [1, 2]
            result = x.take(indices)

            result[0] = 100

            expect(x).to(equal_np_array([10, 20, 30, 40]))

        with it('allows to specify the desired shape with the indices'):
            x = np.array([10, 20, 30, 40])
            indices = [
                [0, 3],
                [1, 2]
            ]

            result = x.take(indices)

            expect(result).to(equal_np_array([
                [10, 40],
                [20, 30]
            ]))

        with it('raises an error by default when the indices are out-of-bounds'):
            x = np.array([10, 20, 30, 40])

            expect(lambda :
                x.take([100])
            ).to(raise_error(IndexError))

        with it('allows to wrap around when indices are out-of-bounds'):
            x = np.array([10, 20, 30, 40])
            indices = [3, 4, 5, 6]

            result = x.take(indices, mode='wrap')

            expect(result).to(equal_np_array([40, 10, 20, 30]))

        with it('allows to clip to the last element when indices are out-of-bounds'):
            x = np.array([10, 20, 30, 40])
            indices = [3, 4, 5, 6]

            result = x.take(indices, mode='clip')

            expect(result).to(equal_np_array([40, 40, 40, 40]))

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

    with description('working with "choose"'):
        with it('selects from different choices'):
            selection = [1, 0]
            choices = [[10, 20], [30, 40]]

            result = np.choose(selection, choices)

            expect(result).to(equal_np_array([30, 20]))

        with it('fails if the selection is bigger than the number of choices'):
            selection = [1, 0, 2]
            choices = [[10, 20], [30, 40]]

            expect(lambda :
                np.choose(selection, choices)
            ).to(raise_error(ValueError))

        with it('fails if the selection index is out-of-bounds'):
            selection = [1, 2]
            choices = [[10, 20], [30, 40]]

            expect(lambda :
                np.choose(selection, choices)
            ).to(raise_error(ValueError))

        with it('allows to wrap around when selection indices are out-of-bounds'):
            selection = [1, 2]
            choices = [[10, 20], [30, 40]]

            result = np.choose(selection, choices, mode='wrap')

            expect(result).to(equal_np_array([30, 20]))

        with it('allows to clip to the last index when selection indices are out-of-bounds'):
            selection = [1, 2]
            choices = [[10, 20], [30, 40]]

            result = np.choose(selection, choices, mode='clip')

            expect(result).to(equal_np_array([30, 40]))

        with it('allows to shape the output as desired'):
            selection = [[1, 0, 1], [0, 1, 0], [1, 0, 1]]
            choices = [10, -10]

            result = np.choose(selection, choices)

            expect(result).to(equal_np_array([
                [-10, 10, -10],
                [10, -10, 10],
                [-10, 10, -10]
            ]))

    with description('working with "sort"'):
        with it('sorts an array in-place'):
            x = np.array([40, 20, 10, 30])

            x.sort()

            expect(x).to(equal_np_array([10, 20, 30, 40]))

        with it('sorts by the last axis by default'):
            x =  np.array([
                [40, 20],
                [10, 30]
            ])

            x.sort()

            expect(x).to(equal_np_array([
                [20, 40],
                [10, 30]
            ]))

        with it('allows to specify the axis to sort by'):
            x = np.array([
                [40, 20],
                [10, 30]
            ])

            x.sort(axis=0)

            expect(x).to(equal_np_array([
                [10, 20],
                [40, 30]
            ]))
