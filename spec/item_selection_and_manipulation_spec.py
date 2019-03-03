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
