from mamba import *
from expects import *
import numpy as np
from ..matchers import equal_np_array

with description('Item selection and manipulation'):
    with description('working with "argsort"'):
        with it('returns the indices that would sort the array'):
            x = np.array([3, 1, 2])

            indices = x.argsort()

            expect(indices).to(equal_np_array([1, 2, 0]))

        with it('assumes the sorting for the last axis by default'):
            x =  np.array([
                [40, 20],
                [10, 30]
            ])

            indices = x.argsort()

            expect(indices).to(equal_np_array([
                [1, 0],
                [0, 1]
            ]))

        with it('allows to specify the axis to sort by'):
            x =  np.array([
                [40, 20],
                [10, 30]
            ])

            indices = x.argsort(axis=0)

            expect(indices).to(equal_np_array([
                [1, 0],
                [0, 1]
            ]))

        with it('considers the flattern array when axis is nullified'):
            x =  np.array([
                [40, 20],
                [10, 30]
            ])

            indices = x.argsort(axis=None)

            expect(indices).to(equal_np_array([2, 1, 3, 0]))
