from mamba import *
from expects import *
import numpy as np
from ..matchers import equal_np_array

with description('Item selection and manipulation'):
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
