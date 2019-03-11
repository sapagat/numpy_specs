from mamba import *
from expects import *
import numpy as np
from ..matchers import *

with description('Item selection and manipulation'):
    with description('working with "non-zero"'):
        with it('provides the indices of the elements that are non-zero'):
            x = np.array([10, 0, 0, -10])

            result = x.nonzero()

            expect(result).to(equal_tuple_of_np_array(([0, 3], )))

        with it('can handle more than one dimension'):
            x = np.array([
                [10, 0, 0, -10],
                [0, 10, 10, 0],
                [10, 10, 0, 0]
            ])

            result = x.nonzero()

            expect(result).to(equal_tuple_of_np_array((
                [0, 0, 1, 1, 2, 2],
                [0, 3, 1, 2, 0, 1]
            )))

        with it('is usefull in order to perform selections'):
            x = np.array([
                [10, 0, 0, -10],
                [0, 10, 10, 0],
                [10, 10, 0, 0]
            ])

            x[x.nonzero()] += 2

            expect(x).to(equal_np_array([
                [12, 0, 0, -8],
                [0, 12, 12, 0],
                [12, 12, 0, 0]
            ]))
