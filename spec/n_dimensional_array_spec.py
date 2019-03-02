from mamba import *
from expects import *
import numpy as np

with description('N-dimensional array (ndarray)') as self:
    with it('can be constructed with np.array'):
        no_elements = []
        expect(np.array(no_elements)).to(be_a(np.ndarray))

    with it('can be a 1-dimensional array'):
        elements = [1, 2, 3]

        one_dimensional = np.array(elements)

        expect(len(one_dimensional.shape)).to(equal(1))
        expect(one_dimensional.shape[0]).to(equal(len(elements)))

    with it('can access the elements by passing an index'):
        one_dimensional = np.array([1 ,2, 3])

        expect(one_dimensional[0]).to(equal(1))
        expect(one_dimensional[1]).to(equal(2))
        expect(one_dimensional[-1]).to(equal(3))

    with it('does not affect the 1-D array when modifying a selected value'):
        one_dimensional = np.array([1 ,2, 3])

        to_modify = one_dimensional[0]
        to_modify += 10

        expect(one_dimensional[0]).to(equal(1))
        expect(to_modify).to(equal(11))

    with it('can slice the array'):
        one_dimensional = np.array([1 ,2, 3])

        subset = one_dimensional[0:2]

        expect(subset).to(have_len(2))
        expect(subset).to(start_with(1))
        expect(subset).to(end_with(2))

    with it('affects to the 1-D array when modifying a sliced value'):
        one_dimensional = np.array([1 ,2, 3])
        subset = one_dimensional[0:2]

        subset[0] = 10

        expect(one_dimensional[0]).to(equal(10))
