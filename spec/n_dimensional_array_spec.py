from mamba import *
from expects import *
import numpy as np
from .matchers import equal_np_array

with description('N-dimensional array (ndarray)') as self:
    with it('can be constructed with np.array'):
        no_elements = []
        expect(np.array(no_elements)).to(be_a(np.ndarray))

    with description('working with a 1-dimensional array') as self:
        with it('has one shape dimension equal to the lenght of array elements'):
            elements = [1, 2, 3]

            one_dimensional = np.array(elements)

            expect(len(one_dimensional.shape)).to(equal(1))
            expect(one_dimensional.shape[0]).to(equal(len(elements)))

        with it('can access the elements by passing an index'):
            one_dimensional = np.array([1 ,2, 3])

            expect(one_dimensional[0]).to(equal(1))
            expect(one_dimensional[1]).to(equal(2))
            expect(one_dimensional[-1]).to(equal(3))

        with it('does not affect to the array when modifying a selected value'):
            one_dimensional = np.array([1 ,2, 3])

            to_modify = one_dimensional[0]
            to_modify += 10

            expect(one_dimensional[0]).to(equal(1))
            expect(to_modify).to(equal(11))

        with it('can slice the array into a view'):
            one_dimensional = np.array([1 ,2, 3])

            view = one_dimensional[0:2]

            expect(view).to(equal_np_array([1, 2]))

        with it('mutates the array when modifying the view'):
            one_dimensional = np.array([1 ,2, 3])
            view = one_dimensional[0:2]

            view[0] = 10

            expect(one_dimensional[0]).to(equal(10))

    with description('working with a 2-dimensional array') as self:
        with it('has shape that represents rows X columns'):
            two_dimensional = np.array([
                [1, 2, 3],
                [4, 5, 6]
            ])

            expect(two_dimensional.shape).to(have_len(2))
            expect(two_dimensional.shape[0]).to(equal(2))
            expect(two_dimensional.shape[1]).to(equal(3))

        with it('can select a single element with an index (row, column)'):
            two_dimensional = np.array([
                [1, 2, 3],
                [4, 5, 6]
            ])

            two_dimensional[1, 1]

            expect(two_dimensional[1, 1]).to(equal(5))
            expect(two_dimensional[0, 2]).to(equal(3))

        with it('can slice the array into a view'):
            two_dimensional = np.array([
                [1, 2, 3],
                [4, 5, 6]
            ])

            view = two_dimensional[:, 1]

            expect(view.shape).to(equal((2,)))
            expect(view).to(equal_np_array([2, 5]))

        with it('mutates the array when modifying the view'):
            two_dimensional = np.array([
                [1, 2, 3],
                [4, 5, 6]
            ])

            view = two_dimensional[:, 1]
            view[1] = 10

            expect(two_dimensional[1,1]).to(equal(10))
