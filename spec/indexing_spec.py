from mamba import *
from expects import *
import numpy as np
from expects.matchers import Matcher

with description('Indexing') as self:
    with description('working with a 1-dimensional array, x') as self:
        with before.each:
            self.x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

        with it('slices with a start:stop:step index'):
            start = 1
            stop = 7
            step = 2

            slice = self.x[start:stop:step]

            expect(slice).to(equal_np_array([1, 3, 5]))

        with it('uses by default a step equal to 1'):
            start = 2
            stop = 6

            slice = self.x[start:stop]

            expect(slice).to(equal_np_array([2, 3, 4, 5]))

        with it('understands a negative start as "len(x) + start"'):
            start = -2
            stop = 10

            slice = self.x[start:stop]

            expect(slice).to(equal_np_array([8, 9]))

        with it('understands a negative stop as "len(x) + stop"'):
            start = 2
            stop = -3

            slice = self.x[start:stop]

            expect(slice).to(equal_np_array([2, 3, 4, 5, 6]))

        with it('understands a negative step as going backwards'):
            start = 9
            stop = 1
            step = -2

            slice = self.x[start:stop:step]

            expect(slice).to(equal_np_array([9, 7, 5, 3]))

        with it('allows to select all indices from start'):
            start = 7

            slice = self.x[start:]

            expect(slice).to(equal_np_array([7, 8, 9]))

        with it('allows to select all indices until a stop'):
            stop = 7

            slice = self.x[:stop]

            expect(slice).to(equal_np_array([0, 1, 2, 3, 4, 5, 6]))


class equal_np_array(Matcher):
    def __init__(self, expected):
        self._expected = expected

    def _match(self, candidate):
        if np.array_equal(self._expected, candidate):
            return True, ['Arrays are equal']

        return False, ['Arrays are different.\n']
