from mamba import *
from expects import *
import numpy as np
from .matchers import equal_np_array

with description('Numerical ranges creation routines') as self:
    with description('working with "arange"'):
        with it('creates evenly spaced values within a half-open interval'):
            start = 2
            stop = 5

            a_range = np.arange(start, stop)

            expect(a_range).to(equal_np_array([2, 3, 4]))

        with it('by default it uses start as 0'):
            a_range = np.arange(3)

            expect(a_range).to(equal_np_array([0, 1, 2]))

        with it('allows to specify a step size'):
            start = 2
            stop = 10
            step = 2

            a_range = np.arange(start, stop, step)

            expect(a_range).to(equal_np_array([2, 4, 6, 8]))

        with it('allows to specify a negative step size'):
            start = 5
            stop = 2
            step = -1

            a_range = np.arange(start, stop, step)

            expect(a_range).to(equal_np_array([5, 4, 3]))

        with it('returns a ndarray rather than a range as the built-in range does'):
            expect(np.arange(3)).to(be_a(np.ndarray))
            expect(range(3)).to(be_a(range))

    with description('working with "linspace"'):
        with it('creates evenly spaced numbers over an interval'):
            start = 0
            stop = 5
            number = 5

            values = np.linspace(start, stop, number)

            expect(values).to(equal_np_array([0, 1.25, 2.5, 3.75, 5]))

        with it('uses by default 50 samples'):
            start = 0
            stop = 5

            values = np.linspace(start, stop)

            expect(values).to(have_len(50))

        with it('does not include the stop point if specified'):
            start = 0
            stop = 5
            number = 5

            values = np.linspace(start, stop, number, endpoint=False)

            expect(values).to(equal_np_array([0, 1, 2, 3, 4]))

        with it('can return the used step'):
            start = 0
            stop = 5
            number = 5

            values, step = np.linspace(start, stop, number, retstep=True)

            expect(step).to(equal(1.25))
