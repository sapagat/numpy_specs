from mamba import *
from expects import *
import numpy as np
from .matchers import equal_np_array

with description('Indexing') as self:
    with description('working with a 1-dimensional array, x') as self:
        with before.each:
            self.x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

        with it('slices with a start:stop:step index'):
            start = 1
            stop = 7
            step = 2

            view = self.x[start:stop:step]

            expect(view).to(equal_np_array([1, 3, 5]))

        with it('uses by default a step equal to 1'):
            start = 2
            stop = 6

            view = self.x[start:stop]

            expect(view).to(equal_np_array([2, 3, 4, 5]))

        with it('understands a negative start as "len(x) + start"'):
            start = -2
            stop = 10

            view = self.x[start:stop]

            expect(view).to(equal_np_array([8, 9]))

        with it('understands a negative stop as "len(x) + stop"'):
            start = 2
            stop = -3

            view = self.x[start:stop]

            expect(view).to(equal_np_array([2, 3, 4, 5, 6]))

        with it('understands a negative step as going backwards'):
            start = 9
            stop = 1
            step = -2

            view = self.x[start:stop:step]

            expect(view).to(equal_np_array([9, 7, 5, 3]))

        with it('allows to select all indices from start'):
            start = 7

            view = self.x[start:]

            expect(view).to(equal_np_array([7, 8, 9]))

        with it('allows to select all indices until a stop'):
            stop = 7

            view = self.x[:stop]

            expect(view).to(equal_np_array([0, 1, 2, 3, 4, 5, 6]))

        with it('changes x if the view is changed'):
            view = self.x[0:2]

            view *= 10

            expect(self.x).to(equal_np_array([0, 10, 2, 3, 4, 5, 6, 7, 8, 9]))

    with description('working with a 2-dimensional matrix'):
        with before.each:
            self.matrix = np.array([
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]
            ])

        with it('allows to slice rows'):
            view = self.matrix[0:2]

            expect(view).to(equal_np_array([
                [1, 2, 3],
                [4, 5, 6]
            ]))

        with it('allows to slice columns'):
            view = self.matrix[:, 1:]

            expect(view).to(equal_np_array([
                [2, 3],
                [5, 6],
                [8, 9]
            ]))

        with it('allows to slice both rows and columns'):
            view = self.matrix[0:2, 1:]

            expect(view).to(equal_np_array([
                [2, 3],
                [5, 6]
            ]))

        with it('has some suprises when chaining slices'):
            view = self.matrix[0:2, 1:]
            alternative = self.matrix[0:2][1:]

            expect(view).not_to(equal_np_array(alternative))
            expect(alternative).to(equal_np_array([
                [4, 5, 6]
            ]))

        with it('can add a new dimension'):
            view = self.matrix[:, np.newaxis, :]

            expect(view.shape).to(equal((3, 1, 3)))
            expect(view).to(equal_np_array([
                [[1, 2, 3]],
                [[4, 5 ,6]],
                [[7, 8, 9]]
            ]))

        with it('changes the matrix if the view is changed'):
            view = self.matrix[0:2, 1:]

            view *= 10

            expect(self.matrix).to(equal_np_array([
                [1, 20, 30],
                [4, 50, 60],
                [7, 8, 9]
            ]))
