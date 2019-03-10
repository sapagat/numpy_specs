from mamba import *
from expects import *
import numpy as np
from ..matchers import equal_np_array

with description('Item selection and manipulation'):
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
