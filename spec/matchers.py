from expects.matchers import Matcher
import numpy as np

class equal_np_array(Matcher):
    def __init__(self, expected):
        self._expected = expected

    def _match(self, candidate):
        if np.array_equal(self._expected, candidate):
            return True, ['Arrays are equal']

        return False, ['Arrays are different.\n']
