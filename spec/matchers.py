from expects.matchers import Matcher
import numpy as np

class equal_np_array(Matcher):
    def __init__(self, expected):
        self._expected = expected

    def _match(self, candidate):
        if np.array_equal(self._expected, candidate):
            return True, ['Arrays are equal']

        return False, ['Arrays are different.\n']

class equal_tuple_of_np_array(Matcher):
    def __init__(self, expected):
        self._expected = expected

    def _match(self, candidate):
        if len(self._expected) != len(candidate):
            return False, ['Tuples have different length.\n']

        for i, expected_item in enumerate(self._expected):
            candidate_item = candidate[i]

            if not np.array_equal(expected_item, candidate_item):
                return False, ['Arrays are different.\n']

        return True, ['Tuple of arrays are the same\n']
