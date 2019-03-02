from mamba import *
from expects import *
import numpy as np

with description('N-dimensional array (ndarray)') as self:
    with it('is a numpy.ndarray'):
        expect(np.ndarray([])).to(be_a(np.ndarray))
