from mamba import *
from expects import *

with description('Set up') as self:
    with it('works'):
        expect(True).to(equal(True))
