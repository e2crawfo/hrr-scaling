
from shovel import task
from mytools import nf, hrr
import numpy as np

@task
def hrrnoise(D, expression, normalize=False):
    normalize = bool(normalize)
    D = int(D)

    f = nf.make_hrr_noise_from_string(D, expression, normalize=normalize, verbose=True)
    i = hrr.HRR(D)

    x = f(i)

    print 'Compare: ', i.compare(x)
    print 'Dot: ', np.dot(i.v, x.v)


