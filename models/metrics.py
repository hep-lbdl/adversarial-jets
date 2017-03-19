import numpy as np
from scipy.spatial.distance import cdist as distance
from pyemd import emd
from scipy.linalg import toeplitz


class AnnoyingError(Exception):
    pass


def _calculate_emd_2D(D1, D2, bins=(40, 40)):
    """
    Args:
    -----
        D1, D2: two np arrays with potentially differing 
            numbers of rows, but two columns. The empirical 
            distributions you want a similarity over
        bins: number of bins in each dim
    """

    try:
        _, bx, by = np.histogram2d(*np.concatenate((D1, D2), axis=0).T, bins=bins)
    except ValueError, e:
        print '[ERROR] found here'

        raise AnnoyingError('Fuck this')

    H1, _, _ = np.histogram2d(*D1.T, bins=(bx, by))
    H2, _, _ = np.histogram2d(*D2.T, bins=(bx, by))

    H1 /= H1.sum()
    H2 /= H2.sum()

    _x, _y = np.indices(H1.shape)

    coords = np.array(zip(_x.ravel(), _y.ravel()))

    D = distance(coords, coords)

    return emd(H1.ravel(), H2.ravel(), D)


def _calculate_emd_1D(D1, D2, bins=40):
    """
    Args:
    -----
        D1, D2: two np arrays with potentially differing 
            numbers of rows, but two columns. The empirical 
            distributions you want a similarity over
        bins: number of bins in each dim
    """

    D1 = D1[np.isnan(D1).sum(axis=-1) < 1]
    D2 = D2[np.isnan(D2).sum(axis=-1) < 1]

    try:
        _, bx = np.histogram(np.concatenate((D1, D2), axis=0), bins=bins)
    except ValueError, e:
        print '[ERROR] found here'

        raise AnnoyingError('Fuck this')

    H1, _ = np.histogram(D1, bins=bx, normed=True)
    H2, _ = np.histogram(D2, bins=bx, normed=True)

    H1 /= H1.sum()
    H2 /= H2.sum()

    # _x, _y = np.indices(H1.shape)

    D = toeplitz(range(len(H1))).astype(float)

    # print coords

    # D = distance(coords, coords)

    return emd(H1, H2, D)


def calculate_metric(D1, signal1, D2, signal2, bins=(40, 40)):
    """
    Args:
    -----
        D1: (nb_rows, 2) array of observations from first distribution
        signal1: (nb_rows, ) array of 1 or 0, indicating class of 
            first distribution
        D2: (nb_rows, 2) array of observations from second distribution
        signal2: (nb_rows, ) array of 1 or 0, indicating class of 
            second distribution

        bins: number of bins in each dim
    """

    try:
        if len(D1.shape) == 2:
            sig_cond = _calculate_emd_2D(D1[signal1 == True], D2[
                                         signal2 == True], bins=bins)
            bkg_cond = _calculate_emd_2D(D1[signal1 == False], D2[
                                         signal2 == False], bins=bins)

        else:
            if not isinstance(bins, int):
                bins = bins[0]
            sig_cond = _calculate_emd_1D(D1[signal1 == True], D2[
                                         signal2 == True], bins=bins)
            bkg_cond = _calculate_emd_1D(D1[signal1 == False], D2[
                                         signal2 == False], bins=bins)

        return max(sig_cond, bkg_cond)
    except AnnoyingError, a:
        return 999
