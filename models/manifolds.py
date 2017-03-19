from joblib import Parallel, delayed
import numpy as np
import math


# form the grids you need to represent the eta, phi coordinates
grid = 0.5 * (np.linspace(-1.25, 1.25, 26)[:-1] + np.linspace(-1.25, 1.25, 26)[1:])
eta = np.tile(grid, (25, 1))
phi = np.tile(grid[::-1].reshape(-1, 1), (1, 25))


def discrete_mass(jet_image):
    '''
    Calculates the jet mass from a pixelated jet image
    Args:
    -----
        jet_image: numpy ndarray of dim (1, 25, 25)
    Returns:
    --------
        M: float, jet mass
    '''
    Px = np.sum(jet_image * np.cos(phi), axis=(1, 2))
    Py = np.sum(jet_image * np.sin(phi), axis=(1, 2))
    Pz = np.sum(jet_image * np.sinh(eta), axis=(1, 2))
    E = np.sum(jet_image * np.cosh(eta), axis=(1, 2))
    PT2 = np.square(Px) + np.square(Py)
    M2 = np.square(E) - (PT2 + np.square(Pz))
    M = np.sqrt(M2)
    return M


def discrete_pt(jet_image):
    '''
    Calculates the jet transverse momentum from a pixelated jet image
    Args:
    -----
        jet_image: numpy ndarray of dim (1, 25, 25)
    Returns:
    --------
        float, jet transverse momentum
    '''
    Px = np.sum(jet_image * np.cos(phi), axis=(1, 2))
    Py = np.sum(jet_image * np.sin(phi), axis=(1, 2))
    return np.sqrt(np.square(Px) + np.square(Py))


def dphi(phi1, phi2):
    '''
    Calculates the difference between two angles avoiding |phi1 - phi2| > 180 degrees
    '''
    import math
    return math.acos(math.cos(abs(phi1 - phi2)))


def _tau1(jet_image):
    '''
    Calculates the normalized tau1 from a pixelated jet image
    Args:
    -----
        jet_image: numpy ndarray of dim (1, 25, 25)
    Returns:
    --------
        float, normalized jet tau1
    '''
    # find coordinate of most energetic pixel, then use formula to compute tau1
    tau1_axis_eta = eta.ravel()[np.argmax(jet_image)]
    tau1_axis_phi = phi.ravel()[np.argmax(jet_image)]
    tau1 = np.sum(jet_image *
                  np.sqrt(np.square(tau1_axis_eta - eta) +
                          np.square([dphi(tau1_axis_phi, p) for p in phi.ravel()]).reshape(25, 25))
                  )
    return tau1 / np.sum(jet_image)  # normalize by the total intensity


def _tau2(jet_image):
    '''
    Calculates the normalized tau2 from a pixelated jet image
    Args:
    -----
        jet_image: numpy ndarray of dim (1, 25, 25)
    Returns:
    --------
        float, normalized jet tau2
    Notes:
    ------
        slow implementation
    '''
    proto = np.array(zip(jet_image[jet_image != 0],
                         eta[jet_image != 0],
                         phi[jet_image != 0]))
    while len(proto) > 2:
        candidates = [
            (
                (i, j),
                (min(pt1, pt2) ** 2) * ((eta1 - eta2) ** 2 + (phi1 - phi2) ** 2)
            )
            for i, (pt1, eta1, phi1) in enumerate(proto)
            for j, (pt2, eta2, phi2) in enumerate(proto)
            if j > i
        ]
        index, value = zip(*candidates)
        pix1, pix2 = index[np.argmin(value)]
        if pix1 > pix2:
            # swap
            pix1, pix2 = pix2, pix1
        (pt1, eta1, phi1) = proto[pix1]
        (pt2, eta2, phi2) = proto[pix2]
        e1 = pt1 / np.cosh(eta1)
        e2 = pt2 / np.cosh(eta2)
        choice = e1 > e2
        eta_add = (eta1 if choice else eta2)
        phi_add = (phi1 if choice else phi2)
        pt_add = (e1 + e2) * np.cosh(eta_add)
        proto[pix1] = (pt_add, eta_add, phi_add)
        proto = np.delete(proto, pix2, axis=0).tolist()

    (_, eta1, phi1), (_, eta2, phi2) = proto
    np.sqrt(np.square(eta - eta1) + np.square(phi - phi1))
    grid = np.array([
        np.sqrt(np.square(eta - eta1) + np.square(phi - phi1)),
        np.sqrt(np.square(eta - eta2) + np.square(phi - phi2))
    ]).min(axis=0)
    # normalize by the total intensity
    return np.sum(jet_image * grid) / np.sum(jet_image)


def tau21(jet_image, nb_jobs=1, verbose=False):
    '''
    Calculates the tau21 from a pixelated jet image using the functions above
    Args:
    -----
        jet_image: numpy ndarray of dim (1, 25, 25)
    Returns:
    --------
        float, jet tau21
    Notes:
    ------
        slow implementation
    '''
    if len(jet_image.shape) == 2:
        tau1 = _tau1(jet_image)
        if tau1 <= 0:
            return 0
        else:
            tau2 = _tau2(jet_image)
            return tau2 / tau1

    return np.array(Parallel(n_jobs=nb_jobs, verbose=verbose)(
        delayed(tau21)(im) for im in jet_image
    ))
