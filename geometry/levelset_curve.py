import numpy as np
from netgen.occ import Pnt

def level_set_quadfoil(theta):
    """
    Compute a point on the zero level set of a specific quadfoil-like function in polar coordinates.

    Parameters
    ----------
    theta : float
        Polar angle (radians).
    
    Description
    -----------
    The level set function φ(r, θ) is implicitly defined as:
        φ(r, θ) = k^8 * r^6 - k^4 * a^2 * r^4 * (1 - cos(4θ))/8 - 0.01*k^2
    For a given θ, we solve φ(r, θ) = 0 for r > 0 and return the corresponding (x, y) point in Cartesian coordinates.
    """
    k = 1.5
    a = 3
    aa = k**8
    bb = -k**4*a**2*(1 - np.cos(4*theta))/8
    cc = -0.01*k**2

    # Solve cubic equation aa*r^6 + bb*r^4 + cc = 0 for r^2
    res = np.roots([aa, bb, 0, cc])
    tmp = [np.real(val) for val in res if np.isreal(val) and val > 0]
    r = tmp[0]**0.5  # r = sqrt(root)

    return Pnt(r*np.cos(theta), r*np.sin(theta), 0)


