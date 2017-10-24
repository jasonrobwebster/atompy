"""Double Bar matrix elements."""

from . import Function, hbar, speed_of_light, e0, pi, sqrt

class DoubleBar(Function):
    """DoubleBar matrix element.

    Returns a double barred matrix element [1] that computes the relative
    transisiton strengths between two atomic states.

    Parameters
    ==========

    Jg : Number, Symbol
        The ground atomic level OAM.

    Je : Number, Symbol
        The excited atomic level OAM.

    E_diff : Number, Symbol
        The energy difference between Jg and Je.

    Gamma : Number, Symbol
        The spontaneous decay rate from Je to Jg.

    Examples
    ========


    
    References
    ==========

    .. [1] Steck, D.A., 2007.
        Quantum and atom optics.
        Oregon Center for Optics and Department of Physics, University of Oregon, p.331.
        http://atomoptics-nas.uoregon.edu/~dsteck/teaching/quantum-optics/quantum-optics-notes.pdf
    """
    nargs = 4

    @classmethod
    def eval(cls, Jg, Je, E, Gamma):
        w0 = E/hbar
        c = speed_of_light
        out = ((3 * pi * e0 * hbar * c**3) / (w0**3)) * ((2*Je + 1) / (2*Jg + 1)) * Gamma
        out = sqrt(out)
        return out

