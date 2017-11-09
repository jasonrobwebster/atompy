"""Double Bar matrix elements."""

from . import Function, hbar, c, e0, pi, sqrt

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

    .. [1] Steck, D.A., 2007. Quantum and atom optics. p.331.
        http://atomoptics-nas.uoregon.edu/~dsteck/teaching/quantum-optics/quantum-optics-notes.pdf
    """
    nargs = 4

    @classmethod
    def eval(cls, jg, je, w0, gamma):
        if gamma.is_number:
            out = ((3 * pi * e0 * hbar * c**3) / (w0**3)) * ((2*je + 1) / (2*jg + 1)) * gamma
            out = sqrt(out)
            return out

