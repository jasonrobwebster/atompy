"""The functions of AtomPy."""

from __future__ import print_function

from atompy.core import AtomicState, SphericalTensor, DoubleBar, clebsch_gordan, sqrt, wigner_6j, sympify

__all__ = [
    'transition_strength'
]

def transition_strength(g_state, e_state, tensor, spin=None, decouple=True):
    """Calculate the transition strengths between two atomic states [1].

    Returns the relative transition strength from the ground state g
    to the excited state e. 

    Parameters
    ==========

    g_state : AtomicLevel
        The ground atomic state.

    e_state : AtomicLevel
        The excited atomic state.

    tensor : SphericalTensor
        An instance of the SphericalTensor class that defines the rank
        and polarization of light.

    spin : Number
        The internal nuclear spin number I. If given, the output returns
        a table of transition strengths from F=|J-I| to F=J+I.

    decouple : Boolean, Optional
        Whether to decouple the resulting <J||..||J'> to <L||..||L'>. 
        Defaults to True.

    Examples
    ========



    References
    ==========

    .. [1] Steck, D.A., 2007. Quantum and atom optics. p. 336
        http://atomoptics-nas.uoregon.edu/~dsteck/teaching/quantum-optics/quantum-optics-notes.pdf
    """

    # assume our states are atomic states
    assert isinstance(g_state, AtomicState)
    assert isinstance(e_state, AtomicState)
    assert isinstance(tensor, SphericalTensor)

    g_ket = g_state.level_ket
    e_ket = e_state.level_ket
    Jg = g_ket.j
    Je = e_ket.j
    Lg = g_ket.l
    Le = e_ket.l
    Sg = g_ket.s
    Se = e_ket.s
    m_Jg = g_ket.m
    m_Je = e_ket.m

    E_diff = e_state.E - g_state.E
    k = tensor.k
    q = tensor.q

    # label the gamma symbol
    label_g = g_state.label
    label_e = e_state.label
    gamma = 'gamma' + label_g  + label_e
    gamma = sympify(gamma)

    # Give the result
    result = (-1)**(Je-Jg+m_Jg-m_Je) * sqrt((2*Jg+1) / (2*Je+1))
    result *= clebsch_gordan(Jg, k, Je, m_Jg, -q, m_Je)
    dbl_bar = DoubleBar(Jg, Je, E_diff, gamma)
    if decouple:
        result *= (-1)**(Je+Lg+1+Sg) * sqrt((2*Je+1)*(2*Lg+1))
        result *= wigner_6j(Lg, Le, 1, Je, Jg, Sg)
        dbl_bar = DoubleBar(Lg, Le, E_diff, gamma)
    
    return result, dbl_bar