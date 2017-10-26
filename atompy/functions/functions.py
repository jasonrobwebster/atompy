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
    Fg = g_ket.f
    Fe = e_ket.f
    Jg = g_ket.j
    Je = e_ket.j
    Lg = g_ket.l
    Le = e_ket.l
    Sg = g_ket.s
    Se = e_ket.s
    m_g = g_ket.m
    m_e = e_ket.m
    i_g = g_ket.i
    i_e = e_ket.i

    assert i_g == i_e
    I = i_g

    E_diff = e_state.E - g_state.E
    k = tensor.k
    q = tensor.q

    # label the gamma symbol
    label_g = g_state.label
    label_e = e_state.label
    gamma = 'Gamma' + label_g  + label_e
    gamma = sympify(gamma)

    # Give the result
    result = (-1)**(Fe-Fg+m_g-m_e) * sqrt((2*Fg+1) / (2*Fe+1))
    result *= clebsch_gordan(Fg, k, Fe, m_g, -q, m_e)
    dbl_bar = DoubleBar(Fg, Fe, E_diff, gamma)
    if decouple:
        # decouple to <J||..||J'>
        result *= (-1)**(Fe+Jg+1+I) * sqrt((2*Fe+1)*(2*Jg+1))
        result *= wigner_6j(Jg, Je, 1, Fe, Fg, I)
        # decouple to <L||..||L'>
        result *= (-1)**(Je+Lg+1+Sg) * sqrt((2*Je+1)*(2*Lg+1))
        result *= wigner_6j(Lg, Le, 1, Je, Jg, Sg)
        dbl_bar = DoubleBar(Lg, Le, E_diff, gamma)

    return result, dbl_bar

def weak_zeeman(atomic_ket, mag_field):
    """Calculates the energy shift due to a weak magnetic field.

    Parameters
    ==========

    atomic_ket: AtomicJxKet, AtomicJyKet, AtomicJzKet
        
