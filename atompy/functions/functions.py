"""The functions of AtomPy."""

from __future__ import print_function

from atompy.core import AtomicState, SphericalTensor, DoubleBar, clebsch_gordan, sqrt, wigner_6j, sympify


__all__ = [
    'transition_strength'
]


# TODO: Extend these to functions to classes

def transition_strength(ground_state, excited_state, tensor, decouple=True):
    """Calculate the transition strengths between two atomic states [1], given by
    <ground_state|tensor|excited_state>.

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

    decouple : Boolean, Optional
        Whether to decouple the resulting <F||tensor||F'> to <L||tensor||L'>. 
        Defaults to True.

    Examples
    ========

    Create an atom with two states and find the transition strength between them.
    >>> import atompy as ap
    >>> # make a hydrogen atom with an S_1/2 and a P_1/2 state 
    >>> h = ap.Atom(name='H')
    >>> ground_state = h.add_level(
    ...     energy='E_0',
    ...     n=1,
    ...     s=ap.S(1)/2,
    ...     l='S',
    ...     j=ap.S(1)/2,
    ...     m=ap.S(1)/2,
    ...     label='S'
    ... )
    >>> excited_state = h.add_level(
    ...     energy='E_1',
    ...     n=2,
    ...     s=ap.S(1)/2,
    ...     l='P',
    ...     j=ap.S(1)/2,
    ...     m=ap.S(1)/2,
    ...     label='P'
    ... )
    >>> # define the tensor as the dipole tensor d_q
    >>> tensor = ap.dipole_tensor(q=0)
    >>> strength = ap.transition_strength(ground_state, excited_state, dipole)
    >>> print(strength)
    1/9

    References
    ==========

    .. [1] Steck, D.A., 2007. Quantum and atom optics. p. 336
        http://atomoptics-nas.uoregon.edu/~dsteck/teaching/quantum-optics/quantum-optics-notes.pdf
    """

    # assume our states are atomic states
    assert isinstance(ground_state, AtomicState)
    assert isinstance(excited_state, AtomicState)
    assert isinstance(tensor, SphericalTensor)

    # get the ground and excited kets and their useful values.
    g_ket = ground_state.ket
    e_ket = excited_state.ket

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

    k = tensor.k
    q = tensor.q

    # label the gamma symbol
    label_g = ground_state.label
    label_e = excited_state.label
    gamma = 'Gamma' + label_g  + label_e
    gamma = sympify(gamma)

    # Give the result
    result = (-1)**(Fe-Fg+m_g-m_e) * sqrt((2*Fg+1) / (2*Fe+1))
    result *= clebsch_gordan(Fg, k, Fe, m_g, -q, m_e)
    if decouple:
        # decouple to <J||..||J'>
        result *= (-1)**(Fe+Jg+1+I) * sqrt((2*Fe+1)*(2*Jg+1))
        result *= wigner_6j(Jg, Je, 1, Fe, Fg, I)
        # decouple to <L||..||L'>
        result *= (-1)**(Je+Lg+1+Sg) * sqrt((2*Je+1)*(2*Lg+1))
        result *= wigner_6j(Lg, Le, 1, Je, Jg, Sg)

    return result

def weak_zeeman(ket, b_z):
    """Calculates the energy shift due to a weak magnetic field.

    Parameters
    ==========

    ket : AtomicJzKet
        An instance of an atomic ket.

    b_z : Number, Symbol
        The magnetic field 
    """
    pass