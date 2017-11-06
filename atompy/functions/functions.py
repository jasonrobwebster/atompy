"""The functions of AtomPy."""

from __future__ import print_function

from atompy.core import (AtomicState, DoubleBar, SphericalTensor, Dagger,
                         OuterProduct, Add, Mul, clebsch_gordan, sqrt,
                         wigner_6j, sympify, S, qapply)


__all__ = [
    'transition_strength',
    'weak_zeeman',
    'lindblad_superop'
]


# TODO: Extend these to functions to classes

def transition_strength(ground_state, excited_state, tensor, decouple_j=True, decouple_l=True):
    """Calculate the transition strengths between two atomic states [1], given by
    <ground_state|tensor|excited_state>.

    Returns the relative transition strength from the ground state g
    to the excited state e. 

    Parameters
    ==========

    g_state : Atomiclevel
        The ground atomic state.

    e_state : Atomiclevel
        The excited atomic state.

    tensor : SphericalTensor
        An instance of the SphericalTensor class that defines the rank
        and polarization of light.

    decouple_j : Boolean, Optional
        Whether to decouple the resulting <F||tensor||F'> to <J||tensor||J'>. 
        Defaults to True.

    decouple_j : Boolean, Optional
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
    >>> # strength = 1/9

    References
    ==========

    .. [1] Steck, D.A., 2007. Quantum and atom optics. p. 336
        http://atomoptics-nas.uoregon.edu/~dsteck/teaching/quantum-optics/quantum-optics-notes.pdf
    """

    # janky way of accounting for additions and multiplications
    # TODO: Fix the jankiness, make it more sympy like.
    out = 0

    if isinstance(tensor, Add):
        for arg in tensor.args:
            out += transition_strength(ground_state, arg, tensor)

    if isinstance(tensor, Mul):
        coeff, tens = tensor.args
        assert isinstance(coeff, SphericalHarmonic) is False
        out += transition_strength(ground_state, tens, excited_state)
    

    # assume our states are atomic states
    assert isinstance(ground_state, AtomicState)
    assert isinstance(excited_state, AtomicState)
    assert isinstance(tensor, SphericalTensor)

    # get the ground and excited kets and their useful values.
    g_ket = ground_state.ket
    e_ket = excited_state.ket

    fg = g_ket.f
    fe = e_ket.f
    jg = g_ket.j
    je = e_ket.j
    lg = g_ket.l
    le = e_ket.l
    sg = g_ket.s
    se = e_ket.s
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
    result = (-1)**(fe-fg+m_g-m_e) * sqrt((2*fg+1) / (2*fe+1))
    result *= clebsch_gordan(fg, k, fe, m_g, -q, m_e)
    dbl_bar = DoubleBar(fg, fe, E_diff, gamma)
    
    if decouple_j or decouple_l:
        # decouple to <J||..||J'>
        result *= (-1)**(fe+jg+1+I) * sqrt((2*fe+1)*(2*jg+1))
        result *= wigner_6j(jg, je, 1, fe, fg, I)
        dbl_bar = DoubleBar(jg, je, E_diff, gamma)
    
    if decouple_l:
        # decouple to <L||..||L'>
        result *= (-1)**(je+lg+1+sg) * sqrt((2*je+1)*(2*lg+1))
        result *= wigner_6j(lg, le, 1, je, jg, sg)
        dbl_bar = DoubleBar(lg, le, E_diff, gamma)
    
    out += result * dbl_bar

    return out

def weak_zeeman(ket, b, **kwargs):
    """Calculates the energy shift due to a weak magnetic field.

    Parameters
    ==========

    ket : AtomicJxKet, AtomicJyKet, AtomicJzKet
        An instance of an atomic ket.

    b : Number, Symbol
        The magnetic field in the direction of the given ket.

    Also accepts the following keyword arguments:
    
    g_l : Number, Symbol
        The L gyromagnetic factor.
        Defaults to 1.

    g_s : Number, Symbol
        The electron spin gyromagnetic factor.
        Defaults to the CODATA [1] value.

    mu_b : Number, Symbol
        The Bohr magnetion.
        Defaults to the SI CODATA value [2] (without the units).

    g_i : Number, Symbol
        The nuclear spin I gyromagnetic factor. Defaults to 0.

    Examples
    ========



    References
    ==========

    .. [1] https://physics.nist.gov/cgi-bin/cuu/Value?gem
    .. [2] https://physics.nist.gov/cgi-bin/cuu/Value?mub
    """
    
    b = sympify(b)
    g_l = sympify(kwargs.get('g_l', 1))
    g_s = sympify(kwargs.get('g_s', 2.00231930436182))
    mu_b = sympify(kwargs.get('mu_b', 927.4009994 * S(10)**(-26)))
    g_i = sympify(kwargs.get('g_i', 0))
    
    s = ket.s
    l = ket.l
    j = ket.j
    i = ket.i
    f = ket.f

    # calculate the lande g factor
    g_j = g_l + (g_s - g_l) * ((j*(j+1) + s*(s+1) - l*(l+1)) / (2*j*(j+1)))
    if i == 0:
        return mu_b * g_j * ket.m * b
    else:
        g_f = g_j * ((f*(f+1) - i*(i+1) + j*(j+1)) / (2*f*(f+1))) 
        g_f += g_i * ((f*(f+1) + i*(i+1) - j*(j+1)) / (2*f*(f+1)))
        if f == 0:
            g_f = 0
        return mu_b * g_f * ket.m * b

def lindblad_superop(operator, rho):
    """Applies the Lindblad superoperator D[operator]rho operator from [1].

    Parameters
    ==========

    operator : Operator, Symbol
        The operator to apply to the density matrix.

    rho : Add, Mul, Outerproduct
        The density matrix of the atom to which to apply this operation to.

    References
    ==========

    .. [1] Steck, D.A., 2007. Quantum and atom optics. p. 145
        http://atomoptics-nas.uoregon.edu/~dsteck/teaching/quantum-optics/quantum-optics-notes.pdf
    """

    op = operator #for brevity
    op_dagger = Dagger(op)

    assert isinstance(op, OuterProduct)
    assert isinstance(op_dagger, OuterProduct)

    # The reason for the op.ket*op.bra seen below is because qapply
    # doesn't work with two operators, so to get the desired behaviour
    # we need to force it to apply to the op's ket before multiplying in
    # bra.

    # Because of the above, we need to act on each element of rho
    # unless it's already an outerproduct

    # TODO: Investigate a fix for this
    # May be able to create a class for rho that extends a hermitian op
    # Which will then have the proper functionality applied to it
    out = 0
    if isinstance(rho, OuterProduct):
        out += qapply(op * rho * op_dagger.ket*op_dagger.bra)
        out -= S.One/2 * qapply(op_dagger*op*rho.ket*rho.bra + rho * op_dagger * op.ket*op.bra)
    elif isinstance(rho, (Add, Mul)):
        if isinstance(rho, Mul):
            coeff, var = rho.args
            if isinstance(coeff, OuterProduct):
                out += var * lindblad_superop(op, coeff)
            elif isinstance(var, OuterProduct):
                out += coeff * lindblad_superop(op, var)
            else:
                raise ValueError('Given mul has too many arguments, got %s' % rho)
        elif isinstance(rho, Add):
            for arg in rho.args:
                # these args should now be Muls, i.e.
                # rho = rho_00|0><0| + rho_01|0><1| + ...
                out += lindblad_superop(op, arg)
    else:
        raise ValueError(
            'Rho should be of type Add, Mul or OuterProduct, got type %s' % type(rho).__name__
        )

    return out
