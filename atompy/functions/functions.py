"""The functions of AtomPy."""

from __future__ import print_function

from atompy.core import (AtomicState, DoubleBar, SphericalTensor, Dagger, Symbol,
                         OuterProduct, Operator, Add, Mul, clebsch_gordan, sqrt,
                         wigner_6j, sympify, S, qapply, hbar, u0)


__all__ = [
    'transition_strength',
    'weak_zeeman',
    'lindblad_superop',
    'operator_commutator'
]


# TODO: Extend these to functions to classes

def transition_strength(ground_state, excited_state, tensor, flip_q=False, decouple_j=True, decouple_l=True, **kwargs):
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

    flip_q : Boolean, Optional
        Whether the output's sign should be flipped based on the value of tensor.q.

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

    subs_list = kwargs.get('subs_list', [])

    if isinstance(tensor, Add):
        for arg in tensor.args:
            out += transition_strength(ground_state, excited_state, arg)
        return out
    elif isinstance(tensor, Mul):
        args = tensor.args
        try:
            out += transition_strength(ground_state, excited_state, args[0]) * Mul(*args[1:])
        except ValueError:
            if len(args) > 2:
                out += transition_strength(ground_state, excited_state, Mul(*args[1:])) * args[0]
            else:
                out += transition_strength(ground_state, excited_state, args[1]) * args[0]
        return out
    elif not isinstance(tensor, SphericalTensor):
        raise ValueError('tensor must be a spherical tensor, got %s' %tensor)
    

    # assume our states are atomic states
    assert isinstance(ground_state, AtomicState)
    assert isinstance(excited_state, AtomicState)

    # get the ground and excited kets and their useful values.
    g_ket = ground_state.ket
    e_ket = excited_state.ket

    e_g, e_e = ground_state.E, excited_state.E
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
    if flip_q:
        q = -q

    # label the gamma symbol
    label_g = ground_state.label
    label_e = excited_state.label
    gamma = 'Gamma_%s%s' %(label_g, label_e)
    gamma = Symbol(gamma)
    w0 = (e_e - e_g)/hbar

    # Give the result
    result = (-1)**(fe-fg+m_g-m_e) * sqrt((2*fg+1) / (2*fe+1))
    result *= clebsch_gordan(fg, k, fe, m_g, -q, m_e)
    dbl_bar = DoubleBar(fg, fe, w0, gamma)
    
    if decouple_j or decouple_l:
        w0 = Symbol('omega_%s%s' %(label_g, label_e))
        # decouple to <J||..||J'>
        result *= (-1)**(fe+jg+1+I) * sqrt((2*fe+1)*(2*jg+1))
        result *= wigner_6j(jg, je, k, fe, fg, I)
        dbl_bar = DoubleBar(jg, je, w0, gamma)
    
    if decouple_l:
        # decouple to <L||..||L'>
        result *= (-1)**(je+lg+1+sg) * sqrt((2*je+1)*(2*lg+1))
        result *= wigner_6j(lg, le, k, je, jg, sg)
        dbl_bar = DoubleBar(lg, le, w0, gamma)
    
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
    mu_b = u0
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
    if isinstance(rho, Add):
        for arg in rho.args:
            out += lindblad_superop(operator, arg)
    elif isinstance(rho, Mul):
        args = rho.args
        try:
            result = Mul(*args[1:]) * lindblad_superop(operator, args[0])
        except ValueError:
            result = args[0] * lindblad_superop(operator, Mul(*args[1:]))
        out += result
    elif not isinstance(rho, Operator):
        raise ValueError('rho is not an operator, got %s' %rho)
    elif isinstance(rho, OuterProduct):
        out += qapply(op * rho * op_dagger.ket*op_dagger.bra)
        out -= S.One/2 * qapply(op_dagger*op*rho.ket*rho.bra + rho * op_dagger * op.ket*op.bra)
    else:
        raise TypeError(
            'Rho should be of type Add, Mul or OuterProduct, got type %s' % type(rho).__name__
        )
    return out

def operator_qapply(a,b):
    """Calculates qapply(a*b) properly.
    Currently if a=|e><g| and b=|g><e|, then the native sympy qapply does
    not produce the correct answer qapply(a*b)!=|e><e|. Likely something wrong
    with the way that qapply handles OuterProducts.

    This function overcomes this problem.
    """

    out = 0

    # TODO: Fix janky handling, perhaps add a .doit method
    if isinstance(b, Add):
        for arg in b.args:
            out += operator_qapply(a, arg)
    elif isinstance(b, Mul):
        args = b.args
        try:
            result = Mul(*args[1:]) * operator_qapply(a, args[0])
        except ValueError:
            result = args[0] * operator_qapply(a, Mul(*args[1:]))
        out += result
    elif not isinstance(b, Operator):
        raise ValueError('b is not an operator, got %s' %b)
    elif not isinstance(b, OuterProduct):
        return qapply(a * b)
    else:
        out += qapply(a* b.ket * b.bra)

    return out

def operator_commutator(a, b):
    """Calculates the commutator for two operators a and b as [a,b].
    Attempts to overcome a janky bug in qapply where |g><e| * |e><g| != |g><g|.

    Parameters
    ==========

    a : Operator
        An Operator. Can also be a sum of operators.

    b : Operator
        An Operator. Can also be a sum of operators.
    """

    # TODO: Rename function
    # TODO: Make a function that just does qapply(a*b) properly

    return operator_qapply(a, b) - operator_qapply(b, a)
