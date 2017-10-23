"""Atomic kets used in states.

Extends the spin class in sympy by expanding the Hilbert space of our atomic
energy levels to incorporate the principle number.

Based off the spin class in Sympy 1.1.1.
"""

from __future__ import print_function, division

from sympy import (Add, binomial, cos, exp, Expr, factorial, I, Integer, Mul,
                   pi, Rational, S, sin, simplify, sqrt, Sum, symbols, sympify,
                   Tuple, Dummy)
from sympy.core.compatibility import u, unicode, range
from sympy.matrices import zeros
from sympy.printing.pretty.stringpict import prettyForm, stringPict
from sympy.printing.pretty.pretty_symbology import pretty_symbol

from sympy.physics.quantum.qexpr import QExpr
from sympy.physics.quantum.operator import (HermitianOperator, Operator,
                                            UnitaryOperator)
from sympy.physics.quantum.state import Bra, Ket, State
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.physics.quantum.constants import hbar
from sympy.physics.quantum.hilbert import ComplexSpace, DirectSumHilbertSpace
from sympy.physics.quantum.tensorproduct import TensorProduct
from sympy.physics.quantum.cg import CG
from sympy.physics.quantum.qapply import qapply

def m_values(j):
    j = sympify(j)
    size = 2*j + 1
    if not size.is_Integer or not size > 0:
        raise ValueError(
            'Only integer or half-integer values allowed for j, got: : %r' % j
        )
    return size, [j - i for i in range(int(2*j + 1))]


#-----------------------------------------------------------------------------
# Spin Operators
#-----------------------------------------------------------------------------


class AtomSpinOpBase(object):
    """Base class for atomic spin operators."""

    @classmethod
    def _eval_hilbert_space(cls, label):
        # We consider all j values so our space is infinite.
        return ComplexSpace(S.Infinity)

    @property
    def name(self):
        return self.args[0]

    def _print_contents(self, printer, *args):
        return '%s%s' % (unicode(self.name), self._coord)

    def _print_contents_pretty(self, printer, *args):
        a = stringPict(unicode(self.name))
        b = stringPict(self._coord)
        return self._print_subscript_pretty(a, b)

    def _print_contents_latex(self, printer, *args):
        return r'%s_%s' % ((unicode(self.name), self._coord))

    def _represent_base(self, basis, **options):
        # TODO: Change how represent works in AtomKets
        # The represent function should return kets that include
        # the space of n, s, and l. 
        n = options.get('n', S.One)
        s = options.get('s', Rational(1,2))
        l = options.get('l', S.Zero)
        j = options.get('j', Rational(1, 2))
        size, mvals = m_values(j)
        result = zeros(size, size)
        for p in range(size):
            for q in range(size):
                me = self.matrix_element(n, n, s, s, l, l, j, mvals[p], j, mvals[q])
                result[p, q] = me
        return result

    def _apply_op(self, ket, orig_basis, **options):
        state = ket.rewrite(self.basis)
        # If the state has only one term
        if isinstance(state, State):
            ret = (hbar*state.m) * state
        # state is a linear combination of states
        elif isinstance(state, Sum):
            ret = self._apply_operator_Sum(state, **options)
        else:
            ret = qapply(self*state)
        if ret == self*state:
            raise NotImplementedError
        return ret.rewrite(orig_basis)

    def _apply_operator_AtomJxKet(self, ket, **options):
        return self._apply_op(ket, 'Jx', **options)

    def _apply_operator_AtomJyKet(self, ket, **options):
        return self._apply_op(ket, 'Jy', **options)

    def _apply_operator_AtomJzKet(self, ket, **options):
        return self._apply_op(ket, 'Jz', **options)

    def _apply_operator_TensorProduct(self, tp, **options):
        # Uncoupling operator is only easily found for coordinate basis spin operators
        # TODO: add methods for uncoupling operators
        if not (isinstance(self, AtomJxOp) or isinstance(self, AtomJyOp) or isinstance(self, AtomJzOp)):
            raise NotImplementedError
        result = []
        for n in range(len(tp.args)):
            arg = []
            arg.extend(tp.args[:n])
            arg.append(self._apply_operator(tp.args[n]))
            arg.extend(tp.args[n + 1:])
            result.append(tp.__class__(*arg))
        return Add(*result).expand()

    # TODO: move this to qapply_Mul
    def _apply_operator_Sum(self, s, **options):
        new_func = qapply(self * s.function)
        if new_func == self*s.function:
            raise NotImplementedError
        return Sum(new_func, *s.limits)

    def _eval_trace(self, **options):
        #TODO: use options to use different j values
        #For now eval at default basis

        # is it efficient to represent each time
        # to do a trace?
        return self._represent_default_basis().trace()

class AtomJplusOp(AtomSpinOpBase, Operator):
    """The J+ operator."""

    _coord = '+'

    basis = 'Jz'

    def _eval_commutator_AtomJminusOp(self, other):
        return 2*hbar*AtomJzOp(self.name)

    def _apply_operator_AtomJzKet(self, ket, **options):
        n = ket.n
        s = ket.s
        l = ket.l
        j = ket.j
        m = ket.m
        if m.is_Number and j.is_Number:
            if m >= j:
                return S.Zero
        return hbar*sqrt(j*(j + S.One) - m*(m + S.One))*AtomJzKet(n, s, l, j, m + S.One)

    def matrix_element(self, n, np, s, sp, l, lp, j, m, jp, mp):
        result = hbar*sqrt(j*(j + S.One) - mp*(mp + S.One))
        result *= KroneckerDelta(n, np)
        result *= KroneckerDelta(m, mp + 1)
        result *= KroneckerDelta(j, jp)
        result *= KroneckerDelta(l, lp)
        result *= KroneckerDelta(s, sp)
        return result

    def _represent_default_basis(self, **options):
        return self._represent_AtomJzOp(None, **options)

    def _represent_AtomJzOp(self, basis, **options):
        return self._represent_base(basis, **options)

    def _eval_rewrite_as_xyz(self, *args):
        return AtomJxOp(args[0]) + I*AtomJyOp(args[0])


class AtomJminusOp(AtomSpinOpBase, Operator):
    """The J- operator."""

    _coord = '-'

    basis = 'Jz'

    def _apply_operator_AtomJzKet(self, ket, **options):
        n = ket.n
        s = ket.a
        l = ket.l
        j = ket.j
        m = ket.m
        if m.is_Number and j.is_Number:
            if m <= -j:
                return S.Zero
        return hbar*sqrt(j*(j + S.One) - m*(m - S.One))*AtomJzKet(n, s, l, j, m - S.One)

    def matrix_element(self, n, np, s, sp, l, lp, j, m, jp, mp):
        result = hbar*sqrt(j*(j + S.One) - mp*(mp - S.One))
        result *= KroneckerDelta(n, np)
        result *= KroneckerDelta(m, mp - 1)
        result *= KroneckerDelta(j, jp)
        result *= KroneckerDelta(l, lp)
        result *= KroneckerDelta(s, sp)
        return result

    def _represent_default_basis(self, **options):
        return self._represent_AtomJzOp(None, **options)

    def _represent_AtomJzOp(self, basis, **options):
        return self._represent_base(basis, **options)

    def _eval_rewrite_as_xyz(self, *args):
        return AtomJxOp(args[0]) - I*AtomJyOp(args[0])


class AtomJxOp(AtomSpinOpBase, HermitianOperator):
    """The Jx operator."""

    _coord = 'x'

    basis = 'Jx'

    def _eval_commutator_AtomJyOp(self, other):
        return I*hbar*AtomJzOp(self.name)

    def _eval_commutator_AtomJzOp(self, other):
        return -I*hbar*AtomJyOp(self.name)

    def _apply_operator_AtomJzKet(self, ket, **options):
        jp = AtomJplusOp(self.name)._apply_operator_AtomJzKet(ket, **options)
        jm = AtomJminusOp(self.name)._apply_operator_AtomJzKet(ket, **options)
        return (jp + jm)/Integer(2)

    def _represent_default_basis(self, **options):
        return self._represent_AtomJzOp(None, **options)

    def _represent_AtomJzOp(self, basis, **options):
        jp = AtomJplusOp(self.name)._represent_AtomJzOp(basis, **options)
        jm = AtomJminusOp(self.name)._represent_AtomJzOp(basis, **options)
        return (jp + jm)/Integer(2)

    def _eval_rewrite_as_plusminus(self, *args):
        return (AtomJplusOp(args[0]) + AtomJminusOp(args[0]))/2


class AtomJyOp(AtomSpinOpBase, HermitianOperator):
    """The Jy operator."""

    _coord = 'y'

    basis = 'Jy'

    def _eval_commutator_AtomJzOp(self, other):
        return I*hbar*AtomJxOp(self.name)

    def _eval_commutator_AtomJxOp(self, other):
        return -I*hbar*AtomJ2Op(self.name)

    def _apply_operator_AtomJzKet(self, ket, **options):
        jp = AtomJplusOp(self.name)._apply_operator_AtomJzKet(ket, **options)
        jm = AtomJminusOp(self.name)._apply_operator_AtomJzKet(ket, **options)
        return (jp - jm)/(Integer(2)*I)

    def _represent_default_basis(self, **options):
        return self._represent_AtomJzOp(None, **options)

    def _represent_AtomJzOp(self, basis, **options):
        jp = AtomJplusOp(self.name)._represent_AtomJzOp(basis, **options)
        jm = AtomJminusOp(self.name)._represent_AtomJzOp(basis, **options)
        return (jp - jm)/(Integer(2)*I)

    def _eval_rewrite_as_plusminus(self, *args):
        return (AtomJplusOp(args[0]) - AtomJminusOp(args[0]))/(2*I)


class AtomJzOp(AtomSpinOpBase, HermitianOperator):
    """The Jz operator."""

    _coord = 'z'

    basis = 'Jz'

    def _eval_commutator_AtomJxOp(self, other):
        return I*hbar*AtomJyOp(self.name)

    def _eval_commutator_AtomJyOp(self, other):
        return -I*hbar*AtomJxOp(self.name)

    def _eval_commutator_AtomJplusOp(self, other):
        return hbar*AtomJplusOp(self.name)

    def _eval_commutator_AtomJminusOp(self, other):
        return -hbar*AtomJminusOp(self.name)

    def matrix_element(self, n, np, s, sp, l, lp, j, m, jp, mp):
        result = hbar*mp
        result *= KroneckerDelta(n, np)
        result *= KroneckerDelta(m, mp)
        result *= KroneckerDelta(j, jp)
        result *= KroneckerDelta(l, lp)
        result *= KroneckerDelta(s, sp)
        return result

    def _represent_default_basis(self, **options):
        return self._represent_AtomJzOp(None, **options)

    def _represent_AtomJzOp(self, basis, **options):
        return self._represent_base(basis, **options)


class AtomJ2Op(AtomSpinOpBase, HermitianOperator):
    """The J^2 operator."""

    _coord = '2'

    def _eval_commutator_AtomJxOp(self, other):
        return S.Zero

    def _eval_commutator_AtomJyOp(self, other):
        return S.Zero

    def _eval_commutator_AtomJzOp(self, other):
        return S.Zero

    def _eval_commutator_AtomJplusOp(self, other):
        return S.Zero

    def _eval_commutator_AtomJminusOp(self, other):
        return S.Zero

    def _apply_operator_AtomJxKet(self, ket, **options):
        j = ket.j
        return hbar**2*j*(j + 1)*ket

    def _apply_operator_AtomJyKet(self, ket, **options):
        j = ket.j
        return hbar**2*j*(j + 1)*ket

    def _apply_operator_AtomJzKet(self, ket, **options):
        j = ket.j
        return hbar**2*j*(j + 1)*ket

    def matrix_element(self, n, np, s, sp, l, lp, j, m, jp, mp):
        result = (hbar**2)*j*(j + 1)
        result *= KroneckerDelta(n, np)
        result *= KroneckerDelta(m, mp)
        result *= KroneckerDelta(j, jp)
        result *= KroneckerDelta(l, lp)
        result *= KroneckerDelta(s, sp)
        return result

    def _represent_default_basis(self, **options):
        return self._represent_AtomJzOp(None, **options)

    def _represent_AtomJzOp(self, basis, **options):
        return self._represent_base(basis, **options)

    def _print_contents_pretty(self, printer, *args):
        a = prettyForm(unicode(self.name))
        b = prettyForm(u('2'))
        return a**b

    def _print_contents_latex(self, printer, *args):
        return r'%s^2' % str(self.name)

    def _eval_rewrite_as_xyz(self, *args):
        return AtomJxOp(args[0])**2 + AtomJyOp(args[0])**2 + AtomJzOp(args[0])**2

    def _eval_rewrite_as_plusminus(self, *args):
        a = args[0]
        return AtomJzOp(a)**2 + \
            Rational(1, 2)*(AtomJplusOp(a)*AtomJminusOp(a) + AtomJminusOp(a)*AtomJplusOp(a))

class AtomS2Op(AtomJ2Op):
    """The S^2 operator."""
    
    def _apply_operator_AtomJxKet(self, ket, **options):
        s = ket.s
        return hbar**2*s*(s + 1)*ket

    def _apply_operator_AtomJyKet(self, ket, **options):
        s = ket.s
        return hbar**2*s*(s + 1)*ket

    def _apply_operator_AtomJzKet(self, ket, **options):
        s = ket.s
        return hbar**2*s*(s + 1)*ket

    def matrix_element(self, n, np, s, sp, l, lp, j, m, jp, mp):
        result = (hbar**2)*s*(s + 1)
        result *= KroneckerDelta(n, np)
        result *= KroneckerDelta(m, mp)
        result *= KroneckerDelta(j, jp)
        result *= KroneckerDelta(l, lp)
        result *= KroneckerDelta(s, sp)
        return result

class AtomL2Op(AtomJ2Op):
    """The L^2 operator."""
    
    def _apply_operator_AtomJxKet(self, ket, **options):
        l = ket.l
        return hbar**2*l*(l + 1)*ket

    def _apply_operator_AtomJyKet(self, ket, **options):
        l = ket.l
        return hbar**2*l*(l + 1)*ket

    def _apply_operator_AtomJzKet(self, ket, **options):
        l = ket.l
        return hbar**2*l*(l + 1)*ket

    def matrix_element(self, n, np, s, sp, l, lp, j, m, jp, mp):
        result = (hbar**2)*l*(l + 1)
        result *= KroneckerDelta(n, np)
        result *= KroneckerDelta(m, mp)
        result *= KroneckerDelta(j, jp)
        result *= KroneckerDelta(l, lp)
        result *= KroneckerDelta(s, sp)
        return result

class Rotation(UnitaryOperator):
    """Wigner D operator in terms of Euler angles.

    Defines the rotation operator in terms of the Euler angles defined by
    the z-y-z convention for a passive transformation. That is the coordinate
    axes are rotated first about the z-axis, giving the new x'-y'-z' axes. Then
    this new coordinate system is rotated about the new y'-axis, giving new
    x''-y''-z'' axes. Then this new coordinate system is rotated about the
    z''-axis. Conventions follow those laid out in [1]_.

    Parameters
    ==========

    alpha : Number, Symbol
        First Euler Angle
    beta : Number, Symbol
        Second Euler angle
    gamma : Number, Symbol
        Third Euler angle

    Examples
    ========

    A simple example rotation operator:

        >>> from sympy import pi
        >>> from sympy.physics.quantum.spin import Rotation
        >>> Rotation(pi, 0, pi/2)
        R(pi,0,pi/2)

    With symbolic Euler angles and calculating the inverse rotation operator:

        >>> from sympy import symbols
        >>> a, b, c = symbols('a b c')
        >>> Rotation(a, b, c)
        R(a,b,c)
        >>> Rotation(a, b, c).inverse()
        R(-c,-b,-a)

    See Also
    ========

    WignerD: Symbolic Wigner-D function
    D: Wigner-D function
    d: Wigner small-d function

    References
    ==========

    .. [1] Varshalovich, D A, Quantum Theory of Angular Momentum. 1988.
    """

    @classmethod
    def _eval_args(cls, args):
        args = QExpr._eval_args(args)
        if len(args) != 3:
            raise ValueError('3 Euler angles required, got: %r' % args)
        return args

    @classmethod
    def _eval_hilbert_space(cls, label):
        # We consider all j values so our space is infinite.
        return ComplexSpace(S.Infinity)

    @property
    def alpha(self):
        return self.label[0]

    @property
    def beta(self):
        return self.label[1]

    @property
    def gamma(self):
        return self.label[2]

    def _print_operator_name(self, printer, *args):
        return 'R'

    def _print_operator_name_pretty(self, printer, *args):
        if printer._use_unicode:
            return prettyForm(u('\N{SCRIPT CAPITAL R}') + u(' '))
        else:
            return prettyForm("R ")

    def _print_operator_name_latex(self, printer, *args):
        return r'\mathcal{R}'

    def _eval_inverse(self):
        return Rotation(-self.gamma, -self.beta, -self.alpha)

    @classmethod
    def D(cls, j, m, mp, alpha, beta, gamma):
        """Wigner D-function.

        Returns an instance of the WignerD class corresponding to the Wigner-D
        function specified by the parameters.

        Parameters
        ===========

        j : Number
            Total angular momentum
        m : Number
            Eigenvalue of angular momentum along axis after rotation
        mp : Number
            Eigenvalue of angular momentum along rotated axis
        alpha : Number, Symbol
            First Euler angle of rotation
        beta : Number, Symbol
            Second Euler angle of rotation
        gamma : Number, Symbol
            Third Euler angle of rotation

        Examples
        ========

        Return the Wigner-D matrix element for a defined rotation, both
        numerical and symbolic:

            >>> from sympy.physics.quantum.spin import Rotation
            >>> from sympy import pi, symbols
            >>> alpha, beta, gamma = symbols('alpha beta gamma')
            >>> Rotation.D(1, 1, 0,pi, pi/2,-pi)
            WignerD(1, 1, 0, pi, pi/2, -pi)

        See Also
        ========

        WignerD: Symbolic Wigner-D function

        """
        return WignerD(j, m, mp, alpha, beta, gamma)

    @classmethod
    def d(cls, j, m, mp, beta):
        """Wigner small-d function.

        Returns an instance of the WignerD class corresponding to the Wigner-D
        function specified by the parameters with the alpha and gamma angles
        given as 0.

        Parameters
        ===========

        j : Number
            Total angular momentum
        m : Number
            Eigenvalue of angular momentum along axis after rotation
        mp : Number
            Eigenvalue of angular momentum along rotated axis
        beta : Number, Symbol
            Second Euler angle of rotation

        Examples
        ========

        Return the Wigner-D matrix element for a defined rotation, both
        numerical and symbolic:

            >>> from sympy.physics.quantum.spin import Rotation
            >>> from sympy import pi, symbols
            >>> beta = symbols('beta')
            >>> Rotation.d(1, 1, 0, pi/2)
            WignerD(1, 1, 0, 0, pi/2, 0)

        See Also
        ========

        WignerD: Symbolic Wigner-D function

        """
        return WignerD(j, m, mp, 0, beta, 0)

    def matrix_element(self, n, np, s, sp, l, lp, j, m, jp, mp):
        result = self.__class__.D(
            jp, m, mp, self.alpha, self.beta, self.gamma
        )
        result *= KroneckerDelta(j, jp)
        result *= KroneckerDelta(n, np)
        result *= KroneckerDelta(s, sp)
        result *= KroneckerDelta(l, lp)
        return result

    def _represent_base(self, basis, **options):
        j = sympify(options.get('j', Rational(1, 2)))
        n = sympify(options.get('n', S.One))
        s = sympify(options.get('s', Rational(1, 2)))
        l = sympify(options.get('l', S.Zero))
        # TODO: move evaluation up to represent function/implement elsewhere
        evaluate = sympify(options.get('doit'))
        size, mvals = m_values(j)
        result = zeros(size, size)
        for p in range(size):
            for q in range(size):
                me = self.matrix_element(n, n, s, s, l, l, j, mvals[p], j, mvals[q])
                if evaluate:
                    result[p, q] = me.doit()
                else:
                    result[p, q] = me
        return result

    def _represent_default_basis(self, **options):
        return self._represent_AtomJzOp(None, **options)

    def _represent_AtomJzOp(self, basis, **options):
        return self._represent_base(basis, **options)

    def _apply_operator_uncoupled(self, state, ket, **options):
        a = self.alpha
        b = self.beta
        g = self.gamma
        j = ket.j
        m = ket.m
        if j.is_number:
            s = []
            size = m_values(j)
            sz = size[1]
            for mp in sz:
                r = Rotation.D(j, m, mp, a, b, g)
                z = r.doit()
                s.append(z * state(j, mp))
            return Add(*s)
        else:
            if options.pop('dummy', True):
                mp = Dummy('mp')
            else:
                mp = symbols('mp')
            return Sum(Rotation.D(j, m, mp, a, b, g) * state(j, mp), (mp, -j, j))

    def _apply_operator_AtomJxKet(self, ket, **options):
        return self._apply_operator_uncoupled(AtomJxKet, ket, **options)

    def _apply_operator_AtomJyKet(self, ket, **options):
        return self._apply_operator_uncoupled(AtomJyKet, ket, **options)

    def _apply_operator_AtomJzKet(self, ket, **options):
        return self._apply_operator_uncoupled(AtomJzKet, ket, **options)

    def _apply_operator_coupled(self, state, ket, **options):
        a = self.alpha
        b = self.beta
        g = self.gamma
        n = ket.n
        s = ket.s
        l = ket.l
        j = ket.j
        m = ket.m
        jn = ket.jn
        coupling = ket.coupling
        if j.is_number:
            s = []
            size = m_values(j)
            sz = size[1]
            for mp in sz:
                r = Rotation.D(j, m, mp, a, b, g)
                z = r.doit()
                s.append(z * state(n, s, l, j, mp, jn, coupling))
            return Add(*s)
        else:
            if options.pop('dummy', True):
                mp = Dummy('mp')
            else:
                mp = symbols('mp')
            return Sum(Rotation.D(j, m, mp, a, b, g) * state(
                n, s, l, j, mp, jn, coupling), (mp, -j, j))

class WignerD(Expr):
    """Wigner-D function

    The Wigner D-function gives the matrix elements of the rotation
    operator in the jm-representation. For the Euler angles `\\alpha`,
    `\\beta`, `\gamma`, the D-function is defined such that:

    .. math ::
        <j,m| \mathcal{R}(\\alpha, \\beta, \gamma ) |j',m'> = \delta_{jj'} D(j, m, m', \\alpha, \\beta, \gamma)

    Where the rotation operator is as defined by the Rotation class [1]_.

    The Wigner D-function defined in this way gives:

    .. math ::
        D(j, m, m', \\alpha, \\beta, \gamma) = e^{-i m \\alpha} d(j, m, m', \\beta) e^{-i m' \gamma}

    Where d is the Wigner small-d function, which is given by Rotation.d.

    The Wigner small-d function gives the component of the Wigner
    D-function that is determined by the second Euler angle. That is the
    Wigner D-function is:

    .. math ::
        D(j, m, m', \\alpha, \\beta, \gamma) = e^{-i m \\alpha} d(j, m, m', \\beta) e^{-i m' \gamma}

    Where d is the small-d function. The Wigner D-function is given by
    Rotation.D.

    Note that to evaluate the D-function, the j, m and mp parameters must
    be integer or half integer numbers.

    Parameters
    ==========

    j : Number
        Total angular momentum
    m : Number
        Eigenvalue of angular momentum along axis after rotation
    mp : Number
        Eigenvalue of angular momentum along rotated axis
    alpha : Number, Symbol
        First Euler angle of rotation
    beta : Number, Symbol
        Second Euler angle of rotation
    gamma : Number, Symbol
        Third Euler angle of rotation

    Examples
    ========

    Evaluate the Wigner-D matrix elements of a simple rotation:

        >>> from sympy.physics.quantum.spin import Rotation
        >>> from sympy import pi
        >>> rot = Rotation.D(1, 1, 0, pi, pi/2, 0)
        >>> rot
        WignerD(1, 1, 0, pi, pi/2, 0)
        >>> rot.doit()
        sqrt(2)/2

    Evaluate the Wigner-d matrix elements of a simple rotation

        >>> rot = Rotation.d(1, 1, 0, pi/2)
        >>> rot
        WignerD(1, 1, 0, 0, pi/2, 0)
        >>> rot.doit()
        -sqrt(2)/2

    See Also
    ========

    Rotation: Rotation operator

    References
    ==========

    .. [1] Varshalovich, D A, Quantum Theory of Angular Momentum. 1988.
    """

    is_commutative = True

    def __new__(cls, *args, **hints):
        if not len(args) == 6:
            raise ValueError('6 parameters expected, got %s' % args)
        args = sympify(args)
        evaluate = hints.get('evaluate', False)
        if evaluate:
            return Expr.__new__(cls, *args)._eval_wignerd()
        return Expr.__new__(cls, *args)

    @property
    def j(self):
        return self.args[0]

    @property
    def m(self):
        return self.args[1]

    @property
    def mp(self):
        return self.args[2]

    @property
    def alpha(self):
        return self.args[3]

    @property
    def beta(self):
        return self.args[4]

    @property
    def gamma(self):
        return self.args[5]

    def _latex(self, printer, *args):
        if self.alpha == 0 and self.gamma == 0:
            return r'd^{%s}_{%s,%s}\left(%s\right)' % \
                (
                    printer._print(self.j), printer._print(
                        self.m), printer._print(self.mp),
                    printer._print(self.beta) )
        return r'D^{%s}_{%s,%s}\left(%s,%s,%s\right)' % \
            (
                printer._print(
                    self.j), printer._print(self.m), printer._print(self.mp),
                printer._print(self.alpha), printer._print(self.beta), printer._print(self.gamma) )

    def _pretty(self, printer, *args):
        top = printer._print(self.j)

        bot = printer._print(self.m)
        bot = prettyForm(*bot.right(','))
        bot = prettyForm(*bot.right(printer._print(self.mp)))

        pad = max(top.width(), bot.width())
        top = prettyForm(*top.left(' '))
        bot = prettyForm(*bot.left(' '))
        if pad > top.width():
            top = prettyForm(*top.right(' ' * (pad - top.width())))
        if pad > bot.width():
            bot = prettyForm(*bot.right(' ' * (pad - bot.width())))
        if self.alpha == 0 and self.gamma == 0:
            args = printer._print(self.beta)
            s = stringPict('d' + ' '*pad)
        else:
            args = printer._print(self.alpha)
            args = prettyForm(*args.right(','))
            args = prettyForm(*args.right(printer._print(self.beta)))
            args = prettyForm(*args.right(','))
            args = prettyForm(*args.right(printer._print(self.gamma)))

            s = stringPict('D' + ' '*pad)

        args = prettyForm(*args.parens())
        s = prettyForm(*s.above(top))
        s = prettyForm(*s.below(bot))
        s = prettyForm(*s.right(args))
        return s

    def doit(self, **hints):
        hints['evaluate'] = True
        return WignerD(*self.args, **hints)

    def _eval_wignerd(self):
        j = sympify(self.j)
        m = sympify(self.m)
        mp = sympify(self.mp)
        alpha = sympify(self.alpha)
        beta = sympify(self.beta)
        gamma = sympify(self.gamma)
        if not j.is_number:
            raise ValueError(
                'j parameter must be numerical to evaluate, got %s' % j)
        r = 0
        if beta == pi/2:
            # Varshalovich Equation (5), Section 4.16, page 113, setting
            # alpha=gamma=0.
            for k in range(2*j + 1):
                if k > j + mp or k > j - m or k < mp - m:
                    continue
                r += (-S(1))**k * binomial(j + mp, k) * binomial(j - mp, k + m - mp)
            r *= (-S(1))**(m - mp) / 2**j * sqrt(factorial(j + m) *
                    factorial(j - m) / (factorial(j + mp) * factorial(j - mp)))
        else:
            # Varshalovich Equation(5), Section 4.7.2, page 87, where we set
            # beta1=beta2=pi/2, and we get alpha=gamma=pi/2 and beta=phi+pi,
            # then we use the Eq. (1), Section 4.4. page 79, to simplify:
            # d(j, m, mp, beta+pi) = (-1)**(j-mp) * d(j, m, -mp, beta)
            # This happens to be almost the same as in Eq.(10), Section 4.16,
            # except that we need to substitute -mp for mp.
            size, mvals = m_values(j)
            for mpp in mvals:
                r += Rotation.d(j, m, mpp, pi/2).doit() * (cos(-mpp*beta) + I*sin(-mpp*beta)) * \
                    Rotation.d(j, mpp, -mp, pi/2).doit()
            # Empirical normalization factor so results match Varshalovich
            # Tables 4.3-4.12
            # Note that this exact normalization does not follow from the
            # above equations
            r = r * I**(2*j - m - mp) * (-1)**(2*m)
            # Finally, simplify the whole expression
            r = simplify(r)
        r *= exp(-I*m*alpha)*exp(-I*mp*gamma)
        return r


AtomJx = AtomJxOp('J')
AtomJy = AtomJyOp('J')
AtomJz = AtomJzOp('J')
AtomJ2 = AtomJ2Op('J')
AtomJplus = AtomJplusOp('J')
AtomJminus = AtomJminusOp('J')

AtomL2 = AtomL2Op('L')
AtomS2 = AtomS2Op('S')


#-----------------------------------------------------------------------------
# Spin States
#-----------------------------------------------------------------------------


class AtomSpinState(State):
    """Base class for angular momentum states."""

    _label_separator = ','

    def __new__(cls, n, s, l, j, m):
        n = sympify(n)
        s = sympify(s)
        l = sympify(l)
        j = sympify(j)
        m = sympify(m)

        if n.is_number:
            if n != int(n):
                raise ValueError(
                    'n must be integer, got: %s' % n)
            if n < 1:
                raise ValueError('n must be >= 1, got: %s' % n)

        if s.is_number:
            if 2*s != int(2*s):
                raise ValueError(
                    's must be integer or half-integer, got: %s' % s)
            if s < 0:
                raise ValueError('s must be >= 0, got: %s' % s)

        if l.is_number:
            if l != int(l):
                raise ValueError(
                    'l must be integer, got: %s' % l)
            if l < 0:
                raise ValueError('l must be >= 0, got: %s' % l)

        if l.is_number and n.is_number:
            if l >= n:
                raise ValueError('l must be < n, got n, l: %s, %s' % (n, l))

        if j.is_number:
            if 2*j != int(2*j):
                raise ValueError(
                    'j must be integer or half-integer, got: %s' % j)
            if j < 0:
                raise ValueError('j must be >= 0, got: %s' % j)

        if s.is_number and l.is_number and j.is_number:
            if abs(l-s) > j:
                raise ValueError(
                    'j must be >= |l-s|, got s, l, j: %s, %s, %s' %(s, l, j)
                )
            if j > l + s:
                raise ValueError(
                    'j must be <= l+s, got s, l, j: %s, %s, %s' %(s, l, j)
                )

        if m.is_number:
            if 2*m != int(2*m):
                raise ValueError(
                    'm must be integer or half-integer, got: %s' % m)

        if j.is_number and m.is_number:
            if abs(m) > j:
                raise ValueError('Allowed values for m are -j <= m <= j, got j, m: %s, %s' % (j, m))
            if int(j - m) != j - m:
                raise ValueError('Both j and m must be integer or half-integer, got j, m: %s, %s' % (j, m))
        return State.__new__(cls, n, s, l, j, m)

    #-------------------------------------------------------------------------
    # Properties
    #-------------------------------------------------------------------------

    @property
    def n(self):
        """The principle number of the state."""
        return self.label[0]

    @property
    def s(self):
        """The total angular momentum of the state."""
        return self.label[1]

    @property
    def l(self):
        """The angular momentum AtomJz eignestate number."""
        return self.label[2]

    @property
    def j(self):
        """The angular momentum AtomJz eignestate number."""
        return self.label[3]

    @property
    def m(self):
        """The angular momentum AtomJz eignestate number."""
        return self.label[4]

    #-------------------------------------------------------------------------
    # Methods
    #-------------------------------------------------------------------------

    @classmethod
    def _eval_hilbert_space(cls, label):
        return ComplexSpace(2*label[0] + 1)

    def _represent_base(self, **options):
        j = self.j
        m = self.m
        alpha = sympify(options.get('alpha', 0))
        beta = sympify(options.get('beta', 0))
        gamma = sympify(options.get('gamma', 0))
        size, mvals = m_values(j)
        result = zeros(size, 1)
        # TODO: Use KroneckerDelta if all Euler angles == 0
        # breaks finding angles on L930
        for p, mval in enumerate(mvals):
            if m.is_number:
                result[p, 0] = Rotation.D(
                    self.j, mval, self.m, alpha, beta, gamma).doit()
            else:
                result[p, 0] = Rotation.D(self.j, mval,
                                          self.m, alpha, beta, gamma)
        return result

    def _eval_rewrite_as_Jx(self, *args, **options):
        if isinstance(self, Bra):
            return self._rewrite_basis(AtomJx, AtomJxBra, **options)
        return self._rewrite_basis(AtomJx, AtomJxKet, **options)

    def _eval_rewrite_as_Jy(self, *args, **options):
        if isinstance(self, Bra):
            return self._rewrite_basis(AtomJy, AtomJyBra, **options)
        return self._rewrite_basis(AtomJy, AtomJyKet, **options)

    def _eval_rewrite_as_Jz(self, *args, **options):
        if isinstance(self, Bra):
            return self._rewrite_basis(AtomJz, AtomJzBra, **options)
        return self._rewrite_basis(AtomJz, AtomJzKet, **options)

    def _rewrite_basis(self, basis, evect, **options):
        from sympy.physics.quantum.represent import represent
        n = self.n
        s = self.s
        l = self.l
        j = self.j
        if j.is_number:
            # TODO: Rewite the _represent_base code.
            # This is to account for the fact that we are a multilevel state.
            # This will change the start value
            start = 0 
            vect = represent(self, basis=basis, **options)
            result = Add(
                *[vect[start + i] * evect(n, s, l, j, j - i) for i in range(2*j + 1)])
            return result
        else:
            i = 0
            mi = symbols('mi')
            # make sure not to introduce a symbol already in the state
            while self.subs(mi, 0) != self:
                i += 1
                mi = symbols('mi%d' % i)
                break
            # TODO: better way to get angles of rotation
            test_args = (n, s, l, 0, mi)
            if isinstance(self, Ket):
                angles = represent(
                    self.__class__(*test_args), basis=basis)[0].args[3:6]
            else:
                angles = represent(self.__class__(
                    *test_args), basis=basis)[0].args[0].args[3:6]
            if angles == (0, 0, 0):
                return self
            else:
                state = evect(n, s, l, j, mi, *args)
                lt = Rotation.D(j, mi, self.m, *angles)
                return Sum(lt * state, (mi, -j, j))

    def _eval_innerproduct_AtomJxBra(self, bra, **hints):
        result = KroneckerDelta(self.j, bra.j)
        result *= KroneckerDelta(self.n, bra.n)
        result *= KroneckerDelta(self.s, bra.s)
        result *= KroneckerDelta(self.l, bra.l)
        if bra.dual_class() is not self.__class__:
            result *= self._represent_AtomJxOp(None)[bra.j - bra.m]
        else:
            result *= KroneckerDelta(self.m, bra.m)
        return result

    def _eval_innerproduct_AtomJyBra(self, bra, **hints):
        result = KroneckerDelta(self.j, bra.j)
        result *= KroneckerDelta(self.n, bra.n)
        result *= KroneckerDelta(self.s, bra.s)
        result *= KroneckerDelta(self.l, bra.l)
        if bra.dual_class() is not self.__class__:
            result *= self._represent_AtomJyOp(None)[bra.j - bra.m]
        else:
            result *= KroneckerDelta(self.m, bra.m)
        return result

    def _eval_innerproduct_AtomJzBra(self, bra, **hints):
        result = KroneckerDelta(self.j, bra.j)
        result *= KroneckerDelta(self.n, bra.n)
        result *= KroneckerDelta(self.s, bra.s)
        result *= KroneckerDelta(self.l, bra.l)
        if bra.dual_class() is not self.__class__:
            result *= self._represent_AtomJzOp(None)[bra.j - bra.m]
        else:
            result *= KroneckerDelta(self.m, bra.m)
        return result

    def _eval_trace(self, bra, **hints):

        # One way to implement this method is to assume the basis set k is
        # passed.
        # Then we can apply the discrete form of Trace formula here
        # Tr(|i><j| ) = \Sum_k <k|i><j|k>
        #then we do qapply() on each each inner product and sum over them.

        # OR

        # Inner product of |i><j| = Trace(Outer Product).
        # we could just use this unless there are cases when this is not true

        return (bra*self).doit()


class AtomJxKet(AtomSpinState, Ket):
    """Eigenket of AtomJx.

    See AtomJzKet for the usage of spin eigenstates.

    See Also
    ========

    AtomJzKet: Usage of spin states

    """

    @classmethod
    def dual_class(self):
        return AtomJxBra

    def _represent_default_basis(self, **options):
        return self._represent_AtomJxOp(None, **options)

    def _represent_AtomJxOp(self, basis, **options):
        return self._represent_base(**options)

    def _represent_AtomJyOp(self, basis, **options):
        return self._represent_base(alpha=3*pi/2, **options)

    def _represent_AtomJzOp(self, basis, **options):
        return self._represent_base(beta=pi/2, **options)


class AtomJxBra(AtomSpinState, Bra):
    """Eigenbra of AtomJx.

    See AtomJzKet for the usage of spin eigenstates.

    See Also
    ========

    AtomJzKet: Usage of spin states

    """

    @classmethod
    def dual_class(self):
        return AtomJxKet


class AtomJyKet(AtomSpinState, Ket):
    """Eigenket of AtomJy.

    See AtomJzKet for the usage of spin eigenstates.

    See Also
    ========

    AtomJzKet: Usage of spin states

    """

    @classmethod
    def dual_class(self):
        return AtomJyBra

    def _represent_default_basis(self, **options):
        return self._represent_AtomJyOp(None, **options)

    def _represent_AtomJxOp(self, basis, **options):
        return self._represent_base(gamma=pi/2, **options)

    def _represent_AtomJyOp(self, basis, **options):
        return self._represent_base(**options)

    def _represent_AtomJzOp(self, basis, **options):
        return self._represent_base(alpha=3*pi/2, beta=-pi/2, gamma=pi/2, **options)


class AtomJyBra(AtomSpinState, Bra):
    """Eigenbra of AtomJy.

    See AtomJzKet for the usage of spin eigenstates.

    See Also
    ========

    AtomJzKet: Usage of spin states

    """

    @classmethod
    def dual_class(self):
        return AtomJyKet


class AtomJzKet(AtomSpinState, Ket):
    """Eigenket of AtomJz.

    Spin state which is an eigenstate of the AtomJz operator. Uncoupled states,
    that is states representing the interaction of multiple separate spin
    states, are defined as a tensor product of states.

    Parameters
    ==========

    n : Number, Symbol
        Principle number

    s : Number, Symbol
        Total electron spin angular momentum

    l : Number, Symbol
        Total orbital angular momentum
    
    j : Number, Symbol
        Total coupled angular momentum

    m : Number, Symbol
        Eigenvalue of the Jz spin operator

    Examples
    ========

    

    See Also
    ========

    Usage of SymPy JzKet

    """

    @classmethod
    def dual_class(self):
        return AtomJzBra

    def _represent_default_basis(self, **options):
        return self._represent_AtomJzOp(None, **options)

    def _represent_AtomJxOp(self, basis, **options):
        return self._represent_base(beta=3*pi/2, **options)

    def _represent_AtomJyOp(self, basis, **options):
        return self._represent_base(alpha=3*pi/2, beta=pi/2, gamma=pi/2, **options)

    def _represent_AtomJzOp(self, basis, **options):
        return self._represent_base(**options)


class AtomJzBra(AtomSpinState, Bra):
    """Eigenbra of AtomJz.

    See the AtomJzKet for the usage of spin eigenstates.

    See Also
    ========

    AtomJzKet: Usage of spin states

    """

    @classmethod
    def dual_class(self):
        return AtomJzKet
