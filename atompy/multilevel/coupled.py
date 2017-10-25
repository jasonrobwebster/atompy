"""We're going to extend the coupled ket class from the sympy spin.

This is better for an atomic state for multiple reasons:
    - It will let the user define what atomic states to work with, be it L, J or
    F states.
    - Can easily create new states.
    - Can decouple.
    - Loads of functionality.
"""

from __future__ import print_function, division

from sympy.physics.quantum.spin import (
    _build_coupled, Bra, CoupledSpinState, hbar, J2Op, Jx, JxBra, JxKet,
    Jy, JyBra, JyKet, Jz, JzBra, JzKet, JzOp, Ket, Rotation, State, uncouple
)

from sympy.core.compatibility import range
from sympy import Tuple, Sum, Add, Mul, sympify, symbols, pi
from sympy.physics.quantum.hilbert import ComplexSpace, DirectSumHilbertSpace
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.printing.pretty.stringpict import prettyForm, stringPict
from sympy.printing.pretty.pretty_symbology import pretty_symbol

__all__ = [
    'S2',
    'L2',
    'J2',
    'F2',
    'Fz',
    'AtomicJxBra',
    'AtomicJyBra',
    'AtomicJzBra',
    'AtomicJxKet',
    'AtomicJyKet',
    'AtomicJzKet'
]


#-----------------------------------------------------------------------------
# Extended J2 Operators
#-----------------------------------------------------------------------------


# TODO: Extend all of the old operators.
# Intergrate the old operators with the new style.

class S2Op(J2Op):
    """The S^2 Operator."""

    def _apply_operator_AtomJxKetCoupled(self, ket, **options):
        s = ket.s
        return hbar**2*s*(s + 1)*ket

    def _apply_operator_AtomJyKetCoupled(self, ket, **options):
        s = ket.s
        return hbar**2*s*(s + 1)*ket

    def _apply_operator_AtomJzKetCoupled(self, ket, **options):
        s = ket.s
        return hbar**2*s*(s + 1)*ket

class J2Op_Ext(J2Op):
    """The extended J^2 Operator."""

    def _apply_operator_AtomJxKetCoupled(self, ket, **options):
        j = ket.j
        return hbar**2*j*(j + 1)*ket

    def _apply_operator_AtomJxKetCoupled(self, ket, **options):
        j = ket.j
        return hbar**2*j*(j + 1)*ket

    def _apply_operator_AtomJxKetCoupled(self, ket, **options):
        j = ket.j
        return hbar**2*j*(j + 1)*ket

class L2Op(J2Op):
    """The F^2 Operator."""

    def _apply_operator_AtomJxKetCoupled(self, ket, **options):
        l = ket.l
        return hbar**2*l*(l + 1)*ket

    def _apply_operator_AtomJyKetCoupled(self, ket, **options):
        l = ket.l
        return hbar**2*l*(l + 1)*ket

    def _apply_operator_AtomJzKetCoupled(self, ket, **options):
        l = ket.l
        return hbar**2*l*(l + 1)*ket

class F2Op(J2Op):
    """The F^2 Operator."""

    def _apply_operator_AtomJxKetCoupled(self, ket, **options):
        f = ket.f
        return hbar**2*f*(f + 1)*ket

    def _apply_operator_AtomJyKetCoupled(self, ket, **options):
        f = ket.f
        return hbar**2*f*(f + 1)*ket

    def _apply_operator_AtomJzKetCoupled(self, ket, **options):
        f = ket.f
        return hbar**2*f*(f + 1)*ket


S2 = S2Op('S')
L2 = L2Op('L')
J2 = J2Op_Ext('J')
F2 = F2Op('F')
Fz = JzOp('F')


#-----------------------------------------------------------------------------
# Uncoupled Classes
#-----------------------------------------------------------------------------


# Hacky fix for uncoupled classes
# TODO: Add proper functionality to uncoupled classes
AtomJxBra = JxBra
AtomJyBra = JyBra
AtomJzBra = JzBra

AtomJxKet = JxKet
AtomJyKet = JyKet
AtomJzKet = JzKet


#-----------------------------------------------------------------------------
# Coupled Base Class
#-----------------------------------------------------------------------------


class AtomSpinState(CoupledSpinState):
    """Base class for atomic spin states."""

    def __new__(cls, n, j, m, jn, *jcoupling):
        # jn is now defined as
        # jn[0] = S
        # jn[1] = L (in which case j = J)
        # jn[2] = I (in which case j = F)

        # check j, m, jn with CoupledSpinState
        state = CoupledSpinState(j, m, jn) if len(jcoupling) == 0 else CoupledSpinState(j, m, jn, *jcoupling)

        # check length of jn, don't need more than three
        if len(jn) > 3:
            raise ValueError('Length of jn is too long for atomic state, got: %s' % jn)

        n = sympify(n)
        j = sympify(j)
        m = sympify(m)
        jn = state.jn
        jcoupling = state.coupling
        l = jn[1] # the l quantum number

        del state

        if n.is_number:
            if n != int(n):
                raise ValueError('n must be an integer, got: %s' % n)
            if n < 1:
                raise ValueError('n must be >= 1, got: %s' % n)

        if l.is_number:
            if l != int(l):
                raise ValueError('l must be an integer, got: %s' % l)
            if l < 0:
                raise ValueError('l must be >= 0, got: %s' % n)

        if n.is_number and l.is_number:
            if l >= n:
                raise ValueError('l must be <= n, got n, l: %s, %s' % (n, l))

        return State.__new__(cls, n, j, m, jn, jcoupling)

    @classmethod
    def _eval_hilbert_space(cls, label):
        j = Add(*label[3])
        if j.is_number:
            return DirectSumHilbertSpace(*[ ComplexSpace(x) for x in range(int(2*j + 1), 0, -2) ])
        else:
            # TODO: Need hilbert space fix, see SymPy issue 5732
            # Desired behavior:
            #ji = symbols('ji')
            #ret = Sum(ComplexSpace(2*ji + 1), (ji, 0, j))
            # Temporary fix:
            return ComplexSpace(2*j + 1)

    #-------------------------------------------------------------------------
    # Redefining Properties
    #-------------------------------------------------------------------------

    @property
    def n(self):
        return self.label[0]

    @property
    def j(self):
        if self.i == 0:
            return self.label[1]
        return self.label[4][0][2]

    @property
    def f(self):
        return self.label[1]

    @property
    def m(self):
        return self.label[2]

    @property
    def jn(self):
        return self.label[3]

    @property
    def s(self):
        return self.jn[0]

    @property
    def l(self):
        return self.jn[1]

    @property
    def i(self):
        if len(self.jn) >= 3:
            return self.jn[2]
        return 0

    @property
    def coupling(self):
        return self.label[4]

    @property
    def coupled_jn(self):
        return _build_coupled(self.label[4], len(self.label[3]))[1]

    @property
    def coupled_n(self):
        return _build_coupled(self.label[4], len(self.label[3]))[0]

    #-------------------------------------------------------------------------
    # Printing
    #-------------------------------------------------------------------------

    def _print_label(self, printer, *args):
        label = [
            'n=%s' %printer._print(self.n),
            'j=%s' %printer._print(self.j),
            'm_j=%s' %printer._print(self.m)]
        for i, ji in enumerate(self.jn):
            if i == 0:
                label.append(
                    'S=%s' % printer._print(ji)
                )
            if i == 1:
                label.append(
                    'L=%s' % printer._print(ji)
                )
            if i == 2:
                label.append(
                    'I=%s' % printer._print(ji)
                )
        return ', '.join(label)

    def _print_label_pretty(self, printer, *args):
        label = [self.n, self.j, self.m]
        for i, ji in enumerate(self.jn):
            if i == 0:
                symb = 'S'
            elif i == 1:
                symb = 'L'
            elif i == 2:
                symb = 'I'
            symb = pretty_symbol(symb)
            symb = prettyForm(symb + '=')
            item = prettyForm(*symb.right(printer._print(ji)))
            label.append(item)
        return self._print_sequence_pretty(label, self._label_separator, printer, *args)

    def _print_label_latex(self, printer, *args):
        label = [self.n, self.j, self.m]
        for i, ji in enumerate(self.jn):
            if i == 0:
                label.append('S=%s' % printer._print(ji) )
            if i == 1:
                label.append('L=%s' % printer._print(ji) )
            if i == 2:
                label.append('I=%s' % printer._print(ji) )
        return self._print_sequence(label, self._label_separator, printer, *args)

    #-------------------------------------------------------------------------
    # _eval_rewrites
    #-------------------------------------------------------------------------

    def _eval_rewrite_as_Jx(self, *args, **options):
        if isinstance(self, Bra):
            return self._rewrite_basis(Jx, AtomJxBraCoupled, **options)
        return self._rewrite_basis(Jx, AtomJxKetCoupled, **options)

    def _eval_rewrite_as_Jy(self, *args, **options):
        if isinstance(self, Bra):
            return self._rewrite_basis(Jy, AtomJyBraCoupled, **options)
        return self._rewrite_basis(Jy, AtomJyKetCoupled, **options)

    def _eval_rewrite_as_Jz(self, *args, **options):
        if isinstance(self, Bra):
            return self._rewrite_basis(Jz, AtomJzBraCoupled, **options)
        return self._rewrite_basis(Jz, AtomJzKetCoupled, **options)

    def _rewrite_basis(self, basis, evect, **options):
        from sympy.physics.quantum.represent import represent
        n = self.n
        j = self.j
        args = self.args[3:]
        if j.is_number:
            if isinstance(self, CoupledSpinState):
                if j == int(j):
                    start = j**2
                else:
                    start = (2*j - 1)*(2*j + 1)/4
            else:
                start = 0
            vect = represent(self, basis=basis, **options)
            result = Add(
                *[vect[start + i] * evect(n, j, j - i, *args) for i in range(2*j + 1)])
            if isinstance(self, CoupledSpinState) and options.get('coupled') is False:
                return uncouple(result)
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
            if isinstance(self, CoupledSpinState):
                test_args = (n, 0, mi, (0, 0))
            else:
                test_args = (0, mi)
            if isinstance(self, Ket):
                angles = represent(
                    self.__class__(*test_args), basis=basis)[0].args[3:6]
            else:
                angles = represent(self.__class__(
                    *test_args), basis=basis)[0].args[0].args[3:6]
            if angles == (0, 0, 0):
                return self
            else:
                state = evect(n, j, mi, *args)
                lt = Rotation.D(j, mi, self.m, *angles)
                return Sum(lt * state, (mi, -j, j))

    #-------------------------------------------------------------------------
    # _eval_innerproducts
    #-------------------------------------------------------------------------

    def _eval_innerproduct_AtomJxBraCoupled(self, bra, **hints):
        result = KroneckerDelta(self.n, bra.n)
        result *= KroneckerDelta(self.j, bra.j)
        result *= KroneckerDelta(self.s, bra.s)
        result *= KroneckerDelta(self.l, bra.l)
        result *= KroneckerDelta(self.i, bra.i)
        if bra.dual_class() is not self.__class__:
            result *= self._represent_JxOp(None)[bra.j - bra.m]
        else:
            result *= KroneckerDelta(self.m, bra.m)
        return result

    def _eval_innerproduct_AtomJyBraCoupled(self, bra, **hints):
        result = KroneckerDelta(self.n, bra.n)
        result *= KroneckerDelta(self.j, bra.j)
        result *= KroneckerDelta(self.s, bra.s)
        result *= KroneckerDelta(self.l, bra.l)
        result *= KroneckerDelta(self.i, bra.i)
        if bra.dual_class() is not self.__class__:
            result *= self._represent_JyOp(None)[bra.j - bra.m]
        else:
            result *= KroneckerDelta(self.m, bra.m)
        return result

    def _eval_innerproduct_AtomJzBraCoupled(self, bra, **hints):
        result = KroneckerDelta(self.n, bra.n)
        result *= KroneckerDelta(self.j, bra.j)
        result *= KroneckerDelta(self.s, bra.s)
        result *= KroneckerDelta(self.l, bra.l)
        result *= KroneckerDelta(self.i, bra.i)
        if bra.dual_class() is not self.__class__:
            result *= self._represent_JzOp(None)[bra.j - bra.m]
        else:
            result *= KroneckerDelta(self.m, bra.m)
        return result


#-----------------------------------------------------------------------------
# Coupled Classes
#-----------------------------------------------------------------------------

#############################################################################
# TODO: Make the associated uncoupled classes.
# Should be a copy paste job.
# TODO: Rename the classes?
# 'AtomJzKetCoupled' seems long and unnecessary.
#############################################################################

class AtomJxKetCoupled(AtomSpinState, Ket):
    """Coupled eigenket of Jx.
    See JzKetCoupled for the usage of coupled spin eigenstates.
    See Also
    ========
    JzKetCoupled: Usage of coupled spin states
    """

    @classmethod
    def dual_class(self):
        return AtomJxBraCoupled

    @classmethod
    def uncoupled_class(self):
        return AtomJxKet

    def _represent_default_basis(self, **options):
        return self._represent_JzOp(None, **options)

    def _represent_JxOp(self, basis, **options):
        return self._represent_coupled_base(**options)

    def _represent_JyOp(self, basis, **options):
        return self._represent_coupled_base(alpha=3*pi/2, **options)

    def _represent_JzOp(self, basis, **options):
        return self._represent_coupled_base(beta=pi/2, **options)

class AtomJxBraCoupled(AtomSpinState, Bra):
    """Coupled eigenbra of Jx.

    See JzKetCoupled for the usage of coupled spin eigenstates.

    See Also
    ========
    JzKetCoupled: Usage of coupled spin states
    """

    @classmethod
    def dual_class(self):
        return AtomJxKetCoupled

    @classmethod
    def uncoupled_class(self):
        return AtomJxBra

class AtomJyKetCoupled(AtomSpinState, Ket):
    """Coupled eigenket of Jy.
    See JzKetCoupled for the usage of coupled spin eigenstates.
    See Also
    ========
    JzKetCoupled: Usage of coupled spin states
    """

    @classmethod
    def dual_class(self):
        return AtomJyBraCoupled

    @classmethod
    def uncoupled_class(self):
        return AtomJyKet

    def _represent_default_basis(self, **options):
        return self._represent_JzOp(None, **options)

    def _represent_JxOp(self, basis, **options):
        return self._represent_coupled_base(gamma=pi/2, **options)

    def _represent_JyOp(self, basis, **options):
        return self._represent_coupled_base(**options)

    def _represent_JzOp(self, basis, **options):
        return self._represent_coupled_base(alpha=3*pi/2, beta=-pi/2, gamma=pi/2, **options)


class AtomJyBraCoupled(AtomSpinState, Bra):
    """Coupled eigenbra of Jz.
    See JzKetCoupled for the usage of coupled spin eigenstates.
    See Also
    ========
    JzKetCoupled: Usage of coupled spin states
    """

    @classmethod
    def dual_class(self):
        return AtomJyKetCoupled

    @classmethod
    def uncoupled_class(self):
        return AtomJyBra


class AtomJzKetCoupled(AtomSpinState, Ket):
    """Coupled eigenket of Jz.

    See JzKetCoupled for the usage of coupled spin eigenstates.

    See Also
    ========
    JzKetCoupled: Usage of coupled spin states
    """

    @classmethod
    def dual_class(self):
        return AtomJzBraCoupled

    @classmethod
    def uncoupled_class(self):
        return AtomJzKet

    def _represent_default_basis(self, **options):
        return self._represent_JzOp(None, **options)

    def _represent_JxOp(self, basis, **options):
        return self._represent_coupled_base(beta=3*pi/2, **options)

    def _represent_JyOp(self, basis, **options):
        return self._represent_coupled_base(alpha=3*pi/2, beta=pi/2, gamma=pi/2, **options)

    def _represent_JzOp(self, basis, **options):
        return self._represent_coupled_base(**options)

class AtomJzBraCoupled(AtomSpinState, Bra):
    """Coupled eigenbra of Jz.
    See JzKetCoupled for the usage of coupled spin eigenstates.
    See Also
    ========
    JzKetCoupled: Usage of coupled spin states
    """

    @classmethod
    def dual_class(self):
        return AtomJzKetCoupled

    @classmethod
    def uncoupled_class(self):
        return AtomJzBra


#-----------------------------------------------------------------------------
# Aliases
#-----------------------------------------------------------------------------


AtomicJxKet = AtomJxKetCoupled
AtomicJyKet = AtomJyKetCoupled
AtomicJzKet = AtomJzKetCoupled

AtomicJxBra = AtomJxBraCoupled
AtomicJyBra = AtomJyBraCoupled
AtomicJzBra = AtomJzBraCoupled
