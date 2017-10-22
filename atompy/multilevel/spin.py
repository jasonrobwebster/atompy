"""Atomic energy level class handler.

Extends the spin class in sympy by expanding the Hilbert space of our atomic
energy levels to incorporate the principle number.

Based off Sympy 1.1.1.
"""

# TODO: Make a functioning multilevel system.
# Should be a copy-paste job from the sympy spin class.
# There must be a smarter way of doing this since all
# I'm adding is the n principle number to the state.
# This is tedious and not needed for my applications,
# but is necessary when taking into account transitions
# between multiple energy levels (e.g. ^1S_1/2 -> ^2S_1/2).
# Would also be nice to have a good way to represent the 
# associated basis (|n,j,m>) with their associated operators.

from __future__ import print_function

from . import (
    Add, Integer, Mul, Sum, Rational, S, symbols, sympify,
    Bra, Ket, State, KroneckerDelta, hbar, ComplexSpace,
    Operator, m_values, Rotation, WignerD, SpinState, 
    TensorProduct, qapply
)


class LevelState(State):
    """Base class for atomic level states."""

    #-------------------------------------------------------------------------
    # Properties
    #-------------------------------------------------------------------------

    @property
    def n(self):
        """The principle number of the state."""
        return self.label[0]

    @property
    def j(self):
        """The total angular momentum of the state."""
        return self.label[1]

    @property
    def m(self):
        """The angular momentum Jz eignestate number."""
        return self.label[2]

    #-------------------------------------------------------------------------
    # Methods
    #-------------------------------------------------------------------------

    def __new__(cls, n, j, m):
        n = sympify(n)
        j = sympify(j)
        m = sympify(m)

        if n.is_number:
            if n != int(n):
                raise ValueError('n must be an integer, got: %s' % n)
            if n < 1:
                raise ValueError('n must be >= 1, got: %s' % n)

        if j.is_number:
            if 2*j != int(2*j):
                raise ValueError(
                    'j must be integer or half-integer, got: %s' % j)
            if j < 0:
                raise ValueError('j must be >= 0, got: %s' % j)

        if m.is_number:
            if 2*m != int(2*m):
                raise ValueError(
                    'm must be integer or half-integer, got: %s' % m)

        if j.is_number and m.is_number:
            if abs(m) > j:
                raise ValueError(
                    'Allowed values for m are -j <= m <= j, got j, m: %s, %s' % (j, m))
            if int(j - m) != j - m:
                raise ValueError(
                    'Both j and m must be integer or half-integer, got j, m: %s, %s' % (j, m))

        return State.__new__(cls, n, j, m)
    
    #-------------------------------------------------------------------------
    # Class Methods
    #-------------------------------------------------------------------------

    @classmethod
    def _eval_hilbert_space(cls, label):
        return ComplexSpace(2*label[1] + 1)

    #-------------------------------------------------------------------------
    # _eval_rewrites
    #-------------------------------------------------------------------------

    def _eval_rewrite_as_Jx(self, *args, **options):
        if isinstance(self, Bra):
            return self._rewrite_basis(Jx, LevelJxBra, **options)
        return self._rewrite_basis(Jx, LevelJxKet, **options)

    def _eval_rewrite_as_Jy(self, *args, **options):
        if isinstance(self, Bra):
            return self._rewrite_basis(Jy, LevelJyBra, **options)
        return self._rewrite_basis(Jy, LevelJyKet, **options)

    def _eval_rewrite_as_Jz(self, *args, **options):
        if isinstance(self, Bra):
            return self._rewrite_basis(Jz, LevelJzBra, **options)
        return self._rewrite_basis(Jz, LevelJzKet, **options)

    def _rewrite_basis(self, basis, evect, **options):
        from sympy.physics.quantum.represent import represent
        n = self.n
        j = self.j
        # TODO: Integrate coupled states
        # see sympy/physics/quantum/spin.py for the way this is implemented
        if j.is_number:
            start = 0
            vect = represent(self, basis=basis, **options)
            result = Add(
                *[vect[start + i] * evect(n, j, j - i) for i in range(2*j + 1)])
            return result
        else:
            i = 0
            mi = symbols('mi')
            # make sure not to introduce a symbol already in the state
            while self.subs(mi, 0) != self:
                i += 1
                mi = symbols('mi%d' % i)
                break
            test_args = (n, 0, mi)
            if isinstance(self, Ket):
                angles = represent(
                    self.__class__(*test_args), basis=basis)[0].args[3:6]
            else:
                angles = represent(self.__class__(
                    *test_args), basis=basis)[0].args[0].args[3:6]
            if angles == (0, 0, 0):
                return self
            else:
                state = evect(n, j, mi)
                lt = Rotation.D(j, mi, self.m, *angles)
                return Sum(lt * state, (mi, -j, j))

class LevelJxKet(LevelState, Ket):
    """
    A Ket defining an atomic energy level.
    """

    @classmethod
    def dual_class(cls):
        return LevelJxBra

class LevelJxBra(LevelState, Bra):
    """
    A Bra defining an atomic energy level.
    """

    @classmethod
    def dual_class(cls):
        return LevelJxKet

class LevelJyKet(LevelState, Ket):
    """
    A Ket defining an atomic energy level.
    """

    @classmethod
    def dual_class(cls):
        return LevelJyBra

class LevelJyBra(LevelState, Bra):
    """
    A Bra defining an atomic energy level.
    """

    @classmethod
    def dual_class(cls):
        return LevelJyKet

class LevelJzKet(LevelState, Ket):
    """
    A Ket defining an atomic energy level.
    """

    @classmethod
    def dual_class(cls):
        return LevelJzBra

class LevelJzBra(LevelState, Bra):
    """
    A Bra defining an atomic energy level.
    """

    @classmethod
    def dual_class(cls):
        return LevelJzKet
