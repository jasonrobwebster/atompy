"""Atomic energy level class handler"""

from . import sympify, JzKet, KetBase, BraBase, Ket, Bra, SpinState, State, ComplexSpace

class LevelState(SpinState):
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

    @classmethod
    def _eval_hilbert_space(cls, label):
        return ComplexSpace(2*label[1] + 1)