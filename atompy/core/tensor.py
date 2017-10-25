"""Spherical tensor class"""

from . import Operator, sympify

__all__=[
    'SphericalTensor'
]

class SphericalTensor(Operator):
    """Returns an instance of a spherical tensor of rank k, subscript q [1].

    Parameters
    ==========

    k : Number, Symbol
        The rank of the spherical tensor.

    q : Number, Symbol
        The subscript of the spherical tensor.

    Examples
    ========



    References
    ==========

    .. [1] Steck, D.A., 2007. Quantum and atom optics. p. 324-329
        http://atomoptics-nas.uoregon.edu/~dsteck/teaching/quantum-optics/quantum-optics-notes.pdf
    """
    # TODO: Flesh out functionality of the spherical tensor

    def __new__(cls, k, q):
        k = sympify(k)
        q = sympify(q)

        if k.is_number:
            if not k.is_Integer:
                raise ValueError('k must be an integer, got: %s' % k)
            if k < 0:
                raise ValueError('k must be >= 0, got: %s' % k)

        if q.is_number:
            if not q.is_Integer:
                raise ValueError('q must be an integer, got: %s' % q)

        if k.is_number and q.is_number:
            if q < -k or q > k:
                raise ValueError('q must be in -k <= q <= k, got k, q: %s, %s' % (k, q))

        return Operator.__new__(cls, k, q)

    @property
    def k(self):
        return self.args[0]
    
    @property
    def q(self):
        return self.args[1]
