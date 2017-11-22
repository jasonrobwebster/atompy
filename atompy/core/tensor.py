"""Spherical tensor class"""

from . import Operator, sympify, Dummy, CG, clebsch_gordan, Sum

__all__=[
    'SphericalTensor',
    'couple_tensor'
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

def couple_tensor(a, b):
    """Couples two spherical tensors together to create a third tensor.
    Equation derived from [1]

    Parameters
    ==========

    a, b: SphericalTensors
        The spherical tensors that will be coupled.

    return_q : Boolean, Optional
        Whether to return the q_vals with the associated tensor sum.

    Examples
    ========

    References
    ==========

    .. [1] Steck, D.A., 2007. Quantum and atom optics. p. 360-375
        http://atomoptics-nas.uoregon.edu/~dsteck/teaching/quantum-optics/quantum-optics-notes.pdf
    """

    # TODO: Get spherical tensors to couple using the TensorProduct function.
    # Similar to how JzKets can coupled using the TensorProduct function along
    # with the couple function.
    from . import CG, clebsch_gordan

    assert isinstance(a, SphericalTensor)
    assert isinstance(b, SphericalTensor)

    # Do A1 in [2]
    k1 = b.k
    m1 = b.q
    k2 = a.k
    m2 = a.q

    if k1.is_number and k2.is_number and m1.is_number and m2.is_number:
        couple_tensor = 0
        for k in range(-abs(k1-k2), k1+k2+1):
            for m in range(-k, k+1):
                couple_tensor += clebsch_gordan(k1, k2, k, m1, m2, m)*SphericalTensor(k, m)
    else:
        k, m = Dummy('k, m')
        couple_tensor = Sum(CG(k1 ,k2, k, m1, m2, m)*SphericalTensor(k, m), (k, abs(k1-k2), k1+k2), (q, -k, k))

    return couple_tensor
        