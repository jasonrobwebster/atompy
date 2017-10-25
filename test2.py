import atompy as ap
from atompy.multilevel.coupled import AtomJzKetCoupled
from sympy.physics.quantum.spin import uncouple

a = AtomJzKetCoupled(1, ap.S(1)/2, ap.S(1)/2, (ap.S(1)/2, 0))
print(a)

print((a.dual * a).doit())