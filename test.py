import atompy as ap
import atompy.multilevel.spin as a
from sympy.physics.quantum.spin import JzKet, Jx
from sympy.physics.quantum import represent

t = a.AtomJzKet(3,1,2,1,0)
j = JzKet(1, 0)

print(t)
ap.pprint(t.rewrite('Jx'))
ap.pprint(represent(a.AtomJx, basis=a.AtomJz))
