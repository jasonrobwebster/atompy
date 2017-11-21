import atompy as ap
from sympy import dsolve, solve

rb = ap.Atom(name='Rb', spin=0)

print(rb)

g = rb.add_level(
    energy = 0,
    n = 5,
    s = ap.S(1)/2,
    l = 'S',
    j = ap.S(1)/2,
    m = ap.S(1)/2,
    label = 'S'
)

e = rb.add_level(
    energy = 1,
    n = 5,
    s = ap.S(1)/2,
    l = 'P',
    j = ap.S(3)/2,
    m = ap.S(1)/2,
    label = 'P'
)

e = rb.add_level(
    energy = 2,
    n = 5,
    s = ap.S(1)/2,
    l = 'D',
    j = ap.S(5)/2,
    m = ap.S(1)/2,
    label = 'D'
)

ap.init_printing()
#ap.pprint(rb.rho)

master_eq = rb.master_equation(0, ap.SphericalTensor(0,0), steady=True)
for i in master_eq:
    ap.pprint(i)

ap.pprint(solve(master_eq, rb.rho_functions))

#t = ap.SphericalTensor(1, 0)
#stren = ap.transition_strength(g, e, t)

#print(g)
