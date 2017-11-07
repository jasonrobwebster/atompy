import atompy as ap

rb = ap.Atom(name='Rb', spin=0)

print(rb)

g = rb.add_level(
    energy = 0,
    n = 5,
    s = ap.S(1)/2,
    l = 'S',
    j = ap.S(1)/2,
    m = ap.S(1)/2
)

e = rb.add_level(
    energy = 1,
    n = 5,
    s = ap.S(1)/2,
    l = 'P',
    j = ap.S(3)/2,
    m = ap.S(1)/2
)

ap.init_printing()
#ap.pprint(rb.rho)

ap.pprint(rb.master_equation(0, ap.SphericalTensor(0,0)))

#t = ap.SphericalTensor(1, 0)
#stren = ap.transition_strength(g, e, t)

#print(g)
