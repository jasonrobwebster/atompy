import atompy as ap

rb = ap.Atom(name='Rb', spin=ap.S(3)/2)

print(rb)

g = rb.add_level(
    energy = 0,
    n = 5,
    s = ap.S(1)/2,
    l = 'S',
    j = ap.S(1)/2,
    f = 1,
    m = 0
)

e = rb.add_m_sublevels(
    energy = 1,
    n = 5,
    s = ap.S(1)/2,
    l = 'D',
    j = ap.S(3)/2,
    f = 2
)

print(e)

#t = ap.SphericalTensor(1, 0)
#stren = ap.transition_strength(g, e, t)

#print(g)
