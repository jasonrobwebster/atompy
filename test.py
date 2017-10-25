import atompy as ap

rb = ap.Atom(
    #name='Rb',
    spin=0,
    mu=0,
    mass='m'
)

print(rb)

g = rb.add_level(
    E = 0,
    n = 1,
    s = ap.S(1)/2,
    l = 'S',
    j = ap.S(1)/2,
    m_j = -ap.S(1)/2
)

e = rb.add_level(
    E = 1,
    n = 2,
    s = ap.S(1)/2,
    l = 'P',
    j = ap.S(1)/2,
    m_j = ap.S(1)/2
)

t = ap.SphericalTensor(1, -1)
stren, dbl = ap.transition_strength(g, e, t)

ap.pprint(type(dbl))
ap.pprint(dbl)
