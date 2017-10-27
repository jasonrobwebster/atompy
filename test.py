import atompy as ap

rb = ap.Atom(name='H')

print(rb)

g = rb.add_level(
    energy = 0,
    n = 1,
    s = ap.S(1)/2,
    l = 'S',
    j = ap.S(1)/2,
    m = ap.S(1)/2
)

e = rb.add_level(
    energy = 1,
    n = 2,
    s = ap.S(1)/2,
    l = 'P',
    j = ap.S(1)/2,
    m = ap.S(1)/2
)

t = ap.SphericalTensor(1, 0)
stren = ap.transition_strength(g, e, t)

ap.pprint(stren**2)
ap.pprint(dbl)
