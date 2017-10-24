import atompy as ap

rb = ap.Atom(
    #name='Rb',
    spin=0,
    mu=0,
    mass='m'
)

print(rb)

rb.add_level(
    E = 'E_0',
    n = 1,
    s = ap.S(1)/2,
    l = 'S',
    j = ap.S(1)/2,
    m_j = ap.S(1)/2
)

rb.add_level(
    E = 'E_1',
    n = 1,
    s = ap.S(1)/2,
    l = 'S',
    j = ap.S(1)/2,
    m_j = -ap.S(1)/2
)

print(rb)

ap.pprint(rb.hamiltonian)
