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

S, D, SS, DD, SD, DS = ap.symbols('S, D, SS, DD, SD, DS')

s_ket = ap.JzKet(0,0)
d_ket = ap.JzKet(2,0)

rho = DD * (d_ket * d_ket.dual)
rho += DS * (d_ket * s_ket.dual)
rho += SD * (s_ket * d_ket.dual)
rho += SS * (s_ket * s_ket.dual)

print(rho)

t = ap.lindblad_superop(s_ket * d_ket.dual, rho)
print(t)

print(e)

#t = ap.SphericalTensor(1, 0)
#stren = ap.transition_strength(g, e, t)

#print(g)
