import atompy as ap

n, j, m = ap.symbols('n, j, m')
l_ket = ap.LevelJzKet(10, 1, 0)
j_ket = ap.JzKet(1,0)
n_ket = ap.Ket(1)

ap.init_printing()

test = l_ket.rewrite('Jx')
print(test)

test1 = ap.qapply(ap.Jz * l_ket)
test2 = ap.qapply(ap.Jz * j_ket)
print(test1, test2)

test3 = j_ket.dual * l_ket
ap.pprint(ap.represent(ap.Jz, basis=ap.Jz))
