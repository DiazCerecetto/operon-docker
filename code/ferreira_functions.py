
k1 = 3.7
k2 = 4
k3 = 3


# Ca = q * Cain / (q + k1)
def ferreira1(q,Cain):
    return q * Cain / (q + k1)

# Cb = k1 * q * Cain / ((q + k1) * (q + k2))
def ferreira2(q,Cain):
    return k1 * q * Cain / ((q + k1) * (q + k2))

# Cc = k2 * k1 * q * Cain / ((q + k1) * (q + k2) * (q + k3))
def ferreira3(q,Cain):
    return k2 * k1 * q * Cain / ((q + k1) * (q + k2) * (q + k3))

# Cd = k3 * k2 * k1 * q * Cain / ((q + k1) * (q + k2) * (q + k3) * (q + k3))
def ferreira4(q,Cain):
    return k3 * k2 * k1 * Cain / ((q + k1) * (q + k2) * (q + k3) * q)

lista_funciones = [ferreira1,ferreira2,ferreira3,ferreira4]
    