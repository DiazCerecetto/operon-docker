
from math import e, exp, sqrt, pi, sin, cos, tanh, tanh, sqrt, exp, pi, log
import numpy as np
import os
import pandas as pd
from .config import PATH_FEYNMAN, PATH_FEYNMAN_TEST, PATH_FEYNMAN_TRAIN, PATH_RESULTADOS_FEYNMAN, generar_datos_modelo
# Este archivo contiene las funciones de las ecuaciones de Feynman
# Las mismas se obtuvieron mediante una recorrida por el csv generado por feynman_generate.py
def feynman0(theta):
    return exp(-theta**2/2)/sqrt(2*pi)

def feynman1(sigma, theta):
    return exp(-(theta/sigma)**2/2)/(sqrt(2*pi)*sigma)

def feynman2(sigma, theta, theta1):
    return exp(-((theta-theta1)/sigma)**2/2)/(sqrt(2*pi)*sigma)

def feynman3(x1, x2, y1, y2):
    return sqrt((x2-x1)**2+(y2-y1)**2)

def feynman4(m1, m2, G, x1, x2, y1, y2, z1, z2):
    return G*m1*m2/((x2-x1)**2+(y2-y1)**2+(z2-z1)**2)

def feynman5(m_0, v, c):
    return m_0/sqrt(1-v**2/c**2)

def feynman6(x1, x2, x3, y1, y2, y3):
    return x1*y1+x2*y2+x3*y3

def feynman7(mu, Nn):
    return mu*Nn

def feynman8(q1, q2, epsilon, r):
    return q1*q2*r/(4*pi*epsilon*r**3)

def feynman9(q1, epsilon, r):
    return q1*r/(4*pi*epsilon*r**3)

def feynman10(q2, Ef):
    return q2*Ef

def feynman11(q, Ef, B, v, theta):
    return q*(Ef+B*v*sin(theta))

def feynman12(m, v, u, w):
    return 1/2*m*(v**2+u**2+w**2)

def feynman13(m1, m2, r1, r2, G):
    return G*m1*m2*(1/r2-1/r1)

def feynman14(m, g, z):
    return m*g*z

def feynman15(k_spring, x):
    return 1/2*k_spring*x**2

def feynman16(x, u, c, t):
    return (x-u*t)/sqrt(1-u**2/c**2)

def feynman17(x, c, u, t):
    return (t-u*x/c**2)/sqrt(1-u**2/c**2)

def feynman18(m_0, v, c):
    return m_0*v/sqrt(1-v**2/c**2)

def feynman19(c, v, u):
    return (u+v)/(1+u*v/c**2)

def feynman20(m1, m2, r1, r2):
    return (m1*r1+m2*r2)/(m1+m2)

def feynman21(r, F, theta):
    return r*F*sin(theta)

def feynman22(m, r, v, theta):
    return m*r*v*sin(theta)

def feynman23(m, omega, omega_0, x):
    return 1/2*m*(omega**2+omega_0**2)*1/2*x**2

def feynman24(q, C):
    return q/C

def feynman25(n, theta2):
    return np.arcsin(n*sin(theta2))

def feynman26(d1, d2, n):
    return 1/(1/d1+n/d2)

def feynman27(omega, c):
    return omega/c

def feynman28(x1, x2, theta1, theta2):
    return sqrt(x1**2+x2**2-2*x1*x2*cos(theta1-theta2))

def feynman29(Int_0, theta, n):
    return Int_0*sin(n*theta/2)**2/sin(theta/2)**2

def feynman30(lambd, d, n):
    return np.arcsin(lambd/(n*d))

def feynman31(q, a, epsilon, c):
    return q**2*a**2/(6*pi*epsilon*c**3)

def feynman32(epsilon, c, Ef, r, omega, omega_0):
    return (1/2*epsilon*c*Ef**2)*(8*pi*r**2/3)*(omega**4/(omega**2-omega_0**2)**2)

def feynman33(q, v, B, p):
    return q*v*B/p

def feynman34(c, v, omega_0):
    return omega_0/(1-v/c)

def feynman35(c, v, omega_0):
    return (1+v/c)/sqrt(1-v**2/c**2)*omega_0

def feynman36(omega, h):
    return (h/(2*pi))*omega

def feynman37(I1, I2, delta):
    return I1+I2+2*sqrt(I1*I2)*cos(delta)

def feynman38(m, q, h, epsilon):
    return 4*pi*epsilon*(h/(2*pi))**2/(m*q**2)

def feynman39(pr, V):
    return 3/2*pr*V

def feynman40(gamma, pr, V):
    return 1/(gamma-1)*pr*V

def feynman41(n, T, V, kb):
    return n*kb*T/V

def feynman42(n_0, m, x, T, g, kb):
    return n_0*exp(-m*g*x/(kb*T))

def feynman43(omega, T, h, kb, c):
    return h/(2*pi)*omega**3/(pi**2*c**2*(exp((h/(2*pi))*omega/(kb*T))-1))

def feynman44(mu_drift, q, Volt, d):
    return mu_drift*q*Volt/d

def feynman45(mob, T, kb):
    return mob*kb*T

def feynman46(gamma, kb, A, v):
    return 1/(gamma-1)*kb*v/A

# Se cambia ln por log
def feynman47(n, kb, T, V1, V2):
    return n*kb*T*log(V2/V1,e)

def feynman48(gamma, pr, rho):
    return sqrt(gamma*pr/rho)

def feynman49(m, v, c):
    return m*c**2/sqrt(1-v**2/c**2)

def feynman50(x1, omega, t, alpha):
    return x1*(cos(omega*t)+alpha*cos(omega*t)**2)

def feynman51(kappa, T1, T2, A, d):
    return kappa*(T2-T1)*A/d

def feynman52(Pwr, r):
    return Pwr/(4*pi*r**2)

def feynman53(q, epsilon, r):
    return q/(4*pi*epsilon*r)

def feynman54(epsilon, p_d, theta, r):
    return 1/(4*pi*epsilon)*p_d*cos(theta)/r**2

def feynman55(epsilon, p_d, r, x, y, z):
    return p_d/(4*pi*epsilon)*3*z/r**5*sqrt(x**2+y**2)

def feynman56(epsilon, p_d, theta, r):
    return p_d/(4*pi*epsilon)*3*cos(theta)*sin(theta)/r**3

def feynman57(q, epsilon, d):
    return 3/5*q**2/(4*pi*epsilon*d)

def feynman58(epsilon, Ef):
    return epsilon*Ef**2/2

def feynman59(sigma_den, epsilon, chi):
    return sigma_den/epsilon*1/(1+chi)

def feynman60(q, Ef, m, omega_0, omega):
    return q*Ef/(m*(omega_0**2-omega**2))

def feynman61(n_0, kb, T, theta, p_d, Ef):
    return n_0*(1+p_d*Ef*cos(theta)/(kb*T))

def feynman62(n_rho, p_d, Ef, kb, T):
    return n_rho*p_d**2*Ef/(3*kb*T)

def feynman63(n, alpha, epsilon, Ef):
    return n*alpha/(1-(n*alpha/3))*epsilon*Ef

def feynman64(n, alpha):
    return 1+n*alpha/(1-(n*alpha/3))

def feynman65(epsilon, c, I, r):
    return 1/(4*pi*epsilon*c**2)*2*I/r

def feynman66(rho_c_0, v, c):
    return rho_c_0/sqrt(1-v**2/c**2)

def feynman67(rho_c_0, v, c):
    return rho_c_0*v/sqrt(1-v**2/c**2)

def feynman68(mom, B, theta):
    return -mom*B*cos(theta)

def feynman69(p_d, Ef, theta):
    return -p_d*Ef*cos(theta)

def feynman70(q, epsilon, r, v, c):
    return q/(4*pi*epsilon*r*(1-v/c))

def feynman71(omega, c, d):
    return sqrt(omega**2/c**2-pi**2/d**2)

def feynman72(epsilon, c, Ef):
    return epsilon*c*Ef**2

def feynman73(epsilon, Ef):
    return epsilon*Ef**2

def feynman74(q, v, r):
    return q*v/(2*pi*r)

def feynman75(q, v, r):
    return q*v*r/2

def feynman76(g_, q, B, m):
    return g_*q*B/(2*m)

def feynman77(q, h, m):
    return q*h/(4*pi*m)

def feynman78(g_, h, Jz, mom, B):
    return g_*mom*B*Jz/(h/(2*pi))

def feynman79(n_0, kb, T, mom, B):
    return n_0/(exp(mom*B/(kb*T))+exp(-mom*B/(kb*T)))

def feynman80(n_rho, mom, B, kb, T):
    return n_rho*mom*tanh(mom*B/(kb*T))

def feynman81(mom, H, kb, T, alpha, epsilon, c, M):
    return mom*H/(kb*T)+(mom*alpha)/(epsilon*c**2*kb*T)*M

def feynman82(mom, B, chi):
    return mom*(1+chi)*B

def feynman83(Y, A, d, x):
    return Y*A*x/d

def feynman84(Y, sigma):
    return Y/(2*(1+sigma))

def feynman85(h, omega, kb, T):
    return 1/(exp((h/(2*pi))*omega/(kb*T))-1)

def feynman86(h, omega, kb, T):
    return (h/(2*pi))*omega/(exp((h/(2*pi))*omega/(kb*T))-1)

def feynman87(mom, B, h):
    return 2*mom*B/(h/(2*pi))

def feynman88(E_n, t, h):
    return sin(E_n*t/(h/(2*pi)))**2

def feynman89(p_d, Ef, t, h, omega, omega_0):
    return (p_d*Ef*t/(h/(2*pi)))*sin((omega-omega_0)*t/2)**2/((omega-omega_0)*t/2)**2

def feynman90(mom, Bx, By, Bz):
    return mom*sqrt(Bx**2+By**2+Bz**2)

def feynman91(n, h):
    return n*(h/(2*pi))

def feynman92(E_n, d, k, h):
    return 2*E_n*d**2*k/(h/(2*pi))

def feynman93(I_0, q, Volt, kb, T):
    return I_0*(exp(q*Volt/(kb*T))-1)

def feynman94(U, k, d):
    return 2*U*(1-cos(k*d))

def feynman95(h, E_n, d):
    return (h/(2*pi))**2/(2*E_n*d**2)

def feynman96(alpha, n, d):
    return 2*pi*alpha/(n*d)

def feynman97(beta, alpha, theta):
    return beta*(1+alpha*cos(theta))

def feynman98(m, q, h, n, epsilon):
    return -m*q**4/(2*(4*pi*epsilon)**2*(h/(2*pi))**2)*(1/n**2)

def feynman99(rho_c_0, q, A_vec, m):
    return -rho_c_0*q*A_vec/m

# Ordenar la lista de funciones como si fuese os.path.listdir,
# para que el orden sea el mismo que el de los archivos

lista_funciones = [feynman0, feynman1, feynman10, feynman11, feynman12, feynman13, 
                    feynman14, feynman15, feynman16, feynman17, feynman18, feynman19, 
                    feynman2, feynman20, feynman21, feynman22, feynman23, feynman24, 
                    feynman25, feynman26, feynman27, feynman28, feynman29, feynman3, 
                    feynman30, feynman31, feynman32, feynman33, feynman34, feynman35, 
                    feynman36, feynman37, feynman38, feynman39, feynman4, feynman40, 
                    feynman41, feynman42, feynman43, feynman44, feynman45, feynman46, 
                    feynman47, feynman48, feynman49, feynman5, feynman50, feynman51, 
                    feynman52, feynman53, feynman54, feynman55, feynman56, feynman57, 
                    feynman58, feynman59, feynman6, feynman60, feynman61, feynman62, 
                    feynman63, feynman64, feynman65, feynman66, feynman67, feynman68, 
                    feynman69, feynman7, feynman70, feynman71, feynman72, feynman73, 
                    feynman74, feynman75, feynman76, feynman77, feynman78, feynman79, 
                    feynman8, feynman80, feynman81, feynman82, feynman83, feynman84, 
                    feynman85, feynman86, feynman87, feynman88, feynman89, feynman9, 
                    feynman90, feynman91, feynman92, feynman93, feynman94, feynman95, 
                    feynman96, feynman97, feynman98, feynman99
                ]

# Ordenar la lista de funciones en orden normal para que la función de generación de datos
# pueda acceder a ellas por índice mientras recorre el dataframe obteniendo los intervalos

lista_funciones_normal = [feynman0, feynman1, feynman2, feynman3, feynman4, feynman5,
                            feynman6, feynman7, feynman8, feynman9, feynman10, feynman11,
                            feynman12, feynman13, feynman14, feynman15, feynman16, feynman17,
                            feynman18, feynman19, feynman20, feynman21, feynman22, feynman23,
                            feynman24, feynman25, feynman26, feynman27, feynman28, feynman29,
                            feynman30, feynman31, feynman32, feynman33, feynman34, feynman35,
                            feynman36, feynman37, feynman38, feynman39, feynman40, feynman41,
                            feynman42, feynman43, feynman44, feynman45, feynman46, feynman47,
                            feynman48, feynman49, feynman50, feynman51, feynman52, feynman53,
                            feynman54, feynman55, feynman56, feynman57, feynman58, feynman59,
                            feynman60, feynman61, feynman62, feynman63, feynman64, feynman65,
                            feynman66, feynman67, feynman68, feynman69, feynman70, feynman71,
                            feynman72, feynman73, feynman74, feynman75, feynman76, feynman77,
                            feynman78, feynman79, feynman80, feynman81, feynman82, feynman83,
                            feynman84, feynman85, feynman86, feynman87, feynman88, feynman89,
                            feynman90, feynman91, feynman92, feynman93, feynman94, feynman95,
                            feynman96, feynman97, feynman98, feynman99
                        ]

def generar_ecuaciones_feynman():
    # Para garantizar reproducibilidad
    SEED_TRAIN = 1234
    SEED_TEST = 5678    
    
    # Generar puntos para train y test
    df = pd.read_csv(os.path.join(PATH_FEYNMAN, 'FeynmanEquations.csv'))

    cantidad_train = int(input('Ingrese la cantidad de puntos para train: '))
    cantidad_test = int(input('Ingrese la cantidad de puntos para test: '))
    generar_datos_modelo(cantidad_train, PATH_FEYNMAN_TRAIN, SEED_TRAIN,df, lista_funciones_normal, 'feynman')
    generar_datos_modelo(cantidad_test, PATH_FEYNMAN_TEST, SEED_TEST,df, lista_funciones_normal, 'feynman')
    print('Listo!')  