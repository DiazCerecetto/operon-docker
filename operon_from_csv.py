from operon.sklearn import SymbolicRegressor
import optuna
import pandas as pd
import numpy as np
import os
import operon
   
from math import e, exp, sqrt, pi, sin, cos, tanh, tanh, sqrt, exp, pi, log
import os
import numpy as np
import pandas as pd

def feynmann0(theta):
    return exp(-theta**2/2)/sqrt(2*pi)

def feynmann1(sigma, theta):
    return exp(-(theta/sigma)**2/2)/(sqrt(2*pi)*sigma)

def feynmann2(sigma, theta, theta1):
    return exp(-((theta-theta1)/sigma)**2/2)/(sqrt(2*pi)*sigma)

def feynmann3(x1, x2, y1, y2):
    return sqrt((x2-x1)**2+(y2-y1)**2)

def feynmann4(m1, m2, G, x1, x2, y1, y2, z1, z2):
    return G*m1*m2/((x2-x1)**2+(y2-y1)**2+(z2-z1)**2)

def feynmann5(m_0, v, c):
    return m_0/sqrt(1-v**2/c**2)

def feynmann6(x1, x2, x3, y1, y2, y3):
    return x1*y1+x2*y2+x3*y3

def feynmann7(mu, Nn):
    return mu*Nn

def feynmann8(q1, q2, epsilon, r):
    return q1*q2*r/(4*pi*epsilon*r**3)

def feynmann9(q1, epsilon, r):
    return q1*r/(4*pi*epsilon*r**3)

def feynmann10(q2, Ef):
    return q2*Ef

def feynmann11(q, Ef, B, v, theta):
    return q*(Ef+B*v*sin(theta))

def feynmann12(m, v, u, w):
    return 1/2*m*(v**2+u**2+w**2)

def feynmann13(m1, m2, r1, r2, G):
    return G*m1*m2*(1/r2-1/r1)

def feynmann14(m, g, z):
    return m*g*z

def feynmann15(k_spring, x):
    return 1/2*k_spring*x**2

def feynmann16(x, u, c, t):
    return (x-u*t)/sqrt(1-u**2/c**2)

def feynmann17(x, c, u, t):
    return (t-u*x/c**2)/sqrt(1-u**2/c**2)

def feynmann18(m_0, v, c):
    return m_0*v/sqrt(1-v**2/c**2)

def feynmann19(c, v, u):
    return (u+v)/(1+u*v/c**2)

def feynmann20(m1, m2, r1, r2):
    return (m1*r1+m2*r2)/(m1+m2)

def feynmann21(r, F, theta):
    return r*F*sin(theta)

def feynmann22(m, r, v, theta):
    return m*r*v*sin(theta)

def feynmann23(m, omega, omega_0, x):
    return 1/2*m*(omega**2+omega_0**2)*1/2*x**2

def feynmann24(q, C):
    return q/C

def feynmann25(n, theta2):
    return np.arcsin(n*sin(theta2))

def feynmann26(d1, d2, n):
    return 1/(1/d1+n/d2)

def feynmann27(omega, c):
    return omega/c

def feynmann28(x1, x2, theta1, theta2):
    return sqrt(x1**2+x2**2-2*x1*x2*cos(theta1-theta2))

def feynmann29(Int_0, theta, n):
    return Int_0*sin(n*theta/2)**2/sin(theta/2)**2

def feynmann30(lambd, d, n):
    return np.arcsin(lambd/(n*d))

def feynmann31(q, a, epsilon, c):
    return q**2*a**2/(6*pi*epsilon*c**3)

def feynmann32(epsilon, c, Ef, r, omega, omega_0):
    return (1/2*epsilon*c*Ef**2)*(8*pi*r**2/3)*(omega**4/(omega**2-omega_0**2)**2)

def feynmann33(q, v, B, p):
    return q*v*B/p

def feynmann34(c, v, omega_0):
    return omega_0/(1-v/c)

def feynmann35(c, v, omega_0):
    return (1+v/c)/sqrt(1-v**2/c**2)*omega_0

def feynmann36(omega, h):
    return (h/(2*pi))*omega

def feynmann37(I1, I2, delta):
    return I1+I2+2*sqrt(I1*I2)*cos(delta)

def feynmann38(m, q, h, epsilon):
    return 4*pi*epsilon*(h/(2*pi))**2/(m*q**2)

def feynmann39(pr, V):
    return 3/2*pr*V

def feynmann40(gamma, pr, V):
    return 1/(gamma-1)*pr*V

def feynmann41(n, T, V, kb):
    return n*kb*T/V

def feynmann42(n_0, m, x, T, g, kb):
    return n_0*exp(-m*g*x/(kb*T))

def feynmann43(omega, T, h, kb, c):
    return h/(2*pi)*omega**3/(pi**2*c**2*(exp((h/(2*pi))*omega/(kb*T))-1))

def feynmann44(mu_drift, q, Volt, d):
    return mu_drift*q*Volt/d

def feynmann45(mob, T, kb):
    return mob*kb*T

def feynmann46(gamma, kb, A, v):
    return 1/(gamma-1)*kb*v/A

def feynmann47(n, kb, T, V1, V2):
    return n*kb*T*log(V2/V1,e)

def feynmann48(gamma, pr, rho):
    return sqrt(gamma*pr/rho)

def feynmann49(m, v, c):
    return m*c**2/sqrt(1-v**2/c**2)

def feynmann50(x1, omega, t, alpha):
    return x1*(cos(omega*t)+alpha*cos(omega*t)**2)

def feynmann51(kappa, T1, T2, A, d):
    return kappa*(T2-T1)*A/d

def feynmann52(Pwr, r):
    return Pwr/(4*pi*r**2)

def feynmann53(q, epsilon, r):
    return q/(4*pi*epsilon*r)

def feynmann54(epsilon, p_d, theta, r):
    return 1/(4*pi*epsilon)*p_d*cos(theta)/r**2

def feynmann55(epsilon, p_d, r, x, y, z):
    return p_d/(4*pi*epsilon)*3*z/r**5*sqrt(x**2+y**2)

def feynmann56(epsilon, p_d, theta, r):
    return p_d/(4*pi*epsilon)*3*cos(theta)*sin(theta)/r**3

def feynmann57(q, epsilon, d):
    return 3/5*q**2/(4*pi*epsilon*d)

def feynmann58(epsilon, Ef):
    return epsilon*Ef**2/2

def feynmann59(sigma_den, epsilon, chi):
    return sigma_den/epsilon*1/(1+chi)

def feynmann60(q, Ef, m, omega_0, omega):
    return q*Ef/(m*(omega_0**2-omega**2))

def feynmann61(n_0, kb, T, theta, p_d, Ef):
    return n_0*(1+p_d*Ef*cos(theta)/(kb*T))

def feynmann62(n_rho, p_d, Ef, kb, T):
    return n_rho*p_d**2*Ef/(3*kb*T)

def feynmann63(n, alpha, epsilon, Ef):
    return n*alpha/(1-(n*alpha/3))*epsilon*Ef

def feynmann64(n, alpha):
    return 1+n*alpha/(1-(n*alpha/3))

def feynmann65(epsilon, c, I, r):
    return 1/(4*pi*epsilon*c**2)*2*I/r

def feynmann66(rho_c_0, v, c):
    return rho_c_0/sqrt(1-v**2/c**2)

def feynmann67(rho_c_0, v, c):
    return rho_c_0*v/sqrt(1-v**2/c**2)

def feynmann68(mom, B, theta):
    return -mom*B*cos(theta)

def feynmann69(p_d, Ef, theta):
    return -p_d*Ef*cos(theta)

def feynmann70(q, epsilon, r, v, c):
    return q/(4*pi*epsilon*r*(1-v/c))

def feynmann71(omega, c, d):
    return sqrt(omega**2/c**2-pi**2/d**2)

def feynmann72(epsilon, c, Ef):
    return epsilon*c*Ef**2

def feynmann73(epsilon, Ef):
    return epsilon*Ef**2

def feynmann74(q, v, r):
    return q*v/(2*pi*r)

def feynmann75(q, v, r):
    return q*v*r/2

def feynmann76(g_, q, B, m):
    return g_*q*B/(2*m)

def feynmann77(q, h, m):
    return q*h/(4*pi*m)

def feynmann78(g_, h, Jz, mom, B):
    return g_*mom*B*Jz/(h/(2*pi))

def feynmann79(n_0, kb, T, mom, B):
    return n_0/(exp(mom*B/(kb*T))+exp(-mom*B/(kb*T)))

def feynmann80(n_rho, mom, B, kb, T):
    return n_rho*mom*tanh(mom*B/(kb*T))

def feynmann81(mom, H, kb, T, alpha, epsilon, c, M):
    return mom*H/(kb*T)+(mom*alpha)/(epsilon*c**2*kb*T)*M

def feynmann82(mom, B, chi):
    return mom*(1+chi)*B

def feynmann83(Y, A, d, x):
    return Y*A*x/d

def feynmann84(Y, sigma):
    return Y/(2*(1+sigma))

def feynmann85(h, omega, kb, T):
    return 1/(exp((h/(2*pi))*omega/(kb*T))-1)

def feynmann86(h, omega, kb, T):
    return (h/(2*pi))*omega/(exp((h/(2*pi))*omega/(kb*T))-1)

def feynmann87(mom, B, h):
    return 2*mom*B/(h/(2*pi))

def feynmann88(E_n, t, h):
    return sin(E_n*t/(h/(2*pi)))**2

def feynmann89(p_d, Ef, t, h, omega, omega_0):
    return (p_d*Ef*t/(h/(2*pi)))*sin((omega-omega_0)*t/2)**2/((omega-omega_0)*t/2)**2

def feynmann90(mom, Bx, By, Bz):
    return mom*sqrt(Bx**2+By**2+Bz**2)

def feynmann91(n, h):
    return n*(h/(2*pi))

def feynmann92(E_n, d, k, h):
    return 2*E_n*d**2*k/(h/(2*pi))

def feynmann93(I_0, q, Volt, kb, T):
    return I_0*(exp(q*Volt/(kb*T))-1)

def feynmann94(U, k, d):
    return 2*U*(1-cos(k*d))

def feynmann95(h, E_n, d):
    return (h/(2*pi))**2/(2*E_n*d**2)

def feynmann96(alpha, n, d):
    return 2*pi*alpha/(n*d)

def feynmann97(beta, alpha, theta):
    return beta*(1+alpha*cos(theta))

def feynmann98(m, q, h, n, epsilon):
    return -m*q**4/(2*(4*pi*epsilon)**2*(h/(2*pi))**2)*(1/n**2)

def feynmann99(rho_c_0, q, A_vec, m):
    return -rho_c_0*q*A_vec/m
     
print("Número de hilos: ", os.environ['OMP_NUM_THREADS'] if 'OMP_NUM_THREADS' in os.environ else 1)
num_threads = int(os.environ['OMP_NUM_THREADS']) if 'OMP_NUM_THREADS' in os.environ else 1
rng = np.random.default_rng(1234)
default_params = {
        'allowed_symbols':"add,sub,mul,div,constant,variable,sin,cos",
        'offspring_generator': 'basic',
        'initialization_method': 'btc',
        'n_threads': 8,
        'objectives':  ['r2', 'length'],
        'epsilon': 1e-5,
        'random_state': rng,
        'reinserter': 'keep-best',
        'max_evaluations': int(1e6),
        'tournament_size': 3,
        'pool_size': None,
        'time_limit': 600
        }


def predecir_desde_csv(est, df, nombre):
    # Leer el archivo csv
    # 30 iteraciones
    
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    print(X.shape, y.shape)
    for i in range(30):
        est = est.fit(X, y)
    m = est.model_
    #print("***************************")
    #print("Original: " + nombre)
    #print("Modelo: " + str(m))
    #print("***************************")
    return est, m

def main():
    print("****************************")
    print("* Operon desde archivo csv *")
    print("****************************")
    
    predicciones = []
    # Leer los archivos csv, en el directorio Feynman
    # para cada archivo CSV, predecir y guardar el modelo
    # Luego, comparar los modelos obtenidos con los modelos reales
    for file in os.listdir("Feynman"):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join("Feynman", file))
            est = SymbolicRegressor(**default_params)
            est, m = predecir_desde_csv(est, df, file)
            predicciones.append(est)
    print("***************************")
    lista_funciones = [feynmann0, feynmann1, feynmann2, feynmann3, feynmann4, feynmann5, feynmann6, feynmann7, feynmann8,
                       feynmann9, feynmann10, feynmann11, feynmann12, feynmann13, feynmann14, feynmann15, feynmann16,
                       feynmann17, feynmann18, feynmann19, feynmann20, feynmann21, feynmann22, feynmann23, feynmann24,
                       feynmann25, feynmann26, feynmann27, feynmann28, feynmann29, feynmann30, feynmann31, feynmann32,
                       feynmann33, feynmann34, feynmann35, feynmann36, feynmann37, feynmann38, feynmann39, feynmann40,
                       feynmann41, feynmann42, feynmann43, feynmann44, feynmann45, feynmann46, feynmann47, feynmann48,
                       feynmann49, feynmann50, feynmann51, feynmann52, feynmann53, feynmann54, feynmann55, feynmann56,
                       feynmann57, feynmann58, feynmann59, feynmann60, feynmann61, feynmann62, feynmann63, feynmann64,
                       feynmann65, feynmann66, feynmann67, feynmann68, feynmann69, feynmann70, feynmann71, feynmann72,
                       feynmann73, feynmann74, feynmann75, feynmann76, feynmann77, feynmann78, feynmann79, feynmann80,
                       feynmann81, feynmann82, feynmann83, feynmann84, feynmann85, feynmann86, feynmann87, feynmann88,
                       feynmann89, feynmann90, feynmann91, feynmann92, feynmann93, feynmann94, feynmann95, feynmann96,
                       feynmann97, feynmann98, feynmann99]
    # Comparar los modelos obtenidos con los modelos reales
    for i, f in enumerate(lista_funciones):
        print("***************************")
        print("Original: " + f.__name__)
        print("Modelo: ")
        print(predicciones[i].get_model_string())
        print("***************************")
        
        
    print("***************************")
    


if __name__ == '__main__':
    #capturar la salida estandar y ponerla en un txt salida.txt
    
    
    # comenzar a capturar la salida estandar
    ##import sys
    #sys.stdout = open('salida.txt', 'w')
    main()
    # terminar de capturar la salida estandar
    #sys.stdout.close()
  