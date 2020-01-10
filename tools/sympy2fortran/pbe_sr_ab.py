# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 17:40:26 2019

@author: Standard
"""

from sympy import Symbol, Function, pi, sympify, log, exp, sqrt, erf
from sympy2fortran import create_Routine_from_Function
from time import time

t_start=time()

# The maximum requested derivative (Hint: the more derivatives requested the more time it needs for creating files and the more code will be produced)
max_deriv=2

# Source file to be created
file=open('pbe_sr.F', 'w')

# Variables
x=Symbol('x')
y=Symbol('y')
z=Symbol('z')

# Density and its derivative (we need Rho as function for sympy and as symbol for the code)
zeta=Symbol('zeta')
rho=Symbol('rho')
drho=Symbol('drho')
Rs=Symbol('Rs')
rhoa=Symbol('rhoa')
rhob=Symbol('rhob')
drhoa=Symbol('drhoa')
drhob=Symbol('drhob')

# Scaling factors
sx=Symbol('sx')
sc=Symbol('sc')

zero=sympify(0)
one=sympify(1)
two=sympify(2)
three=sympify(3)
four=sympify(4)
five=sympify(5)
six=sympify(6)
eight=sympify(8)
half=sympify('1/2')
third=sympify('1/3')

# Correlation energy density (e.g. from PW92)
Ec=Function('Ec')

# mu dependent Correlation energy density
Ecmu=Function('Ecmu')

# Range-separation parameter
mu=Symbol('mu')

# Functional specific constants
gamma=Symbol('gamma')
gamma_=sympify(0.031091)
beta_PBE=Symbol('beta_PBE')
beta_PBE_=sympify(0.066725)
alpha_C=Symbol('alpha_C')
alpha_C_=sympify(2.78)
kappa=Symbol('kappa')
kappa_=sympify(0.840)
alpha_X=Symbol('alpha_X')
alpha_X_=sympify(19.0)
b_PBE=Symbol('b_PBE')
b_PBE_=sympify(0.21951)
c=Symbol('c')
c_=sympify((one+six*sqrt(three))**half)

# Correlation energy density (e.g. from PW92)
class Ex(Function):
    nargs=1
    name='Ex'
    
    @classmethod
    def eval(cls, rho):
        return Function('Ex')(rho)

subs_ex=[[Ex, 'calc_Ex', rho]]

# Correlation energy density (e.g. from PW92)
class Ec(Function):
    nargs=1
    name='Ec'
    
    @classmethod
    def eval(cls, rho):
        return Function('Ec')(rho)

subs_ec=[[Ec, 'calc_Ec', rho]]

# Correlation energy density (e.g. from PW92)
class Ecab(Function):
    nargs=2
    name='Ecab'
    
    @classmethod
    def eval(cls, rho, zeta):
        return Function('Ecab')(rho, zeta)

subs_ecab=[[Ecab, 'calc_Ecab', rho, zeta]]

# Correlation energy density (e.g. from PW92)
class Ecmu(Function):
    nargs=1
    name='Ecmu'
    
    @classmethod
    def eval(cls, rho):
        return Function('Ecmu')(rho)

subs_ecmu=[[Ecmu, 'calc_Ecmu', rho]]

# Correlation energy density (e.g. from PW92)
class Ecmuab(Function):
    nargs=2
    name='Ecmuab'
    
    @classmethod
    def eval(cls, rho, zeta):
        return Function('Ecmuab')(rho, zeta)

subs_ecmuab=[[Ecmuab, 'calc_Ecmuab', rho, zeta]]

# helper routines for exchange part

class kf(Function):
    nargs=1
    
    @classmethod
    def eval(cls, rho):
        return (three*pi**two*rho)**third

class s_(Function):
    nargs=2
    name='s'
    
    @classmethod
    def eval(cls, rho, drho):
        return drho/two/kf(rho)/rho

class s(Function):
    nargs=2
    name='s'
    
    @classmethod
    def eval(cls, rho, drho):
        return Function('s')(rho, drho)

subs_s=[[s, 'calc_s', rho, drho]]
create_Routine_from_Function(s_, file, max_deriv, [], [], [], 'calc_s', [], rho, drho)

class c_1_(Function):
    nargs=1
    name='c_1'
    
    @classmethod
    def eval(cls, mu):
        nu=mu/c
        return one+sympify(22)*nu**two+sympify(144)*nu**four

class c_1(Function):
    nargs=1
    name='a2'
    
    @classmethod
    def eval(cls, mu):
        return Function('c_1')(mu)

subs_c1=[[c_1, 'calc_c_1', mu]]
create_Routine_from_Function(c_1_, file, 0, [], [], [], 'calc_c_1', [], mu)

class c_2_(Function):
    nargs=1
    name='c_2'
    
    @classmethod
    def eval(cls, mu):
        nu=mu/c
        return two*nu**two*(-sympify(7)+sympify(72)*nu**two)

class c_2(Function):
    nargs=1
    name='c_2'
    
    @classmethod
    def eval(cls, rs):
        return Function('c_2')(rs)

subs_c2=[[c_2, 'calc_c_2', mu]]
create_Routine_from_Function(c_2_, file, 0, [], [], [], 'calc_c_2', [], mu)

class c_3_(Function):
    nargs=1
    name='c_3'
    
    @classmethod
    def eval(cls, mu):
        nu=mu/c
        return -sympify(864)*nu**four*(two*nu**two-one)

class c_3(Function):
    nargs=1
    name='c_3'
    
    @classmethod
    def eval(cls, mu):
        return Function('c_3')(mu)

subs_c3=[[c_3, 'calc_c_3', mu]]
create_Routine_from_Function(c_3_, file, 0, [], [], [], 'calc_c_3', [], mu)

class c_4_(Function):
    nargs=1
    name='c_4'
    
    @classmethod
    def eval(cls, mu):
        nu=mu/c
        return nu**two*(-three-sympify(24)*nu**two+sympify(32)**four+eight*nu*sqrt(pi)*erf(half/nu))

class c_4(Function):
    nargs=1
    name='c_4'
    
    @classmethod
    def eval(cls, mu):
        return Function('c_4')(mu)

subs_c4=[[c_4, 'calc_c_4', mu]]
create_Routine_from_Function(c_4_, file, 0, [], [], [], 'calc_c_4', [], mu)

class b_T_(Function):
    nargs=1
    name='b_T'
    
    @classmethod
    def eval(cls, mu):
        return (c_2(mu)*exp(one/four/mu**two)-c_1(mu))/(c_3(mu)+sympify(54)*c_4(mu)*exp(one/four/mu**two))

class b_T(Function):
    nargs=1
    name='b_T'
    
    @classmethod
    def eval(cls, mu):
        return Function('b_T')(mu)

subs_bT=[[b_T, 'calc_b_T', mu]]
create_Routine_from_Function(b_T_, file, 0, subs_c1+subs_c2+subs_c3+subs_c4, [], [], 'calc_b_T', [], mu)

class b_(Function):
    nargs=1
    name='b'
    
    @classmethod
    def eval(cls, mu):
        return b_PBE*b_T(mu)/sympify('7/81')*exp(-alpha_X*mu**two)

class b(Function):
    nargs=1
    name='b'
    
    @classmethod
    def eval(cls, mu):
        return Function('b')(mu)

subs_b=[[b, 'calc_b', mu]]
create_Routine_from_Function(b_, file, 0, subs_bT, [], [], 'calc_b', [], mu)

class Fx_(Function):
    nargs=2
    name='Fx'
    
    @classmethod
    def eval(cls, rho, drho):
        return one+kappa-kappa*(one+b(mu)*s(rho, drho)**two/kappa)

class Fx(Function):
    nargs=2
    name='a2'
    
    @classmethod
    def eval(cls, rho, drho):
        return Function('Fx')(rho, drho)

subs_Fx=[[Fx, 'calc_Fx', rho, drho]]
create_Routine_from_Function(Fx_, file, max_deriv, subs_b+subs_s, [], [], 'calc_Fx', [mu,], rho, drho)

# several helper routines for the correlation part

class ks(Function):
    nargs=1
    
    @classmethod
    def eval(cls, rho):
        return sqrt(four*kf(rho)/pi)

class phi_(Function):
    nargs=1
    name='phi'
    
    @classmethod
    def eval(cls, zeta):
        return ((one+zeta)**(two*third)+(one-zeta)**(two*third))/two

class phi(Function):
    nargs=1
    name='phi'
    
    @classmethod
    def eval(cls, zeta):
        return Function('phi')(zeta)

subs_phi=[[phi, 'calc_phi', zeta]]
create_Routine_from_Function(phi_, file, max_deriv, [], [], [], 'calc_phi', [], zeta)

class t_ab_(Function):
    nargs=3
    name='t_ab'
    
    @classmethod
    def eval(cls, rho, zeta, drho):
        return drho/two/rho/phi(zeta)/ks(rho)

class t_ab(Function):
    nargs=3
    name='tab'
    
    @classmethod
    def eval(cls, rho, zeta, drho):
        return Function('t_ab')(rho, zeta, drho)

subs_tab=[[t_ab, 'calc_t_ab', rho, zeta, drho]]
create_Routine_from_Function(t_ab_, file, max_deriv, subs_phi, [], [], 'calc_t_ab', [], rho, zeta, drho)

class beta_ab_(Function):
    nargs=2
    name='beta_ab'
    
    @classmethod
    def eval(cls, rho, zeta):
        return beta_PBE*(Ecmuab(rho, zeta)/Ecab(rho, zeta))**alpha_C

class beta_ab(Function):
    nargs=2
    name='beta_ab'
    
    @classmethod
    def eval(cls, rho, zeta):
        return Function('beta_ab')(rho, zeta)

subs_betaab=[[beta_ab, 'calc_beta_ab', rho, zeta]]
create_Routine_from_Function(beta_ab_, file, max_deriv, subs_ecmuab+subs_ecab, [], [], 'calc_beta_ab', [mu,], rho, zeta)

class A_ab_(Function):
    nargs=2
    name='A_ab'
    
    @classmethod
    def eval(cls, rho, zeta):
        return beta_ab(rho, zeta)/gamma/(exp(-Ecmuab(rho, zeta)/(gamma*phi(zeta)**three))-one)

class A_ab(Function):
    nargs=2
    name='A_ab'
    
    @classmethod
    def eval(cls, rho, zeta):
        return Function('A_ab')(rho, zeta)

subs_Aab=[[A_ab, 'calc_A_ab', rho, zeta]]
create_Routine_from_Function(A_ab_, file, max_deriv, subs_ecmuab+subs_betaab+subs_phi, [], [], 'calc_A_ab', [mu,], rho, zeta)

class H_ab_(Function):
    nargs=3
    name='H_ab'
    
    @classmethod
    def eval(cls, rho, zeta, drho):
        At2=A_ab(rho, zeta)*t_ab(rho, zeta, drho)**two
        return gamma*phi(zeta)*log(one+beta_ab(rho, zeta)*t_ab(rho, zeta, drho)**two/gamma*(one+At2)/(one+At2+At2**two))

class H_ab(Function):
    nargs=3
    name='H_ab'
    
    @classmethod
    def eval(cls, rho, zeta, drho):
        return Function('H_ab')(rho, zeta, drho)

subs_Hab=[[H_ab, 'calc_H_ab', rho, zeta, drho]]
create_Routine_from_Function(H_ab_, file, max_deriv, subs_betaab+subs_tab+subs_Aab+subs_phi, [], [], 'calc_H_ab', [mu,], rho, zeta, drho)

# The actual energy density

# spin-polarized
class ec_pbe_mu_ab(Function):
    nargs=3
    name='lsd_pbe_sr_c'
    
    @classmethod
    def eval(cls, rho, zeta, drho):
        return Ecmuab(rho, zeta)+H_ab(rho, zeta, drho)


# Simplified routines for spin unpolarized systems

class t_(Function):
    nargs=2
    name='t'
    
    @classmethod
    def eval(cls, rho, drho):
        return drho/two/rho/ks(rho)

class t(Function):
    nargs=2
    name='t'
    
    @classmethod
    def eval(cls, rho, drho):
        return Function('t')(rho, drho)

subs_t=[[t, 'calc_t', rho, drho]]
create_Routine_from_Function(t_, file, max_deriv, [], [], [], 'calc_t', [], rho, drho)

class beta_(Function):
    nargs=1
    name='beta'
    
    @classmethod
    def eval(cls, rho):
        return beta_PBE*(Ecmu(rho)/Ec(rho))**alpha_C

class beta(Function):
    nargs=1
    name='beta'
    
    @classmethod
    def eval(cls, rho):
        return Function('beta')(rho)

subs_beta=[[beta, 'calc_beta', rho]]
create_Routine_from_Function(beta_, file, max_deriv, subs_ecmu+subs_ec, [], [], 'calc_beta', [mu,], rho)

class A_(Function):
    nargs=1
    name='A'
    
    @classmethod
    def eval(cls, rho):
        return beta(rho)/gamma/(exp(-Ecmu(rho)/gamma)-one)

class A(Function):
    nargs=1
    name='A'
    
    @classmethod
    def eval(cls, rho):
        return Function('a2')(rho)

subs_A=[[A, 'calc_A', rho]]
create_Routine_from_Function(A_, file, max_deriv, subs_ecmu+subs_beta, [], [], 'calc_A', [mu,], rho)

class H_(Function):
    nargs=2
    name='H'
    
    @classmethod
    def eval(cls, rho, drho):
        At2=A(rho)*t(rho, drho)**two
        return gamma*log(one+beta(rho)*t(rho, drho)**two/gamma*(one+At2)/(one+At2+At2**two))

class H(Function):
    nargs=2
    name='H'
    
    @classmethod
    def eval(cls, rho, drho):
        return Function('H')(rho, drho)

subs_H=[[H, 'calc_H', rho, drho]]
create_Routine_from_Function(H_, file, max_deriv, subs_beta+subs_t+subs_A, [], [], 'calc_H', [mu,], rho, drho)

# The actual energy density

# spin-unpolarized
class ec_pbe_mu(Function):
    nargs=2
    name='lda_pbe_sr_c'
    
    @classmethod
    def eval(cls, rho, drho):
        return Ecmu(rho)+H(rho, drho)

# The actual energy density

# spin-polarized
class ex_pbe_mu(Function):
    nargs=2
    name='pbe_sr_x'
    
    @classmethod
    def eval(cls, rho, drho):
        return Ex(rho)*Fx(rho, drho)

# List with constants
constants=[(gamma, gamma_), (beta_PBE, beta_PBE_), (alpha_C, alpha_C_)]
constants.extend([(b_PBE, b_PBE_), (c, c_), (alpha_X, alpha_X_), (kappa, kappa_)])
    
create_Routine_from_Function(ec_pbe_mu_ab, file, max_deriv, subs_ecmuab+subs_Hab, [], [], ec_pbe_mu_ab.name, [mu,], rho, zeta, drho)
create_Routine_from_Function(ec_pbe_mu, file, max_deriv, subs_ecmu+subs_H, [], [], ec_pbe_mu.name, [mu,], rho, drho)
create_Routine_from_Function(ex_pbe_mu, file, max_deriv, subs_ex+subs_Fx, [], [], ex_pbe_mu.name, [mu,], rho, drho)

file.close()

print(time()-t_start)