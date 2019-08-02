# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 17:40:26 2019
@author: Standard
"""

from sympy import Symbol, Function, pi, sympify, log, exp, sqrt
from sympy2fortran import create_Routine_from_Function, get_deriv

# The maximum requested derivative (Hint: the more derivatives requested the more it needs for creating files and the more code will be produced)
max_deriv=2

filename='lda_sr_c.F'

file=open(filename, 'w')

# List with the created routines
routines=[]

# Density and its derivative (we need Rho as function for sympy and as symbol for the code)
rs=Symbol('rs')

zero=sympify('0')
one=sympify('1')
two=sympify('2')
three=sympify('3')
four=sympify('4')
five=sympify('5')
six=sympify('6')
eight=sympify('8')
half=sympify('1/2')
third=sympify('1/3')

# Correlation energy density (e.g. from PW92)
class Ec(Function):
    nargs=1
    name='Ec'
    
    @classmethod
    def eval(cls, rs):
        return Function('Ec')(rs)

subs_ec=list(zip(*get_deriv(Ec, max_deriv, rs)))[::-1]

alpha=Symbol('alpha')
alpha_=((two/three)**two/pi)**third

constants_alpha=[(alpha, alpha_),]

# Range-separation parameter
mu=Symbol('mu')

# Low-Level Helper Functions

beta=Symbol('beta')
beta_=sympify(0.784949)

constants_beta=[(beta, beta_)]

# We will not introduce a helper because that function is too easy
class b0(Function):
    nargs=1
    
    @classmethod
    def eval(cls, rs):
        return beta*rs
    
d3_1=Symbol('d3_1')
d3_2=Symbol('d3_2')

constants_D3=[(d3_1, sympify(0.31)), (d3_2, sympify(4.95))]

class D3_(Function):
    nargs=1
    name='D3'
    
    @classmethod
    def eval(cls, rs):
        return exp(-d3_1*rs)/rs**two*(-d3_2+rs)

class D3(Function):
    nargs=1
    name='D3'
    
    @classmethod
    def eval(cls, rs):
        return Function('D3')(rs)

subs_D3=list(zip(*get_deriv(D3, max_deriv, rs)))[::-1]
create_Routine_from_Function(D3_, file, max_deriv, [], constants_D3, 'calc_D3', [], rs)

d2_1=Symbol('d2_1')
d2_2=Symbol('d2_2')
d2_3=Symbol('d2_3')
constants_D2=[(d2_1, sympify(0.547)), (d2_2, sympify(0.388)), (d2_3, sympify(0.676))]

class D2_(Function):
    nargs=1
    name='D2'
    
    @classmethod
    def eval(cls, rs):
        return exp(-d2_1*rs)/rs*(-d2_2+d2_3*rs)

class D2(Function):
    nargs=1
    name='D2'
    
    @classmethod
    def eval(cls, rs):
        return Function('D2')(rs)

subs_D2=list(zip(*get_deriv(D2, max_deriv, rs)))[::-1]
    
create_Routine_from_Function(D2_, file, max_deriv, [], constants_D2, 'calc_D2', [], rs)

# Second derivative of the hole

g2_1=Symbol('g2_1')
g2_2=Symbol('g2_2')
g2_3=Symbol('g2_3')

constants_g2=[(g2_1, sympify(0.02267)), (g2_2, sympify(0.4319)), (g2_3, sympify(0.04))]

class g2(Function):
    nargs=1
    name='g2'
    
    @classmethod
    def eval(cls, rs):
        return two**(five/three)/five/alpha**two/rs**two*(one-g2_1*rs)/(one+g2_2*rs+g2_3*rs**two)

# It is easier to deal with corner cases of this function(fully spin-polarized gas)
class g2z_(Function):
    nargs=1
    name='g2z'
    
    @classmethod
    def eval(cls, rs):
        return one/two**two*g2(rs*two**third)

class g2z(Function):
    nargs=1
    name='g2z'
    
    @classmethod
    def eval(cls, rs):
        return Function('g2z')(rs)
    
subs_g2z=list(zip(*get_deriv(g2z, max_deriv, rs)))[::-1]

create_Routine_from_Function(g2z_, file, max_deriv, [], constants_g2, 'calc_g2z', [], rs)

# The hole function
g0_1=Symbol('g0_1')
g0_2=Symbol('g0_2')
g0_3=Symbol('g0_3')
g0_4=Symbol('g0_4')
g0_5=Symbol('g0_5')

constants_g0=[(g0_1, sympify(0.0207)), (g0_2, sympify(0.08193)), (g0_3, sympify(0.01277)), (g0_4, sympify(0.001859)), (g0_5, sympify(0.7524))]

class g0_(Function):
    nargs=1
    name='g0'
    
    @classmethod
    def eval(cls, rs):
        return half*(one+g0_1*rs+g0_2*rs**two-g0_3*rs**three+g0_4*rs**four)*exp(-g0_5*rs)

create_Routine_from_Function(g0_, file, max_deriv, [], constants_g0, 'calc_g0', [], rs)

class g0(Function):
    nargs=1
    name='g0'
    
    @classmethod
    def eval(cls, rs):
        return Function('g0')(rs)

subs_g0=list(zip(*get_deriv(g0, max_deriv, rs)))[::-1]
    
class gc0(Function):
    nargs=1
    
    @classmethod
    def eval(cls, rs):
        return g0(rs)-half

# High-Level Helper Functions for the coefficients

class C2_(Function):
    nargs=1
    name='C2'
    
    @classmethod
    def eval(cls, rs):
        return -three/eight/rs**three*gc0(rs)

class C2(Function):
    nargs=1
    name='C2'
    
    @classmethod
    def eval(cls, rs):
        return Function('C2')(rs)
create_Routine_from_Function(C2_, file, max_deriv, subs_g0, [], 'calc_C2', [], rs)

subs_C2=list(zip(*get_deriv(C2, max_deriv, rs)))[::-1]

class C3_(Function):
    nargs=1
    name='C3'
    
    @classmethod
    def eval(cls, rs):
        return -one/sqrt(two*pi)/rs**three*g0(rs)

class C3(Function):
    nargs=1
    name='C3'
    
    @classmethod
    def eval(cls, rs):
        return Function('C3')(rs)
create_Routine_from_Function(C3_, file, max_deriv, subs_g0, [], 'calc_C3', [], rs)

subs_C3=list(zip(*get_deriv(C3, max_deriv, rs)))[::-1]

class C4_(Function):
    nargs=1
    name='C4'
    
    @classmethod
    def eval(cls, rs):
        return -sympify('9/64')/rs**three*(two*g2z(rs)+D2(rs)-one/five/alpha**two/rs**two)

class C4(Function):
    nargs=1
    name='C4'
    
    @classmethod
    def eval(cls, rs):
        return Function('C4')(rs)

subs_C4=list(zip(*get_deriv(C4, max_deriv, rs)))[::-1]

create_Routine_from_Function(C4_, file, max_deriv, subs_g2z+subs_D2, [], 'calc_C4', [], rs)

class C5_(Function):
    nargs=1
    name='C5'
    
    @classmethod
    def eval(cls, rs):
        return -sympify('9/40')/sqrt(two*pi)/rs**three*(two*g2z(rs)+D3(rs))

class C5(Function):
    nargs=1
    name='C5'
    
    @classmethod
    def eval(cls, rs):
        return Function('C5')(rs)

subs_C5=list(zip(*get_deriv(C5, max_deriv, rs)))[::-1]
create_Routine_from_Function(C5_, file, max_deriv, subs_g2z+subs_D3, [], 'calc_C5', [], rs)

# Coefficients for the energy density

class a1_(Function):
    nargs=1
    name='a1'
    
    @classmethod
    def eval(cls, rs):
        return four*b0(rs)**six*C3(rs)+b0(rs)**eight*C5(rs)

class a1(Function):
    nargs=1
    name='a1'
    
    @classmethod
    def eval(cls, rs):
        return Function('a1')(rs)
subs_a1=list(zip(*get_deriv(a1, max_deriv, rs)))[::-1]
create_Routine_from_Function(a1_, file, max_deriv, subs_C3+subs_C5, constants_beta, 'calc_a1', [], rs)

class a2_(Function):
    nargs=1
    name='a2'
    
    @classmethod
    def eval(cls, rs):
        return four*b0(rs)**six*C2(rs)+b0(rs)**eight*C4(rs)+six*b0(rs)**four*Ec(rs)

class a2(Function):
    nargs=1
    name='a2'
    
    @classmethod
    def eval(cls, rs):
        return Function('a2')(rs)
subs_a2=list(zip(*get_deriv(a2, max_deriv, rs)))[::-1]
create_Routine_from_Function(a2_, file, max_deriv, subs_ec+subs_C2+subs_C4, constants_beta, 'calc_a2', [], rs)

class a3_(Function):
    nargs=1
    name='a3'
    
    @classmethod
    def eval(cls, rs):
        return b0(rs)**eight*C3(rs)

class a3(Function):
    nargs=1
    name='a3'
    
    @classmethod
    def eval(cls, rs):
        return Function('a3')(rs)
subs_a3=list(zip(*get_deriv(a3, max_deriv, rs)))[::-1]
create_Routine_from_Function(a3_, file, max_deriv, subs_C3, constants_beta, 'calc_a3', [], rs)

class a4_(Function):
    nargs=1
    name='a4'
    
    @classmethod
    def eval(cls, rs):
        return b0(rs)**eight*C2(rs)+four*b0(rs)**six*Ec(rs)

class a4(Function):
    nargs=1
    name='a4'
    
    @classmethod
    def eval(cls, rs):
        return Function('a4')(rs)
subs_a4=list(zip(*get_deriv(a4, max_deriv, rs)))[::-1]
create_Routine_from_Function(a4_, file, max_deriv, subs_C2+subs_ec, constants_beta, 'calc_a4', [], rs)

class a5_(Function):
    nargs=1
    name='a5'
    
    @classmethod
    def eval(cls, rs):
        return b0(rs)**eight*Ec(rs)

class a5(Function):
    nargs=1
    name='a5'
    
    @classmethod
    def eval(cls, rs):
        return Function('a5')(rs)
subs_a5=list(zip(*get_deriv(a5, max_deriv, rs)))[::-1]
create_Routine_from_Function(a5_, file, max_deriv, subs_ec, constants_beta, 'calc_a5', [], rs)

# constants for the functional
a_=sympify(5.84605)
c_=sympify(3.91744)
d_=sympify(3.44851)
b_=d_-sympify('3')*pi*a_/sympify('4*log(2)-4')
q_=sympify('2*(log(2)-1)/pi**2')

# List with substitutions
list_subs=[(Ec(rs), Symbol('ed0'))]
        
a=Symbol('a')
b=Symbol('b')
c=Symbol('c')
d=Symbol('d')
q=Symbol('q')
x=Symbol('x')

constants_Q=[(a, a_), (c, c_), (d, d_), (q, q_), (b, b_)]

class Q_(Function):
    nargs=1
    name='Q'
    
    @classmethod
    def eval(cls, rs):
        x=mu*sqrt(rs) 
        return q*log((one+a*x+b*x**2+c*x**three)/(one+a*x+d*x**two))

create_Routine_from_Function(Q_, file, max_deriv, [], constants_Q+constants_alpha, 'calc_Q', [mu,], rs)

class Q(Function):
    nargs=1
    name='Q'
    
    @classmethod
    def eval(cls, rs):
        return Function('Q')(rs)

subs_Q=list(zip(*get_deriv(Q, max_deriv, rs)))[::-1]

# The actual energy density

# spin-polarized
class ec_mu(Function):
    nargs=1
    name='ec_lsd'
    
    @classmethod
    def eval(cls, rs):
        return Ec(rs)-(Q(rs)+a1(rs)*mu**three+a2(rs)*mu**four+a3(rs)*mu**five+a4(rs)*mu**six+a5(rs)*mu**eight)/(one+b0(rs)**two*mu**two)**four

create_Routine_from_Function(ec_mu, file, max_deriv, subs_ec+subs_Q+subs_a1+subs_a2+subs_a3+subs_a4+subs_a5, [], 'ec_mu', [mu,], rs)

file.close()

