# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 17:40:26 2019

@author: Standard
"""

from sympy import Symbol, Function, pi, sympify, log, exp, sqrt
from sympy2fortran import create_Routine_from_Function, My_Function

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
class Ec(My_Function):
    nargs=1

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
    
subs_b0=[(beta, Symbol('b0')/rs)]
    
d3_1=Symbol('d3_1')
d3_2=Symbol('d3_2')

constants_D3=[(d3_1, sympify(0.31)), (d3_2, sympify(-4.95))]

class D3(My_Function):
    nargs=1
    constants=constants_D3
    
    @classmethod
    def eval(cls, rs):
        return exp(-d3_1*rs)/rs**two*(d3_2+rs)

subs_D3=[[D3, 'calc_D3', rs]]
create_Routine_from_Function(D3, file, max_deriv, [], False, rs)

d2_1=Symbol('d2_1')
d2_2=Symbol('d2_2')
d2_3=Symbol('d2_3')
constants_D2=[(d2_1, sympify(0.547)), (d2_2, sympify(-0.388)), (d2_3, sympify(0.676))]

class D2(My_Function):
    nargs=1
    
    @classmethod
    def eval(cls, rs):
        return exp(-d2_1*rs)/rs*(d2_2+d2_3*rs)

create_Routine_from_Function(D2, file, max_deriv, [], False, rs)

# Second derivative of the hole

g2_1=Symbol('g2_1')
g2_2=Symbol('g2_2')
g2_3=Symbol('g2_3')

constants_g2=[(g2_1, sympify(0.02267)), (g2_2, sympify(0.4319)), (g2_3, sympify(0.04))]

class g2(My_Function):
    nargs=1
    constants=constants_g2
    
    @classmethod
    def eval(cls, rs_):
        rs=rs_*two**third
        return two**(five/three)/five/alpha**two/rs**two*(one-g2_1*rs)/(one+g2_2*rs+g2_3*rs**two)/two
    
create_Routine_from_Function(g2, file, max_deriv, [], False, rs)

# The hole function
g0_1=Symbol('g0_1')
g0_2=Symbol('g0_2')
g0_3=Symbol('g0_3')
g0_4=Symbol('g0_4')
g0_5=Symbol('g0_5')

constants_g0=[(g0_1, sympify(0.0207)), (g0_2, sympify(0.08193)), (g0_3, sympify(0.01277)), (g0_4, sympify(0.001859)), (g0_5, sympify(0.7524))]

class g0(My_Function):
    nargs=1
    constants=constants_g0
    
    @classmethod
    def eval(cls, rs):
        return half*(one+g0_1*rs+g0_2*rs**two-g0_3*rs**three+g0_4*rs**four)*exp(-g0_5*rs)

create_Routine_from_Function(g0, file, max_deriv, [], False, rs)
    
class gc0(My_Function):
    nargs=1
    
    @classmethod
    def eval(cls, rs):
        return g0(rs)-half
    
    @classmethod
    def dummy(cls, rs):
        return g0.dummy(rs)-half

# High-Level Helper Functions for the coefficients

class C2(My_Function):
    nargs=1
    needed_functions=[(g0, lambda rs: (rs,), (rs,))]
    
    @classmethod
    def eval(cls, rs):
        return -three/eight/rs**three*(g0.dummy(rs)-half)

create_Routine_from_Function(C2, file, max_deriv, [], False, rs)

c3_1=Symbol('c3_1')
c3_1_=sympify('one/sqrt(two*pi)', evaluate=False)

constants_C3=[(c3_1, c3_1_)]

class C3(My_Function):
    nargs=1
    needed_functions=[(g0, lambda rs: (rs,), (rs,))]
    
    @classmethod
    def eval(cls, rs):
        return -c3_1/rs**three*g0.dummy(rs)

create_Routine_from_Function(C3, file, max_deriv, [], False, rs)

class C4(My_Function):
    nargs=1
    needed_functions=[(g2, lambda rs: (rs,), (rs,)),
                      (D2, lambda rs: (rs,), (rs,))]
    
    @classmethod
    def eval(cls, rs):
        return -sympify('9/64')/rs**three*(g2.dummy(rs)+D2.dummy(rs)-one/five/alpha**two/rs**two)

create_Routine_from_Function(C4, file, max_deriv, [], False, rs)

class C5(My_Function):
    nargs=1
    needed_functions=[(g2, lambda rs: (rs,), (rs,)),
                      (D3, lambda rs: (rs,), (rs,))]
    
    @classmethod
    def eval(cls, rs):
        return -sympify('9/40')/sqrt(two*pi)/rs**three*(g2.dummy(rs)+D3.dummy(rs))

create_Routine_from_Function(C5, file, max_deriv, [], False, rs)

# Coefficients for the energy density

class a1(My_Function):
    nargs=1
    needed_functions=[(C3, lambda rs: (rs,), (rs,)),
                      (C5, lambda rs: (rs,), (rs,))]
    
    @classmethod
    def eval(cls, rs):
        return four*b0(rs)**six*C3.dummy(rs)+b0(rs)**eight*C5.dummy(rs)

create_Routine_from_Function(a1, file, max_deriv, subs_b0, False, rs)

class a2(My_Function):
    nargs=1
    needed_functions=[(C2, lambda rs: (rs,), (rs,)),
                      (C4, lambda rs: (rs,), (rs,)),
                      (Ec, lambda rs: (rs,), (rs,))]
    
    @classmethod
    def eval(cls, rs):
        return four*b0(rs)**six*C2.dummy(rs)+b0(rs)**eight*C4.dummy(rs)+six*b0(rs)**four*Ec.dummy(rs)

create_Routine_from_Function(a2, file, max_deriv, subs_b0, False, rs)

class a3(My_Function):
    nargs=1
    needed_functions=[(C3, lambda rs: (rs,), (rs,))]
    
    @classmethod
    def eval(cls, rs):
        return b0(rs)**eight*C3.dummy(rs)

create_Routine_from_Function(a3, file, max_deriv, subs_b0, False, rs)

class a4(My_Function):
    nargs=1
    needed_functions=[(C2, lambda rs: (rs,), (rs,)),
                      (Ec, lambda rs: (rs,), (rs,))]
    
    @classmethod
    def eval(cls, rs):
        return b0(rs)**eight*C2.dummy(rs)+four*b0(rs)**six*Ec.dummy(rs)

create_Routine_from_Function(a4, file, max_deriv, subs_b0, False, rs)

class a5(My_Function):
    nargs=1
    needed_functions=[(Ec, lambda rs: (rs,), (rs,))]
    
    @classmethod
    def eval(cls, rs):
        return b0(rs)**eight*Ec.dummy(rs)

create_Routine_from_Function(a5, file, max_deriv, subs_b0, False, rs)

# constants for the functional
a_=sympify(5.84605)
c_=sympify(3.91744)
d_=sympify(3.44851)
b_=d_-sympify('3')*pi*a_/sympify('4*log(2)-4', evaluate=False)
q_=sympify('2*(log(2)-1)/pi**2', evaluate=False)
        
a=Symbol('aa')
b=Symbol('bb')
c=Symbol('cc')
d=Symbol('dd')
q=Symbol('qq')
x=Symbol('xx')

constants_Q=[(a, a_), (c, c_), (d, d_), (q, q_), (b, b_)]

class Q(My_Function):
    nargs=1
    add_args=(mu,)
    constants=constants_Q
    
    @classmethod
    def eval(cls, rs): 
        x=mu*sqrt(rs)
        return q*log((one+a*x+b*x**2+c*x**three)/(one+a*x+d*x**two))

create_Routine_from_Function(Q, file, max_deriv, [(mu, Symbol('xx')/sqrt(rs))], True, rs)

# The actual energy density

# spin-polarized
class ec_mu(My_Function):
    nargs=1
    add_args=(mu,)
    needed_functions=[(Ec, lambda rs: (rs,), (rs,)),
                      (Q, lambda rs: (rs,), (rs,)),
                      (a1, lambda rs: (rs,), (rs,)),
                      (a2, lambda rs: (rs,), (rs,)),
                      (a3, lambda rs: (rs,), (rs,)),
                      (a4, lambda rs: (rs,), (rs,)),
                      (a5, lambda rs: (rs,), (rs,))]
    
    @classmethod
    def eval(cls, rs):
        return Ec.dummy(rs)-(Q.dummy(rs)+a1.dummy(rs)*mu**three+a2.dummy(rs)*mu**four+a3.dummy(rs)*mu**five+a4.dummy(rs)*mu**six+a5.dummy(rs)*mu**eight)/(one+b0(rs)**two*mu**two)**four

create_Routine_from_Function(ec_mu, file, max_deriv, [], False, rs)

file.close()