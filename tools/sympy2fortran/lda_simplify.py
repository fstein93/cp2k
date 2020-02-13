# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 17:40:26 2019

@author: Standard
"""

from sympy import Symbol, Function, pi, sympify, log, exp, sqrt
from sympy2fortran import create_Routine_from_Function, My_Function

# The maximum requested derivative (Hint: the more derivatives requested the more it needs for creating files and the more code will be produced)
max_deriv=2

filename='lda_sr_c_ab.F'

file=open(filename, 'w')

# List with the created routines
routines=[]

# Density and its derivative (we need Rho as function for sympy and as symbol for the code)
zeta=Symbol('zeta')
rho=Symbol('rho')
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
class Ec_ab(My_Function):
    nargs=2

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

class D3(My_Function):
    nargs=1
    constants=constants_D3
    
    @classmethod
    def eval(cls, rs):
        return exp(-d3_1*rs)/rs**two*(-d3_2+rs)

create_Routine_from_Function(D3, file, max_deriv, [], False, rs)

d2_1=Symbol('d2_1')
d2_2=Symbol('d2_2')
d2_3=Symbol('d2_3')
constants_D2=[(d2_1, sympify(0.547)), (d2_2, sympify(0.388)), (d2_3, sympify(0.676))]

class D2(My_Function):
    nargs=1
    constants=constants_D2
    
    @classmethod
    def eval(cls, rs):
        return exp(-d2_1*rs)/rs*(-d2_2+d2_3*rs)
    
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
    def eval(cls, rs):
        return two**(five/three)/five/alpha**two/rs**two*(one-g2_1*rs)/(one+g2_2*rs+g2_3*rs**two)

# It is easier to deal with corner cases of this function(fully spin-polarized gas)
class g2z_ab(My_Function):
    nargs=2
    
    @classmethod
    def eval(cls, rs, zeta):
        return ((one+zeta)/two)**two*g2(rs*(two/(one+zeta))**third)+((one-zeta)/two)**two*g2(rs*(two/(one-zeta))**third)

create_Routine_from_Function(g2z_ab, file, max_deriv, [], False, rs, zeta)

# Version to deal with special case zeta >= 1
class g2z_p1_ab(My_Function):
    nargs=2
    
    @classmethod
    def eval(cls, rs, zeta):
        return ((one+zeta)/two)**two*g2(rs*(two/(one+zeta))**third)

create_Routine_from_Function(g2z_p1_ab, file, max_deriv, [(zeta, one)], False, rs, zeta)

# Version to deal with special case zeta <= -1
class g2z_m1_ab(My_Function):
    nargs=2
    
    @classmethod
    def eval(cls, rs, zeta):
        return ((one-zeta)/two)**two*g2(rs*(two/(one-zeta))**third)

create_Routine_from_Function(g2z_m1_ab, file, max_deriv, [(zeta, -one)], True, rs, zeta)

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

# Some helper routine concerning spin polarization
n=Symbol('n')
        
class phin(My_Function):
    nargs=(1, 2)
    
    @classmethod
    def eval(cls, zeta, n_=n):
        return half*((one+zeta)**(n_*third)+(one-zeta)**(n_*third))
    
    @classmethod
    def dummy(cls, zeta, n_=n):
        return Function('phin', nargs=2)(zeta, n_)

class phi8(My_Function):
    nargs=1
    
    @classmethod
    def eval(cls, zeta):
        return phin(zeta, 8)
    
    @classmethod
    def dummy(cls, zeta):
        return phin.dummy(zeta, 8)

class phi2(My_Function):
    nargs=1
    
    @classmethod
    def eval(cls, zeta):
        return phin(zeta, 2)
    
    @classmethod
    def dummy(cls, zeta):
        return phin.dummy(zeta, 2)

create_Routine_from_Function(phin, file, max_deriv, [], False, zeta)

# High-Level Helper Functions for the coefficients

class C2_ab(My_Function):
    nargs=2
    needed_functions=[(g0, lambda rs: (rs,), (rs,))]
    
    @classmethod
    def eval(cls, rs, zeta):
        return -three*(one-zeta**two)/eight/rs**three*gc0.dummy(rs)

create_Routine_from_Function(C2_ab, file, max_deriv, [], False, rs, zeta)

c3_1=Symbol('c3_1')
c3_1_=sympify('one/sqrt(two*pi)', evaluate=False)

constants_C3=[(c3_1, c3_1_)]

class C3_ab(My_Function):
    nargs=2
    constants=constants_C3
    needed_functions=[(g0, lambda rs: (rs,), (rs,))]
    
    @classmethod
    def eval(cls, rs, zeta):
        return -(one-zeta**two)*c3_1/rs**three*g0.dummy(rs)

create_Routine_from_Function(C3_ab, file, max_deriv, [], False, rs, zeta)

class C4_ab(My_Function):
    nargs=2
    needed_functions=[(g2z_ab, lambda rs, zeta: (rs, zeta), (rs, zeta)),
                      (D2, lambda rs: (rs,), (rs,)),
                      (phi8, lambda zeta: (zeta,), (zeta,))]
    
    @classmethod
    def eval(cls, rs, zeta):
        return -sympify('9/64')/rs**three*(g2z_ab.dummy(rs, zeta)+(one-zeta**two)*D2.dummy(rs)-phi8.dummy(zeta)/five/alpha**two/rs**two)

create_Routine_from_Function(C4_ab, file, max_deriv, [], False, rs, zeta)

class C5_ab(My_Function):
    nargs=2
    needed_functions=[(g2z_ab, lambda rs, zeta: (rs, zeta), (rs, zeta)),
                      (D3, lambda rs: (rs,), (rs,))]
    
    @classmethod
    def eval(cls, rs, zeta):
        return -sympify('9/40')/sqrt(two*pi)/rs**three*(g2z_ab.dummy(rs, zeta)+(one-zeta**2)*D3.dummy(rs))

create_Routine_from_Function(C5_ab, file, max_deriv, [], False, rs, zeta)

# Coefficients for the energy density

class a1_ab(My_Function):
    nargs=2
    needed_functions=[(C3_ab, lambda rs, zeta: (rs, zeta), (rs, zeta)),
                      (C5_ab, lambda rs, zeta: (rs, zeta), (rs, zeta))]
    
    @classmethod
    def eval(cls, rs, zeta):
        return four*b0(rs)**six*C3_ab.dummy(rs, zeta)+b0(rs)**eight*C5_ab.dummy(rs, zeta)

create_Routine_from_Function(a1_ab, file, max_deriv, [(beta, Symbol('b0')/rs)], False, rs, zeta)

class a2_ab(My_Function):
    nargs=2
    needed_functions=[(C2_ab, lambda rs, zeta: (rs, zeta), (rs, zeta)),
                      (C4_ab, lambda rs, zeta: (rs, zeta), (rs, zeta)),
                      (Ec_ab, lambda rs, zeta: (rs, zeta), (rs, zeta))]
    
    @classmethod
    def eval(cls, rs, zeta):
        return four*b0(rs)**six*C2_ab.dummy(rs, zeta)+b0(rs)**eight*C4_ab.dummy(rs, zeta)+six*b0(rs)**four*Ec_ab.dummy(rs, zeta)

create_Routine_from_Function(a2_ab, file, max_deriv, [(beta, Symbol('b0')/rs)], False, rs, zeta)

class a3_ab(My_Function):
    nargs=2
    needed_functions=[(C3_ab, lambda rs, zeta: (rs, zeta), (rs, zeta)),
                      (Ec_ab, lambda rs, zeta: (rs, zeta), (rs, zeta))]
    
    @classmethod
    def eval(cls, rs, zeta):
        return b0(rs)**eight*C3_ab.dummy(rs, zeta)

create_Routine_from_Function(a3_ab, file, max_deriv, [(beta, Symbol('b0')/rs)], False, rs, zeta)

class a4_ab(My_Function):
    nargs=2
    needed_functions=[(C2_ab, lambda rs, zeta: (rs, zeta), (rs, zeta)),
                      (Ec_ab, lambda rs, zeta: (rs, zeta), (rs, zeta))]
    
    @classmethod
    def eval(cls, rs, zeta):
        return b0(rs)**eight*C2_ab.dummy(rs, zeta)+four*b0(rs)**six*Ec_ab.dummy(rs, zeta)

create_Routine_from_Function(a4_ab, file, max_deriv, [], False, rs, zeta)

class a5_ab(My_Function):
    nargs=2
    needed_functions=[(Ec_ab, lambda rs, zeta: (rs, zeta), (rs, zeta))]
    
    @classmethod
    def eval(cls, rs, zeta):
        return b0(rs)**eight*Ec_ab.dummy(rs, zeta)

create_Routine_from_Function(a5_ab, file, max_deriv, [(beta, Symbol('b0')/rs)], False, rs, zeta)

# constants for the functional
a_=sympify(5.84605)
c_=sympify(3.91744)
d_=sympify(3.44851)
b_=d_-sympify('3*pi*a/(4*log(2)-4)', evaluate=False)
q_=sympify('2*(log(2)-1)/pi**2', evaluate=False)
        
a=Symbol('aa')
b=Symbol('bb')
c=Symbol('cc')
d=Symbol('dd')
q=Symbol('qq')
x=Symbol('xx')

constants_Q=[(a, a_), (c, c_), (d, d_), (q, q_), (b, b_)]

# Q(x), x=mu*sqrt(rs)/phi2(zeta)
# Workaround because sympy does not perform the necessary substitutions when
# the calculation of x is moved to ec_mu
class Q_ab(My_Function):
    nargs=2
    add_args=(mu,)
    constants=constants_Q
    needed_functions=[(phi2, lambda zeta: (zeta,), (zeta,))]
    
    @classmethod
    def eval(cls, rs, zeta):
        x=mu*sqrt(rs)/phi2.dummy(zeta)
        return q*log((one+a*x+b*x**2+c*x**three)/(one+a*x+d*x**two))

create_Routine_from_Function(Q_ab, file, max_deriv, [(mu, Symbol('xx')/sqrt(rs)/phi2.dummy(zeta))], False, rs, zeta)

# The actual energy density

# spin-polarized
class ec_mu_ab(My_Function):
    nargs=2
    add_args=(mu,)
    needed_functions=[(Q_ab, lambda rs, zeta: (rs, zeta), (rs, zeta)),
                      (phi2, lambda zeta: (zeta,), (zeta,)),
                      (a1_ab, lambda rs, zeta: (rs, zeta), (rs, zeta)),
                      (a2_ab, lambda rs, zeta: (rs, zeta), (rs, zeta)),
                      (a3_ab, lambda rs, zeta: (rs, zeta), (rs, zeta)),
                      (a4_ab, lambda rs, zeta: (rs, zeta), (rs, zeta)),
                      (a5_ab, lambda rs, zeta: (rs, zeta), (rs, zeta)),
                      (Ec_ab, lambda rs, zeta: (rs, zeta), (rs, zeta))]
    
    @classmethod
    def eval(cls, rs, zeta):
        return Ec_ab.dummy(rs, zeta)-(phi2.dummy(zeta)**3*Q_ab.dummy(rs, zeta)+a1_ab.dummy(rs, zeta)*mu**three+a2_ab.dummy(rs, zeta)*mu**four+a3_ab.dummy(rs, zeta)*mu**five+a4_ab.dummy(rs, zeta)*mu**six+a5_ab.dummy(rs, zeta)*mu**eight)/(one+b0(rs)**two*mu**two)**four

create_Routine_from_Function(ec_mu_ab, file, max_deriv, [(beta, Symbol('b0')/rs/mu)], False, rs, zeta)

file.close()