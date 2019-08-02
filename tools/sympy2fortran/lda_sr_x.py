# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 17:40:26 2019
@author: Standard
"""

from sympy import Symbol, Function, pi, sympify, exp, sqrt, erf
from sympy2fortran import create_Routine_from_Function

# The maximum requested derivative (Hint: the more derivatives requested the more it needs for creating files and the more code will be produced)
max_deriv=2

filename='lda_sr_x.F'

file=open(filename, 'w')

# List with the created routines
routines=[]

# Density and its derivative (we need Rho as function for sympy and as symbol for the code)
zeta=Symbol('zeta')
rho=Symbol('rho')
rs=Symbol('rs')

one=sympify('1')
two=sympify('2')
three=sympify('3')
four=sympify('4')
eight=sympify('8')
half=sympify('1/2')
third=sympify('1/3')

# Range-separation parameter
mu=Symbol('mu')
    
# Exchange functional

class A_(Function):
    nargs=1
    name='A'
    
    @classmethod
    def eval(cls, rs):
        return mu/((sympify('18')*pi)**third)*rs

class A(Function):
    nargs=1
    name='A'
    
    @classmethod
    def eval(cls, rs):
        return Function('A')(rs)
create_Routine_from_Function(A_, file, max_deriv, [], [], 'calc_A', [mu], rs)

class ex_mu(Function):
    nargs=1
    name='ex_lda'
    
    @classmethod
    def eval(cls, rs):
        return -(sympify('18')/pi**2)**third/rs*(three/eight-A(rs)*(sqrt(pi)*erf(half/A(rs))+(two*A(rs)-four*A(rs)**three)*exp(-one/four/A(rs)**two)-three*A(rs)+four*A(rs)**three))

create_Routine_from_Function(ex_mu, file, max_deriv, [(A(rs), A_(rs))], [], 'ex_mu', [mu,], rs)

file.close()

