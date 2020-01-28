# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 17:40:26 2019

@author: Standard
"""

from sympy import Symbol, pi, sympify, exp, sqrt, erf
from sympy2fortran import create_Routine_from_Function, My_Function

# The maximum requested derivative (Hint: the more derivatives requested the more it needs for creating files and the more code will be produced)
max_deriv=2

filename='lda_sr_x.F'

file=open(filename, 'w')

# Density and its derivative (we need Rho as function for sympy and as symbol for the code)
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

facx_A_=sympify('(18*pi)**(-1/3)', evaluate=False)
facx_=sympify('(18/pi**2)**(1/3)', evaluate=False)

facx_A=Symbol('facx_A')
facx=Symbol('facx')

constants_sr_x=[(facx_A, facx_A_), (facx, facx_)]
    
# Exchange functional

class Ax(My_Function):
    nargs=1
    add_args=(mu,)
    constants=[(facx_A, facx_A_)]
    
    @classmethod
    def eval(cls, rs):
        return rs*facx_A*mu

class ex_mu(My_Function):
    nargs=1
    add_args=(mu,)
    constants=constants_sr_x
    
    @classmethod
    def eval(cls, rs):
        return -facx/rs*(three/eight-Ax(rs)*(sqrt(pi)*erf(half/Ax(rs))+(two*Ax(rs)-four*Ax(rs)**three)*exp(-one/four/Ax(rs)**two)-three*Ax(rs)+four*Ax(rs)**three))

create_Routine_from_Function(ex_mu, file, max_deriv, [], False, rs)

file.close()