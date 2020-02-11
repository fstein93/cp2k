# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 17:40:26 2019

@author: Standard
"""

from sympy import Symbol, cse, fcode, powdenest, Function
from copy import deepcopy

'''
Hints:
    Build your function(s) with sympy.
    Wrap all constants. 
    When you have unspecified helper functions, provide a list of tuples with 
    possible substitutions.
    Provide a list with all included constants and, if necessary, a list with 
    additional arguments. These will be included in the created routine.
    
    This script is very verbose. It shall help the user to see whether all
    substitutions work properly. If necessary, try additional substitutions.
'''

class My_Function(Function):
    '''
    This Helper class is used to simplify the calculation of all derivatives.
    It can be used as an ordinary Function class object.
    
    Attributes:
    -----------
    name                Name of the function, used to create variable and
                        routine names of output code (Default: empty string)
    nargs               Number of arguments (without those in add_args)
    add_args            Additional arguments if you do not need derivatives wrt
                        to these variables, usually parameters of a function
                        (for example the range-separation
                        parameter mu of short-ranged functionals)
    needed_functions    Functions that are needed to evaluate beforehand with
                        actual arguments and needed arguments for derivatives
                        Format of each element:
                            (Function object, function providing tuple of passed values
                            given the actual arguments of the original function, tuple of actual arguments to call
                            the function)
                        Example:
                            Function f(x, y) needs function g(z) and calls g via g(x+y)
                            then intermediates reads
                                (f, lambda x, y: (x+y,), (x, y))
                            or alternatively
                                (f, lambda *args: (args[0]+args[1],), (args[0], args[1]))
    constants           Used to provide connections between constant symbols
                        and the actual values, in eval() one should use the
                        constant name
    
    Methods:
    --------
    eval                Ordinary routine to actually evaluate the function
                        Default: Use name to create a function without body
    dummy               Creates a dummy function without body using the function name
                        Used in routines to split calculation of derivatives of
                        a function in smaller chunks
    '''
    nargs=0
    add_args=()
    needed_functions=[]
    constants=[]
    
    @classmethod
    def eval(self, *args):
        for na in self.nargs:
            return Function(self.__name__, nargs=na)(*args)
    
    @classmethod
    def dummy(self, *args):
        for na in self.nargs:
            return Function(self.__name__, nargs=na)(*args)

def create_Routine_from_Function(func: My_Function, file, max_deriv: int, list_subs, do_simplify: bool, *args):
    deriv=get_deriv(func, max_deriv,True, *args)
    
    common, deriv=postprocessing(deriv, func.needed_functions, list_subs, func.__name__, max_deriv, do_simplify)
    
    print('Number of common subexpressions for routine {}: {}'.format(func.__name__, len(common)))
    
    write_file(common, deriv, func, max_deriv, file, *args)

def get_deriv(func: My_Function, max_deriv: int, be_verbose=True, *args):
    ''' This function calculates recursively all partial derivatives of a sympy
    function func depending on the sympy variables in list var up to order 
    max_deriv. It assumes continouity of the function and its derivatives up 
    to the given order.
    '''
    
    if be_verbose:
        print()
        print('Calculate derivatives of function '+func.__name__)
    
    # result array with already calculated derivatives
    deriv=[func(*args)]
    
    # Array with last variable w.r.t that we derived lastly to take symmetries into account
    config=[0]
    
    offset=0
    
    # Loop over the needed derivatives
    for i in range(1, max_deriv+1):
        # Counter counts number of new derivatives
        new_config=[]
        # Loop over the calculated derivatives of the last run
        for i, d in enumerate(deriv[offset:]):
            # Assume continouity of all functions and their derivatives
            # Thus, we can take symmetries into account
            for j, arg in enumerate(args[config[i]:]):
                deriv.append(d.diff(arg))
                new_config.append(i+j)
        config=new_config
        offset=len(deriv)-len(new_config)
    
    if be_verbose:
        print('Finished calculating derivatives')
        print()
    
    return deriv

def get_deriv_symbols(func: My_Function, passed_args, actual_args, max_deriv):
    ''' This function calculates recursively all partial derivatives of a sympy
    function func depending on the sympy variables in list var up to order 
    max_deriv. It assumes continouity of the function and its derivatives up 
    to the given order.
    '''
    
    # result array with already calculated derivatives
    deriv=[[func.dummy(*(passed_args(*actual_args))),func.__name__]]
    
    # Array with last variable w.r.t that we derived lastly to take symmetries into account
    config=[0]
    
    offset=0
    
    # Loop over the needed derivatives
    for i in range(1, max_deriv+1):
        # Counter counts number of new derivatives
        new_config=[]
        # Loop over the calculated derivatives of the last run
        for i, d in enumerate(deriv[offset:]):
            # Assume continouity of all functions and their derivatives
            # Thus, we can take symmetries into account
            for j, arg in enumerate(actual_args[config[i]:]):
                deriv.append([d[0].diff(arg),d[1]+'_'+arg.name])
                new_config.append(i+j)
        config=new_config
        offset=len(deriv)-len(new_config)
    
    return deriv

def postprocessing(deriv, intermediates, list_subs, name, max_deriv: int, do_simplify: bool):
    '''
    This function does substitutions and performs a common subexpression 
    elimination (CSE) afterwards. To support the creation of good expressions, 
    the routine is very verbose.
    
    deriv               list of derivatives (sympy expressions)
    intermediates       list with intermediate functions with dependent
                        variables and actual arguments (compare needed_functions in My_Function)
    list_subs           list of additional substitutions
    name                name of the routine
    max_deriv           maximum needed derivative
    '''
    
    print('Start postprocessing of function '+name)
    print(len(list_subs))
    
    # Perform substitutitons and do some simplifications
    for i, d in enumerate(deriv):
        print()
        print('Substitute', i)
        print(d)
        
        for token, sub in list_subs:
            d=d.subs(token, sub).doit()
        
        # Perform substitutions in sympy representation
        for intermediate in intermediates:
            # Invert order
            derivs=get_deriv_symbols(*intermediate, max_deriv)[::-1]
            for sub, dname in derivs:
                d=d.subs(sub, Symbol(dname)).doit()
                d=powdenest(d, force=True)
            
        # Do some simplifications (hint: simplify is quite expensive and 
        # does strange things from time to time, if it is appropriate for your 
        # problem, decomment the following line!)
        if do_simplify:
            d=d.simplify()
        
        # Finally copy the result
        deriv[i]=d.doit()
        
        print()
        print(i, 'after all substitutions')
        print(d)
    
    # Perform a common subexpression elimination
    return cse(deriv)
    
def write_file(common, deriv, func: My_Function, max_deriv: int, file, *args):
    '''
    This function generates the code to calculate a set of functions in 
    Fortran 2008. It uses the general format from CP2K. It does not include 
    the correct line length (-> make pretty). If you have the sympy constants 
    like pi, you will get an additional line like 
    'parameter (pi = 3.14159265358979_dp)'. Derivations might not be fully 
    substituted. (Then, you either have to substitute them by hand or find 
    an appropriate substitution.)
    
    common          list with common subexpressions
    deriv           list of derivatives (sympy expressions)
    func            our function object
    max_deriv       Number of needed derivatives
    file            File object where to write everything
    add_args        list with additional arguments of func
    args            list with arguments of the func object
    '''
    
    # Get string representation of the arguments
    output_var=''
    for subs, sname in get_deriv_symbols(func, lambda *args: args, args, max_deriv):
        output_var+=sname+', '
    else:
        output_var=output_var[:-2]
        
    # actual argument list
    arguments=tuple([a for a in args])+func.add_args
    
    input_var=''
    for s in arguments:
        input_var+=str(s)+', '
    else:
        input_var=input_var[:-2]
        
    # Get a string of local variables
    local_var=''
    for s, d in common:
        local_var+=str(s)+', '
    for s, d in func.constants:
        local_var+=str(s)+', '
    for intermediate in func.needed_functions:
        for subs, sname in get_deriv_symbols(*intermediate, max_deriv):
            local_var+=sname+', '
    else:
        # Remove trailing comma
        local_var=local_var[:-2]
        
    # List with all needed expressions
    # Use a deep copy to prevent surprises
    expr=deepcopy(func.constants)
    expr.extend(common)
    symb=get_deriv_symbols(func, lambda *args: args, args, max_deriv)
    for i, f in enumerate(deriv):
        expr+=[(Symbol(symb[i][1]), f)]
    
    print('Write file')
    
    # Write out everything
    file.write('\n')
    file.write(('   ELEMENTAL SUBROUTINE calc_'+func.__name__+'('+input_var+', max_deriv, '+output_var+')\n').replace('_,', ',').replace('__', '_'))
    file.write(('      REAL(KIND=dp), INTENT(IN) :: '+input_var+'\n').replace('_,', ',').replace('__', '_'))
    file.write('      INTEGER, INTENT(IN) :: max_deriv\n')
    file.write(('      REAL(KIND=dp), INTENT(OUT) :: '+output_var+'\n').replace('_(', '(').replace('_,', ',').replace('__', '_'))
    file.write('\n')
    # character 34 is '"'
    file.write('      CHARACTER(LEN=*), PARAMETER :: routineN = '+chr(34)+'calc_'+func.__name__+chr(34)+', routineP = moduleN//'+chr(34)+':'+chr(34)+'//routineN\n')
    file.write('\n')
    if local_var != '':
        file.write('      REAL(KIND=dp) :: '+local_var+'\n')
        file.write('\n')
    for s, passed_args, myargs in func.needed_functions:
        inp_var=''
        for i in myargs+s.add_args:
            inp_var+=str(i)+', '
        inp_var+='max_deriv'
        out_var=''
        for subs, sname in get_deriv_symbols(s, passed_args, myargs, max_deriv):
            out_var+=sname+', '
        else:
            out_var=out_var[:-2]
        file.write('      CALL calc_'+s.__name__+'('+inp_var+', '+out_var+')\n')
    if len(func.needed_functions)>0:
        file.write('\n')
    
    for s, sfunc in expr:
        file.write('      '+fcode(sfunc, assign_to=s, source_format='free', standard=2008).replace('_(', '(').replace('_ ', ' ').replace('__', '_').replace('d0', '_dp').replace('1.0/', '1.0_dp/').replace(' 1/', ' 1.0_dp/').replace('parameter (pi = 3.14159265358979_dp)\n', '').replace('parameter (pi = 3.1415926535897932_dp)\n', '')+'\n')
    file.write('\n')
    file.write('   END SUBROUTINE calc_'+func.__name__+'\n')
    
    print('Finished writing function '+func.__name__)