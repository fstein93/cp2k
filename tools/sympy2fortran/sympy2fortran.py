# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 17:40:26 2019

@author: Standard
"""

from sympy import Symbol, cse, fcode, powdenest

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

def create_Routine_from_Function(func, file, max_deriv, intermediates, list_subs, constants, name, add_args, *args):
    deriv=get_deriv(func, max_deriv,True, *args)
    
    common, deriv=postprocessing(deriv, intermediates, list_subs, name, max_deriv)
    
    print('Number of common subexpressions for routine {}: {}'.format(name, len(common)))
    
    arguments=[a for a in args]
    arguments.extend(add_args)
    
    write_file(common, deriv, func, intermediates, constants, max_deriv, name, len(args), arguments, file)

def get_deriv(func, max_deriv, be_verbose=True, *args):
    ''' This function calculates recursively all partial derivatives of a sympy
    function func depending on the sympy variables in list var up to order 
    max_deriv. It assumes continouity of the function and its derivatives up 
    to the given order.
    '''
    
    if be_verbose:
        print()
        print('Calculate derivatives of function '+func.name)
    
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
            # Assume continouity of all functions
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

def postprocessing(deriv, intermediates, list_subs, name, max_deriv):
    '''
    This function does substitutions and performs a common subexpression 
    elimination (CSE) afterwards. To support the creation of good expressions, 
    the routine is very verbose.
    
    deriv               list of derivatives (sympy expressions)
    intermediates       list with intermediate functions with dependent variables
    name                name of the routine
    max_deriv           maximum needed derivative
    '''
    
    print('Start postprocessing of function '+name)
    
    # Perform substitutitons and do some simplifications
    for i, d in enumerate(deriv):
        print()
        print('Substitute', i)
        print(d)
        
        # Perform substitutions in sympy representation
        for func, call, *args in intermediates:
            derivs=get_deriv(func, max_deriv, False, *args)[::-1]
            for j, sub in enumerate(derivs):
                d=d.subs(sub, Symbol(str(func)+'('+str(len(derivs)-j-1)+')')).doit()
                d=powdenest(d, force=True)
        
        for token, sub in list_subs:
            d=d.subs(token, sub)
            
        # Do some simplifications (hint: simplify is quite expensive and 
        # does strange things from time to time, if it is appropriate for your 
        # problem, decomment the following line!)
        #d=simplify(d)
        
        # Finally copy the result
        deriv[i]=d.doit()
        
        print()
        print(i, 'after all substitutions')
        print(d)
    
    # Perform a common subexpression elimination
    return cse(deriv)
    
def write_file(common, deriv, func, intermediates, constants, max_deriv, name, nargs, arguments, file):
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
    intermediates   list with intermediate functions
    constants       list of constants of format (Symbol, value)
    name            name of the routine
    nargs           number of arguments of func
    arguments       list with arguments
    file            File object where to write everything
    '''
    
    # Get string representation of the arguments
    input_var=''
    for s in arguments:
        input_var+=str(s)+', '
    else:
        input_var=input_var[:-2]
        
    # Get a string of local variables
    local_var=''
    for s, d in common:
        local_var+=str(s)+', '
    for s, d in constants:
        local_var+=str(s)+', '
    for f, call, *args in intermediates:
        local_var+=str(f)+'(0:'+str(len(get_deriv(f, max_deriv, False, *args))-1)+'), '
    else:
        local_var=local_var[:-2]
        
    # List with all needed expressions
    expr=constants
    expr.extend(common)
    expr.extend([(Symbol(str(func)+'('+str(i)+')'), f) for i, f in enumerate(deriv)])
    
    print('Write file')
    
    # Write out everything
    file.write('\n')
    file.write(('   PURE SUBROUTINE '+name+'('+input_var+', '+str(func)+', max_deriv)\n').replace('_,', ',').replace('__', '_'))
    file.write(('      REAL(KIND=dp), INTENT(IN) :: '+input_var+'\n').replace('_,', ',').replace('__', '_'))
    file.write('      INTEGER, INTENT(IN) :: max_deriv\n')
    file.write(('      REAL(KIND=dp), INTENT(OUT) :: '+str(func)+'(0:)\n').replace('_(', '(').replace('_,', ',').replace('__', '_'))
    file.write('\n')
    # character 34 is '"'
    file.write('      CHARACTER(LEN=*), PARAMETER :: routineN = '+chr(34)+name+chr(34)+', routineP = moduleN//'+chr(34)+':'+chr(34)+'//routineN\n')
    file.write('\n')
    file.write('      REAL(KIND=dp) :: '+local_var+'\n')
    file.write('\n')
    for s, call, *args in intermediates:
        inp_var=''
        for i in args:
            inp_var+=str(i)+', '
        inp_var+=str(s)+', max_deriv'
        file.write('      CALL '+call+'('+inp_var+')\n')
    file.write('\n')
    for s, func in expr:
        file.write('      '+fcode(func, assign_to=s, source_format='free', standard=2008).replace('_(', '(').replace('_ ', ' ').replace('__', '_').replace('d0', '_dp').replace('1.0/', '1.0_dp/').replace(' 1/', ' 1.0_dp/').replace('parameter (pi = 3.14159265358979_dp)\n', '').replace('parameter (pi = 3.1415926535897932_dp)\n', '')+'\n')
    file.write('\n')
    file.write('   END SUBROUTINE '+name+'\n')
    
    print('Finished writing function '+name)