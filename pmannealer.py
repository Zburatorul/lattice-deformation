#!/usr/bin/python

import argparse
import os
import pty
import sys
import time
import numpy as np
import numpy.matlib as matlib
from scipy.optimize import basinhopping
import scipy
import subprocess
import copy

# Number of significant digits to keep in each generating parameter
sig = 3

# Number of dimensions of each signature
n = 8

# Basis of initial lattice of signature (n,n). In this case E8
basis = np.array([ 2,0,0,0,0,0,0,0, -1,1,0,0,0,0,0,0, 0,-1,1,0,0,0,0,0, 0,0,-1,1,0,0,0,0, 0,0,0,-1,1,0,0,0, 0,0,0,0,-1,1,0,0, 0,0,0,0,0,-1,1,0, 0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5 ])
basis.resize(8,8)
basis = basis.transpose()

''' Start with hypercubic instead '''
#basis = np.identity(n)


# Can check that np.dot(basis.transpose(), basis) is the Gram matrix of E8 given by Mathematica, but not the usual
'''[[ 4. -2.  0.  0.  0.  0.  0.  1.]
 [-2.  2. -1.  0.  0.  0.  0.  0.]
 [ 0. -1.  2. -1.  0.  0.  0.  0.]
 [ 0.  0. -1.  2. -1.  0.  0.  0.]
 [ 0.  0.  0. -1.  2. -1.  0.  0.]
 [ 0.  0.  0.  0. -1.  2. -1.  0.]
 [ 0.  0.  0.  0.  0. -1.  2.  0.]
 [ 1.  0.  0.  0.  0.  0.  0.  2.]]
'''


# Now, double the basis. 
# A perfectly acceptable starting point is just the hypercubic, a good basis for which is the identity matrix 
# basis = np.identity( 2 * n)
basis = np.bmat( [[basis,0 * basis], [0 * basis, basis ]])


smallid = np.identity(n)
eta = np.bmat( [[ smallid, 0 * smallid], [0 * smallid, -smallid]] )
leftid =  np.bmat( [[ smallid, 0 * smallid], [0 * smallid, 0 * smallid]] )
rightid = np.bmat( [[ 0 * smallid, 0 * smallid], [0 * smallid, smallid]] )



''' This generates code to be bed to MAGMA '''

preamble = 'B := Matrix(RealField(), ' + str(2 * n) + ',' + str(2 * n) + ',' # [3,-1,-1,5]);
postamble = 'L := LatticeWithBasis(B); \n time Min(L); \n'


# Create the magma process and read out the useless lines
magma = subprocess.Popen(['/usr/local/magma/magma'], shell=False, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
for i in range(13):
    line = magma.stdout.readline()
    #print line


def max_params(params):
    ''' The largest value of the relevant parameters. This is important because these parameters get exponentiated, 
so numerical precision suffers if they are large. This leads to non-unimodular lattices after the transformation. '''
    r = params.reshape(120,1)
    
    return (abs(r[28 + 28 :])).max()


def build_mat(params):
    ''' Returns an SO(n,n) matrix '''

    def antisym(lst):
        ''' Constructs an n x n antisymmetric matrix out of n(n-1)/2 parameters ''' 
        mat = matlib.zeros([8,8])
        curr = 0
        for i in range(7):
            for j in range(i+1,8):
                mat[i,j] = lst[curr]
                curr += 1
        mat -= mat.transpose()
        return np.asarray(mat)
    
    ''' It is important to simplify the exact values of the generating parameters,
        by keeping less dynamic range in their digits        
        '''    
    params = params.round(sig)
    

    M = params[:28]
    N = params[ 28:56]
    P = params[56:]
    P = P.reshape(8,8)

    
    
    M = antisym(M)
    N = antisym(N)
    
    # Build the generator matrix
    genmat = np.bmat( [[M, P], [P.transpose(), N]])

    # Exponentiate
    Omat = scipy.linalg.expm(genmat)
    return Omat, params




def cost(params):
    ''' This routine evaluates the cost function that we are trying to minimize.
        It creates an SO(n,n) orthogonal matrix out of the parameters. It then applies this transformation to the basis vectors of some (n,n)-lattice.
        Using the new basis vectors it computes the matrix of scaling dimensions; this is the Gram matrix of the basis with respect to a Euclidean inner product.
        It feeds this Gram matrix to Magma and asks Magma to constructs a lattice out of it.
        Finally, it called the Magma routine for finding the shortest non-zero vector of a positive-definite lattice, and returns this norm? or norm^2? '''

    # Make sure we are given a nice object
    #assert( type(genmat) == np.ndarray)
    
    #genmat = np.asarray(genmat)
    #if len(genmat.shape) != 2 or genmat.shape[0] != genmat.shape[1]:
    #    raise ValueError('expected a square matrix')
    

    #
   
    if (abs(params)).max() > 10:
        return 10.01

    Omat, params = build_mat(params)

    # Check it is indeed an orthogonal matrix
    res = np.dot( np.transpose(Omat), np.dot(eta, Omat))
    res = res.round(8)

    if ( res != eta * 1.0 ).all():
        '''
        print res
        print '#######################'
        print M - M.transpose()
        print '#######################'
        print N - N.transpose()        
        print '#######################'
        print P
        print genmat
        '''
        print 'Bad orthogonal matrix with max param size = ', max_params(params)
        #dumpout(params, flag='.bad.')
        return 0.01
        
        

    # Dump matrix
    #Omat.dump('/home/eugeniu/omat.mat')

    # Obtain the new Gram matrix, aka the matrix of scaling dims
    newbasis = np.dot(Omat,basis)
    newgram  = np.dot( newbasis.transpose(), newbasis)

    if abs(np.linalg.det(newbasis)) - 1 > 10**-5:
        print 'New basis is not unimodular'
        assert False

    #lbasis = np.dot(leftid, newbasis)
    #leftgram = np.dot( lbasis.transpose(), lbasis )
    #rbasis = np.dot(rightid, newbasis)
    #rightgram = np.dot( rbasis.transpose(), rbasis)
    #oldgram = np.dot( newbasis.transpose(), np.dot( eta, newbasis))

    #print lbasis
    '''
    for i in range(2*n):
        for j in range(2*n):
            if abs(oldgram[i,j]) < 10**-8:
                oldgram[i,j] = 0
    #print oldgram
    #print '###########################'

    oldgram = leftgram
    for i in range(2*n):
        for j in range(2*n):
            if abs(oldgram[i,j]) < 10**-8:
                oldgram[i,j] = 0
    '''
    #print oldgram
    #print '###########################'



    # Convert into format appropriate for magma input
    lst = newbasis.tolist()
    text = str(lst)
    text = text.replace('[','')
    text = text.replace(']','')
    text = '[' + text + ']'

    # Final code to run
    '''
        Example: 
        B := Matrix(RealField(), 2,2, [3.2,-1,-1,2.5]);
        L := LatticeWithBasis(m);
        time Min(L);
    '''
    code = preamble + text + '); \n' + postamble
    magma.stdin.write(code)

    #print code
    
    result = magma.stdout.readline()
    #print result + '\n'

    try:
        line  = magma.stdout.readline()
        result2 = line.split(' ')[1]
    except:
        print line
        time.sleep(4)
        return 0.99

    # Try to interpret the result as an integer
    try:
        result = float(result)
    except:
        

        print 'Magma returned a non-number value'
        return 0.02

        print magma.stdout.readline()
        print result
        print code
        res = np.dot( np.transpose(Omat), np.dot(eta, Omat))
        res = res.round(12)
        if ( res != eta * 1.0 ).all():
            print 'Omat not orthogonal'
            res = np.dot( np.transpose(Omat), np.dot(eta, Omat))
            res = res.round(2)
            print res
        
        dumpout(params)
    
    print 'Minimal norm-squared = ', result, ' \t \t with max param size = ', max_params(params)
    #print 'Computed in: ', result2

    if (result > 2):
        dumpout(params, flag='IMPROVEMENT')
        folder = 'parameters/'

        now = int( time.time() )
        fout = file(folder + 'magma-code' +'.%d.txt' % now, 'w+')
        for line in code:
            fout.write(line)
        fout.flush()
        fout.close()
        
        print code
        assert False

    if (result > 3):
        dumpout(params, flag='DECENT')
        folder = '/home/eugeniu/parameters/'

        now = int( time.time() )
        fout = file(folder + 'magma-code' +'.%d.txt' % now, 'w+')
        for line in code:
            fout.write(line)
        fout.flush()
        fout.close()

    if (result > 4):
        dumpout(params, flag='MAGIC')

    return -result
   
#gmat = randomgen()

def writeout(tab, fname):
    folder = 'parameters/'

    now = int( time.time() )
    print 'Time now is %d' % now

    fout = file(folder + fname +'.%d.txt' % now, 'w+')

    for i in range(tab.shape[0]):
        line = ''
        for j in range(tab.shape[1]):
            line += str(tab[i,j]) + '  '
        line += '\n'
        fout.write(line)
    fout.flush()
    fout.close()
 

def dumpout(params, flag=''):
    def antisym(lst):
        ''' Constructs an n x n antisymmetric matrix out of n(n-1)/2 parameters ''' 
        mat = matlib.zeros([8,8])
        curr = 0
        for i in range(7):
            for j in range(i+1,8):
                mat[i,j] = lst[curr]
                curr += 1
        mat -= mat.transpose()
        return np.asarray(mat)
         

    M = params[:28]
    N = params[ 28:56]
    P = params[56:]
    P = P.reshape(8,8)

    
    
    M = antisym(M)
    N = antisym(N)
    
    # Build the generator matrix
    genmat = np.bmat( [[M, P], [P.transpose(), N]])

    # Exponentiate
    Omat = scipy.linalg.expm(genmat)    

    writeout(params.reshape(60,2), 'params' + flag)
    writeout(genmat, 'genmat' + flag)
    writeout(Omat, 'omat' + flag)


def my_accept_test(f_new, x_new, f_old, x_old):
    if (f_new < 0) and (f_old >= 0):
        print '-----------\t Forced Accept \t ----------------'
        time.sleep(2)
        return "force accept"

    if max_params(x_new) < max_params(x_old) and max_params(x_old) > 8:
        return False
    return True
    


params = np.asarray(matlib.rand(1, 28 + 28))
params = np.append(params,  np.append( 2 * np.asarray(matlib.rand(1, 32)),  np.asarray(matlib.rand(1, 32))  )  )
params = params.reshape(120,)

print cost(params)

res = basinhopping(cost, params, accept_test = my_accept_test)
print res
print 'Result is ', res.fun






    
''' 
    What I really should check is that among the vectors of Euclidean norm < 2 (4?) none are spin-zero.
    For this I have to list all the vectors.

 '''




