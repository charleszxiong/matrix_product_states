#!/usr/bin/env python3
"""Matrix product states"""

from math import *
import numpy as np
import pandas as pd
from scipy.linalg import eig
from scipy.linalg import orth
from scipy.linalg import sqrtm
from scipy.linalg import svd
    
def transfer_matrix(A, O=None, optimize=False, flattened_output=False):
    '''Compute the transfer matrix of a given tensor
    
    Input:
    
    A - tensor
        This is either a rank-3 tensor A[i,j,k], where i is the physical index and j and k are the virtual indices,
        or a higher-rank tensor A[...,i,j,k], representing a collection of such rank-3 tensors.
        
    O - physical operator
        When this argument is present, the output will be the transfer matrix with physical operator inserted.
        
    optimize - optimization of Einstein summation
    
    flattened_output - whether E is flattened (default: not flattened)
    
    Output:
    
    E - transfer matrix
        This is the rank-4 tensor E[a,a',b,b'] = A[i,a,b] * O[i,i'] * A[i',a',b'].conj() when O is present, or
        A[i,a,b] * A[i',a',b'].conj() when O is absent, where the Einstein summation rule applies.
        
    Note:
    
    The broadcasting rule applies when A has rank > 3.
    '''
    d = A.shape[-3]         # get physical dimension
    chi = A.shape[-2]       # get bond dimension
    if O is None:
        O = np.identity(d)  # set O to identity if no operator insertion
    if len(A.shape) == 3:   # if the input is a single tensor
        E = np.einsum('iab,ij,jcd->acbd', A, O, A.conj(), optimize=optimize)
        if flattened_output == False:
            return E
        else:
            return E.reshape((chi*chi,chi*chi))
    elif len(A.shape) == 4: # if the input is a collection of tensors
        E = np.einsum('...iab,ij,...jcd->...acbd', A, O, A.conj(), optimize=optimize)
        if flattened_output == False:
            return E
        else:
            m = A.shape[0]
            return E.reshape((m,chi*chi,chi*chi))
        
def spectrum(E):
    '''Compute the spectrum of a given transfer matrix
    
    Input:
    
    E - transfer matrix
        This is either the rank-4 tensor E[a,a',b,b'] = A[i,a,b] * A[i,a',b'].conj(), where the Einstein summation rule
        applies, or a rank-5 tensor representing a collection of transfer matrices E[...,a,a',b,b'].
        
    Output:
    
    w - eigenvalues
        The eigenvalues are repeated according to multiplicity, but not necessarily sorted.
        
    vl - left eigenvectors (matrix acts from right)
        The eigenvectors are normalized, and vl[i] has eigenvalue w[i]. The eigenvectors may be linearly dependent.
        
    vr - right eigenvectors (matrix acts from left)
        The eigenvectors are normalized, and vr[i] has eigenvalue w[i]. The eigenvectors may be linearly dependent.
        
    Note:
    
    Broadcasting rule applies when E has rank 5.
    '''
    if len(E.shape) == 4:   # if the input is a single transfer matrix
        chi = E.shape[0]                                   # get the bond dimension chi
        E_flattened = E.reshape(chi*chi, chi*chi)          # treat two input indices as one, and two output indices as one
        (w,vl,vr) = eig(E_flattened, left=True, right=True)  # eigenvalues, left eigenvectors, and right eigenvectors
        vl = vl.T
        vr = vr.T
    elif len(E.shape) == 5:  # if the input is a collection of transfer matrices
        (m,chi,chi,chi,chi) = E.shape                      # get the number of transfer matrices m as well as chi
        E_flattened = E.reshape(m, chi*chi, chi*chi)
        w = np.zeros((m,chi*chi), dtype=np.complex_)
        vl = np.zeros((m,chi*chi,chi*chi), dtype=np.complex_)
        vr = np.zeros((m,chi*chi,chi*chi), dtype=np.complex_)
        for n in range(m):
            (w[n],vl[n],vr[n]) = eig(E_flattened[n], left=True, right=True)
        vl = vl.transpose((0,2,1))
        vr = vr.transpose((0,2,1))
    return w,vl,vr

def spectral_radius(E):
    '''Compute the spectral radius of a transfer matrix
    
    Input:
    E - transfer matrix
        This is either the rank-4 tensor E[a,a',b,b'] = A[i,a,b] * A[i,a',b'].conj(), where the Einstein summation rule
        applies, or a rank-5 tensor representing a collection of transfer matrices E[...,a,a',b,b'].
        
    Output:
    rho - spectral radius
        This is either a nonnegative real number (when E has rank 4) or a vector of nonnegative real numbers
        (when E has rank 5).
        
    Note:
    This routines uses the routine spectrum(E), which computes the entire spectrum and all eigenvectors of E.
    '''
    w = spectrum(E)[0]              # get eigenvalues
    return np.amax(abs(w), axis=-1)   # find the largest magnitude

def fixed_point(E, tol=1e-10, max_iter=inf, optimize=False, flattened_output=True, hermitianize=True, power_iteration=False):
    '''Compute the fixed point of a transfer matrix under renormalization
    
    Input:
    
    E - transfer matrix
        This is either the rank-4 tensor E[a,a',b,b'] = A[i,a,b] * A[i,a',b'].conj(), where the Einstein summation rule
        applies, or a rank-5 tensor representing a collection of transfer matrices E[...,a,a',b,b'].
        
    tol - tolerance
        Routine converges when change in vectors is less than tol in norm.
        
    max_iter - maximum number of iterations
    
    optimize - optimization of Einstein summation
    
    flattened_output - whether r, l, and E_inf are flattened
    
    hermitianize - whether to hermitianize dominant eigenvectors at every step
    
    power_iteration - whether to use power iteration, default False
        
    Output:
    
    z - dominant eigenvalue
        If hermitianize is True, z will be made > 0.
    
    l - dominant normalized left eigenvector
        This is the flattened version of l[a,a'] by default.
        If hermitianize is True, the matrix l will be made >= 0.
    
    r - dominant normalized right eigenvector
        This is the flattened version of r[b,b'] by default.
        If hermitianize is True, the matrix r will be made >= 0.
    
    E_inf - normalized (spectral radius 1) fixed-point transfer matrix
        This is related to z, r, l by E_inf[a,a';b,b'] = r[a,a'] * l[b,b'] / np.dot(l.conj(), r).
        This is the flattened version of E_inf[a,a';b,b'] by default.
        
    Note:
    
    This routine uses the power iteration method and converges only if E is known to be a normal tensor.
    Broadcasting rule applies when E has rank 5.
    '''
    if power_iteration:     # use power iteration method
        if len(E.shape) == 4:      # if the input is a single transfer matrix
            # get bond dimension chi
            chi = E.shape[0]                                
            # prepares flattened E
            E_flattened = E.reshape((chi*chi, chi*chi))       
            # initialize right vector
            r = np.random.rand(chi,chi)
            if hermitianize:
                r = (r + np.einsum('...ab->...ba', r.conj()))/2 # hermitianize
            r = r.reshape(chi*chi)
            r = r/np.linalg.norm(r)   # normalize
            # initialize left vector
            l = np.random.rand(chi,chi)
            if hermitianize:
                l = (l + np.einsum('...ab->...ba', l.conj()))/2 # hermitianize
            l = l.reshape(chi*chi)
            l = l/np.linalg.norm(l)   # normalize
            i = 0
            while i < max_iter:
                # update right vector
                r_old = r                                     
                r = np.dot(E_flattened, r)
                if hermitianize:
                    r = r.reshape((chi,chi))
                    r = (r + np.einsum('...ab->...ba', r.conj()))/2 # hermitianize
                    r = r.reshape(chi*chi)
                r = r/np.linalg.norm(r)
                # update left vector
                l_old = l                                     
                l = np.dot(E_flattened.conj().T, l)
                if hermitianize:
                    l = l.reshape((chi,chi))
                    l = (l + np.einsum('...ab->...ba', l.conj()))/2 # hermitianize
                    l = l.reshape(chi*chi)
                l = l/np.linalg.norm(l)
                # check for convergence
                if np.linalg.norm(r - r_old) < tol and np.linalg.norm(l - l_old) < tol:
                    break
            # dominant eigenvalue
            z = r.conj() @ E_flattened @ r
            # fixed-point transfer matrix
            E_inf = np.einsum('i,j', r, l.conj()) / np.dot(l.conj(), r)
        elif len(E.shape) == 5:     # if the input is a collection of transfer matrices
            # get number of transfer matrices m as well as chi
            (m,chi,chi,chi,chi) = E.shape                     
            # prepares flattened E
            E_flattened = E.reshape((m, chi*chi, chi*chi))    
            # initialize right vector
            r = np.random.rand(m, chi, chi)
            if hermitianize:
                r = (r + np.einsum('...ab->...ba', r.conj()))/2 # hermitianize
            r = r.reshape((m, chi*chi))
            r = np.einsum('ni,n->ni', r, 1/np.linalg.norm(r,axis=1), optimize=optimize)   # normalize
            # initialize left vector
            l = np.random.rand(m, chi, chi)
            if hermitianize:
                l = (l + np.einsum('...ab->...ba', l.conj()))/2 # hermitianize
            l = l.reshape((m, chi*chi))
            l = np.einsum('ni,n->ni', l, 1/np.linalg.norm(l,axis=1), optimize=optimize)   # normalize
            i = 0
            while i < max_iter:
                # update right vector
                r_old = r                                     
                r = np.einsum('nij,nj->ni', E_flattened, r, optimize=optimize)
                if hermitianize:
                    r = r.reshape((m,chi,chi))
                    r = (r + np.einsum('...ab->...ba', r.conj()))/2 # hermitianize
                    r = r.reshape((m,chi*chi))
                r = np.einsum('ni,n->ni', r, 1/np.linalg.norm(r,axis=1), optimize=optimize)
                # update left vector
                l_old = l                                     
                l = np.einsum('nji,nj->ni', E_flattened.conj(), l, optimize=optimize)
                if hermitianize:
                    l = l.reshape((m,chi,chi))
                    l = (l + np.einsum('...ab->...ba', l.conj()))/2 # hermitianize
                    l = l.reshape((m,chi*chi))
                l = np.einsum('ni,n->ni', l, 1/np.linalg.norm(l,axis=1), optimize=optimize)
                # check for convergence
                if np.linalg.norm(r - r_old) < tol and np.linalg.norm(l - l_old) < tol:
                    break
            # dominant eigenvalue
            z = np.einsum('ni,nij,nj->n',r.conj(), E_flattened, r, optimize=optimize)
            # fixed-point transfer matrix
            E_inf = np.einsum('nij,n->nij',
                np.einsum('ni,nj->nij', r, l.conj(), optimize=optimize),
                1/np.einsum('ni,ni->n', l.conj(), r, optimize=optimize))
        # make l and r >= 0 and z > 0 if necessary
        if hermitianize:
            z = np.abs(z)
            if len(E.shape) == 4:
                l = l * np.sign( np.trace( l.reshape((chi,chi)) ) )
                r = r * np.sign( np.trace( r.reshape((chi,chi)) ) )
            elif len(E.shape) == 5:
                l = np.einsum( 'ni,n->ni', l, np.sign( np.trace( l.reshape((m,chi,chi)), axis1=-2, axis2=-1 ) ) )
                r = np.einsum( 'ni,n->ni', r, np.sign( np.trace( r.reshape((m,chi,chi)), axis1=-2, axis2=-1 ) ) )
    else: # if not using power iteration
        (w, vl, vr) = spectrum(E)
        where = np.argmax(np.abs(w), axis=-1)
        if len(E.shape) == 4:
            chi = E.shape[0]
            # dominant eigenvalue
            z = w[where]
            if hermitianize:
                z = np.abs(z)
            # left dominant eigenvector
            l = vl[where]
            if hermitianize:
                l = l.reshape((chi,chi))
                l = (l + np.einsum('...ab->...ba', l.conj()))/2 # hermitianize
                l = l * np.sign( np.trace(l) ) # make >= 0
                l = l.reshape(chi*chi)
            l = l/np.linalg.norm(l) # normalize
            # right dominant eigenvector
            r = vr[where]
            if hermitianize:
                r = r.reshape((chi,chi))
                r = (r + np.einsum('...ab->...ba', r.conj()))/2 # hermitianize
                r = r * np.sign( np.trace(r) ) # make >= 0
                r = r.reshape(chi*chi)
            r = r/np.linalg.norm(r) # normalize
            # fixed-point transfer matrix
            E_inf = np.einsum('i,j', r, l.conj()) / np.dot(l.conj(), r)
        elif len(E.shape) == 5:
            (m, chi, chi, chi, chi) = E.shape
            # dominant eigenvalue
            if hermitianize:
                z = np.amax(np.abs(w), axis=-1)
            else:
                z = np.zeros(m, dtype=np.complex_)
                for n in range(m):
                    z[n] = w[n, where[n]]
            # left dominant eigenvector
            l = np.zeros((m,chi*chi), dtype=np.complex_)
            for k in range(m):
                l[k] = vl[k][where[k]]
            if hermitianize:
                l = l.reshape((m,chi,chi))
                l = (l + np.einsum('...ab->...ba', l.conj()))/2 # hermitianize
                l = l.reshape((m,chi*chi))
                l = np.einsum('ni,n->ni', l, np.sign( np.trace( l.reshape((m,chi,chi)), axis1=-2, axis2=-1 ) ) ) # make >= 0
            l = np.einsum('ni,n->ni', l, 1/np.linalg.norm(l,axis=1), optimize=optimize) # normalize
            # right dominant eigenvector
            r = np.zeros((m,chi*chi), dtype=np.complex_)
            for k in range(m):
                r[k] = vr[k][where[k]]
            if hermitianize:
                r = r.reshape((m,chi,chi))
                r = (r + np.einsum('...ab->...ba', r.conj()))/2 # hermitianize
                r = r.reshape((m,chi*chi))
                r = np.einsum('ni,n->ni', r, np.sign( np.trace( r.reshape((m,chi,chi)), axis1=-2, axis2=-1 ) ) ) # make >= 0
            r = np.einsum('ni,n->ni', r, 1/np.linalg.norm(r,axis=1), optimize=optimize) # normalize
            # fixed-point transfer matrix
            E_inf = np.einsum('nij,n->nij',
                np.einsum('ni,nj->nij', r, l.conj(), optimize=optimize),
                1/np.einsum('ni,ni->n', l.conj(), r, optimize=optimize))
    # return results
    if flattened_output == True:
        return z, l, r, E_inf
    else:
        if len(E.shape) == 4:
            return z, l.reshape((chi,chi)), r.reshape((chi,chi)), E_inf.reshape((chi,chi,chi,chi))
        elif len(E.shape) == 5:
            return z, l.reshape((m,chi,chi)), r.reshape((m,chi,chi)), E_inf.reshape((m,chi,chi,chi,chi))

def normality_indicators(E):
    '''Check if a transfer matrix comes from a normal tensor
    
    Input:
    
    E - transfer matrix
        This is either the rank-4 tensor E[a,a',b,b'] = A[i,a,b] * A[i,a',b'].conj(), where the Einstein summation rule
        applies, or a rank-5 tensor representing a collection of transfer matrices E[...,a,a',b,b'].
        
    Output:
    
    gap - spectral gap
        Let w0 >= w1 be largest two in absolute value of all the eigenvalues of the transfer matrix, which can contain
        repetitions. Then g is defined to be (w0-w1)/w0.
    
    det_l - absolute value of the determinant of the dominant normalized left eigenvector l of E
    
    det_r - absolute value of the determinant of the dominant normalized right eigenvector r of E
    
    det_r_left_canonical - absolute value of the determinant of the dominant unit-trace right dominant eigenvector
    in left-canonical form
        In the left-canonical form, the left dominant eigenvector l' is set to the identity, whereas the right
        dominant eigenvector r' is set to be diagonal and have unit trace. They are obtained as follows:
        l' = U X_inv_dag l X_inv U_dag, r' = U X r X_dag U_dag / c, where X = sqrt(l), U is some unitary, and c is some
        normalization factor. One can show that c = tr(lr). Therefore, det(r') = det(l'r') = det(lr/c) = det(lr)/c**chi.
    
    Note:
    
    gap, det_l, det_r, det_r_left_canonical >= 0.
    The transfer matrix comes from a normal tensor iff gap, det_l, det_r > 0, iff gap, det_r_left_canonical > 0.
    Broadcasting rule applies when E has rank 5.
    '''
    (w,vl,vr) = spectrum(E)
    if len(E.shape) == 4:   # if the input is a single transfer matrix
        chi = E.shape[0]               # get bond dimension
        temp = abs(w)
        i0 = np.argmax(temp)           # locate eigenvalue of largest magnitude
        temp[i0] = 0            
        i1 = np.argmax(temp)           # locate eigenvalue of next largest magnitude
        gap = (abs(w[i0])-abs(w[i1]))/abs(w[i0])
                                       # compute spectral gap
        l = vl[i0].reshape(chi,chi)    # view dominant left eigenvector as matrix
        r = vr[i0].reshape(chi,chi)    # view dominant right eigenvector as matrix
        det_l = abs(np.linalg.det(l))  # compute absolute value of determinant
        det_r = abs(np.linalg.det(r))  # compute absolute value of determinant
        det_r_left_canonical = abs(np.linalg.det(l.conj().T @ r / np.trace(l.conj().T @ r)))
                                       # the would-be value in left canonical form
    elif len(E.shape) == 5:  # if the input is a collection of transfer matrices
        (m,chi,chi,chi,chi) = E.shape  # get the number m of transfer matrices as well as bond dimension
        gap = np.zeros(m)
        det_l = np.zeros(m)
        det_r = np.zeros(m)
        det_r_left_canonical = np.zeros(m)
        for n in range(m):             # loop through different transfer matrices
            temp = abs(w[n])
            i0 = np.argmax(temp)                  # locate eigenvalue of largest magnitude
            temp[i0] = 0            
            i1 = np.argmax(temp)                  # locate eigenvalue of next largest magnitude
            gap[n] = (abs(w[n,i0])-abs(w[n,i1]))/abs(w[n,i0])
                                                  # compute spectral gap
            l = vl[n,i0].reshape(chi,chi)         # view dominant left eigenvector as matrix
            r = vr[n,i0].reshape(chi,chi)         # view dominant right eigenvector as matrix
            det_l[n] = abs(np.linalg.det(l))      # compute absolute value of determinant
            det_r[n] = abs(np.linalg.det(r))      # compute absolute value of determinant
            det_r_left_canonical[n] = abs(np.linalg.det(l.conj().T @ r / np.trace(l.conj().T @ r)))
                                                  # the would-be value in left canonical form
    return gap, det_l, det_r, det_r_left_canonical

def injectivity_indicator(A, optimize=False):
    '''Check if a tensor is an injective tensor
    
    Input:
    
    A - tensor
        This is either a rank-3 tensor A[i,j,k], where i is the physical index and j and k are the virtual indices,
        or a rank-4 tensor A[h,i,j,k], representing a collection of such rank-3 tensors.
        
    optimize - optimization of Einstein summation
        
    Output:
    
    det - measure of injectiveness
        In case A has rank 3, we view A as a map from the virtual space to the physical space, and consider the
        chi*chi vectors A[:,j,k]. Return the absolute value of the Gram determinant of these vectors.
        In case A has rank 4, returns a collection of such absolute values.
        
    Note:
    
    det >= 0. The tensor is injective iff det > 0.
    '''
    if len(A.shape) == 3:               # if the input is a single tensor
        (d,chi,chi) = A.shape               # get the physical dimension d and bond dimension chi
        mat = A.reshape((d,chi*chi))        # view A as a map from the virtual space to the physical space
        gram = mat.conj().T @ mat           # the Gramian matrix
        return abs(np.linalg.det(gram))     # return the Gram determinant
    elif len(A.shape) == 4:             # if the input is a collection of tensors
        (m,d,chi,chi) = A.shape             # get the number of tensors m, as well as d and chi
        mat = A.reshape((m,d,chi*chi))
        gram = np.einsum('ikj,ikl->ijl', mat.conj(), mat, optimize=optimize)
        return abs(np.linalg.det(gram))

def make_left_canonical(A, tol=1e-10, max_iter=inf, optimize=False, full_output=False, power_iteration=False):
    '''Make a normal tensor left-canonical
    
    Input:
    
    A - tensor
        This is either a rank-3 tensor A[i,j,k], where i is the physical index and j and k are the virtual indices,
        or a higher-rank tensor A[...,i,j,k], representing a collection of such rank-3 tensors.
        
    tol - tolerance
        Routine converges when change in vectors is less than tol in norm.
        
    max_iter - maximum number of iterations
    
    optimize - optimization of Einstein summation
    
    full_output - whether to return new dominant eigenvectors as well as new tensor
        
    power_iteration - whether to use power iteration method, default True
    
    Output:
    
    newA - gauge-equivalent left-canonical tensor
        We say a normal tensor is left-canonical if its transfer matrix has the identity matrix as the dominant
        left eigenvector and a positive diagonal matrix as the dominant right eigenvector. See page 48 of
        arXiv:1210.7710.
        
    newL - new left dominant eigenvector (identity matrix)
    
    newR - new right dominant eigenvector, normalized to have unit trace
    
    Note:
    
    This routine assumes the given tensor is normal.
    Broadcasting rule applies when A has higher rank.
    '''
    # Initialize new tensor
    newA = A
    # Get dominant eignevalue and left eigenvector
    E = transfer_matrix(A, optimize=optimize)                # transfer matrix
    (z, L, R, _) = fixed_point(E, tol=tol, max_iter=max_iter,
                              optimize=optimize, flattened_output=False, power_iteration=power_iteration)
    # Diagonalize L
    L = (L + np.einsum('...ab->...ba', L.conj()))/2 # hermitianize
    (w, v) = np.linalg.eigh(L)                      # eigenvalues and eigenvectors
    # Apply gauge transformation
    w = abs(w)                                                                # reverse sign in case L < 0
    X = np.einsum('...ij,...j,...kj->...ik', v, np.sqrt(w), v.conj())         # square root of L
    X_inv = np.einsum('...ij,...j,...kj->...ik', v, 1/np.sqrt(w), v.conj())   # inverse square root of L
    newA = np.einsum('...ab,...ibc,...cd->...iad', X, newA, X_inv)            # conjugate A by X and X_inv
    newA = np.einsum('...iab,...->...iab', newA, 1/np.sqrt(z))                # make spectral radius 1
    R = np.einsum('...ab,...bc,...cd->...ad', X, R, X)                        # conjugate R by X and X
    #L = np.einsum('...ab,...bc,...cd->...ad', X_inv, L, X_inv)                # conjugate L by X_inv and X_inv
    # Diagonalize R
    R = (R + np.einsum('...ab->...ba', R.conj()))/2 # hermitianize
    (_, v) = np.linalg.eigh(R)                      # eigenvalues and eigenvectors
    # Apply gauge transformation
    X_inv = v                                                      # the unitary part only
    X = np.einsum('...ab->...ba', v.conj())                        # conjugate transpose of v
    newA = np.einsum('...ab,...ibc,...cd->...iad', X, newA, X_inv) # conjugate A by X and X_inv
    R = np.einsum('...ab,...bc,...cd->...ad', X, R, X_inv)         # conjugate R by X and X_inv
    #L = np.einsum('...ab,...bc,...cd->...ad', X, L, X_inv)         # conjugate L by X and X_inv
    # Return result
    if full_output:
        if len(A.shape) == 3:
            (d,chi,chi) = A.shape
            newL = np.identity(chi)                                       # set left dominant eigenvector to identity
            R = (R + np.einsum('...ab->...ba', R.conj()))/2               # hermitianize
            newR = R / np.trace(R, axis1=-2, axis2=-1, dtype=np.complex_) # make right dominant eigenvector trace 1
            return newA, newL, newR
        elif len(A.shape) == 4:
            (m,d,chi,chi) = A.shape
            newL = np.repeat(np.array([np.identity(chi)]), m, axis=0)     # set left dominant eigenvector to identity
            R = (R + np.einsum('...ab->...ba', R.conj()))/2               # hermitianize
            newR = np.einsum('...ab,...->...ab', R, 1 / np.trace(R, axis1=-2, axis2=-1, dtype=np.complex_)) # make right dominant eigenvector trace 1
            return newA, newL, newR
    else:
        return newA
        
def prepare_tangent_space(A, L, R, optimize=False):
    '''Get left-canonical tangent vector
    
    Input:
    
    A - tensor
        This is a rank-3 tensor A[i,j,k], where i is the physical index and j and k are the virtual indices.
        
    L - left dominant eigenvector
        This is a rank-2 tensor L[i,j]. Needs L>=0.
        
    R - right dominant eigenvector
        This is a rank-2 tensor R[i,j]. Needs R>=0.
        
    optimize - optimization of Einstein summation
    
    Output:
    
    L_inv_sqrt - inverse square root of L
    
    R_inv_sqrt - inverse square root of R
    
    V_L - au auxiliary tensor
        This is a complex ndarray of shape (d, chi, (d - 1) * chi). The vectors V_L[:, :, i] for different i
        are orthonormal. See page 65 of arXiv:1210.7710 for detail.
    
    Note:
    
    Broadcasting rule does NOT apply.
    
    For the left canonical tangents to have the desired physical properties, we need to assume that rho(E) = 1
    and Tr(L^\dag R) = 1.
    '''
    # Construct auxiliary tensor LA
    L = (L + L.conj().T)/2                   # hermitianize
    L_sqrt = sqrtm(L)                        # square root of L
    LA = np.einsum('ab,ibc->iac', L_sqrt, A,
                  optimize=optimize)         # this is related to the L in Eq. (169) of
                                             # arXiv:1210.7710 by LA[i,b,a].conj() = L^i_{ab}
    # Construct auxiliary tensor V_L (columns of V_L are orthonormal and orthogonal to columns of LA)
    (d,chi,chi) = LA.shape
    LA_reshaped = LA.reshape((d*chi, chi))  # A[i,a,b] with i,a as row index and b as column index
    P_range = orth(LA_reshaped)             # projector onto the range defined by the colums of A_reshaped
    P_kernel = np.identity(d*chi) - P_range @ P_range.conj().T # projector onto the kernel of A_reshaped
    (V_L, _, _) = svd(P_kernel)
    V_L = V_L[:,:(d-1)*chi]                 # V_L, as appears under Eq. (169) of arXiv:1210.7710
    V_L = V_L.reshape(d,chi,(d-1)*chi)      # reshape V_L to rank-3 tensor
    # Compute inverse square root of L and R
    L_inv_sqrt = np.linalg.inv(L_sqrt)       # inverse square root of L
    R = (R + R.conj().T)/2                   # hermitianize
    R_inv_sqrt = np.linalg.inv(sqrtm(R))     # inverse square root of R
    # Return inverse square root of L and R, and V_L
    return L_inv_sqrt, R_inv_sqrt, V_L

def left_canonical_tangent(L_inv_sqrt, R_inv_sqrt, V_L, X, optimize=False):
    '''Get left-canonical tangent vector
    
    Input:
        
    L_inv_sqrt - the inverse square root of the left dominant eigenvector
        The left dominant eigenvector L is a rank-2 tensor L[i,j]. Needs L>=0.
        L_inv_sqrt can be computed using prepare_tangent_space.
        
    R_inv_sqrt - right dominant eigenvector
        The right dominant eigenvector R is a rank-2 tensor R[i,j]. Needs R>=0.
        R_inv_sqrt can be computed using prepare_tangent_space.
        
    V_L - au auxiliary tensor
        This is a complex ndarray of shape (d, chi, (d - 1) * chi). The vectors V_L[:, :, i] for different i
        are orthonormal.
        V_L can be computed using prepare_tangent_space.
        
    X - complex coordinates
        This is either a rank-2 tensor X[i,j], or a rank-3 tensor representing a collection of such rank-2
        tensors. In X[i,j], the index i has dimension (d-1)*chi, where d is the physical dimension and chi
        is the bond dimension, whereas the index j has dimension chi.
        
    optimize - optimization of Einstein summation
    
    Output:
    
    B - left-canonical tangent vector at A parameterized by X in the tangent space of the space of MPS tensors
        This tangent vector preserves the left dominant eigenvector L, not not necessarily the right dominant
        eigenvector R, of the transfer matrix. See page 65 of arXiv:1210.7710 for detail.
    
    Note:
    
    Broadcasting rule applies when X has rank 3.
    
    For the left canonical tangents to have the desired physical properties, we need to assume that the A, L,
    and R from which L_inv_sqrt, R_inv_sqrt, and V_L were derived satisfy rho(E) = 1 and Tr(L^\dag R) = 1.
    '''
    B = np.einsum('ab,ibc,...cd,de->...iae', L_inv_sqrt, V_L, X, R_inv_sqrt, optimize=optimize)
    return B
    
def correlation_length(E):
    '''Compute correlation length
    
    Input:
    
    E - transfer matrix
        This is either the rank-4 tensor E[a,a',b,b'] = A[i,a,b] * A[i,a',b'].conj(), where the Einstein summation rule
        applies, or a rank-5 tensor representing a collection of transfer matrices E[...,a,a',b,b'].
        
    Output:
    
    xi - correlation length
        Let w0 >= w1 be largest two in absolute value of all the eigenvalues of the transfer matrix, which can contain
        repetitions. Then xi = -1/log(abs(w1/w0)).
    
    Note:
    
    The routine assumes the tensor is normal.
    Broadcasting rule applies when E has rank 5.
    '''
    (w,_,_) = spectrum(E)
    if len(E.shape) == 4:   # if the input is a single transfer matrix
        temp = abs(w)
        i0 = np.argmax(temp)           # locate eigenvalue of largest magnitude
        temp[i0] = 0            
        i1 = np.argmax(temp)           # locate eigenvalue of next largest magnitude
        ratio = abs(w[i1])/abs(w[i0])  # ratio of second largest eigenvalue absolute value to largest
        xi = - 1/log(ratio)            # compute correlation length
    elif len(E.shape) == 5:  # if the input is a collection of transfer matrices
        m = len(E)
        ratio = np.zeros(m)
        for n in range(m):             # loop through different transfer matrices
            temp = abs(w[n])
            i0 = np.argmax(temp)                  # locate eigenvalue of largest magnitude
            temp[i0] = 0            
            i1 = np.argmax(temp)                  # locate eigenvalue of next largest magnitude
            ratio[n] = abs(w[n,i1])/abs(w[n,i0])  # ratio of second largest eigenvalue absolute value to largest
        xi = - 1/np.log(ratio)         # compute correlation length
    return xi

def one_point_function(A, O, rho=None, l=None, r=None, E=None, E_O=None, optimize=False, tol=1e-10, max_iter=inf, power_iteration=False):
    '''Compute the one-point function
    
    Input:
    
    A - tensor
        This is either a rank-3 tensor A[i,j,k], where i is the physical index and j and k are the virtual indices,
        or a rank-4 tensor A[h,i,j,k], representing a collection of such rank-3 tensors.
        
    O - 1-local physical operator
    
    rho - dominant eigenvalue
    
    l - flattened normalized left dominant eigenvector
    
    r - flattened normalized right dominant eigenvector
    
    E - unflattened transfer matrix
    
    E_O - unflattened transfer matrix with insertion O
        
    optimize - optimization of Einstein summation
    
    tol - tolerance
        Routine converges when change in vectors is less than tol in norm.
        
    max_iter - maximum number of iterations
        
    power_iteration - whether to use power iteration method, default False
    
    Output:
    
    ex - expectation value of O with respect to MPS defined by A
    
    Note:
    
    Broadcasting rule applies when A has rank 4.
    '''
    # get number of examples, physical dimension, and bond dimension
    if len(A.shape) == 3:
        (d, chi, chi) = A.shape
    else:
        (m, d, chi, chi) = A.shape        
    
    # compute dominant eigenvectors and eigenvalues
    if l is None or r is None:
        if E is None:
            E = transfer_matrix(A, optimize=optimize, flattened_output=False)
        (rho,l,r,_) = fixed_point(E, optimize=optimize, flattened_output=True,
                                 tol=tol, max_iter=max_iter, power_iteration=power_iteration)
    else:
        if rho is None:
            if E is None:
                E = transfer_matrix(A, optimize=optimize, flattened_output=True)
            else:
                if len(E.shape) == 4:
                    E = E.reshape((chi*chi, chi*chi))
                elif len(E.shape) == 5:
                    E = E.reshape((m, chi*chi, chi*chi))
            rho = np.einsum('...i,...ij,...j->...', r.conj(), E, r, optimize=optimize)
    
    # compute transfer matrix with insertion
    if E_O is None:
        E_O = transfer_matrix(A, O=O, optimize=optimize, flattened_output=True)
    else:
        if len(E_O.shape) == 4:
            E_O = E_O.reshape((chi*chi, chi*chi))
        elif len(E_O.shape) == 5:
            E_O = E_O.reshape((m, chi*chi, chi*chi))
    
    # compute expectation value
    if len(A.shape) == 3:       
        ex = (l.conj() @ E_O @ r) / np.dot(l.conj(), r) / rho
    elif len(A.shape) == 4:     
        ex = np.einsum('...i,...ij,...j->...', l.conj(), E_O, r, optimize=optimize) \
             / np.einsum('...i,...i->...', l.conj(), r, optimize=optimize) / rho
    
    # return expectation value(s)
    return ex

def two_point_function(A, O0, x0, O1, x1, connected=True, rho=None, l=None, r=None, E=None, E0=None, E1=None, optimize=False, tol=1e-10, max_iter=inf, power_iteration=False):
    '''Compute the connected two-point function
    
    Input:
    
    A - tensor
        This is either a rank-3 tensor A[i,j,k], where i is the physical index and j and k are the virtual indices,
        or a rank-4 tensor A[h,i,j,k], representing a collection of such rank-3 tensors.
        
    O0 - first 1-local physical operator
    
    x0 - coordinate of first operator
    
    O1 - second 1-local physical operator
    
    x1 - coordinate of second operator (x1>x0 are both integers)
    
    connected - whether to compute the connected two-point function
    
    rho - dominant eigenvalue
    
    l - flattened normalized left dominant eigenvector
    
    r - flattened normalized right dominant eigenvector
    
    E - unflattened transfer matrix
    
    E0 - unflattened transfer matrix with insertion O0
    
    E1 - unflattened transfer matrix with insertion O1
        
    optimize - optimization of Einstein summation
    
    tol - tolerance
        Routine converges when change in vectors is less than tol in norm.
        
    max_iter - maximum number of iterations
        
    power_iteration - whether to use power iteration method, default False
    
    Output:
    
    ex - expectation value <O0(x0)O1(x1)> - <O0><O1> with respect to MPS defined by A
    
    Note:
    
    Broadcasting rule applies when A has rank 4.
    '''
    # Compute transfer matrix
    if E is None:
        E = transfer_matrix(A, optimize=optimize, flattened_output=False)
    
    # Compute flattened version of transfer matrix
    if len(E.shape) == 4:
        (chi,chi,chi,chi) = E.shape
        E_flattened = E.reshape((chi*chi, chi*chi))
    elif len(E.shape) == 5:
        (m,chi,chi,chi,chi) = E.shape
        E_flattened = E.reshape((m, chi*chi, chi*chi))
    
    # Compute dominant eigenvectors and eigenvalue
    if l is None or r is None:
        (rho,l,r,_) = fixed_point(E, optimize=optimize, flattened_output=True,
                                  tol=tol, max_iter=max_iter, power_iteration=power_iteration)
    else:
        if rho is None:
            rho = np.einsum('...i,...ij,...j->...', r.conj(), E_flattened, r, optimize=optimize)
    
    # Compute (flattened) transfer matrices with insertions
    if E0 is None:
        E0 = transfer_matrix(A, O0, optimize=optimize, flattened_output=True)
    else:
        if len(E.shape) == 4:
            E0 = E0.reshape((chi*chi, chi*chi))
        elif len(E.shape) == 5:
            E0 = E0.reshape((m, chi*chi, chi*chi))
    if E1 is None:
        E1 = transfer_matrix(A, O1, optimize=optimize, flattened_output=True)
    else:
        if len(E.shape) == 4:
            E1 = E1.reshape((chi*chi, chi*chi))
        elif len(E.shape) == 5:
            E1 = E1.reshape((m, chi*chi, chi*chi))
    
    # Compute two-point function
    if len(A.shape) == 3:    # input is a single tensor
        E_flattened = np.matrix(E_flattened)
        ex01 = (l.conj() @ E0 @ E_flattened**int(x1-x0-1) @ E1 @ r) / np.dot(l.conj(), r) / rho**int(x1-x0+1)
        ex01 = ex01[0,0]
        if connected:
            ex0 = (l.conj() @ E0 @ r) / np.dot(l.conj(), r) / rho
            ex1 = (l.conj() @ E1 @ r) / np.dot(l.conj(), r) / rho
            ex = ex01 - ex0 * ex1
        else:
            ex = ex01
    elif len(A.shape) == 4:  # input is a collection of tensors
        denominator = np.einsum('...i,...i->...', l.conj(), r, optimize=optimize) * rho**int(x1-x0+1)
        numerator = np.einsum('...ij,...j->...i', E1, r, optimize=optimize)
        for x in range(int(x1-x0-1)):
            numerator = np.einsum('...ij,...j->...i', E_flattened, numerator, optimize=optimize)
        numerator = np.einsum('...ij,...j->...i', E0, numerator, optimize=optimize)
        numerator = np.einsum('...i,...i->...', l.conj(), numerator, optimize=optimize)
        ex01 = numerator / denominator
        if connected:
            ex0 = np.einsum('...i,...ij,...j->...', l.conj(), E0, r,
                            optimize=optimize) / np.einsum('...i,...i->...', l.conj(), r, optimize=optimize) / rho
            ex1 = np.einsum('...i,...ij,...j->...', l.conj(), E1, r,
                            optimize=optimize) / np.einsum('...i,...i->...', l.conj(), r, optimize=optimize) / rho
            ex = ex01 - ex0 * ex1
        else:
            ex = ex01
        
    # Return expectation value(s)
    return ex

def fidelity(A0, A1, tol=1e-10, max_iter=inf, optimize=False, power_iteration=False):
    '''Compute fidelity per site between two tensors
    
    Fidelity per site, fid, is defined by the asymptotic <Psi_0|Psi_1> ~ fid ** n, where n is the number of sites,
    and Psi_0, Psi_1 are two many-body states.
    
    Input:
    
    A0 - tensor 0
    
    A1 - tensor 1
    
    tol - tolerance
        Routine converges when change in vectors is less than tol in norm.
        
    max_iter - maximum number of iterations
    
    optimize - optimization of Einstein summation
    
    power_iteration - whether to use power iteration to compute spectral radii for A0 and A1
    
    Output:
    
    fid - fidelity per site
        This is defined as rho01 / sqrt(rho0) / sqrt(rho1), where rho0 and rho1 are the spectral radii for A0 and A1,
        respectively, whereas rho01 is the spectral radius of the generalized transfer matrix which is bilinear in
        A0.conj() and A1. This quantity is within [0,1], and equals 1 iff A0 and A1 are gauge equivalent. See
        Eq. (141) of arXiv:1210.7710.
        
    Note:
    
    Broadcasting rule applies when A0 and A1 have rank 4.
    '''
    # Get transfer matrix of A0
    E0 = transfer_matrix(A0, optimize=optimize)
    # Get transfer matrix of A1
    E1 = transfer_matrix(A1, optimize=optimize)
    # Get cross-transfer matrix between A0 and A1
    E01 = np.einsum('...iab,...icd->...acbd', A1, A0.conj(), optimize=optimize)
    # Compute dominant eigenvalues
    if power_iteration:
        (z0, _, _, _) = fixed_point(E0, tol=tol, max_iter=max_iter, optimize=optimize, hermitianize=True,
                                   power_iteration=power_iteration)
        (z1, _, _, _) = fixed_point(E1, tol=tol, max_iter=max_iter, optimize=optimize, hermitianize=True,
                                   power_iteration=power_iteration)
    else:
        z0 = spectral_radius(E0)
        z1 = spectral_radius(E1)
    z01 = spectral_radius(E01)
    # Compute and return fidelity per site
    fid = abs( z01 / np.sqrt(z0) / np.sqrt(z1) )
    return fid

def orthogonality(A0, A1, tol=1e-10, max_iter=inf, optimize=False, power_iteration=False):
    '''Compute the orthogonality per site between two tensors
    
    Orthogonality is in the range [0, 1]. It is the effective angle between two vectors in a Hilbert space calculated
    from the fidelity per site between two tensors. A value of 1 indicates orthogonality and a value of 0 indicates
    proportionality.    
    
    Input:
    
    A0 - tensor 0
    
    A1 - tensor 1
    
    tol - tolerance
        Routine converges when change in vectors is less than tol in norm.
        
    max_iter - maximum number of iterations
    
    optimize - optimization of Einstein summation
    
    power_iteration - whether to use power iteration to compute spectral radii for A0 and A1
    
    Output:
    
    dis - orthogonality per site between two tensors
        This is defined as 2/pi*arccos(fid), where fid is the fidelity per site between A0 and A1.
        
    Note:
    
    Broadcasting rule applies when A0 and A1 have rank 4.
    '''
    fid = fidelity(A0, A1, tol=tol, max_iter=max_iter, optimize=optimize, power_iteration=power_iteration)
    fid = np.clip(fid, 0, 1) # clip values outside [0,1]
    dis = 2/pi*np.arccos(fid)
    return dis
    
def operator_basis(d, hermitian=True):
    '''Return operator basis in the physical space
    
    Input:
    
    d - physical dimension, integer
    
    hermitian - whether to return hermitian operator basis, boolean, default True
    
    Output:
    
    O - operator basis, ndarray of shape (d, d, d, d)
        If hermitian is False, then O[i, j] is the matrix with a single 1 at the (i, j) position.
        If hermitian is True, then O[i, j] has a single 1 at the (i, j) position if i == j; has 1
        at (i, j) and (j, i) if i < j; and has -1J at (i, j) and 1J at (j, i) if i > j.
    '''
    # Non-Hermitian basis
    operators = np.zeros((d,d,d,d))
    for i in range(d):
        for j in range(d):
            operators[i,j,i,j] = 1
            
    if not hermitian:
        return operators
        
    # Hermitianize
    O = []    # container of hermitian operators, ndarray of shape (d, d, d, d)
    for i in range(d):
        temp = []
        for j in range(d):
            if i == j:
                temp.append(operators[i,j])
            elif i < j:
                temp.append(operators[i,j]+operators[j,i])
            elif i > j:
                temp.append(-1.j*operators[i,j]+1.j*operators[j,i])
        O.append(temp)
    O = np.array(O)
    
    return O

class Ensemble:
    '''MPS ensemble
    
    Instantiation:
    
        Ensemble(A0, [E0,] ...)
        Ensemble(A0, [E0,] L0, R0, ...)
        Ensemble(A0, [E0,] L0, R0, L0_inv_sqrt, R0_inv_sqrt, V_L, ...)
        
    where ... are additional optional parameters:
    
        radius=1.
        max_dist=10
        optimize=False
        power_iteration=False
        
    --
    
    Parameters:
        
    A0 - basepoint tensor, ndarray of shape (d, chi, chi)
    
    E0 - basepoint transfer matrix, ndarray of shape (chi, chi, chi, chi)
    
    L0 - left dominant eigenvector, ndarray of shape (chi, chi)
        If provided, must be >=0 as a matrix and such that Tr(L0^\dag R0) = 1.
    
    R0 - right dominant eignevector, ndarray of shape (chi, chi)
        If provided, must be >=0 as a matrix and such that Tr(L0^\dag R0) = 1.
    
    L0_inv_sqrt - inverse square root of L0, ndarray of shape (chi, chi)
        If provided, must be calculated using prepare_tangent_space(A0, L0, R0) with parameters satisfying
        rho(E0) = 1, L0 >= 0, R0 >= 0, and Tr(L0^\dag R0) = 1, where E0 is the transfer matrix of A0.

    R0_inv_sqrt - inverse square root of R0, ndarray of shape (chi, chi)
        If provided, must be calculated using prepare_tangent_space(A0, L0, R0) with parameters satisfying
        rho(E0) = 1, L0 >= 0, R0 >= 0, and Tr(L0^\dag R0) = 1, where E0 is the transfer matrix of A0.

    V_L - auxiliary tensor, ndarray of shape (d, chi, (d - 1) * chi)
        The vectors V_L[:, :, i] for different i are orthonormal. See page 65 of arXiv:1210.7710 for detail.
        If provided, must be calculated using prepare_tangent_space(A0, L0, R0) with parameters satisfying
        rho(E0) = 1, L0 >= 0, R0 >= 0, and Tr(L0^\dag R0) = 1, where E0 is the transfer matrix of A0.
    
    radius - radius of ensemble, float, default 1.
    
    max_dist - maximum distance (inclusive) up to which 2-point correlations are calculated, integer, default 10
    
    optimize - whether to optimize numpy.einsum, boolean, default False
    
    power_iteration - whether to use power iteration to obtain fixed point, boolean, default False
    
    --
    
    Attributes:
    
    A0 - basepoint tensor, ndarray of shape (d, chi, chi)
        This is normalized after instantiation so that the spectral radius of E0 is 1.
    
    E0 - basepoint transfer matrix, ndarray of shape (chi, chi, chi, chi)
        This is normalized after instantiation to have spectral radius 1.
    
    L0 - left dominant eigenvector, ndarray of shape (chi, chi)
        This is >= 0 as a matrix and such that Tr(L0^\dag R0) = 1. If calculated in __init__ rather than given,
        then this additionally satisfies that L0 and R0 are normalized in Frobenius norm.
    
    R0 - right dominant eignevector, ndarray of shape (chi, chi)
        This is >= 0 as a matrix and such that Tr(L0^\dag R0) = 1. If calculated in __init__ rather than given,
        then this additionally satisfies that L0 and R0 are normalized in Frobenius norm.
        
    L0_inv_sqrt - inverse square root of L0, ndarray of shape (chi, chi)
    
    R0_inv_sqrt - inverse square root of R0, ndarray of shape (chi, chi)
    
    V_L - auxiliary tensor, ndarray of shape (d, chi, (d - 1) * chi)
        The vectors V_L[:, :, i] for different i are orthonormal. See page 65 of arXiv:1210.7710 for detail.
        
    d - physical dimension, integer
    
    chi - bond dimension, integer
    
    N_coordinates - real dimension of space of projective physical states described by MPS, integer
    
    N_correlations - number of correlations to be computed for each tensor
    
    radius - radius of ensemble, float
    
    max_dist - maximum distance (inclusive) up to which 2-point correlations are calculated, integer
    
    optimize - whether to optimize numpy.einsum, boolean
    
    power_iteration - whether to use power iteration to obtain fixed point, boolean
    
    O - basis of Hermitian physical operators on one site, ndarray of shape (d, d, d, d)
        This is calculated using operator_basis.
        
    --
    '''
    
    def __init__(self, A0, E0=None, L0=None, R0=None, L0_inv_sqrt=None, R0_inv_sqrt=None, V_L=None, radius=1.,
                max_dist=10, optimize=False, power_iteration=False):
        '''Initialize self'''
        # Basepoint tensor
        self.A0 = A0
        
        # Basepoint transfer matrix
        if E0 is None:
            self.E0 = transfer_matrix(self.A0, optimize=optimize)
        else:
            self.E0 = E0
        
        # Normalize A0 and E0 so that E0 has spectral radius 1
        rho0 = spectral_radius(self.E0)
        self.A0 = self.A0 / np.sqrt(rho0)
        self.E0 = self.E0 / rho0
        
        # Dominant eigenvectors 
        if L0 is None or R0 is None:
            # >= 0 as matrix and normalized in Frobenius norm
            _, self.L0, self.R0, _ = fixed_point(self.E0, flattened_output=False, optimize=optimize, power_iteration=power_iteration)
        else:
            # Must be >= 0 as matrix and such that Tr(L^\dag R) = 1
            self.L0 = L0
            self.R0 = R0
        
        # Prepare tangent space
        if L0_inv_sqrt is None or R0_inv_sqrt is None or V_L is None:
            self.L0_inv_sqrt, self.R0_inv_sqrt, self.V_L = prepare_tangent_space(self.A0, self.L0, self.R0, optimize=optimize)
        else:
            self.L0_inv_sqrt = L0_inv_sqrt
            self.R0_inv_sqrt = R0_inv_sqrt
            self.V_L = V_L
        
        # Ensemble radius
        self.radius = radius
        
        # 2-point function maximum distance (inclusive)
        self.max_dist = max_dist
        
        # Whether to optimize np.einsum
        self.optimize = optimize
        
        # Whether to use power_iteration
        self.power_iteration = power_iteration
                
        # Dimensions
        (d, chi, chi) = self.A0.shape
        self.d = d                            # physical dimension
        self.chi = chi                        # bond dimension
        self.N_coordinates = 2*(d-1)*chi**2   # real dimension of space of projective physical states described by MPS
        self.N_correlations = d**2 + d**4 * self.max_dist # number of correlations to be computed for each tensor
        
        # Initialize physical operators
        self.O = operator_basis(self.d, hermitian=True)
    
    def __repr__(self):
        return ("MPS ensemble\n\nParameters:\n\n" +
                "d = " + str(self.d) + "\n\n" +
                "chi = " + str(self.chi) + "\n\n" +
                "A0 = \n" + str(self.A0) + "\n\n" +
                "E0 = \n" + str(self.E0) + "\n\n" +
                "L0 = \n" + str(self.L0) + "\n\n" +
                "R0 = \n" + str(self.R0) + "\n\n" +
                "L0_inv_sqrt = \n" + str(self.L0_inv_sqrt) + "\n\n" +
                "R0_inv_sqrt = \n" + str(self.R0_inv_sqrt) + "\n\n" +
                "V_L = \n" + str(self.V_L) + "\n\n" +
                "radius = " + str(self.radius) + "\n\n" +
                "max_dist = " + str(self.max_dist)
               )
    
    def __str__(self):
        return "MPS ensemble with basepoint\n" + str(self.A0)
        
    def sample_coordinates_in_ball(self, n_samples):
        '''Sample coordinates uniformly from a ball of radius self.radius
        
        Input:
        
        self
        
        n_samples - sample size, integer
        
        Output:
        
        labels - sampled coordinates, ndarray of shape (n_samples, self.N_coordinates)
        '''
        labels = np.random.normal(size=(n_samples, self.N_coordinates))              # normal distribution
        length = np.linalg.norm(labels, axis=1)                                        # length
        normalization = np.random.rand(n_samples)                                    # normalization        
        normalization = normalization ** (1/self.N_coordinates)                        # larger radius more likely to occur
        labels = np.einsum('i,ij->ij', self.radius * (normalization / length), labels) # uniform distribution in ball
        return labels
    
    def sample_coordinates_on_sphere(self, n_samples):
        '''Sample coordinates uniformly from a sphere of radius self.radius
        
        Input:
        
        self
        
        n_samples - sample size, integer
        
        Output:
        
        labels - sampled coordinates, ndarray of shape (n_samples, self.N_coordinates)
        '''
        labels = np.random.normal(size=(n_samples, self.N_coordinates))              # normal distribution
        length = np.linalg.norm(labels, axis=1)                                        # length
        labels = np.einsum('i,ij->ij', self.radius / length, labels)                   # uniform distribution in ball
        return labels
    
    def sample_coordinates_uniform_in_radius(self, n_samples):
        '''Sample coordinates from a ball such that radius is uniform in [0, self.radius]
        
        Input:
        
        self
        
        n_samples - sample size, integer
        
        Output:
        
        labels - sampled coordinates, ndarray of shape (n_samples, self.N_coordinates)
        '''
        labels = np.random.normal(size=(n_samples, self.N_coordinates))              # normal distribution
        length = np.linalg.norm(labels, axis=1)                                        # length
        normalization = np.random.rand(n_samples)                                    # normalization
        labels = np.einsum('i,ij->ij', self.radius * (normalization / length), labels) # radius is uniform in [0, self.radius]
        return labels
    
    def convert_coordinates_to_tensors(self, coordinates):
        '''Convert coordinates to tensors
        
        Input:
        
        self
        
        coordinates - coordinates, ndarray of shape (n_samples, self.N_coordinates)
            coordinates[i,::2] and coordinates[i,1::2] are the real and imaginary part of
            flattened X for example i, where X is the input to left_canonical_tangent
            that parameterizes tangent vectors.
        
        Output:
        
        A - corresponding tensors, ndarray of shape (n_samples, self.d, self.chi, self.chi)
        '''
        # Assemble real vector into complex array
        coordinates_flattened = np.ndarray.flatten(coordinates)
        X = coordinates_flattened[::2] + 1J * coordinates_flattened[1::2]
        X = X.reshape((-1, (self.d-1)*self.chi, self.chi))
        # Compute tangents
        B = left_canonical_tangent(self.L0_inv_sqrt, self.R0_inv_sqrt, self.V_L, X, optimize=self.optimize)     
        # Add tangents to A0
        A = np.repeat([self.A0], len(coordinates), axis=0) + B
        return A
    
    def convert_tensors_to_normality_indicators(self, A):
        '''Convert tensors to normality indicators
        
        Input:
        
        self
        
        A - tensors, ndarray of shape (n_samples, d, chi, chi)
        
        Output:
        
        w - relative spectral gap, ndarray of shape (n_samples)
        
        det_r_lc - determinant of dominant right eigenvector in left-canonical form, ndarray of shape (n_samples)
        '''
        (w, _, _, det_r_lc) = normality_indicators(transfer_matrix(A))
        return (w, det_r_lc)
    
    def convert_tensors_to_correlations(self, A):
        '''Convert tensors to correlations
        
        Input:
        
        self
        
        A - tensors, ndarray of shape (n_samples, d, chi, chi)
        
        Output:
        
        cor - correlations, ndarray of shape (n_samples, N_correlations)
        '''        
        # List to store 1-point, 2-point, ... correlation functions
        cor = []
        
        # Compute dominant eigenvectors
        E = transfer_matrix(A, optimize=self.optimize)
        (rho, L, R, _) = fixed_point(E, optimize=self.optimize, power_iteration=self.power_iteration)
        
        # Compute transfer matrices with insertions
        E_ins = []
        for i in range(self.d):
            temp = []
            for j in range(self.d):
                temp.append(transfer_matrix(A, O=self.O[i,j], optimize=self.optimize))
            E_ins.append(temp)
        E_ins = np.array(E_ins)
        
        # Compute 1-point correlation functions
        for i in range(self.d):
            for j in range(self.d):
                ex = one_point_function(A, self.O[i,j], rho=rho, l=L, r=R, E_O=E_ins[i,j],
                                          optimize=self.optimize).real
                     # only keep real part since operator is hermitian
                cor.append(ex) # ex is an n_samples-dimensional vector
        
        # Compute connected 2-point correlation functions
        for i in range(self.d):
            for j in range(self.d):
                for k in range(self.d):
                    for l in range(self.d):
                        for x in range(1, self.max_dist + 1):
                            ex = two_point_function(A, self.O[i,j], 0, self.O[k,l], x, connected=True,
                                                    rho=rho, l=L, r=R, E=E, E0=E_ins[i,j], E1=E_ins[k,l],
                                                    optimize=self.optimize).real
                                                    # only keep real part since operator is hermitian
                            cor.append(ex)
        
        # Return correlation functions
        cor = np.array(cor).T  # cor[i] contains the correlations functions of the i-th tensor
        return cor
        
    def make_data(self, n_samples, sampling_func=None, as_dataframe=True):
        '''Make data for training or testing purposes
        
        Input:
        
        n_samples - sample size, integer
        
        sampling_func - a callable such that sampling_func(n_samples) returns sampled coordinates
        
        as_dataframe - whether to return dataframe or a tuple, boolean, default True
        
        Output:
        
        If as_dataframe is False, this returns a tuple X, y, where X are the correlations (input) and
        y are the coordinates (output), as calculated using self.convert_tensors_to_correlations and
        sampling_func, respectively.
        
        If as_dataframe is True, this returns a pandas DataFrame with columns
        
            O_00, O_01, O_02, ...,
            O_00(0)O_00(1), O_00(0)O_00(2), ...,
            Re(X_00), Im(X_00), Re(X_01), Im(X_01), ...
            
        where the first N_correlations columns are the X and the last N_coordinates columns
        are the y mentioned above.
        '''
        # Set default sampling function
        if sampling_func is None:
            sampling_func = self.sample_coordinates_uniform_in_radius
        
        # Make data
        labels = sampling_func(n_samples)
        correlations = self.convert_tensors_to_correlations(self.convert_coordinates_to_tensors(labels))
        
        if not as_dataframe:
            # Return input and output
            return correlations, labels
        else:
            # Column names for dataframe
            columns_one_point = [f'O_{i}{j}' for i in range(self.d) for j in range(self.d)]
            columns_two_point = [f'O_{i}{j}(0)O_{k}{l}({x})' for i in range(self.d)
                                                             for j in range(self.d)
                                                             for k in range(self.d)
                                                             for l in range(self.d)
                                                             for x in range(self.max_dist)]
            columns_coordinates = [f'{p}(X_{i}{j})' for i in range((self.d - 1) * self.chi)
                                                    for j in range(self.chi)
                                                    for p in ['Re', 'Im']]
            columns = columns_one_point + columns_two_point + columns_coordinates
            
            # Concatenate data
            data = np.concatenate((correlations, labels), axis=1)
            
            # Return dataframe
            return pd.DataFrame(data=data, columns=columns)
            
        
