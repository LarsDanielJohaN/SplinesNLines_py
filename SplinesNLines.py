
import numpy as np
import scipy.stats as sts
import torch
from torch import linalg as LA

import scipy.stats as sts
import torch
from torch import linalg as LA


"""
X_a: array of design matrices, each of dimension (Ko x B), of length O
f_a: array of observation values, vectors of dimension Ko,  of length O
T_a: array of precision matrices, dimensions (Ko x Ko),  of length O

Note: Ko can be different for different values of o = 1,2,...,O. 
"""
def get_mat_comp(X_a, Tf_a, T_a, idx_down, idx_up):
    O = len(X_a)
    B = X_a[0].shape[1]
    XtXf = torch.zeros((O, B)).float()
    XtTX = torch.zeros((O, B, B)).float()

    for o in range(O):
        XtXf[o, :] = X_a[o].T@(Tf_a[idx_down[o]:idx_up[o]+1]).squeeze()
        XtTX[o, :, :] = X_a[o].T@torch.diag(T_a[idx_down[o]:idx_up[o]+1])@X_a[o]

    return {'XtXf':XtXf, 'XtTX':XtTX}


def get_mat_comp(X_a, f_a, T_a):
    O = len(X_a)
    B = X_a[0].shape[1]
    XtXf = np.zeros((O, B))
    XtTX = np.zeros((O, B, B))
    for o in range(O):
        XtXf[o, :] = X_a[o].T@T_a[o]@f_a[o].squeeze()
        XtTX[o, :, :] = X_a[o].T@T_a[o]@X_a[o]
    return {'XtXf':XtXf, 'XtTX':XtTX}

"""

def EM_alg(XtXf, XtTX, n_max, tol ):
    B = XtTX.shape[1]
    O = XtXf.shape[0]
    S =  torch.from_numpy(np.eye(B) ) #Gets first estimate for S
    mu = torch.from_numpy(np.random.uniform(size = B).reshape(-1,1) )#Gets first estimate for mu. 
    m0 = torch.zeros(O, B)
    XtTX = torch.from_numpy(XtTX) #Converts XtTX to a torch tensor. 
    XtXf = torch.from_numpy(XtXf)
    m0_outer = torch.zeros(O, B, B) #Creates a torch tensor to store outer products of m0's. 

    i = 0 
    diff_S = tol + np.pi 
    diff_mu = tol + np.pi
    while (i <n_max ) and ( (diff_S**2/B**2 > tol) or (diff_mu > tol) ):
        if i % 1000 == 0:
            print("Iteration:  ", i)
        Si = LA.inv(S) #Calculates inverse of current iteration for variance covariance. 

        C0 = LA.inv(Si + XtTX) #Calculates C0's = inv(Si + XtXT), I hope (O, B, B)

        f = (Si.float()@mu.float()).squeeze(-1)

        m0 = torch.bmm(C0, (XtXf + f).unsqueeze(-1)).squeeze(-1)

        m0_outer = torch.bmm(m0.unsqueeze(-1), m0.unsqueeze(-2))

        #for o in range(O): #Gets the outer product of individual mean estimates. 
        #    m0[o,:] = C0[o, :, :]@(XtXf[o,:] + f)
        #    m0_outer[o,:,:] = m0[o,:]@m0[o,:].T #Gets inner product of the o-th mean estimate.

        mu_n = torch.mean(m0, 0).reshape(-1,1) #Gets new mean estimate. 
        S_n = torch.mean(C0 + m0_outer, axis = 0) - mu_n@mu_n.T #Gets new iteration for S_n
        diff_S = LA.matrix_norm(S_n - S) #Computes matrix norm of current iteration and previous one. 
        diff_mu = LA.matrix_norm(mu_n - mu) #Computes matrix norm of current iteration and previous one. 
        mu = mu_n
        S = S_n 
        i+= 1
    return {'mu_h':mu_n, 'S_h':S_n, 'm0':m0, 'Si_h':Si}

"""

"""
x_e: numpy array of values to evaluate. 
loc_vals: numpy array of location parameters for the line profiles. 
width_vals: numpy array of the widths for the line profiles. 

returns a len(x_e) x len(valid profiles) with evaluated line profiles with gaussian kernels. 

Note: The profiles considered as valid are those with width greater than zero. 
"""


def eval_Line_Profiles(x_e, loc_vals, width_vals):
    width_greater_than_zero = (width_vals > 0) #Selects the valid profiles. 
    width_vals_e = width_vals[width_greater_than_zero] #Selects the valid widths. 
    loc_vals_e = loc_vals[width_greater_than_zero] #Selects the valid loc values. 

    L = len(loc_vals_e) #Takes the total number of valid profiles. 
    L1 = lambda x: sts.norm.pdf(x, loc = loc_vals_e[0], scale = width_vals_e[0]) #Defines first gaussian kernel. 
    L1 = np.vectorize(L1) 
    X = L1(x_e)

    for i in range(1,L):
        Li = lambda x: sts.norm.pdf(x, loc = loc_vals_e[i], scale = width_vals_e[i]) #Defines i-th gaussian kernel. 
        Li = np.vectorize(Li)
        X = np.vstack((X, Li(x_e)))

    return X.T


def eval_Line_Profiles_opt(x_e, loc_vals, width_vals):
    #width_greater_than_zero = (width_vals > 0) #Selects the valid profiles. 
    #width_vals_e = width_vals[width_greater_than_zero] #Selects the valid widths. 
    #loc_vals_e = loc_vals[width_greater_than_zero] #Selects the valid loc values. 
    X = sts.norm.pdf(x_e[:, np.newaxis], loc=loc_vals, scale= width_vals )

    return X


"""
T: numpy array of knots
x: Value to evaluate
i: element in the basis
m: value such that m+1 is the degree of B^m+1(x)
"""

def eval_B_Spline(x, T, i, m):
    if m == -1:
        return float( (x < T[i]) & (x >= T[i-1])    )
    else:
        z0 = (x- T[i-1])/(T[i+m] - T[i-1])
        z1 = (T[i+m+1] - x)/(T[i+m+1] - T[i])
        return z0*eval_B_Spline(x, T, i, m-1) + z1*eval_B_Spline(x, T, i+1, m-1)


"""
get_basis_mat_B_Spline, gets design matrix for a B-Spline basis.
T: numpy array of knots
x_e: numpy array of values to evaluate.
B: Number of elements in basis. 
m: Values such that m+1 is the degree of B^m+1(x)

returns a (len(x_e) x B) matrix whose entries correspond to the elements in x_e evaluated at the 
basis functions. 

Note: Be aware that, as B-Splines are evaluated recursivelly, x_e's of great length may take long to compute. 
"""
def get_basis_mat_B_Spline(x_e, B, m, T):
    B1 = lambda x: eval_B_Spline(x, T, 1, m)
    B1 = np.vectorize(B1)
    X = B1(x_e)
    for i in range(1, B):
        Bi = lambda x: eval_B_Spline(x, T, i+1, m)
        Bi = np.vectorize(Bi)
        X = np.vstack((X, Bi(x_e)))
    return X.T

def get_basis_mat_B_Spline_opt(x_e, B, m, T):
    X = np.zeros((len(x_e), B))
    # Compute each basis function in a vectorized way
    for i in range(B):
        X[:, i] = eval_B_Spline_opt(x_e, T, i+1, m)  # Assuming eval_B_Spline is vectorized
    return X



def eval_B_Spline_opt(x, T, i, m):
    """
    Vectorized B-Spline evaluation (degree m+1).
    Works for scalar or array x.
    """
    if m == -1:
        # Base case: indicator function (vectorized)
        return np.where((T[i-1] <= x) & (x < T[i]), 1.0, 0.0)
    else:
        # Initialize terms
        term1 = np.zeros_like(x)
        term2 = np.zeros_like(x)
        
        # Compute denominators safely (avoid division by zero)
        denom1 = T[i + m] - T[i - 1]
        denom2 = T[i + m + 1] - T[i]
        
        # Masks for valid denominators
        mask1 = denom1 != 0
        mask2 = denom2 != 0
        
        # Recursive terms (only computed where denominators are valid)
        term1[mask1] = (x[mask1] - T[i - 1]) / denom1 * eval_B_Spline_opt(x[mask1], T, i, m - 1)
        term2[mask2] = (T[i + m + 1] - x[mask2]) / denom2 * eval_B_Spline_opt(x[mask2], T, i + 1, m - 1)
        
        return term1 + term2
"""
get_basis_mat_Splines, gets design matrix for Spline basis with knots at T, evaluated at x_e
x_e: points to evaluate, expected to be of dimension (len(x_e),)
m: degree of Splines 
T: Knots, expected to be of dimension (len(T), )

splines are of the form 
S(x) = x**0 + x**1 + ... + x**m + (k1 - x)_+**2 + (k2 - x)_+**2 + ... + (k_len(T) - x)_+**2
where (x)_+ = x if x >= 0 and 0 e.o.c. 

Note: B-Splines are a more stable alternative, the use of this method is not recomended. 
"""


def get_basis_mat_Splines(x_e, m, T):
    K_rep = np.ones((len(x_e), len(T)))@np.diag(T) # len(x_e) x len(T) i.e. evaluation points x knots
    X_S = np.repeat(x_e.reshape(-1,1), len(T), axis = 1) - K_rep
    X_S = X_S*( X_S >= 0)
    X_P = np.ones(len(x_e))

    for i in range(m):
        X_P = np.vstack((X_P, x_e**(i+1)))

    X = np.hstack((X_P.T, X_S))
    return X





def EM_alg(XtXf, XtTX, n_max, tol ):
    B = XtTX.shape[1]
    O = XtXf.shape[0]
    L = np.random.normal(size = B).reshape(-1,1)
    S =  torch.from_numpy(10*L@L.T + np.eye(B) ) #Gets first estimate for S
    mu = torch.from_numpy(np.random.uniform(10, 100, size = B).reshape(-1,1) )#Gets first estimate for mu. 
    m0 = torch.zeros(O, B)
    XtTX = torch.from_numpy(XtTX) #Converts XtTX to a torch tensor. 
    XtXf = torch.from_numpy(XtXf)
    m0_outer = torch.zeros(O, B, B) #Creates a torch tensor to store outer products of m0's. 

    i = 0 
    diff_S = tol + np.pi 
    diff_mu = tol + np.pi
    while (i <n_max ) and ( (diff_S**2/B**2 > tol) or (diff_mu > tol) ):
        if i % 1000 == 0:
            print("Iteration:  ", i)

        Si = LA.inv(S) #Calculates inverse of current iteration for variance covariance. 
        C0 = LA.inv(Si + XtTX) #Calculates C0's = inv(Si + XtXT), I hope (O, B, B)
        f = (Si.float()@mu.float()).squeeze(-1)
        m0 = torch.bmm(C0, (XtXf + f).unsqueeze(-1)).squeeze(-1)
        m0_outer = torch.bmm(m0.unsqueeze(-1), m0.unsqueeze(-2))
        mu_n = torch.mean(m0, 0).reshape(-1,1) #Gets new mean estimate. 
        S_n = torch.mean(C0 + m0_outer, axis = 0) - mu_n@mu_n.T #Gets new iteration for S_n
        diff_S = LA.matrix_norm(S_n - S) #Computes matrix norm of current iteration and previous one. 
        diff_mu = LA.matrix_norm(mu_n - mu) #Computes matrix norm of current iteration and previous one. 
        mu = mu_n
        S = S_n 
        i+= 1

    return {'mu_h':mu_n, 'S_h':S_n, 'm0':m0, 'Si_h':Si}




