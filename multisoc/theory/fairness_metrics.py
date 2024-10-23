import numpy as np
from scipy.stats import skellam

from multisoc.generate.utils import make_composite_index
from multisoc.generate.utils import comp_index_to_integer

#############################################################################
## Analytical computation of Common Language (CL) delta measure for ER-style
## networks. DIRECTED VERSION.
#############################################################################

def analytical_multidim_expected_indegree(r,H,F,N):
    """
	Compute expected degree of multidimensional group r in an Erdos-Renyi 
    or SBM type of network given the connection probabilities H, the population
    distribution F, and the number of nodes N.

	Parameters
	----------
	r: tuple
		Attribute vector of group r.
    H: numpy.ndarray
        Matrix of connection probabilities where entry (R,S) is the probability
        of group R to connect to group S.
    F: numpy.ndarray
        Tensor where the element (r1,r2,r3,...,rd) contains the population fraction
        of group r with attribute vector (r1,r2,...,rd).
    N: int
        Number of nodes
	Returns
	-------
	result: float
		Expected in-degree for group r.
	"""
    ## Extract H values as vector
    g_vec = F.shape
    R = comp_index_to_integer(r,g_vec)
    H_col = H[:,R]
    ## Convert F tensor to vector
    comp_indices = make_composite_index(g_vec)
    F_vec = [F[s] for s in comp_indices]
    expected_in_degree = N * np.dot(H_col, F_vec)
    return expected_in_degree

def analytical_multidimensional_delta_num_cdf(r,s,H,F,N,get_probs=False):
    """
	Compute the delta inequality metric for the degree of two 
    groups r, s measuring how much advantage has r over s.
    Notice that the metric is antisymmetric.

	Parameters
	----------
	r: tuple
		Attribute vector of group r.
    s: tuple
        Attribute vector of group s.
    H: numpy.ndarray
        Matrix of connection probabilities where entry (R,S) is the probability
        of group R to connect to group S.
    F: numpy.ndarray
        Tensor where the element (r1,r2,r3,...,rd) contains the population fraction
        of group r with attribute vector (r1,r2,...,rd).
    N: int
        Number of nodes
	Returns
	-------
	result: float
		Value for the delta inequality metric.
	"""
    
    ## Poissonian parameters (expected number of links)
    lambda1 = analytical_multidim_expected_indegree(r,H,F,N)
    lambda2 = analytical_multidim_expected_indegree(s,H,F,N)
    
    ## Cumulative probability for >0 and <0 according to Skellam
    p_upper = skellam.sf(0,lambda1,lambda2) ## p(x1 > x2) = p(x1-x2 > 0) = 1 - p(x1-x2 <= 0)
    p_lower = skellam.cdf(-1,lambda1,lambda2) ## p(x1 < x2) = p(x1-x2 < 0) = p(x1-x2 <= -1)

    if get_probs:
        return p_upper - p_lower, p_upper, p_lower
    else:
        return p_upper - p_lower
    
def analytical_onedim_1vRest_delta_from_multidim_delta(d,x,F,multidim_deltas):
    """
	Compute the one vs rest delta inequality metric for one dimensional groups.

	Parameters
	----------
	d: int
		Dimension index, starting from 0.
    x: int
        Particular one dimensional group or attribute value within dimension d.
    F: numpy.ndarray
        Tensor where the element (r1,r2,r3,...,rd) contains the population fraction
        of group r with attribute vector (r1,r2,...,rd).
    multidim_deltas: dict 
        Dictionary with {key:value} being {(r_vec, s_vec):delta}. Keys are tuples
        of tuples.
	Returns
	-------
	result: float
		Value for the delta inequality metric.
	"""
    g_vec = F.shape
    indices_lst = make_composite_index(g_vec)
    
    norm_constant = 0
    total_delta = 0
    for i_vec in indices_lst:
        for j_vec in indices_lst:
            if i_vec[d] == x and j_vec[d] != x:
                ## VERIFY THIS!! I'm not sure if it is reasonable to 
                ## replace NaN by 0 here!!
                if np.isnan(multidim_deltas[(i_vec, j_vec)]):
                    pass
                else:
                    total_delta += F[i_vec]*F[j_vec]*multidim_deltas[(i_vec, j_vec)]
                    norm_constant += F[i_vec] * F[j_vec]
    
    if total_delta == norm_constant == 0:
        return np.nan
    else:
        return total_delta / norm_constant

def analytical_multidimensional_1vRest_delta_from_multidim_delta(r,F,multidim_deltas):
    """
	Compute the one vs rest delta inequality metric for multidimensional groups.

	Parameters
	----------
	r: tuple
		Attribute vector of group r.
    F: numpy.ndarray
        Tensor where the element (r1,r2,r3,...,rd) contains the population fraction
        of group r with attribute vector (r1,r2,...,rd).
    multidim_deltas: dict 
        Dictionary with {key:value} being {(r_vec, s_vec):delta}. Keys are tuples
        of tuples.
	Returns
	-------
	result: float
		Value for the delta inequality metric.
	"""
    g_vec = F.shape
    indices_lst = make_composite_index(g_vec)
    
    norm_constant = 0
    total_delta = 0
    
    for i_vec in indices_lst:
        if np.all(i_vec != r):
            ## VERIFY THIS!! I'm not sure if it is reasonable to 
            ## replace NaN by 0 here!!
            if np.isnan(multidim_deltas[(r, i_vec)]):
                pass
            else:
                total_delta += F[i_vec] * multidim_deltas[(r, i_vec)]
            norm_constant += F[i_vec]
    
    return total_delta / norm_constant

def analytical_1v1_multidimensional_deltas(H,F,N,get_probs=False):
    """
	Compute all the one vs one delta inequality metric for one-dimensional groups.

	Parameters
	----------
	H: numpy.ndarray
        Matrix of connection probabilities where entry (R,S) is the probability
        of group R to connect to group S.
    F: numpy.ndarray
        Tensor where the element (r1,r2,r3,...,rd) contains the population fraction
        of group r with attribute vector (r1,r2,...,rd).
    N: int 
        Number of nodes.
	Returns
	-------
	result: dict
		{(r,s):delta}
	"""
    g_vec = F.shape
    indices_lst = make_composite_index(g_vec)
    
    multidim_deltas = {}
    if get_probs:
        multidim_deltas_upper = {}
        multidim_deltas_lower = {}
    for r in indices_lst:
        for s in indices_lst:
            if get_probs:
                multidim_deltas[(r, s)], multidim_deltas_upper[(r,s)], multidim_deltas_lower[(r,s)] = analytical_multidimensional_delta_num_cdf(r,s,H,F,N,get_probs=get_probs)
            else:
                multidim_deltas[(r, s)] = analytical_multidimensional_delta_num_cdf(r,s,H,F,N,get_probs=get_probs)
    
    if get_probs:
        return multidim_deltas, multidim_deltas_upper, multidim_deltas_lower
    else:
        return multidim_deltas

def analytical_1vRest_onedimensional_deltas(H,F,N):
    """
	Compute all the one vs rest delta inequality metric for one-dimensional groups.

	Parameters
	----------
	H: numpy.ndarray
        Matrix of connection probabilities where entry (R,S) is the probability
        of group R to connect to group S.
    F: numpy.ndarray
        Tensor where the element (r1,r2,r3,...,rd) contains the population fraction
        of group r with attribute vector (r1,r2,...,rd).
    N: int 
        Number of nodes.
	Returns
	-------
	result: dict
		{r:delta}
	"""
    multidim_deltas = analytical_1v1_multidimensional_deltas(H,F,N)
    
    ndim = F.ndim
    g_vec = F.shape
    
    onedim_deltas_1vRest = {}
    for d in range(ndim):
        onedim_deltas_1vRest[d] = {}
        for vi in range(g_vec[d]):
            onedim_deltas_1vRest[d][vi] = analytical_onedim_1vRest_delta_from_multidim_delta(d,vi,F,multidim_deltas)
    
    return onedim_deltas_1vRest

def analytical_1vRest_multidimensional_deltas(H,F,N):
    """
	Compute all the one vs rest delta inequality metric for multidimensional groups.

	Parameters
	----------
	H: numpy.ndarray
        Matrix of connection probabilities where entry (R,S) is the probability
        of group R to connect to group S.
    F: numpy.ndarray
        Tensor where the element (r1,r2,r3,...,rd) contains the population fraction
        of group r with attribute vector (r1,r2,...,rd).
    N: int 
        Number of nodes.
	Returns
	-------
	result: dict
		{dimension:{attribute_value:delta}}
	"""
    multidim_deltas = analytical_1v1_multidimensional_deltas(H,F,N)
    
    g_vec = F.shape
    indices_lst = make_composite_index(g_vec)
    
    multidim_deltas_1vRest = {}
    for r in indices_lst:
        multidim_deltas_1vRest[r] = analytical_multidimensional_1vRest_delta_from_multidim_delta(r,F,multidim_deltas)

    return multidim_deltas_1vRest