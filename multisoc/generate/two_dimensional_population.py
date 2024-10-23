import numpy as np

##############################################################################
## Tunable consolidation for two binary populations
##############################################################################

def two_bin_comp_pop_frac_tnsr(bf,pop_fracs_lst):
    """Two-dimensional population distribution from multidimensional minority
    
    This parameters' names come from my original presentation on how to
    tune consolidation in a system with two binary populations
    bf-black females (intersectional minority)
    wf-white females
    bm-black males
    wm-white males (intersectional majority)
    
    Parameters
    ----------
    bf : float
        Size of the smallest group (multidimensional minority)

    pop_fracs_lst : Iterable[np.ndarray]
        List of population marginals
    
    Returns
    -------
    np.ndarray
        2D population distribution
    """
    wf = pop_fracs_lst[1,0] - bf
    bm = pop_fracs_lst[0,0] - bf
    wm = 1 + bf - pop_fracs_lst[1,0] - pop_fracs_lst[0,0]
    return np.array([[bf,bm],[wf,wm]])

def consol_comp_pop_frac_tnsr(pop_fracs_lst,consol):
    """Generate a two-dimensional population distribution tuning correlation

    
    Parameters
    ----------
    pop_fracs_lst : Iterable[np.ndarray]
        List of population marginals

    consol : float
        Correlation parameter (minimum=0, maximum=1.)
    
    Returns
    -------
    np.ndarray
        2D opulation distribution
    """
    assert 0 <= consol <= 1

    pop_fracs_lst = np.array(pop_fracs_lst)

    ## Verify population fractions are ordered in increasing order
    ## Easier to compute next stuff
    for pop_fracs in pop_fracs_lst:
        assert np.all(np.sort(pop_fracs) == pop_fracs)

    bf_MC = min(pop_fracs_lst[0,0],pop_fracs_lst[1,0])
    bf_AC = 0.0

    comp_pop_frac_tnsr_MC = two_bin_comp_pop_frac_tnsr(bf_MC,pop_fracs_lst)
    comp_pop_frac_tnsr_AC = two_bin_comp_pop_frac_tnsr(bf_AC,pop_fracs_lst)

    return consol * comp_pop_frac_tnsr_MC + (1.0-consol) * comp_pop_frac_tnsr_AC


def relative_correlation(f2m: float,corr: float) -> float:
    """Compute relative correlation parameter
    
    Computes relative correlation parameter in interval [-1,1] with neutral
    value 0 from original correlation paramter in interval [0,1] with neutral
    value max(f1m,f2m)=f2m.
    
    Parameters
    ----------
    f2m : float
        Minority fraction from dimension 2 (largest minority)
    corr : float
        Original correlation value
    
    Returns
    -------
    float
        Relative correlation value between -1 and 1
    """
    assert 0 <= f2m <= 1
    assert 0 <= corr <= 1
    return (corr-f2m)/(1-f2m) if corr>f2m else (corr-f2m)/f2m

def relative_correlation_inv(f2m: float,rel_corr: float) -> float:
    """Compute original correlation parameter from relative
    
    Computes the original correlation parameter in interval [0,1] with neutral
    value f2m from relative correlation paramter in interval [-1,+1] with 
    neutral value 0.
    
    Parameters
    ----------
    f2m : float
        Minority fraction from dimension 2 (largest minority)
    rel_corr : float
        Relative correlation parameter
    
    Returns
    -------
    float
        Original correlation parameter
    """
    assert 0 <= f2m <= 1
    assert -1 <= rel_corr <= 1
    return (1-f2m)*rel_corr+f2m if rel_corr>0 else f2m*rel_corr+f2m