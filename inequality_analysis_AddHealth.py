import pandas as pd
import itertools
import numpy as np

from multisoc.infer import aux_functions
from multisoc.metrics import fairness_metrics
from multisoc.theory.fairness_metrics import analytical_1vRest_onedimensional_deltas
from multisoc.generate.utils import make_composite_index

def degree_seq_from_nodes_edges_df(
    nodes_list,
    edges_list,
    dimensions_list):

    nodes_list2 = nodes_list.join(pd.DataFrame(edges_list.target.value_counts()),how="left",validate="one_to_one")
    nodes_list2["count"] = nodes_list2["count"].fillna(0)
    deg_seq_dict = {}

    # Create an empty dictionary to store the result
    deg_seq_dict = {}

    for attribute in dimensions_list:
        deg_seq_dict[attribute] = {}
        for category in nodes_list2[attribute].unique():

            subset_df = nodes_list2[nodes_list2[attribute] == category]
            
            deg_seq_dict[attribute][category] = subset_df["count"].to_numpy()
            
    return deg_seq_dict

def multi_degree_seq_from_nodes_edges_df(
    nodes_list,
    edges_list,
    multidim_groups,
    dimensions_list):
    
    nodes_list2 = nodes_list.join(pd.DataFrame(edges_list.target.value_counts()),how="left",validate="one_to_one")
    nodes_list2["count"] = nodes_list2["count"].fillna(0)
    deg_seq_dict = {}
    
    # Create an empty dictionary to store the result
    deg_seq_dict = {}
    
    for group in multidim_groups:
        msk = nodes_list2[dimensions_list[0]] == group[0]
        for i in range(1,len(group)):
            msk = np.logical_and(msk, nodes_list2[dimensions_list[i]] == group[i])
        subset_df = nodes_list2[msk]
        deg_seq_dict[group] = subset_df["count"].to_numpy()

    return deg_seq_dict

def inequalities_from_degree_seq_dict(degree_seq_dict):
    attr_vals = list(degree_seq_dict.keys()) ## Fix order of attributes
    x_lst = [degree_seq_dict[v] for v in attr_vals]
    res = fairness_metrics.CL_delta_groups_1vRest(x_lst)
    res_dct = dict(zip(attr_vals,res))
    return res_dct

def all_emp_ineq_from_nodes_edges_df(
    nodes_list,
    edges_list,
    dimensions_list):

    deg_seq_dict = degree_seq_from_nodes_edges_df(nodes_list,edges_list,dimensions_list)
    result_dict = {}
    for d, seq_dict in deg_seq_dict.items():
        result_dict[d] = inequalities_from_degree_seq_dict(seq_dict)
            
    return result_dict

def all_multi_emp_ineq_from_nodes_edges_df(
    nodes_list,
    edges_list,
    multidim_groups,
    dimensions_list):
    
    deg_seq_dict = multi_degree_seq_from_nodes_edges_df(nodes_list,edges_list,multidim_groups,dimensions_list)
    result_dict = inequalities_from_degree_seq_dict(deg_seq_dict)
            
    return result_dict
    

def get_H_from_results_df(results_df_row,multidim_groups,aggr_fun="and_1d-simple"):
    if len(results_df_row) != 1:
        raise ValueError(f"A dataframe of length 1 is needed for results_df_row. A data frame of length f{len(results_df_row)} was passed.") ## Fails for dataframes with more than 1 row or with 0 rows
    H = np.zeros((len(multidim_groups),len(multidim_groups))) + np.nan
    for i, col1 in enumerate(multidim_groups):
        for j, col2 in enumerate(multidim_groups):
            col1col2 = "H_" + aggr_fun + "_" + "|".join(col1) + "-" + "|".join(col2)
            H[i,j] = results_df_row[col1col2].iloc[0]
    return H

def H_df_from_matrix(H,multidim_groups):
    return pd.DataFrame(H,columns=multidim_groups,index=multidim_groups)

def get_H_from_H_df(H_df):
    H = np.zeros((len(multidim_groups),len(multidim_groups))) + np.nan
    for i, col1 in enumerate(multidim_groups):
        for j, col2 in enumerate(multidim_groups):
            ## To extract info from a pandas dataframe it goes like df[column][row] instead of [row][column] like an array
            try:
                H[i,j] = H_df[col2][col1]
            except KeyError:
                pass
    return H

def get_F_from_data(nodes_list,edges_list,attributes_dict,multidim_groups,dimensions_list):
    
    n, _ = aux_functions.get_n_and_counts(nodes_list,edges_list,dimensions_list)
    F = np.zeros(tuple([len(attributes_dict[i]) for i in dimensions_list]))
    g_vec = F.shape
    indices_lst = make_composite_index(g_vec)
    
    for i, g in enumerate(multidim_groups):
        try:
            F[indices_lst[i]] = n[g].iloc[0]
        except KeyError:
            pass
            
    if np.sum(F) == 0:
        raise Error("Something went wrong in the computation of F: everything is 0")
        
    return F
            
def anal_inequalities_1D_from_data(nodes_list,edges_list,results_df_row,all_attributes_dict,dimensions_list,aggr_fun="and_1d-simple"):
    ## Buid attributes_dict and multidim_groups with nonempty groups
    n, _ = aux_functions.get_n_and_counts(nodes_list,edges_list,dimensions_list)
    attributes_dict = {k:[i for i in v if i in n.columns.get_level_values(k)] for k,v in all_attributes_dict.items()}
    multidim_groups = list(itertools.product(*[attributes_dict[d] for d in dimensions_list]))
    
    H = get_H_from_results_df(results_df_row,multidim_groups,aggr_fun=aggr_fun)
        
    F = get_F_from_data(nodes_list,edges_list,attributes_dict,multidim_groups,dimensions_list)
    
    N = np.sum(F)
    F = F/N
    
    onedim_deltas_1vRest0 = analytical_1vRest_onedimensional_deltas(H,F,N)

    onedim_deltas_1vRest = {dimensions_list[d]:{attributes_dict[dimensions_list[d]][k]:v for k,v in kv.items()} for d,kv in onedim_deltas_1vRest0.items()}
    return onedim_deltas_1vRest