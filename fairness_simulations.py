from sklearn.model_selection import ParameterGrid
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import re
import multiprocessing
import importlib

from multisoc.metrics import fairness_metrics as fm
from multisoc.generate.two_dimensional_population import consol_comp_pop_frac_tnsr
from multisoc.generate.utils import make_composite_index
from multisoc.theory.multidimensional_preferences import composite_H

##############################################################################
## General function
##############################################################################

def simul_iter_multiprocess(
	pre_params_dct, ## Preliminary parameters to generate model parameters
	model_params_fun,
	sv_model_params_fun,
	model,
	n_simul,
	analysis_fun,
	v=2,
	n_proc = 1
	):
	"""
	Example of parameter dictionary:
	params_dct={
		"N":[],
		"h_mtrx_lst":[],
		"comp_pop_frac_tnsr":[],
		"kind":[],
		"pop_fracs_lst":[],
		...
		}
	"""
	## Build parameters grid (list of dicts)
	param_grid = ParameterGrid(pre_params_dct)
	## Dictionary to store parameters and results
	simul_info_dct = {}
	simul_info_dct["simul_i"] = []
	## For parallel processing - classical maxtasksperchild to avoid RAM leak
	pool = multiprocessing.Pool(n_proc,maxtasksperchild=5) 
	for param_i in tqdm(param_grid,disable=v<1):
		## Compute model parameters from preliminary
		model_params_i = model_params_fun(param_i)
		## Convert into convenient format
		model_params_sv_i = sv_model_params_fun(model_params_i)
		## Include n_simul copies of the parameters in the results dict
		for simul_i in range(n_simul):
			simul_info_dct["simul_i"].append(simul_i)
			## Update info dct adding preliminary parameters
			simul_info_dct = add_data_to_info_dct(simul_info_dct,param_i,pref="prm_")
			## Update info dct adding model parameters
			simul_info_dct = add_data_to_info_dct(simul_info_dct,model_params_sv_i,pref="mdl_")
		## Perform computation in parallel and callback the analysis function
		model_params_lst = [(model,model_params_i)]*n_simul
		res_lst = pool.map(hlpr_model_multiprocessing,model_params_lst)
		## Analyze and store results
		for res_i in res_lst:
			anal_res_i = analysis_fun(res_i)
			simul_info_dct = add_data_to_info_dct(simul_info_dct,anal_res_i,pref="res_")
	pool.close()
	pool.join()
	## Convert info dict into dataframe
	simul_info_df = pd.DataFrame(simul_info_dct)
	return simul_info_df

def hlpr_model_multiprocessing(params):
	model,model_params = params
	return model(**model_params)

def simul_iter(
	pre_params_dct, ## Preliminary parameters to generate model parameters
	model_params_fun,
	sv_model_params_fun,
	model,
	n_simul,
	analysis_fun,
	v=2
	):
	"""
	Example of parameter dictionary:
	params_dct={
		"N":[],
		"h_mtrx_lst":[],
		"comp_pop_frac_tnsr":[],
		"kind":[],
		"pop_fracs_lst":[],
		...
		}
	"""
	## Build parameters grid (list of dicts)
	param_grid = ParameterGrid(pre_params_dct)
	## Dictionary to store parameters and results
	simul_info_dct = {}
	simul_info_dct["simul_i"] = []
	for param_i in tqdm(param_grid,disable=v<1):
		## Compute model parameters from preliminary
		model_params_i = model_params_fun(param_i)
		## Convert into convenient format
		model_params_sv_i = sv_model_params_fun(model_params_i)
		for simul_i in tqdm(range(n_simul),disable=v<2):
			simul_info_dct["simul_i"].append(simul_i)
			## Update info dct adding preliminary parameters
			simul_info_dct = add_data_to_info_dct(simul_info_dct,param_i,pref="prm_")
			## Update info dct adding model parameters
			simul_info_dct = add_data_to_info_dct(simul_info_dct,model_params_sv_i,pref="mdl_")
			## Compute model
			res = model(**model_params_i)
			## Perform analysis on result of computation
			anal_res = analysis_fun(res)
			## Update info dict adding analysis of result
			simul_info_dct = add_data_to_info_dct(simul_info_dct,anal_res,pref="res_")
	## Convert info dict into dataframe
	simul_info_df = pd.DataFrame(simul_info_dct)
	return simul_info_df

def add_data_to_info_dct(info_dct,new_dct,pref=""):
	for k,v in new_dct.items():
		try:
			info_dct[pref+k].append(v)
		except KeyError:
			info_dct[pref+k] = [v]
	return info_dct

# def build_results_dict(params_dct,sv_model_params_fun,analysis_fun):
# 	"""
# 	To intialize the dictionary to store parameters and results of the
# 	simulations
# 	"""
# 	simul_info_dct = {}
# 	## Include preliminary parameters in the dictionary (with prefix "prm_")
# 	for k in params_dct.keys():
# 		simul_info_dct["prm_"+k] = []
# 	## Include model parameters in the dictionary (with prefix "prm_")
# 	model_params_dct = sv_model_params_fun()
# 	for k in model_params_dct.keys():
# 		simul_info_dct["mdl_"+k] = []
# 	## Include results of analysis in the dictionary (with prefix "res_")
# 	res_dct = analysis_fun()
# 	for k in res_dct.keys():
# 		simul_info_dct["res_"+k] = []
# 	return simul_info_dct

##############################################################################
## Aggregate results from repeated simulations
##############################################################################

def aggr_repeated_simul(
	simul_info_df,
	aggr_fun_lst=[np.mean, np.std],
	group_cols = None
	):
	"""
	Apply a list of aggregation functions to some columns of simul_info_df and
	store the aggregated results in a new dataframe.
	TO DO: Might be worth to include a rounding or cut step to "bin" the 
	grouping columns so that there are no problems with numerical values such 
	as 3.090000000001 and 3.09.
	"""

	## List of columns to group by
	if group_cols is None:
		parameter_columns =  [col for col in simul_info_df if col.startswith("prm_")]
	else:
		parameter_columns = group_cols
	## Columns to aggregate (and aggregation functions)
	res_columns_compute = {col:aggr_fun_lst for col in simul_info_df if col.startswith("res_")}
	## Apply group and aggregation steps
	group_and_aggr = simul_info_df.groupby(parameter_columns).agg(res_columns_compute)
	## Flatten multicolumns
	group_and_aggr.columns = group_and_aggr.columns.to_flat_index()
	## Rename multicolumns
	group_and_aggr.columns = ['_'.join(col) for col in group_and_aggr.columns.values]
	## Flatten multirows
	group_and_aggr = group_and_aggr.reset_index()
	## Rename all columns to remove prefixes prm_ mdl_ res_
	group_and_aggr.columns = [col[4:] for col in group_and_aggr.columns.values]

	return group_and_aggr

##############################################################################
## ER 2 binary attributes
##############################################################################

# def pre_params_ER_2_dim_2_attr():
# 	"""
# 	To build pre_params_dct
# 	"""
# 	pre_params_dct = {
# 		"N":[500],
# 		"kind":["all"],
# 		"consol":[0.1,0.3,0.5,0.7,0.9],
# 		"h_sym":np.linspace(0,1,11),
# 		"f0_0":0.5,
# 		"f1_0":[0.1,0.3,0.5],
# 	}
# 	return pre_params_dct

def model_params_ER_2_dim_2_attr_h_sym(pre_params_dct):
	"""
	For model_params_fun (convert preliminary parameters to valid model parameters)
	"""
	model_params = {}

	special_pre_params = ["f0_0","f1_0","h_sym","consol"]
	## Copy all params but special_pre_params directly
	model_params = {}
	for k,v in pre_params_dct.items():
		if k not in special_pre_params:
			model_params[k] = v

	## Compute population fractions
	f0_0 = pre_params_dct["f0_0"]
	f1_0 = pre_params_dct["f1_0"]
	pop_fracs_lst = [[f0_0,1-f0_0],[f1_0,1-f1_0]]
	model_params["pop_fracs_lst"] = pop_fracs_lst
	## Compute population tensor
	consol = pre_params_dct["consol"]
	comp_pop_frac_tnsr = consol_comp_pop_frac_tnsr(pop_fracs_lst,consol)
	model_params["comp_pop_frac_tnsr"] = comp_pop_frac_tnsr
	## Compute 1D h matrices
	h_sym = pre_params_dct["h_sym"]
	h_mtrx_lst = [ np.array([[h_sym,1-h_sym],[1-h_sym,h_sym]]), 
				   np.array([[h_sym,1-h_sym],[1-h_sym,h_sym]]) ]
	model_params["h_mtrx_lst"] = h_mtrx_lst
	return model_params

def model_params_ER_2_dim_2_attr_consol(pre_params_dct):
	"""
	For model_params_fun (convert preliminary parameters to valid model parameters)
	"""
	model_params = {}
	model_params["N"] = pre_params_dct["N"]
	model_params["kind"] = pre_params_dct["kind"]
	pop_fracs_lst = pre_params_dct["pop_fracs_lst"]
	model_params["pop_fracs_lst"] = pop_fracs_lst
	model_params["h_mtrx_lst"] = pre_params_dct["h_mtrx_lst"]
	## Compute population tensor
	consol = pre_params_dct["consol"]
	comp_pop_frac_tnsr = consol_comp_pop_frac_tnsr(pop_fracs_lst,consol)
	model_params["comp_pop_frac_tnsr"] = comp_pop_frac_tnsr
	return model_params

def model_params_ER_1_dim_2_attr_h_sym(
	pre_params_dct
	):
	"""
	For model_params_fun (convert preliminary parameters to valid model parameters)
	"""
	special_pre_params = ["f0_0","h_sym"]
	## Copy all params but special_pre_params directly
	model_params = {}
	for k,v in pre_params_dct.items():
		if k not in special_pre_params:
			model_params[k] = v
	## Kind is a mandatory argument, but in 1D it does not matter
	model_params["kind"] = "all" 
	## Compute population fractions
	f0_0 = pre_params_dct["f0_0"]
	pop_fracs_lst = [ [f0_0,1-f0_0] ]
	model_params["pop_fracs_lst"] = pop_fracs_lst
	## Compute population tensor
	model_params["comp_pop_frac_tnsr"] = np.array( pop_fracs_lst[0] )
	## Compute 1D h matrices
	h_sym = pre_params_dct["h_sym"]
	h_mtrx_lst = [ np.array([[h_sym,1-h_sym],[1-h_sym,h_sym]]) ]
	model_params["h_mtrx_lst"] = h_mtrx_lst
	return model_params

def model_params_ER_2_dim_2_attr_identity(pre_params_dct):
	"""
	For model_params_fun (convert preliminary parameters to valid model parameters)
	"""
	model_params = {}
	model_params["N"] = pre_params_dct["N"]
	model_params["kind"] = pre_params_dct["kind"]
	pop_fracs_lst = pre_params_dct["pop_fracs_lst"]
	model_params["pop_fracs_lst"] = pop_fracs_lst
	model_params["h_mtrx_lst"] = pre_params_dct["h_mtrx_lst"]
	model_params["comp_pop_frac_tnsr"] = pre_params_dct["comp_pop_frac_tnsr"]
	return model_params

def sv_model_params_ER_2_dim_2_attr(model_params=None):
	"""
	For sv_model_params_fun
	"""
	params_sv = {}
	comp_pop_frac_tnsr = model_params["comp_pop_frac_tnsr"]
	g_vec = comp_pop_frac_tnsr.shape
	h_mtrx_lst = model_params["h_mtrx_lst"]
	kind = model_params["kind"]
	## Save h_mtrx_lst
	for d,v_d in enumerate(g_vec):
		for i in range(v_d):
			for j in range(v_d):
				params_sv[f"dim{d}_h{i,j}"] = h_mtrx_lst[d][i][j]
	## Save theoretical multi H matrix
	comp_indices = make_composite_index(g_vec)
	H_theor = composite_H(
		h_mtrx_lst,
		kind,
		p_d = None,
		alpha = None,
		)
	for I, gi in enumerate(comp_indices):
		## Save population tensor
		params_sv[f"pop{gi}"] = comp_pop_frac_tnsr[gi]
		for J, gj in enumerate(comp_indices):
			## Save theoretical multi H matrix
			params_sv[f"multi_H{gi,gj}"] = H_theor[I,J]
	return params_sv

def analysis_fun_ER_2_dim_2_attr(
	G,
	g_vec = [2,2],
	dim = 2
	):
	"""
	For analysis_fun
	"""
	N = G.order()
	node_info_arr = np.zeros((N,dim+1))
	comp_indices = make_composite_index(g_vec)
	## Store info about degree and group membership of each node
	degs = G.in_degree()
	node_attr = G.nodes(data=True)
	for n,degi in degs:
		for d,i in enumerate(node_attr[n]["attr"]):
			node_info_arr[n,d] = i
		node_info_arr[n,dim] = degi
		
	## Compute fairness for each dimension
	theil_1d = []
	delta_1d = []
	delta_1d_1vr = []
	for d,v_d in enumerate(g_vec):
		wlth_lst = []
		for i in range(v_d):
			msk = node_info_arr[:,d] == i
			wlth_lst.append(node_info_arr[msk,dim])
		theil_1d.append(fm.theil_index_groups(wlth_lst))
		delta_1d.append(fm.CL_delta_groups_1v1(wlth_lst))
		delta_1d_1vr.append(fm.CL_delta_groups_1vRest(wlth_lst))

	## Compute fairness for intersectional groups
	wlth_lst = []
	for g in comp_indices:
		msk = np.full(N,True)
		for d,i in enumerate(g):
			msk = np.logical_and(msk,node_info_arr[:,d]==i)
		wlth_lst.append(node_info_arr[msk,dim])
	theil_multi = fm.theil_index_groups(wlth_lst)
	delta_multi = fm.CL_delta_groups_1v1(wlth_lst)
	delta_multi_1vr = fm.CL_delta_groups_1vRest(wlth_lst)

	fairness_info_dct = {}

	fairness_info_dct["multi_theil"] = theil_multi[0]
	fairness_info_dct["multi_theil_bet"] = theil_multi[1]
	fairness_info_dct["multi_theil_wit"] = np.nansum(theil_multi[2])

	for d,v_d in enumerate(g_vec):
		fairness_info_dct[f"dim{d}_theil"] = theil_1d[d][0]
		fairness_info_dct[f"dim{d}_theil_bet"] = theil_1d[d][1]
		fairness_info_dct[f"dim{d}_theil_wit"] = np.nansum(theil_1d[d][2])
		for i in range(v_d):
			fairness_info_dct[f"dim{d}_g{i}_theil"] = theil_1d[d][3][i]
			fairness_info_dct[f"dim{d}_g{i}_theil_wit"] = theil_1d[d][2][i]
			fairness_info_dct[f"dim{d}_g{i}_delta"] = delta_1d_1vr[d][i]

			for j in range(v_d):
				fairness_info_dct[f"dim{d}_delta{i,j}"] = delta_1d[d][i][j]

	for I,gi in enumerate(comp_indices):
		fairness_info_dct[f"multi{gi}_theil"] = theil_multi[3][I]
		fairness_info_dct[f"multi{gi}_theil_wit"] = theil_multi[2][I]
		fairness_info_dct[f"multi{gi}_delta"] = delta_multi_1vr[I]
		for J,gj in enumerate(comp_indices):
			fairness_info_dct[f"multi{gi,gj}_delta"] = delta_multi[I][J]

	return fairness_info_dct

def get_g_vec_from_pop_dct(pop_dct):
	some_group_str = next(iter(pop_dct.keys()))
	some_group_str = re.sub(r"[^\)\(\d,]", "", some_group_str) ## keep only tuple
	some_group = eval(some_group_str)
	dim = len(some_group)
	vals_set = [set() for _ in range(dim)]
	for gi in pop_dct.keys():
		gi_str = re.sub(r"[^\)\(\d,]", "", gi)
		gi_tpl = eval(gi_str)
		for d,i in enumerate(gi_tpl):
			vals_set[d].add(i)
	g_vec = [len(i) for i in vals_set]
	return g_vec

def dct_to_comp_pop_frac_tnsr(pop_dct):
	## Compute g_vec
	g_vec = get_g_vec_from_pop_dct(pop_dct)
	pop_frac_tnsr = np.zeros(g_vec)
	for k,v in pop_dct.items():
		gi_str = re.sub(r"[^\)\(\d,]", "", k)
		gi_tpl = eval(gi_str)
		pop_frac_tnsr[gi_tpl] = v
	return pop_frac_tnsr