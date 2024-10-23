from datetime import datetime
import importlib

from multisoc.generate.multidimensional_network import multidimensional_network
from multisoc.generate.two_dimensional_population import relative_correlation_inv

import fairness_simulations

if __name__ == "__main__":
	now = datetime.now()
	current_time = now.strftime("%Y_%m_%d_%H_%M")

	## System configuration
	number_processors = 4

	## Model parameters
	## Values used for Fig. 4
	N = [500]
	n_simul = 100
	kind = "all"

	h_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
	correlation_rs_list = [-0.9,-0.3,0,0.3,0.9]

	correlation_list = [relative_correlation_inv(0.48,i) for i in correlation_rs_list] ## Notice that I need to use the right f2m value (0.48), otherwise I won't get the correct correlation value
	f1m_list = [0.32]
	f2m_list = [0.48]

	parameter_grid_dict_1 = {
		    "N":N,
		    "kind":[kind],
		    "consol":correlation_list,
		    "h_sym":h_list,
		    "f0_0":f1m_list,
		    "f1_0":f2m_list,
		    "get_aggr_res":[False], ## If I use True I can not use analysis_fun_ER_dim_2_attr for the analyses
		    "directed":[True]
		    }


	f1m_list = [0.4]
	f2m_list = [0.41]
	correlation_list = [0.1]

	parameter_grid_dict_2 = {
		    "N":N,
		    "kind":[kind],
		    "consol":correlation_list,
		    "h_sym":h_list,
		    "f0_0":f1m_list,
		    "f1_0":f2m_list,
		    "get_aggr_res":[False], ## If I use True I can not use analysis_fun_ER_dim_2_attr for the analyses
		    "directed":[True]
		    }

	list_of_parameter_grids = [parameter_grid_dict_1, parameter_grid_dict_2]

	## To simulate many different model parameter combinations, this function 
	## performs simulations by building all possible parameter combinations
	## contained within each dictionary in the dict of parameter lists
	## (each dictionary is a parameter grid, so all parameter combinations
	## WITHIN the dictionary will be tried)
 
	res_df = fairness_simulations.simul_iter_multiprocess(
	    list_of_parameter_grids,
	    fairness_simulations.model_params_ER_2_dim_2_attr_h_sym,
	    fairness_simulations.sv_model_params_ER_2_dim_2_attr,
	    multidimensional_network,
	    n_simul,
	    fairness_simulations.analysis_fun_ER_2_dim_2_attr,
	    v=2,
	    n_proc=number_processors
	    )
	res_df.to_csv(f"paper_results/{current_time}_n_realiz_{n_simul}_sym_h_N_{N}_{kind}.gz",
		compression='gzip',
		index=False
		)