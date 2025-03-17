import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from tqdm.auto import tqdm
import pandas as pd
import copy
import json
import os
import warnings

from multisoc.generate.multidimensional_network import multidimensional_network_fix_av_degree
from multisoc.generate.utils import make_composite_index
from multisoc.generate.two_dimensional_population import relative_correlation

def draw_2d_network(
	G,
	pos = None,
	color_dict = {
		(0,0):"#8A2846",
		(0,1):"#03045E",
		(1,0):"#FFC2D4",
		(1,1):"#CAF0F8",
		(0,):"#c63963",
		(1,):"#3A6CC6"	
		}
	):

	## Node positions
	if pos is None:
		pos = nx.kamada_kawai_layout(G,scale=3)

	# Setup visualization
	nodelist = G.nodes()
	node_colors = [color_dict[G.nodes[i]["attr"]] for i in nodelist]
	n = len(G.nodes())
	node_size = 40000*(1/(n+200))

	degs = np.array([G.degree()[i] for i in nodelist])
    
    ## Build ranking
	temp = degs.argsort()
	ranks = np.empty_like(temp)
	ranks[temp] = np.arange(len(degs))
	degs_rank_dct = {nodelist[i]:ranks[i] for i in range(len(nodelist))}
    
	node_size_list =  node_size*0.1+(node_size*4-node_size*0.1)*(degs-min(degs))/(max(degs)-min(degs))
	nodelist = list(nodelist)
	

	## Draw network (takes a while)
	fig = plt.figure(figsize=(10,10))

	nx.draw_networkx(G,
						 with_labels = True,
                         labels = degs_rank_dct,
						 pos=pos,
						 nodelist=nodelist,
						 node_color=node_colors,
						 node_size=node_size_list,
						 # node_shape=shape_dict[key],
						 width=0.1,
						 alpha = .8,
						 arrowstyle = '-|>',
						 linewidths = 1,
						 edgecolors = 'black',
						 edge_color = 'grey', ## v2
						 # ax=ax
					)

	try: 
		plt.plot([],[],"o",label="fC",color=color_dict[(0,0)])
		plt.plot([],[],"o",label="fD",color=color_dict[(0,1)])
		plt.plot([],[],"o",label="mC",color=color_dict[(1,0)])
		plt.plot([],[],"o",label="mD",color=color_dict[(1,1)])

		plt.plot([],[],"o",label="C",color=color_dict[(0,)])
		plt.plot([],[],"o",label="D",color=color_dict[(1,)])

		plt.legend(loc="upper left",ncol=2)
	except KeyError:
		pass

	return fig

def av_deg_n_network_simulations(
	N,
	m,
	h_mtrx_lst,
	pop_fracs_lst,
	comp_pop_frac_tnsr,
	g_vec,
	directed=True,
	kind = "all",
	p_d = [0.5,0.5],
	iterations = 100,
	):

	assert np.all(np.array(g_vec) == np.array([len(i) for i in pop_fracs_lst]))

	comp_indices = make_composite_index(g_vec)
	
	mean_degrees_lst = {i:[] for i in comp_indices}
	group_deg_rank_lst = np.zeros((iterations,N,len(g_vec)))
	mean_degrees_1d_lst = {i:{j:[] for j in range(g_vec[i])} for i in range(len(g_vec))}

	for it in range(iterations):
		print (it,"/",iterations,"h")
		G = multidimensional_network_fix_av_degree(
						h_mtrx_lst,
						comp_pop_frac_tnsr,
						kind,
						directed=directed,
						pop_fracs_lst = pop_fracs_lst,
						N=N,
						m=m,
						v = 0,
						p_d = p_d,
                        random_pop = False
						)

		group_degrees, group_degrees_1d, group_deg_rank = analyze_network_degree(G, g_vec)

		group_deg_rank_lst[it,:,:] = group_deg_rank

		for gi in mean_degrees_lst:
			mean_degrees_lst[gi].append(np.mean(group_degrees[gi]))

		for di in mean_degrees_1d_lst:
			for gi in mean_degrees_1d_lst[di]:
				mean_degrees_1d_lst[di][gi].append(np.mean(group_degrees_1d[di][gi]))

	return mean_degrees_lst, mean_degrees_1d_lst, group_deg_rank_lst

def analyze_network_degree(G, g_vec):
	degree_dct = dict(G.in_degree)

	## Average degree of multidim groups
	comp_indices = make_composite_index(g_vec)
	group_degrees = {}
	for gi in comp_indices:
		group_degrees[gi] = [di for ni,di in degree_dct.items() if G.nodes[ni]["attr"] == gi]

	## Average degree of 1D groups
	group_degrees_1d = {dim: {} for dim in range(len(g_vec))}
	for dim, v_d in enumerate(g_vec):
		for gi in range(v_d):
			group_degrees_1d[dim][gi] = [di for ni,di in degree_dct.items() if G.nodes[ni]["attr"][dim] == gi]


	## Ranking of multidim groups
	sorted_degs = sorted(degree_dct.items(), key=lambda x:x[1],reverse=True)
	group_deg_rank = [G.nodes[n[0]]["attr"] for n in sorted_degs]

	return group_degrees, group_degrees_1d, group_deg_rank

def analyze_degree_ranking(group_deg_rank, g_vec):

	assert group_deg_rank.ndim == 3
	assert group_deg_rank.shape[2] == len(g_vec)

	comp_indices = make_composite_index(g_vec)
	n_iter = 1.0*group_deg_rank.shape[0]

	rank_prob_dct_multi = {}
	for gi in comp_indices:
		rank_prob_dct_multi[gi] = np.sum(np.all(group_deg_rank == gi,axis=2),axis=0) / n_iter

	rank_prob_dct_1D = {}
	for d, v_d in enumerate(g_vec):
		rank_prob_dct_1D[d] = {}
		for gi in range(v_d):
			rank_prob_dct_1D[d][gi] = np.sum(group_deg_rank[:,:,d] == gi,axis=0) / n_iter

	return rank_prob_dct_multi, rank_prob_dct_1D

def aggregate_degree_avs(mean_degrees_lst,mean_degrees_1d_lst,group_labels):

	ndim = len(group_labels)

	group_degrees_av_std = {f"dim_{d+1}":[] for d in range(len(group_labels))}
	group_degrees_av_std["mean"] = []
	group_degrees_av_std["std"] = []

	for gmul, vals in mean_degrees_lst.items():
		for d, gi in enumerate(gmul):
			group_degrees_av_std[f"dim_{d+1}"].append(group_labels[d][gi])
		group_degrees_av_std["mean"].append(np.nanmean(vals))
		group_degrees_av_std["std"].append(np.nanstd(vals))

	group_degrees_1D_av_std = {
		"dim":[],
		"group":[],
		"mean":[],
		"std":[]
		}
		
	for d in mean_degrees_1d_lst:
		for gi, vals in mean_degrees_1d_lst[d].items():
			group_degrees_1D_av_std["dim"].append(d+1)
			group_degrees_1D_av_std["group"].append(group_labels[d][gi])
			group_degrees_1D_av_std["mean"].append(np.nanmean(vals))
			group_degrees_1D_av_std["std"].append(np.nanstd(vals))

	return group_degrees_av_std, group_degrees_1D_av_std

def label_rankings(rank_prob_dct_multi, rank_prob_dct_1D, group_labels):

	rank_prob_dct_multi_RLBL = {}
	for k,v in rank_prob_dct_multi.items():
		knew = [group_labels[d][gi] for d,gi in enumerate(k)]
		rank_prob_dct_multi_RLBL[tuple(knew)] = v

	rank_prob_dct_1D_RLBL = {}
	for d in rank_prob_dct_1D:
		for gi, vals in rank_prob_dct_1D[d].items():
			rank_prob_dct_1D_RLBL[group_labels[d][gi]] = vals

	return rank_prob_dct_multi_RLBL, rank_prob_dct_1D_RLBL

def prepare_model_params(
	h_mtrx_lst,
	pop_fracs_lst,
	comp_pop_frac_tnsr,
	group_labels,
	g_vec
	):

	model_params = {}
	
	for d, gr_lst in enumerate(group_labels):

		for i, gi in enumerate(gr_lst):

			## Homophily
			model_params[f"h_{gi}"] = h_mtrx_lst[d][i,i]

			## 1D Population fractions
			model_params[f"f_{gi}"] = pop_fracs_lst[d][i]

	## Multidim population fractions
	comp_indices = make_composite_index(g_vec)
	for multi_group in comp_indices:
		multi_group_lbl = "".join([group_labels[d][v] for d, v in enumerate(multi_group)])
		model_params[f"F_{multi_group_lbl}"] = comp_pop_frac_tnsr[multi_group]

	## Correlation
	if comp_pop_frac_tnsr.shape == (2,2):
		abs_corr = comp_pop_frac_tnsr[0,0] / pop_fracs_lst[0][0]
		model_params["corr"] = relative_correlation(pop_fracs_lst[1][0],abs_corr)
	else:
		model_params["corr"] = np.nan
		warnings.warn("Correlation is only uniquely defined for 2x2 populations. Returning NaN.")

	return json.dumps(model_params, indent = 4)

def generate_nw_data_for_viz(
	N,
	m,
	h_mtrx_lst,
	pop_fracs_lst,
	comp_pop_frac_tnsr,
	g_vec,
	group_labels,
	iterations,
	directed=True,
	kind = "all",
	p_d = [0.5,0.5],
	):

	
	model_params_json = prepare_model_params(
		h_mtrx_lst,
		pop_fracs_lst,
		comp_pop_frac_tnsr,
		group_labels,
		g_vec
		)

	mean_degrees_lst, mean_degrees_1d_lst,group_deg_rank = av_deg_n_network_simulations(
		N,
		m,
		h_mtrx_lst,
		pop_fracs_lst,
		comp_pop_frac_tnsr,
		g_vec,
		directed=directed,
		kind = kind,
		p_d = p_d,
		iterations = iterations
		)

	mean_degrees_multi, mean_degrees_1d = aggregate_degree_avs(mean_degrees_lst,mean_degrees_1d_lst,group_labels)
	rank_prob_dct_multi, rank_prob_dct_1D = analyze_degree_ranking(group_deg_rank, g_vec)
	rank_prob_dct_multi_RLBL, rank_prob_dct_1D_RLBL = label_rankings(rank_prob_dct_multi, rank_prob_dct_1D, group_labels)

	return {
		"parameters": model_params_json, 
		"average_degrees_multi":pd.DataFrame(mean_degrees_multi),
		"average_degrees_1D":pd.DataFrame(mean_degrees_1d),
		"degree_ranking_multi":pd.DataFrame.from_dict(rank_prob_dct_multi_RLBL,"index").T,
		"degree_ranking_1D":pd.DataFrame.from_dict(rank_prob_dct_1D_RLBL,"index").T
		}

def save_viz_data(res,folder,fname):
	save_path = folder + "/" + fname
	
	if not os.path.exists(save_path):
		os.mkdir(save_path)
	else:
		warnings.warn(f"File {save_path} exists. Overwriting.")

	with open(save_path + "/parameters.json","w") as f:
		f.write(res["parameters"])

	for name, results in res.items():
		if name in ["parameters"]:
			continue
		results.to_csv(save_path+f"/{name}.csv")

def save_graph_json(G,group_labels,folder,fname):

	save_path = folder + "/" + fname + "/example_network.json"

	if os.path.isfile(save_path):
		warnings.warn(f"File {save_path} exists. Overwriting.")

	G_export = copy.deepcopy(G)
	for n,info in G_export.nodes.items():
	    G_export.nodes[n]["attr_name"] = tuple([group_labels[d][gi] for d,gi in enumerate(G_export.nodes[n]["attr"])])
	G_json = nx.node_link_data(G_export)

	## Convert np.int32 to int bc np.int32 are not json serializable
	for i,n in enumerate(G_json["nodes"]):
	    for k, v in n.items():
	        if type(v) == np.int32:
	            G_json["nodes"][i][k] = int(v)

	for i,n in enumerate(G_json["links"]):
	    for k, v in n.items():
	        if type(v) == np.int32:
	            G_json["links"][i][k] = int(v)

	with open(save_path,"w") as f:
	    f.write(json.dumps(G_json, indent = 4))

def generate_viz_package(
	N,
	m,
	h_mtrx_lst,
	pop_fracs_lst,
	comp_pop_frac_tnsr,
	g_vec,
	group_labels,
	iterations,
	folder,
	fname,
	seed = None,
	draw_network =True
	):

	## Compute degrees statistics
	res = generate_nw_data_for_viz(
		N,
		m,
		h_mtrx_lst,
		pop_fracs_lst,
		comp_pop_frac_tnsr,
		g_vec,
		group_labels,
		iterations=iterations,
		directed=True,
		kind = "all",
		p_d = [0.5,0.5],
		)

	save_viz_data(res,folder,fname)

	if seed is not None:
		np.random.seed(seed)

	## Generate example network
	G = multidimensional_network_fix_av_degree(
                h_mtrx_lst,
                comp_pop_frac_tnsr,
                "all",
                directed=True, ## Directed or undirected network
                pop_fracs_lst = pop_fracs_lst,
                N=N,
                m=m,
                v = 0,
                p_d = [0.5,0.5],
                random_pop = False
                )

	## Remove isolated nodes
	G.remove_nodes_from(list(nx.isolates(G)))

	## Save network
	save_graph_json(G,group_labels,folder,fname)

	## Draw network
	if draw_network:
		_ = draw_2d_network(G)
		plt.savefig(folder + "/" + fname + "/network.png",dpi=300)
		plt.show()