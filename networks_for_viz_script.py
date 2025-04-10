import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import copy
import networkx as nx
import pandas as pd
from matplotlib.patches import Patch
import time
from multiprocessing import Pool
from tqdm.auto import tqdm
import os

from multisoc.generate.multidimensional_network import multidimensional_network_fix_av_degree
from multisoc.generate.two_dimensional_population import consol_comp_pop_frac_tnsr
from multisoc.generate.two_dimensional_population import relative_correlation_inv
from multisoc.generate.utils import make_composite_index
from multisoc.theory.multidimensional_preferences import composite_H
from multisoc.metrics.fairness_metrics import *
from multisoc.theory.fairness_metrics import *

from helpers_for_viz import *

color_dict = {
		(0,0):"#8A2846",
		(0,1):"#03045E",
		(1,0):"#FFC2D4",
		(1,1):"#CAF0F8"
		}
groups_lst = [(0,0),(0,1),(1,0),(1,1)]
group_labels = [["f","m"],["C","D"]]
colors_race = ["#595959","#d9d9d9"]
colors_gender = ["#c63963","#3A6CC6"]

h_CD_values = np.linspace(0.1,0.9,9)
h_fm_values = np.linspace(0.1,0.9,9)
corr_values = np.linspace(-0.8,0.8,9)
f_f_values = np.linspace(0.1,0.5,9)
f_C_values = np.linspace(0.1,0.5,9)

## Common parameters
## Number of nodes and links
N = 200 ## Number of nodes 200
m = 10  ## Average number of connections per node 10
g_vec = [2,2]
iterations = 50 ## 50

kind = "all" ## Aggregation function: {all->and, one->mean, any->or}
p_d = [0.5, 0.5] ## Weight of each dimension for "mean" aggregation function

def worker_function(h_CD,h_fm,correlation,f_f,f_cat):
	if f_f > f_cat:
		return

	## Preferences
	h_cat = h_CD
	h_dog = h_CD
	
	h_f = h_fm
	h_m = h_fm
	
	## List of 1d homophily matrices (2 for a two-dimensional system)
	h_mtrx_lst = [
		
		np.array([[h_f,1-h_f],
				  [1-h_m,h_m]]),
		
		np.array([[h_cat,1-h_cat],
				  [1-h_dog,h_dog]])
	]
	
	assert f_f <= f_cat
	
	pop_fracs_lst = [
		[f_f,1-f_f],
		[f_cat,1-f_cat]
	]
	
	consol = relative_correlation_inv(max(f_f,f_cat),correlation)
	comp_pop_frac_tnsr = consol_comp_pop_frac_tnsr(pop_fracs_lst,consol)
	
	fname = f"initial_multi_example_HighCorrelation_ff{int(100*f_f):02d}_fC{int(100*f_cat):02d}_c{int(100*consol):02d}_hf{int(100*h_f):02d}_hm{int(100*h_m):02d}_hC{int(100*h_cat):02d}_hD{int(100*h_dog):02d}"
	
	## If file exists, skip computation instead of overwriting
	save_path = folder + "/" + fname
	if os.path.exists(save_path):
		print ("WARNING!!!! \n ********************************\nFile ",save_path," exists!! Skipping computation instead of rewriting!!")
		return

	generate_viz_package(
		N,
		m,
		h_mtrx_lst,
		pop_fracs_lst,
		comp_pop_frac_tnsr,
		g_vec,
		group_labels,
		iterations=iterations,
		folder="./viz_data/all_parameters",
		fname=fname,
		draw_network =False,
		seed = 12 ## 18
		)

if __name__ == "__main__":
	with Pool(processes=20,maxtasksperchild=200) as pool: 
		for h_CD in tqdm(h_CD_values):
			for h_fm in h_fm_values:
				for correlation in corr_values:
					for f_f in f_f_values:
						for f_cat in f_C_values:
							pool.apply_async(worker_function,args=(h_CD,h_fm,correlation,f_f,f_cat))
							# result.get()
		pool.close()
		pool.join()

	## Non parallel version for tests
	# for h_CD in h_CD_values:
	# 	for h_fm in h_fm_values:
	# 		for correlation in corr_values:
	# 			for f_f in f_f_values:
	# 				for f_cat in f_C_values:
	# 					worker_function(h_CD,h_fm,correlation,f_f,f_cat)