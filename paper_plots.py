import numpy as np
import matplotlib.pyplot as plt
## https://osxastrotricks.wordpress.com/2014/12/02/add-border-around-text-with-matplotlib/
import matplotlib.patheffects as PathEffects
from sklearn import linear_model
import copy

from multisoc.generate.utils import make_composite_index
from multisoc.theory.multidimensional_preferences import composite_H
from multisoc.generate.two_dimensional_population import consol_comp_pop_frac_tnsr,relative_correlation_inv
from multisoc.theory.fairness_metrics import analytical_1v1_multidimensional_deltas,analytical_1vRest_multidimensional_deltas,analytical_1vRest_onedimensional_deltas,analytical_multidim_expected_indegree

##############################################################################
## Fig. 4
##############################################################################

def plot_multidimensional_inequalities(
    N,
    f1m,
    f2m,
    corr,
    group_labels = None,
    multidim_colors = None,
    axs = None
    ):

    ## Preliminaries
    N = int(N)
    assert f1m <= f2m
    marginal_distributions = [[f1m,1-f1m],[f2m,1-f2m]]

    ## All homophily values
    h_sym_lst = np.linspace(0.01,0.99,100)

    ## Multidimensional population distribution
    F = consol_comp_pop_frac_tnsr(marginal_distributions,corr)

    comp_indices = make_composite_index(F.shape)

    ## Compute deltas
    multidim_deltas_lst = {r:[] for r in comp_indices}

    onedim_deltas_hypothetical_lst = {
        d:{
            vi:[] 
            for vi in range(F.shape[d])
        } 
        for d in range(F.ndim)
    }
    
    onedim_deltas_lst = {
        d:{
            vi:[] 
            for vi in range(F.shape[d])
        } 
        for d in range(F.ndim)
    }

    ## Computations
    for h_sym in h_sym_lst:

        h_mtrx_lst = [
            np.array([
                [h_sym, 1-h_sym],
                [1-h_sym, h_sym]]),
            np.array([
                [h_sym, 1-h_sym],
                [1-h_sym, h_sym]])
                     ]

        H_theor = composite_H(
                h_mtrx_lst,
                "all",
                p_d = None,
                alpha = None,
                )

        multidim_deltas_i = analytical_1vRest_multidimensional_deltas(H_theor,F,N)
        onedim_deltas_i = analytical_1vRest_onedimensional_deltas(H_theor,F,N)
        for r, delta in multidim_deltas_i.items():
            multidim_deltas_lst[r].append(delta)
        for d in range(F.ndim):
            H_theor_1D = h_mtrx_lst[d]
            F_1D = np.array(marginal_distributions[d])
            onedim_deltas_hypothetical_i = analytical_1vRest_onedimensional_deltas(H_theor_1D,F_1D,N)
            for vi in range(F.shape[d]):
                onedim_deltas_lst[d][vi].append(onedim_deltas_i[d][vi])
                onedim_deltas_hypothetical_lst[d][vi].append(onedim_deltas_hypothetical_i[0][vi])

    ## Plot
    if axs is None:
        fig, axs = plt.subplots(1,3,figsize=(.4*4*3,.4*4))
    plt.sca(axs[-1])
    for r, yplt in multidim_deltas_lst.items():
        multidim_label = "".join(group_labels[d][r_d] for d,r_d in enumerate(r))
        if multidim_colors is not None:
            clr = multidim_colors[r]
        else:
            clr = None
        p = plt.plot(h_sym_lst, yplt,"-",label=multidim_label,color=clr)
        ## Label lines with group sizes
        c = p[0].get_color()
        lbl_str = int(np.ceil(100*F[r]))
#             plt.plot(
#                 1.05*h_sym_lst[-1],
#                 yplt[-1],
#                 marker=f"${lbl_str}$",
#                 ms=30,
#                 color=c
#             )s
        axs[-1].annotate(
                lbl_str,
                (
                #1.01,
                1.01*h_sym_lst[-1], 
                yplt[-1]),
                # xycoords=('axes fraction', 'data'), 
                xycoords=('data', 'data'), 
                color=c,
                va="center",
                ha="left")

    plt.legend(bbox_to_anchor=(0.05, 1.05),loc="lower left")
    plt.axvline(0.5,color="grey",alpha=0.3,lw=2,ls=":",zorder=0)
    plt.axhline(0,color="grey",alpha=0.3,lw=2,ls=":",zorder=0)
#     plt.xlabel("$h$")
    plt.ylim(-1.2,1.2)
    plt.xlim(-.05,1.1)
    
    for d in range(F.ndim):
#         plt.figure()
        plt.sca(axs[d])
        for vi in range(F.shape[d]):
            yplt = onedim_deltas_lst[d][vi]
            p = plt.plot(h_sym_lst,yplt,"-",label=group_labels[d][vi])
            ## Label lines with group sizes
            c = p[0].get_color()
            yplt_hypo = onedim_deltas_hypothetical_lst[d][vi]
            plt.plot(h_sym_lst, yplt_hypo,"--",color=c,alpha=0.7,zorder=1)
            lbl_str = int(np.ceil(100*marginal_distributions[d][vi]))
#             plt.plot(
#                 1.05*h_sym_lst[-1],
#                 yplt[-1],
#                 marker=f"${lbl_str}$",
#                 ms=30,
#                 color=c
#             )
            axs[d].annotate(
					lbl_str,
					(
					#1.01,
					1.01*h_sym_lst[-1], 
					yplt[-1]),
                    # xycoords=('axes fraction', 'data'), 
                    xycoords=('data', 'data'), 
                    color=c,
	                va="center",
	                ha="left")

            if d == 0:
                plt.xlabel("$h$")
                plt.ylabel(r"$\delta$")
        
        if d == 0:
            plt.legend(bbox_to_anchor=(0.05, 1.05),loc="lower left")
        else:
            plt.setp( axs[d].get_yticklabels(), visible=False)

        plt.axvline(0.5,color="grey",alpha=0.3,lw=2,ls=":",zorder=0)
        plt.axhline(0,color="grey",alpha=0.3,lw=2,ls=":",zorder=0)
        plt.ylim(-1.2,1.2)
        plt.xlim(-.05,1.1)
        plt.xticks([0,0.5,1])
        plt.gca().spines.right.set_visible(False)
        plt.gca().spines.top.set_visible(False)

    for i, (r, yy) in enumerate(multidim_deltas_lst.items()):
        xx_lst = []
        for d, vi in enumerate(r):
            xx = onedim_deltas_lst[d][vi]
            xx_lst.append(xx)

        clf = linear_model.LinearRegression()
        X = np.array(xx_lst)
        reg = clf.fit(X.T, yy)

#         plt.figure("all_multidimensional")
        plt.sca(axs[-1])
        if multidim_colors is not None:
            clr = multidim_colors[r]
        else:
            clr = f"C{i}"
        plt.plot(h_sym_lst, reg.coef_[0]*X[0] + reg.coef_[1]*X[1] + reg.intercept_,"--",color=clr,alpha=0.7,zorder=1)
        plt.setp( axs[-1].get_yticklabels(), visible=False)
        plt.xticks([0,0.5,1])
        plt.gca().spines.top.set_visible(False)
        plt.gca().spines.right.set_visible(False)

    return fig, axs

def plot_simulations_results(res_aggr_df,axs,corr,f1m,f2m,multidim_colors):
    ## Include simulations results
    msk = np.abs(res_aggr_df["consol"] - corr) < 1e-14 ## Stored correlation values might not be exactly the same due to roundoff error
    msk = np.logical_and(msk, res_aggr_df["f0_0"] == f1m)
    msk = np.logical_and(msk, res_aggr_df["f1_0"] == f2m)

    if res_aggr_df[msk].empty:
        print ("WARNING! The dataframe does not contain simulation data matching the model paramters")
    
    ## Dim 1
    plt.sca(axs[0])
    xplt = res_aggr_df[msk]["h_sym"]
    yplt = res_aggr_df[msk]["dim0_g0_delta_mean"]
    plt.plot(xplt,yplt,"o",markeredgecolor="none",color="C0",markersize=4,markeredgewidth=0.5)
    
    xplt = res_aggr_df[msk]["h_sym"]
    yplt = res_aggr_df[msk]["dim0_g1_delta_mean"]
    plt.plot(xplt,yplt,"o",markeredgecolor="none",color="C1",markersize=4,markeredgewidth=0.5)
    
    ## Dim 2
    plt.sca(axs[1])
    xplt = res_aggr_df[msk]["h_sym"]
    yplt = res_aggr_df[msk]["dim1_g0_delta_mean"]
    plt.plot(xplt,yplt,"o",markeredgecolor="none",color="C0",markersize=4,markeredgewidth=0.5)
    
    xplt = res_aggr_df[msk]["h_sym"]
    yplt = res_aggr_df[msk]["dim1_g1_delta_mean"]
    plt.plot(xplt,yplt,"o",markeredgecolor="none",color="C1",markersize=4,markeredgewidth=0.5)
    
    ## Multi
    plt.sca(axs[2])
    xplt = res_aggr_df[msk]["h_sym"]
    yplt = res_aggr_df[msk]["multi(0, 0)_delta_mean"]
    plt.plot(xplt,yplt,"o",markeredgecolor="none",color=multidim_colors[0,0],markersize=4,markeredgewidth=0.5)
    
    xplt = res_aggr_df[msk]["h_sym"]
    yplt = res_aggr_df[msk]["multi(0, 1)_delta_mean"]
    plt.plot(xplt,yplt,"o",markeredgecolor="none",color=multidim_colors[0,1],markersize=4,markeredgewidth=0.5)
    
    xplt = res_aggr_df[msk]["h_sym"]
    yplt = res_aggr_df[msk]["multi(1, 0)_delta_mean"]
    plt.plot(xplt,yplt,"o",markeredgecolor="none",color=multidim_colors[1,0],markersize=4,markeredgewidth=0.5)
    
    xplt = res_aggr_df[msk]["h_sym"]
    yplt = res_aggr_df[msk]["multi(1, 1)_delta_mean"]
    plt.plot(xplt,yplt,"o",markeredgecolor="none",color=multidim_colors[1,1],markersize=4,markeredgewidth=0.5)

##############################################################################
## Fig. S3
##############################################################################

def get_analytical_inequalities_2D_Sym_h(
    N,
    f1m,
    f2m,
    corr,
    get_probs = False,
    get_av_degs = False,
    ):
    
    N = int(N)
    assert 0<= f1m <= f2m <= 1
    marginal_distributions = [[f1m,1-f1m],[f2m,1-f2m]]

    ## All homophily values
    h_sym_lst = np.linspace(0.01,0.99,100)

    ## Multidimensional population distribution
    F = consol_comp_pop_frac_tnsr(marginal_distributions,corr)

    comp_indices = make_composite_index(F.shape)

    ## Compute deltas
    multidim_deltas_lst = {r:[] for r in comp_indices}

    if get_av_degs:
        av_degs_lst = {r:[] for r in comp_indices}

    multidim_deltas_1v1_lst = {}
    for r in comp_indices:
        for s in comp_indices:
            multidim_deltas_1v1_lst[(r,s)] = []

    if get_probs:
        multidim_deltas_1v1_prob_upper_lst = copy.deepcopy(multidim_deltas_1v1_lst)
        multidim_deltas_1v1_prob_lower_lst = copy.deepcopy(multidim_deltas_1v1_lst)

    onedim_deltas_hypothetical_lst = {
        d:{
            vi:[] 
            for vi in range(F.shape[d])
        } 
        for d in range(F.ndim)
    }
    
    onedim_deltas_lst = {
        d:{
            vi:[] 
            for vi in range(F.shape[d])
        } 
        for d in range(F.ndim)
    }

    for h_sym in h_sym_lst:

        h_mtrx_lst = [
            np.array([
                [h_sym, 1-h_sym],
                [1-h_sym, h_sym]]),
            np.array([
                [h_sym, 1-h_sym],
                [1-h_sym, h_sym]])
                     ]

        H_theor = composite_H(
                h_mtrx_lst,
                "all",
                p_d = None,
                alpha = None,
                )

        multidim_deltas_1v1_i = analytical_1v1_multidimensional_deltas(H_theor,F,N,get_probs=get_probs)
        multidim_deltas_i = analytical_1vRest_multidimensional_deltas(H_theor,F,N)
        onedim_deltas_i = analytical_1vRest_onedimensional_deltas(H_theor,F,N)

        if get_av_degs:
            for r in comp_indices:
                av_degs_lst[r].append(analytical_multidim_expected_indegree(r,H_theor,F,N))
        
        for r, delta in multidim_deltas_i.items():
            multidim_deltas_lst[r].append(delta)

        if get_probs:
            
            for rs, delta_rs in multidim_deltas_1v1_i[0].items():
                multidim_deltas_1v1_lst[rs].append(delta_rs)
                
            for rs, delta_rs in multidim_deltas_1v1_i[1].items():
                multidim_deltas_1v1_prob_upper_lst[rs].append(delta_rs)
                
            for rs, delta_rs in multidim_deltas_1v1_i[2].items():
                multidim_deltas_1v1_prob_lower_lst[rs].append(delta_rs)

        else:
            
            for rs, delta_rs in multidim_deltas_1v1_i.items():
                multidim_deltas_1v1_lst[(r,s)].append(delta_rs)
            
        for d in range(F.ndim):
            H_theor_1D = h_mtrx_lst[d]
            F_1D = np.array(marginal_distributions[d])
            onedim_deltas_hypothetical_i = analytical_1vRest_onedimensional_deltas(H_theor_1D,F_1D,N)
            for vi in range(F.shape[d]):
                onedim_deltas_lst[d][vi].append(onedim_deltas_i[d][vi])
                onedim_deltas_hypothetical_lst[d][vi].append(onedim_deltas_hypothetical_i[0][vi])

    if get_probs:
        if get_av_degs:
            return multidim_deltas_lst, onedim_deltas_lst, multidim_deltas_1v1_lst, multidim_deltas_1v1_prob_upper_lst, multidim_deltas_1v1_prob_lower_lst, av_degs_lst
        else:
            return multidim_deltas_lst, onedim_deltas_lst, multidim_deltas_1v1_lst, multidim_deltas_1v1_prob_upper_lst, multidim_deltas_1v1_prob_lower_lst
    else:
        return multidim_deltas_lst, onedim_deltas_lst, multidim_deltas_1v1_lst

##############################################################################
## Figs. S4-S9
##############################################################################

def plot_one_axes(ax,ZZ,region_lbl,h_list,rel_corr_list):
    
    plt.sca(ax)
    plt.imshow(np.flip(ZZ,axis=0),cmap="seismic",vmin=-1,vmax=1,interpolation="nearest", 
           extent=[min(h_list), max(h_list), min(rel_corr_list),max(rel_corr_list)],aspect="auto")
    
    plt.axhline(0,ls="--",color="grey")
    plt.axvline(0.5,ls=":",color="grey")
    
    plt.xticks([],[])
    plt.yticks([],[])

    txt = plt.text(0.1,0.9,region_lbl,va="top",ha="left",color="k",alpha=0.5)
    txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w')])

def identify_region(f1m,f2m):
    assert f2m>=0.9999999999*f1m ## To avoid roundoff error issues
    if f2m>2*f1m and f2m>=0.5*(1-f1m):
        return "a"
    elif f2m<=2*f1m and f2m>0.5*(1-f1m) and f2m<(1-2*f1m):
        return "b"
    elif f2m>=(1-2*f1m):
        return "c"
    elif f2m<0.5*(1-f1m) and f2m>2*f1m:
        return "d"
    elif f2m<=0.5*(1-f1m) and f2m<=2*f1m:
        return "e"
    else:
        print ("Out of bounds?? Why??",f1m,f2m)