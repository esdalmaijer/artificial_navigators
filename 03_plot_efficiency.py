#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import numpy
import matplotlib
from matplotlib import pyplot
from mpl_toolkits.axes_grid1 import make_axes_locatable

from simulation_settings import parameter_range, simulation_types, \
    n_repetitions, n_generations, n_flights_per_generation, \
    p_goal_range, p_social_range, p_memory_range, p_continuity_range_used, \
    sd_goal_range, sd_continuity, sd_social, \
    sd_memory_max_range, sd_memory_min, sd_memory_steps, \
    start_pos, goal_pos, n_landmarks, start_heading, \
    stepsize, max_steps, goal_threshold, landmark_threshold

# Ignore all empty-slice warnings.
import warnings
warnings.filterwarnings('ignore')


# Colour maps.
CMAP = { \
    "experimental": "Reds", \
    "solo":         "Greys", \
    "pair":         "Blues", \
    }

# Files and folders.
DIR = os.path.dirname(os.path.abspath(__file__))
PIGEON_DIR = os.path.join(DIR, "output", "valenti")
OUTDIR = os.path.join(DIR, "output_simulation_{}".format(parameter_range))
TMP_DIR = os.path.join(OUTDIR, "reduced_data")
EFF_OUTDIR = os.path.join(OUTDIR, "efficiency")
HM_OUTDIR = os.path.join(EFF_OUTDIR, "heatmaps")
LN_OUTDIR = os.path.join(EFF_OUTDIR, "line-plots")
SC_OUTDIR = os.path.join(EFF_OUTDIR, "scatter-plots")

for outdir in [EFF_OUTDIR, HM_OUTDIR, LN_OUTDIR, SC_OUTDIR]:
    if not os.path.isdir(outdir):
        os.mkdir(outdir)

# Load efficiency data.
print("Loading efficiency data...")
var_shape = (len(p_goal_range), len(p_social_range), \
    len(p_memory_range), len(sd_goal_range), len(sd_memory_max_range), \
    len(simulation_types), n_repetitions, 2, n_generations, \
    n_flights_per_generation)
efficiency = numpy.memmap(os.path.join(TMP_DIR, "efficiency.dat"), \
    dtype=numpy.float64, mode="r", shape=var_shape)
print("...Efficiency data loaded!")

# Load pigeon efficiency data.
pigeon = numpy.memmap(os.path.join(PIGEON_DIR, "real_data_eff.dat"), \
    dtype=numpy.float64, mode="r", shape=(3, 10, 2, 5, 12))
# Compute the distance between the simulated and the pigeon data. We average
# across runs first, then across birds, and then compute the distances between
# simulations and pigeons in the first generation and over all generations.
m_sim = numpy.nanmean(numpy.nanmean(efficiency, axis=6), axis=6)
m_pig = numpy.nanmean(numpy.nanmean(pigeon, axis=1), axis=1)
d = numpy.abs(m_sim - m_pig)
# Compute the sum of distances across all flights.
d = numpy.nansum(d, axis=7)
d /= numpy.max(d)
# Compute the sum for all generations.
d_all = numpy.sum(d, axis=6)
d_all /= numpy.max(d_all)

# BEST PATHS
print("Sorting simulations by path efficiency.")
for eff_type in ["final", "generational-increase"]:
    for ci, con in enumerate(simulation_types):
        if eff_type == "final":
            val = numpy.max(efficiency[:,:,:,:,:,ci,:,:,-1,:], axis=7)
        elif eff_type == "generational-increase":
            val = numpy.nansum(numpy.diff(numpy.max( \
                efficiency[:,:,:,:,:,ci,:,:,1:,:], \
                axis=8), axis=7), axis=7) / (efficiency.shape[8]-1)
        # Average over birds.
        m_eff = numpy.nanmean(val, axis=6)
        # Average over runs.
        n_runs = m_eff.shape[5]
        m_eff = numpy.nansum(m_eff, axis=5) / n_runs
        # Sort the simulations by efficiency.
        sorted_eff = numpy.argsort(m_eff, axis=None)
        # Write the sorted efficiencies to file.
        fpath = os.path.join(EFF_OUTDIR, \
            "sorted_efficiency_{}_{}.csv".format(eff_type, con))
        with open(fpath, "w") as f:
            header = ["efficiency_{}".format(eff_type), "flat_index", \
                "p_goal", "p_social", "p_memory", "sd_goal", "sd_memory_max", \
                "condition", "pigeon_distance_first_gen", "pigeon_distance"]
            f.write(",".join(header))
            for i in sorted_eff[::-1]:
                pgi, psi, pmi, sdgi, sdmmi = numpy.unravel_index(i, m_eff.shape)
                line = [m_eff[pgi, psi, pmi, sdgi, sdmmi], i, \
                    p_goal_range[pgi], p_social_range[psi], p_memory_range[pmi], \
                    sd_goal_range[sdgi], sd_memory_max_range[sdmmi], \
                    simulation_types[ci], d[pgi, psi, pmi, sdgi, sdmmi, ci, 0], \
                    d_all[pgi, psi, pmi, sdgi, sdmmi, ci]]
                f.write("\n" + ",".join(map(str, line)))

# HISTOGRAM
print("Plotting histograms for path efficiency per condition.")
for eff_type in ["final", "generational-increase"]:
    fig, ax = pyplot.subplots()
    for ci, con in enumerate(simulation_types):
        if eff_type == "final":
            val = numpy.max(efficiency[:,:,:,:,:,ci,:,:,-1,:], axis=7)
            val_range = (0.0, 1.0)
        elif eff_type == "generational-increase":
            val = numpy.nansum(numpy.diff(numpy.max( \
                efficiency[:,:,:,:,:,ci,:,:,1:,:], \
                axis=8), axis=7), axis=7) / (efficiency.shape[8]-1)
            val_range = (-0.025, 0.075)
        # Average over birds.
        m_eff = numpy.nanmean(val, axis=6)
        # Average over runs.
        n_runs = m_eff.shape[5]
        m_eff = numpy.nansum(m_eff, axis=5) / n_runs
        # Create a histogram.
        hist, bin_edges = numpy.histogram(m_eff, bins=50, range=val_range, \
            density=True)
        bin_centres = bin_edges[:-1] + numpy.diff(bin_edges) / 2.0
        # Plot the histogram.
        ax.plot(bin_centres, hist, "-", color=CMAP[con][:-1], alpha=0.8, \
            label=con.capitalize())
        ax.fill_between(bin_centres, hist, numpy.zeros(hist.shape), \
            color=CMAP[con][:-1], alpha=0.2)
    # Finish the plot.
    ax.set_xlim(val_range)
    ax.set_xlabel("Efficiency ({})".format(eff_type.replace("-", " ")), \
        fontsize=18)
    if parameter_range == "narrow" and eff_type == "generational-increase":
        ax.set_ylim(0, 50)
    else:
        ax.set_ylim(0, 12)
    ax.set_ylabel("Probability density", fontsize=18)
    ax.legend(fontsize=14)
    # Save and close the plot.
    fig.savefig(os.path.join(EFF_OUTDIR, "histogram_{}.png".format(eff_type)))
    pyplot.close(fig)

# SCATTER PLOTS
if parameter_range == "wide":
    marker_size_max = 900
elif parameter_range == "narrow":
    marker_size_max = 1100
else:
    marker_size_max = 2000
plot_types = ["efficiency-final", "efficiency-start", "efficiency-increase", \
    "difference-from-solo", "difference-from-pair", \
    "p-difference-from-solo", "p-difference-from-pair", \
    "first-second-difference", "generational-efficiency-increase"]
for plot_type in plot_types:
    # Create a single plot for each SD(goal) and SD(memory-max) value.
    for sdgi, sd_goal in enumerate(sd_goal_range):
        for sdmmi, sd_memory_max in enumerate(sd_memory_max_range):
            
            print("Plotting scatter plot for SD(goal)={} and SD(memory-max)={}" \
                .format(sd_goal, sd_memory_max))
    
            # Create big plots.
            n_rows = 1
            n_cols = len(simulation_types)
            if plot_type in ["difference-from-solo", "difference-from-pair", \
                "p-difference-from-solo", "p-difference-from-pair"]:
                n_cols -= 1
            fig, ax = pyplot.subplots(nrows=n_rows, ncols=n_cols, \
                figsize=(n_cols*6.0, n_rows*5.5), dpi=100.0)
            if n_rows == 1:
                ax = ax.reshape((n_rows,n_cols))
            fig.subplots_adjust(left=0.07, bottom=0.11, right=0.93, top=0.94, \
                wspace=0.1, hspace=0.1)
    
            # Run through all conditions.
            for ci, con in enumerate(simulation_types):
                
                # Skip plots that wouldn't make sense, e.g. due to being a
                # difference from itself.
                if (plot_type in ["difference-from-solo", \
                    "p-difference-from-solo"]) and (con == "solo"):
                    continue
                if (plot_type in ["difference-from-pair", \
                    "p-difference-from-pair"]) and (con == "pair"):
                    continue
    
                # Choose the axis to draw in.
                row = 0
                column = ci
                if plot_type in ["difference-from-solo", \
                    "p-difference-from-solo"]:
                    if ci > simulation_types.index("solo"):
                        column = ci - 1
                elif plot_type in ["difference-from-pair", \
                    "p-difference-from-pair"]:
                    if ci > simulation_types.index("pair"):
                        column = ci - 1
                
                # Add an axis for the colour bar.
                divider = make_axes_locatable(ax[row,column])
                bax = divider.append_axes("right", size="5%", pad=0.05)

                # Set the axis title.
                ax[row,column].set_title(con.capitalize(), fontsize=18)
                # Set the axis labels.
                if row == n_rows - 1:
                    ax[row,column].set_xlabel("$w_{social}$", fontsize=20)
                if column == 0:
                    ax[row,column].set_ylabel("$w_{goal}$", fontsize=20)
                # Set axis limits.
                xlim = [min(p_social_range), max(p_social_range)]
                xrange = xlim[1] - xlim[0]
                xlim[0] -= 0.05*xrange
                xlim[1] += 0.05*xrange
                ax[row,column].set_xlim(xlim)
                ylim = [min(p_goal_range), max(p_goal_range)]
                yrange = ylim[1] - ylim[0]
                ylim[0] -= 0.05*yrange
                ylim[1] += 0.05*yrange
                ax[row,column].set_ylim(ylim)
                # Remove tick labels after the left-most y-axis and lower-most
                # x-axis.
                if column > 0:
                    ax[row,column].set_yticklabels([])
                if row < n_rows-1:
                    ax[row,column].set_xticklabels([])
                # Add labelled data (out of sight) for the legend.
                if column == n_cols-1:
                    for p_memory in [min(p_memory_range), " ", \
                        max(p_memory_range)]:
                        if p_memory == " ":
                            ax[row,column].scatter(-1,-1, c="black", \
                                label="$\u2193$", alpha=0.0)
                        else:
                            if parameter_range == "narrow":
                                n_char = 5
                            else:
                                n_char = 4
                            ax[row,column].scatter(-1,-1, \
                                s=marker_size_max*p_memory, c="black", \
                                label="$w_{memory}$=" + "{}".format( \
                                "{}".format(p_memory).ljust(n_char,"0")))
                    ax[row,column].legend(loc="upper right", fontsize=14)
                
                # Plot each set of dots, moving from high to low P(memory).
                for p_memory in p_memory_range[::-1]:
                    
                    # Get P(memory) index.
                    pmi = p_memory_range.index(p_memory)
    
                    # Get x values, P(social)
                    x_ = numpy.repeat(p_social_range, len(p_goal_range))
                    psi = numpy.array([p_social_range.index(p_social) for \
                        p_social in x_], dtype=numpy.int64)
                    # Get y values, P(goal).
                    y_ = numpy.hstack(len(p_social_range)*[p_goal_range])
                    pgi = numpy.array([p_goal_range.index(p_goal) for \
                        p_goal in y_], dtype=numpy.int64)
                    # Set marker size, P(memory).
                    s_ = marker_size_max * p_memory * \
                        numpy.ones(x_.shape, dtype=numpy.float64)
                    # Get colour values, efficiency.
                    vadj = 3
                    if plot_type == "efficiency-final":
                        eff_ = numpy.max( \
                            efficiency[pgi,psi,pmi,sdgi,sdmmi,ci,:,:,-1,:], \
                            axis=3)
                        clbl = "$Efficiency_{final}$"
                        cmap_name = "viridis"
                        vlim = (0.0, 1.0)
                    elif plot_type == "efficiency-start":
                        eff_ = numpy.copy( \
                            efficiency[pgi,psi,pmi,sdgi,sdmmi,ci,:,:,0,0])
                        clbl = "$Efficiency_{first}$"
                        cmap_name = "viridis"
                        vlim = (0.0, 1.0)
                    elif plot_type == "efficiency-increase":
                        eff_ = \
                            efficiency[pgi,psi,pmi,sdgi,sdmmi,ci,:,:,-1,-1] \
                            - efficiency[pgi,psi,pmi,sdgi,sdmmi,ci,:,:,0,0]
                        clbl = "$\Delta Efficiency_{total}$"
                        cmap_name = "PiYG"
                        vlim = (-1.0, 1.0)
                    elif plot_type == "first-second-difference":
                        eff_ = \
                            efficiency[pgi,psi,pmi,sdgi,sdmmi,ci,:,:,0,1] \
                            - efficiency[pgi,psi,pmi,sdgi,sdmmi,ci,:,:,0,0]
                        clbl = "$\Delta Efficiency_{first-second}$"
                        cmap_name = "PiYG"
                        vlim = (-0.2, 0.2)
                    elif plot_type == "difference-from-solo":
                        cci = simulation_types.index("solo")
                        eff_ = \
                            efficiency[pgi,psi,pmi,sdgi,sdmmi,ci,:,:,-1,-1] \
                            - efficiency[pgi,psi,pmi,sdgi,sdmmi,cci,:,:,-1,-1]
                        clbl = "$\Delta Efficiency_{solo}$"
                        cmap_name = "PiYG"
                        vlim = (-0.3, 0.3)
                    elif plot_type == "p-difference-from-solo":
                        cci = simulation_types.index("solo")
                        eff_ = \
                            efficiency[pgi,psi,pmi,sdgi,sdmmi,ci,:,:,-1,-1] \
                            - efficiency[pgi,psi,pmi,sdgi,sdmmi,cci,:,:,-1,-1]
                        eff_ /= \
                            efficiency[pgi,psi,pmi,sdgi,sdmmi,cci,:,:,-1,-1]
                        clbl = "$\Delta Efficiency_{solo}$ (proportion)"
                        cmap_name = "PiYG"
                        vlim = (-1.0, 1.0)
                    elif plot_type == "difference-from-pair":
                        cci = simulation_types.index("pair")
                        eff_ = \
                            efficiency[pgi,psi,pmi,sdgi,sdmmi,ci,:,:,-1,-1] \
                            - efficiency[pgi,psi,pmi,sdgi,sdmmi,cci,:,:,-1,-1]
                        clbl = "$\Delta Efficiency_{pair}$"
                        cmap_name = "PiYG"
                        vlim = (-0.3, 0.3)
                    elif plot_type == "p-difference-from-pair":
                        cci = simulation_types.index("pair")
                        eff_ = \
                            efficiency[pgi,psi,pmi,sdgi,sdmmi,ci,:,:,-1,-1] \
                            - efficiency[pgi,psi,pmi,sdgi,sdmmi,cci,:,:,-1,-1]
                        eff_ /= \
                            efficiency[pgi,psi,pmi,sdgi,sdmmi,cci,:,:,-1,-1]
                        clbl = "$\Delta Efficiency_{pair}$ (proportion)"
                        cmap_name = "PiYG"
                        vlim = (-1.0, 1.0)
                    elif plot_type == "generational-efficiency-increase":
                        eff_ = numpy.nansum(numpy.diff(numpy.max( \
                            efficiency[pgi,psi,pmi,sdgi,sdmmi,ci,:,:,1:,:], \
                            axis=4), axis=3), axis=3) / (efficiency.shape[8]-1)
                        eff_ = numpy.nansum(numpy.diff(numpy.max( \
                            efficiency[pgi,psi,pmi,sdgi,sdmmi,ci,:,:,1:,:], \
                            axis=4), axis=3), axis=3) / (efficiency.shape[8]-1)
                        clbl = "$\Delta Efficiency_{generational}$"
                        cmap_name = "PiYG"
                        vlim = (-0.05, 0.05)
                        vadj = 4
                    # Set colour bar ticks.
                    if vlim[0] < 0  and vlim[1] > 0:
                        cticks = (vlim[0], 0.0, vlim[1])
                    else:
                        cticks = (vlim[0], sum(vlim)/2.0, vlim[1])
                    cticklabels = []
                    for tick in cticks:
                        cticklabels.append(str(round(tick, vadj-2)).ljust( \
                            vadj,"0"))
                    
                    # Count the all-NaNs.
                    nan_count = numpy.sum(numpy.sum(numpy.isnan(eff_), \
                        axis=2), axis=1)
                    nan_mask = nan_count == eff_.shape[2]*eff_.shape[1]
                    
                    # First, average over birds.
                    eff_ = numpy.nanmean(eff_, axis=2)
                    # Next, average over runs. (Here we count NaN runs as 0, 
                    # as they simply did not return home, and so have an
                    # efficiency that is infinitely close to 0.)
                    n_runs = eff_.shape[1]
                    eff_ = numpy.nansum(eff_, axis=1)
                    eff_ /= float(n_runs)
                    
                    # Nan out the all-NaN.
                    eff_[nan_mask] = numpy.NaN
                    
                    # Get the colour map.
                    cmap = matplotlib.cm.get_cmap(cmap_name)
                    norm = matplotlib.colors.Normalize(vmin=vlim[0], \
                        vmax=vlim[1])
                    
                    # Plot the points.
                    ax[row,column].scatter(x_, y_, s=s_, c=eff_, cmap=cmap, \
                        norm=norm, edgecolors="none", alpha=1.0)

                    # Add the colour bar.
                    if pmi == 0:
                        cbar = matplotlib.colorbar.ColorbarBase(bax, \
                            cmap=cmap, norm=norm, ticks=cticks, \
                            orientation='vertical')
                        if column == n_cols-1:
                            cbar.set_ticklabels(cticklabels)
                            cbar.ax.tick_params(labelsize=12)
                            cbar.set_label(clbl, fontsize=18)
                        else:
                            cbar.set_ticklabels([])

            # Save and close the figure.
            save_path = os.path.join(SC_OUTDIR, \
                "{}_SDgoal-{}_SDmemoryMax-{}.png".format(plot_type, \
                round(sd_goal*100), round(sd_memory_max*100)))
            fig.savefig(save_path)
            pyplot.close(fig)
