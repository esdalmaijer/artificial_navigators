#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import numpy
import matplotlib
from matplotlib import pyplot
from mpl_toolkits.axes_grid1 import make_axes_locatable

from simulation_settings import simulation_types, n_repetitions, \
    n_generations, n_flights_per_generation, sd_goal_range, \
    sd_continuity, sd_social, sd_memory_max_range, sd_memory_min, \
    sd_memory_steps, start_pos, goal_pos, n_landmarks, start_heading, \
    stepsize, max_steps, goal_threshold, landmark_threshold

# Colour maps.
CMAP = { \
    "experimental": "Reds", \
    "solo":         "Greys", \
    "pair":         "Blues", \
    }

# Files and folders.
DIR = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(DIR, "figures")
CSVDIR = os.path.join(DIR, "csv_output")
TMP_DIR = { \
    "wide": os.path.join(DIR, "output_simulation_wide", "reduced_data"), \
    "narrow": os.path.join(DIR, "output_simulation_narrow", "reduced_data"), \
    "pigeon": os.path.join(DIR, "output", "valenti")
    }

for outdir in [OUTDIR, CSVDIR]:
    if not os.path.isdir(outdir):
        os.mkdir(outdir)

# Parameter ranges.
parameter_range = { \
    "wide": { \
        "p_goal_range": list(numpy.round( \
            numpy.arange(0.1, 0.75, 0.05), 3)), \
        "p_continuity_range": list(numpy.round( \
            numpy.arange(0.1, 0.75, 0.05), 3)), \
        "p_social_range": list(numpy.round( \
            numpy.arange(0.1, 0.75, 0.05), 3)), \
        "p_memory_range": [0.04, 0.96], \
        }, \
    "narrow": { \
        "p_goal_range": list(numpy.round( \
            numpy.arange(0.025, 0.26, 0.025), 3)), \
        "p_continuity_range": list(numpy.round( \
            numpy.arange(0.0, 1.01, 0.025), 3)), \
        "p_social_range": list(numpy.round( \
            numpy.arange(0.025, 0.26, 0.025), 3)), \
        "p_memory_range": [0.01, 0.71], \
        }, \
    }
# Compute the range of P(memory) parameter values that will be used, and count
# the number of simulations that the current settings will result in.
for pr in parameter_range.keys():
    parameter_range[pr]["p_memory_range_used"] = []
    for p_goal in parameter_range[pr]["p_goal_range"]:
        for p_social in parameter_range[pr]["p_social_range"]:
            for p_continuity in parameter_range[pr]["p_continuity_range"]:
                p_memory = round(1.0 - (p_goal + p_social + p_continuity), 3)
                if (p_memory >= parameter_range[pr]["p_memory_range"][0]) \
                    & (p_memory <= parameter_range[pr]["p_memory_range"][1]):
                    if p_memory not in \
                        parameter_range[pr]["p_memory_range_used"]:
                        parameter_range[pr]["p_memory_range_used"].append( \
                            p_memory)
    parameter_range[pr]["p_memory_range_used"].sort()

# Load efficiency data.
print("Loading efficiency data...")
var_shape = {}
efficiency = {}
for pr in parameter_range.keys():
    var_shape[pr] = ( \
        len(parameter_range[pr]["p_goal_range"]), \
        len(parameter_range[pr]["p_social_range"]), \
        len(parameter_range[pr]["p_memory_range_used"]), \
        len(sd_goal_range), len(sd_memory_max_range), \
        len(simulation_types), n_repetitions, 2, n_generations, \
        n_flights_per_generation)
    efficiency[pr] = numpy.memmap(os.path.join(TMP_DIR[pr], \
        "efficiency.dat"), dtype=numpy.float64, mode="r", shape=var_shape[pr])
print("...Efficiency data loaded!")


# BEST PATHS
print("Sorting simulations by path efficiency.")
for pr in efficiency.keys():
    for eff_type in ["final", "generational-increase"]:
        for ci, con in enumerate(simulation_types):
            if eff_type == "final":
                val = numpy.max(efficiency[pr][:,:,:,:,:,ci,:,:,-1,:], axis=7)
            elif eff_type == "generational-increase":
                val = numpy.nansum(numpy.diff(numpy.max( \
                    efficiency[pr][:,:,:,:,:,ci,:,:,1:,:], \
                    axis=8), axis=7), axis=7) / (efficiency[pr].shape[8]-1)
            # Average over birds.
            m_eff = numpy.nanmean(val, axis=6)
            # Average over runs.
            n_runs = m_eff.shape[5]
            m_eff = numpy.nansum(m_eff, axis=5) / n_runs
            # Sort the simulations by efficiency.
            sorted_eff = numpy.argsort(m_eff, axis=None)
            # Write the sorted efficiencies to file.
            fpath = os.path.join(CSVDIR, \
                "sorted_efficiency_{}_{}_{}.csv".format(pr, eff_type, con))
            with open(fpath, "w") as f:
                header = ["efficiency_{}".format(eff_type), "flat_index", \
                    "p_goal", "p_social", "p_memory", "sd_goal", \
                    "sd_memory_max", "condition"]
                f.write(",".join(header))
                for i in sorted_eff[::-1]:
                    pgi, psi, pmi, sdgi, sdmmi = numpy.unravel_index(i, m_eff.shape)
                    line = [m_eff[pgi, psi, pmi, sdgi, sdmmi], i, \
                        parameter_range[pr]["p_goal_range"][pgi], \
                        parameter_range[pr]["p_social_range"][psi], \
                        parameter_range[pr]["p_memory_range_used"][pmi], \
                        sd_goal_range[sdgi], sd_memory_max_range[sdmmi], \
                        simulation_types[ci]]
                    f.write("\n" + ",".join(map(str, line)))

# HISTOGRAM
print("Plotting histograms for path efficiency per condition.")
fig, axes = pyplot.subplots(nrows=2, ncols=2, figsize=(8.268,6.201), dpi=300.0)
fig.subplots_adjust(left=0.12, bottom=0.1, right=0.97, top=0.95, \
    wspace=0.2, hspace=0.2)
for pri, pr in enumerate(["wide", "narrow"]):
    for eti, eff_type in enumerate(["final", "generational-increase"]):
        ax = axes[pri, eti]
        for ci, con in enumerate(simulation_types):
            if eff_type == "final":
                val = numpy.max(efficiency[pr][:,:,:,:,:,ci,:,:,-1,:], axis=7)
                val_range = (0.0, 1.0)
            elif eff_type == "generational-increase":
                val = numpy.nansum(numpy.diff(numpy.max( \
                    efficiency[pr][:,:,:,:,:,ci,:,:,1:,:], \
                    axis=8), axis=7), axis=7) / (efficiency[pr].shape[8]-1)
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
        if pri == 1:
            ax.set_xlabel("Efficiency ({})".format( \
                eff_type.replace("-", " ")), fontsize=14)
        if eff_type == "generational-increase":
            ax.set_ylim(0, 50)
        else:
            ax.set_ylim(0, 10)
        if eti == 0:
            ax.set_ylabel("{} parameter range\n(Probability density)".format( \
                pr.capitalize()), fontsize=14)
# Set the legend only in one subplot.
axes[0,1].legend(loc="upper right", fontsize=12)
# Save and close the plot.
fig.savefig(os.path.join(OUTDIR, "figure-s02_efficiency_histograms.png"))
pyplot.close(fig)

# SCATTER PLOTS
for scatter_plot in [ \
    "figure-02_efficiency", \
    "figure-03_generational_increase", \
    "figure-s01_efficiency_narrow_parameter_range", \
    ]:
    
    # Create a new figure.
    if scatter_plot == "figure-02_efficiency":
        plot_types = ["efficiency-start", "efficiency-final", \
            "efficiency-increase"]
        prs = ["wide", "wide", "wide"]
    elif scatter_plot == "figure-03_generational_increase":
        plot_types = ["generational-efficiency-increase", \
            "generational-efficiency-increase"]
        prs = ["wide", "narrow"]
    elif scatter_plot == "figure-s01_efficiency_narrow_parameter_range":
        plot_types = ["efficiency-start", "efficiency-final", \
            "efficiency-increase"]
        prs = ["narrow", "narrow", "narrow"]

    # Create figure.
    n_rows = len(plot_types)
    n_cols = len(simulation_types)
    fig, ax = pyplot.subplots(nrows=n_rows, ncols=n_cols, \
        figsize=(n_cols*2.756, n_rows*2.526), dpi=900.0)
    if n_rows == 1:
        ax = ax.reshape((n_rows,n_cols))
    if n_rows == 2:
        fig.subplots_adjust(left=0.08, bottom=0.1, right=0.92, top=0.94, \
            wspace=0.1, hspace=0.15)
    if n_rows == 3:
        fig.subplots_adjust(left=0.08, bottom=0.08, right=0.92, top=0.95, \
            wspace=0.1, hspace=0.1)

    # Set the SDs for goal and memory max. (Should only be one, but flexibly
    # written if we ever want to simulate more.)    
    sdgi = 0
    sd_goal = sd_goal_range[sdgi]
    sdmmi = 0
    sd_memory_max = sd_memory_max_range[0]

    # Loop through the plot types (subplots).
    for pi, plot_type in enumerate(plot_types):
        # Choose the parameter range for this plot.
        pr = prs[pi]

        # Set the size of the memory disks.
        if pr == "wide":
            marker_size_max = 190
        elif pr == "narrow":
            marker_size_max = 270

        # Run through all conditions.
        for ci, con in enumerate(simulation_types):
            
            # Choose the axis to draw in.
            row = pi
            column = ci
            
            # Add an axis for the colour bar.
            divider = make_axes_locatable(ax[row,column])
            bax = divider.append_axes("right", size="5%", pad=0.05)

            # Set the axis title.
            if row == 0:
                ax[row,column].set_title(con.capitalize(), fontsize=16)

            # Set the axis labels.
            if row == n_rows - 1:
                ax[row,column].set_xlabel("$w_{social}$", fontsize=16)
            if column == 0:
                ax[row,column].set_ylabel("$w_{goal}$", fontsize=16)

            # Set axis limits.
            xlim = [min(parameter_range[pr]["p_social_range"]), \
                max(parameter_range[pr]["p_social_range"])]
            xrange = xlim[1] - xlim[0]
            xlim[0] -= 0.06*xrange
            xlim[1] += 0.06*xrange
            ax[row,column].set_xlim(xlim)
            ylim = [min(parameter_range[pr]["p_goal_range"]), \
                max(parameter_range[pr]["p_goal_range"])]
            yrange = ylim[1] - ylim[0]
            ylim[0] -= 0.06*yrange
            ylim[1] += 0.06*yrange
            ax[row,column].set_ylim(ylim)

            # Add labelled data (out of sight) for the legend.
            if column == n_cols-1:
                for p_memory in [ \
                    min(parameter_range[pr]["p_memory_range_used"]), " ", \
                    max(parameter_range[pr]["p_memory_range_used"])]:
                    if p_memory == " ":
                        ax[row,column].scatter(-1,-1, c="black", \
                            label="$\u2193$", alpha=0.0)
                    else:
                        if pr == "narrow":
                            n_char = 5
                        else:
                            n_char = 4
                        ax[row,column].scatter(-1,-1, \
                            s=marker_size_max*p_memory, c="black", \
                            label="$w_{memory}$=" + "{}".format( \
                            "{}".format(p_memory).ljust(n_char,"0")))
                if ((row == 0) or (len(numpy.unique(prs)) > 1)) and \
                    (column == n_cols-1):
                    ax[row,column].legend(loc="upper right", fontsize=10)
            
            # Plot each set of dots, moving from high to low P(memory).
            for p_memory in parameter_range[pr]["p_memory_range_used"][::-1]:
                
                # Get P(memory) index.
                pmi = parameter_range[pr]["p_memory_range_used"].index( \
                    p_memory)

                # Get x values, P(social)
                x_ = numpy.repeat(parameter_range[pr]["p_social_range"], \
                    len(parameter_range[pr]["p_goal_range"]))
                psi = numpy.array([ \
                    parameter_range[pr]["p_social_range"].index(p_social) for \
                    p_social in x_], dtype=numpy.int64)
                # Get y values, P(goal).
                y_ = numpy.hstack(len( \
                    parameter_range[pr]["p_social_range"]) * \
                    [parameter_range[pr]["p_goal_range"]])
                pgi = numpy.array([parameter_range[pr]["p_goal_range"].index( \
                    p_goal) for p_goal in y_], dtype=numpy.int64)
                # Set marker size, P(memory).
                s_ = marker_size_max * p_memory * \
                    numpy.ones(x_.shape, dtype=numpy.float64)
                # Get colour values, efficiency.
                vadj = 3
                if plot_type == "efficiency-final":
                    eff_ = numpy.max( \
                        efficiency[pr][pgi,psi,pmi,sdgi,sdmmi,ci,:,:,-1,:], \
                        axis=3)
                    clbl = "$Efficiency_{final}$"
                    cmap_name = "viridis"
                    vlim = (0.0, 1.0)
                elif plot_type == "efficiency-start":
                    eff_ = numpy.copy( \
                        efficiency[pr][pgi,psi,pmi,sdgi,sdmmi,ci,:,:,0,0])
                    clbl = "$Efficiency_{first}$"
                    cmap_name = "viridis"
                    vlim = (0.0, 1.0)
                elif plot_type == "efficiency-increase":
                    eff_ = \
                        efficiency[pr][pgi,psi,pmi,sdgi,sdmmi,ci,:,:,-1,-1] \
                        - efficiency[pr][pgi,psi,pmi,sdgi,sdmmi,ci,:,:,0,0]
                    clbl = "$\Delta Efficiency$"
                    cmap_name = "PiYG"
                    vlim = (-1.0, 1.0)
                elif plot_type == "generational-efficiency-increase":
                    eff_ = numpy.nansum(numpy.diff(numpy.max( \
                        efficiency[pr][pgi,psi,pmi,sdgi,sdmmi,ci,:,:,1:,:], \
                        axis=4), axis=3), axis=3) / (efficiency[pr].shape[8]-1)
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
                # NaN out all the impossible weights (sum>1).
                too_high = x_ + y_ + p_memory > 1
                eff_[too_high] = numpy.NaN
                
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
                        cbar.ax.tick_params(labelsize=7)
                        cbar.set_label(clbl, fontsize=12)
                    else:
                        cbar.set_ticklabels([])
                
                # Set the ticks.
                if pr == "wide":
                    step = 0.1
                    xticklim = [parameter_range[pr]["p_social_range"][0], \
                        parameter_range[pr]["p_social_range"][-1]+step/2.0]
                    yticklim = [parameter_range[pr]["p_goal_range"][0], \
                        parameter_range[pr]["p_goal_range"][-1]+step/2.0]
                elif pr == "narrow":
                    step = 0.05
                    xticklim = [parameter_range[pr]["p_social_range"][1], \
                        parameter_range[pr]["p_social_range"][-1]+step/2.0]
                    yticklim = [parameter_range[pr]["p_goal_range"][1], \
                        parameter_range[pr]["p_goal_range"][-1]+step/2.0]
                xticks = numpy.round(numpy.arange(xticklim[0], xticklim[1], \
                    step), 2)
                yticks = numpy.round(numpy.arange(yticklim[0], yticklim[1], \
                    step), 2)
                ax[row,column].set_xticks(xticks)
                ax[row,column].set_yticks(yticks)
                if (row == n_rows-1) or (len(numpy.unique(prs)) > 1):
                    ax[row,column].set_xticklabels(xticks, fontsize=7)
                else:
                    ax[row,column].set_xticklabels([])
                if column == 0:
                    ax[row,column].set_yticklabels(yticks, fontsize=7)
                else:
                    ax[row,column].set_yticklabels([])

    # Save and close the figure.
    save_path = os.path.join(OUTDIR, "{}.png".format(scatter_plot))
    fig.savefig(save_path)
    pyplot.close(fig)

# LINE PLOTS
plot_parameters = [ \
    {"title":"Highest final efficiency", "range":"narrow", \
        "p_goal":0.2, "p_social":0.175, "p_memory":0.3}, \
    {"title":"Highest generational increase", "range":"narrow", \
        "p_goal":0.025, "p_social":0.125, "p_memory":0.375}, \
    {"title":"Pigeon data", "range":"real-data", \
        "p_goal":0.14, "p_social":0.16, "p_memory":0.12}, \
    ]
# Create a new figure.
fig, axes = pyplot.subplots(nrows=3, ncols=1, figsize=(8.268,5.846), dpi=900.0)
fig.subplots_adjust(left=0.08, bottom=0.09, right=0.98, top=0.95, \
    wspace=0.1, hspace=0.3)
# Compute numbers of flights so that generations can be plotted in one line,
# with a notch for every change-over.
nx = n_generations * n_flights_per_generation
x_ = numpy.linspace(1, nx, nx)
xticks = numpy.linspace(2, nx, nx//2)
xticklabels = n_generations * list(range(2, 1+n_flights_per_generation, 2))
yticks = numpy.arange(0, 1.01, 0.2)
yticklabels = [str(ytick).ljust(3, "0") for ytick in numpy.round(yticks, 1)]
for pi, plot in enumerate(plot_parameters):
    
    # Choose the axis to draw in.
    ax = axes[pi]
    # Set the axis title.
    ax.set_title(plot["title"], fontsize=16, loc="left")
    
    # Load the data.
    if plot["range"] == "real-data":
        # Load pigeon efficiency data.
        fpath = os.path.join(TMP_DIR["pigeon"], "real_data_eff.dat")
        eff = numpy.memmap(fpath, dtype=numpy.float64, mode="r", \
            shape=(3, 10, 2, 5, 12))
    else:
        pr = plot["range"]
        pgi = parameter_range[pr]["p_goal_range"].index(plot["p_goal"])
        psi = parameter_range[pr]["p_social_range"].index(plot["p_social"])
        pmi = parameter_range[pr]["p_memory_range_used"].index( \
            plot["p_memory"])
        # SD for goal and memory max set to first entry, which is currently
        # the only option. Flexibly written, though, in case future efforts
        # will focus on different SDs.
        sdgi = 0
        sd_goal = sd_goal_range[sdgi]
        sdmmi = 0
        sd_memory_max = sd_memory_max_range[sdmmi]
        # Get the efficiency data.
        eff = numpy.copy( \
            efficiency[plot["range"]][pgi,psi,pmi,sdgi,sdmmi,:,:,:,:,:])
    
    # Loop through conditions.
    for ci, con in enumerate(simulation_types):
        # Create the colour map for this condition.
        cmap = matplotlib.cm.get_cmap(CMAP[con])
        norm = matplotlib.colors.Normalize(vmin=0.0, vmax=1.0)

        # Average between birds.
        eff_ = numpy.nanmean(eff[ci,:,:,:,:], axis=1)
        
        # Count number of valid runs.
        n = numpy.sum(numpy.invert(numpy.isnan(eff_)), axis=0)
        if numpy.max(n) < 1:
            continue
        # Compute mean and standard error of the mean, across
        # repeated runs.
        m = numpy.nanmean(eff_, axis=0)
        sem = numpy.nanstd(eff_, axis=0) / numpy.sqrt(n)
        ci95 = 1.96 * sem
        
        # Plot the generations.
        for j in range(m.shape[0]):
            if j == 0:
                lbl = con.capitalize()
            else:
                lbl = None
            si = j*m.shape[1]
            ei = si+m.shape[1]
            ax.plot(x_[si:ei], m[j,:], "o-", lw=3, color=cmap(norm(1.0)), \
               label=lbl, alpha=1.0)
            if m.shape[1] > 1:
                ax.fill_between(x_[si:ei], m[j,:]-ci95[j,:], \
                    m[j,:]+ci95[j,:], color=cmap(norm(1.0)), alpha=0.3)
            # Annotate generation number.
            if (pi == len(plot_parameters)-1) and (ci == 0):
                xy = (x_[si], 0.05)
                ax.annotate("Generation {}".format(j+1), xy, fontsize=10, \
                    color="black", alpha=0.8)

        # Finish the plot.
        ax.set_ylim(0, 1)
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels, fontsize=7)
        ax.set_xlim(0, nx + 1)
        ax.set_xticks(xticks)
        if pi == 0:
            ax.legend(loc="lower right", fontsize=10)
        ax.set_ylabel("Efficiency", fontsize=14)
        if pi == len(plot_parameters)-1:
            ax.set_xlabel("Flight number", fontsize=14)
            ax.set_xticklabels(xticklabels, fontsize=7, rotation="vertical")
        else:
            ax.set_xticklabels([])

# Save and close the figure.
fig.savefig(os.path.join(OUTDIR, "figure-04_efficiency_increases.png"))
pyplot.close(fig)

