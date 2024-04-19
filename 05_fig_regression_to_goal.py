#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import copy

import numpy
import matplotlib
from matplotlib import pyplot
from mpl_toolkits.axes_grid1 import make_axes_locatable

from simulation_settings import parameter_range, simulation_types, \
    n_repetitions, n_generations, n_flights_per_generation, \
    p_goal_range, p_social_range, p_memory_range, \
    p_continuity_range_used, sd_goal_range, sd_continuity, sd_social, \
    sd_memory_max_range, sd_memory_min, sd_memory_steps, \
    start_pos, goal_pos, n_landmarks, start_heading, \
    stepsize, max_steps, goal_threshold, landmark_threshold

from agents import circular_distance


# Overwrite the temporary data? (This ONLY pertains to the relative bearing 
# data computed here, NOT to the path data computed before.)
OVERWRITE_TMP = False

# Set the maximum path lengths.
MAX_PATH_LENGTH = 200

# Compute ALL the bearings (NOTE: takes FOR EVER) or only those for the plot?
COMPUTE_PLOT_BEARINGS_ONLY = True

# Files and folders.
DIR = os.path.dirname(os.path.abspath(__file__))
DATADIR = os.path.join(DIR, "data_simulation_{}".format(parameter_range))
OUTDIR = os.path.join(DIR, "output_simulation_{}".format(parameter_range))
FIGDIR = os.path.join(DIR, "figures")
TMP_DIR = os.path.join(OUTDIR, "reduced_data")
for outpath in [OUTDIR, TMP_DIR]:
    if not os.path.isdir(outpath):
        os.mkdir(outpath)

# Set parameter combinations to plot.
if parameter_range == "revision":
    plot_parameters = { \
        "p_goal":   [0.1, 0.2, 0.3], \
        "p_social": [0.1, 0.2, 0.3], \
        "p_memory": p_memory_range, \
        }
    best_fit = { \
        "efficiency": { \
            "p_goal":           0.30, \
            "p_social":         0.15, \
            "p_memory":         0.05, \
            }, \
        "generational_improvement": { \
            "p_goal":           0.15, \
            "p_social":         0.20, \
            "p_memory":         0.40, \
            }, \
        }

elif parameter_range == "narrow":
    plot_parameters = { \
        "p_goal":   [0.05, 0.15, 0.25], \
        "p_social": [0.05, 0.15, 0.25], \
        "p_memory": p_memory_range, \
        }
    best_fit = { \
        "efficiency": { \
            "p_goal":           0.15, \
            "p_social":         0.20, \
            "p_memory":         0.30, \
            }, \
        "generational_improvement": { \
            "p_goal":           0.025, \
            "p_social":         0.125, \
            "p_memory":         0.375, \
            }, \
        }
elif parameter_range == "wide":
    plot_parameters = { \
        "p_goal":   [0.1, 0.2, 0.3], \
        "p_social": [0.1, 0.2, 0.3], \
        "p_memory": p_memory_range, \
        }
    best_fit = { \
        "efficiency": { \
            "p_goal":           0.20, \
            "p_social":         0.20, \
            "p_memory":         0.35, \
            }, \
        "generational_improvement": { \
            "p_goal":           0.10, \
            "p_social":         0.20, \
            "p_memory":         0.15, \
            }, \
        }

elif parameter_range == "control":
    plot_parameters = { \
        "range":    "control", \
        "p_goal":   [0.05, 0.15, 0.25], \
        "p_social": [0.0, 0.15, 0.25], \
        "p_memory": p_memory_range, \
        }
    best_fit = { \
        "efficiency": { \
            "p_goal":           0.0, \
            "p_social":         0.0, \
            "p_memory":         0.0, \
            }, \
        "generational_improvement": { \
            "p_goal":           0.0, \
            "p_social":         0.0, \
            "p_memory":         0.0, \
            }, \
        }

# Load existing path data.
print("Loading path data...")
path_shape = (len(p_goal_range), len(p_social_range), \
    len(p_memory_range), len(sd_goal_range), len(sd_memory_max_range), \
    len(simulation_types), n_repetitions, 2, n_generations, \
    n_flights_per_generation, MAX_PATH_LENGTH)
x = numpy.memmap(os.path.join(TMP_DIR, "x.dat"), dtype=numpy.float32, \
    mode="r", shape=path_shape)
y = numpy.memmap(os.path.join(TMP_DIR, "y.dat"), dtype=numpy.float32, \
    mode="r", shape=path_shape)


# # # # #
# CREATE HISTOGRAMS

if COMPUTE_PLOT_BEARINGS_ONLY:
    b_shape = (len(plot_parameters["p_goal"]), \
        len(plot_parameters["p_social"]), len(plot_parameters["p_memory"]), \
        len(sd_goal_range), len(sd_memory_max_range), n_repetitions, \
        n_generations-1, n_flights_per_generation, MAX_PATH_LENGTH)
else:
    b_shape = (len(p_goal_range), len(p_social_range), len(p_memory_range), \
        len(sd_goal_range), len(sd_memory_max_range), n_repetitions, \
        n_generations-1, n_flights_per_generation, MAX_PATH_LENGTH)

# Load existing bearing data.
if os.path.isfile(os.path.join(TMP_DIR, "b.dat")) and (not OVERWRITE_TMP):
    print("Loading bearing data...")
    b = numpy.memmap(os.path.join(TMP_DIR, "b.dat"), dtype=numpy.float32, \
        mode="r", shape=b_shape)
# Compute new bearing data.
else:
    print("Computing bearing data...")
    b = numpy.memmap(os.path.join(TMP_DIR, "b.dat"), dtype=numpy.float32, \
        mode="w+", shape=b_shape)
    b[:] = numpy.NaN

    # Loop through goal and social parameter combinations.
    if COMPUTE_PLOT_BEARINGS_ONLY:
        enum_range = { \
            "p_goal": plot_parameters["p_goal"], \
            "p_social": plot_parameters["p_social"], \
            "p_memory": plot_parameters["p_memory"], \
            }
    else:
        enum_range = { \
            "p_goal": p_goal_range, \
            "p_social": p_social_range, \
            "p_memory": p_memory_range, \
            }
    for p_pgi, p_goal in enumerate(enum_range["p_goal"]):
        pgi = p_goal_range.index(p_goal)
        for p_psi, p_social in enumerate(enum_range["p_social"]):
            psi = p_social_range.index(p_social)
            # Loop through memory parameters.
            for p_pmi, p_memory in enumerate(enum_range["p_memory"]):
                pmi = p_memory_range.index(p_memory)
                print("\tp_goal={}, p_social={}, p_memory={}".format(p_goal, \
                    p_social, p_memory))
                # Find the indices in the array with ALL simulation results.
                # SD for goal and memory max set to first entry, which is currently
                # the only option. Flexibly written, though, in case future efforts
                # will focus on different SDs.
                sdgi = 0
                sd_goal = sd_goal_range[sdgi]
                sdmmi = 0
                sd_memory_max = sd_memory_max_range[sdmmi]
    
                # Loop through all the things.
                con = "experimental"
                ci = simulation_types.index(con)
                for rep in range(n_repetitions):
                    # Skip the first generation, as there is only a naive bird
                    # in that one.
                    for gen in range(1, n_generations):
                        for flight in range(n_flights_per_generation):
                            for i in range(1, MAX_PATH_LENGTH):
                                
                                # Quit if we hit a NaN.
                                x_nan = numpy.isnan( \
                                    x[pgi,psi,pmi,sdgi,sdmmi,ci,rep,1,gen,flight,i])
                                y_nan = numpy.isnan( \
                                    y[pgi,psi,pmi,sdgi,sdmmi,ci,rep,1,gen,flight,i])
                                if x_nan or y_nan:
                                    break

                                # Compute current heading for naive agent.
                                naive_pos = ( \
                                    x[pgi,psi,pmi,sdgi,sdmmi,ci,rep,1,gen,flight,i], \
                                    y[pgi,psi,pmi,sdgi,sdmmi,ci,rep,1,gen,flight,i])
                                naive_pos_prev = ( \
                                    x[pgi,psi,pmi,sdgi,sdmmi,ci,rep,1,gen,flight,i-1], \
                                    y[pgi,psi,pmi,sdgi,sdmmi,ci,rep,1,gen,flight,i-1])
                                naive_heading = numpy.arctan2( \
                                    naive_pos[1]-naive_pos_prev[1], \
                                    naive_pos[0]-naive_pos_prev[0])
                                # Compute the expected next position for the 
                                # naive agent on the basis of its current 
                                # heading and stepsize.
                                naive_pos_next = ( \
                                    naive_pos[0] + stepsize \
                                        * numpy.cos(naive_heading), \
                                    naive_pos[1] + stepsize \
                                        * numpy.sin(naive_heading))

                                # Compute the experienced agent's current 
                                # heading.
                                experienced_pos = ( \
                                    x[pgi,psi,pmi,sdgi,sdmmi,ci,rep,0,gen,flight,i], \
                                    y[pgi,psi,pmi,sdgi,sdmmi,ci,rep,0,gen,flight,i])
                                experienced_pos_prev = ( \
                                    x[pgi,psi,pmi,sdgi,sdmmi,ci,rep,0,gen,flight,i-1], \
                                    y[pgi,psi,pmi,sdgi,sdmmi,ci,rep,0,gen,flight,i-1])
                                experienced_heading = numpy.arctan2( \
                                    experienced_pos[1]-experienced_pos_prev[1], \
                                    experienced_pos[0]-experienced_pos_prev[0])

                                # For the experienced agent, compute the 
                                # bearing to the naive agent's expected 
                                # position.
                                b_naive = numpy.arctan2( \
                                    naive_pos_next[1]-experienced_pos[1], \
                                    naive_pos_next[0]-experienced_pos[0])
                                # Compute the bearing towards the goal for the 
                                # experienced agent.
                                b_goal = numpy.arctan2( \
                                    goal_pos[1]-experienced_pos[1], \
                                    goal_pos[0]-experienced_pos[0])

                                # Compute the bearings relative to the agent's 
                                # current heading.
                                b_naive_ = circular_distance(b_naive, \
                                    experienced_heading)
                                b_goal_ = circular_distance(b_goal, \
                                    experienced_heading)

                                # Code all bearings towards the goal as 
                                # positive, and away as negative.
                                if ((b_naive_ < 0) and (b_goal_ < 0)) \
                                    or ((b_naive_ > 0) and (b_goal_ > 0)):
                                    b[p_pgi,p_psi,p_pmi,sdgi,sdmmi,rep,gen-1,flight,i] = \
                                        numpy.abs(b_naive_)
                                else:
                                    b[p_pgi,p_psi,p_pmi,sdgi,sdmmi,rep,gen-1,flight,i] = \
                                        -1 * numpy.abs(b_naive_)


# # # # #
# FIGURE

# Open a new figure.
n_rows = len(plot_parameters["p_goal"])
n_cols = len(plot_parameters["p_social"])
fig_wh_ratio = 1.3
fig_w = 8.268
fig_h = n_cols * ((fig_w / n_cols) / fig_wh_ratio)
fig, axes = pyplot.subplots(nrows=n_rows, ncols=n_cols, \
    figsize=(fig_w,fig_h), dpi=900.0)
if n_rows == 1:
    axes = axes.reshape((1,n_cols))
elif n_cols == 1:
    axes = axes.reshape((n_rows,1))
fig.subplots_adjust(left=0.11, bottom=0.12, right=0.93, top=0.95, \
    wspace=0.1, hspace=0.1)
# Set limits and x ticks.
x_ = numpy.linspace(-numpy.pi, numpy.pi, 5)
xticklabels = ["-$\pi$", "-$\pi$/2", "0", "$\pi$/2", "$\pi$"]
xlim = (-numpy.pi, numpy.pi)
ylim = (0, 0.42)
bin_edges = numpy.linspace(xlim[0], xlim[1], round(numpy.pi*20))
# Choose the colour map.
cmap = matplotlib.colormaps.get_cmap("viridis")
norm = matplotlib.colors.Normalize(vmin=0.0, vmax=1.0)
cticks = (0.0, 0.5, 1.0)

# Loop through goal and social parameter combinations.
for row, p_goal in enumerate(plot_parameters["p_goal"][::-1]):
    for col, p_social in enumerate(plot_parameters["p_social"]):

        # Choose the axis to draw in.
        ax = axes[row,col]
        
        # Add an axis for the colour bar.
        divider = make_axes_locatable(ax)
        bax = divider.append_axes("right", size="5%", pad=0.05)
        # Set the axis labels.
        if col == 0:
            ax.text( \
                xlim[0] - 0.40*(xlim[1]-xlim[0]), \
                ylim[0] + 0.19*(ylim[1]-ylim[0]), \
                "$w_{goal}=$" + str(round(p_goal,3)).ljust(4,"0"), \
                fontsize=14, rotation="vertical")
            ax.text( \
                xlim[0] - 0.25*(xlim[1]-xlim[0]), \
                ylim[0] + 0.13*(ylim[1]-ylim[0]), \
                "Probability density", fontsize=10, \
                rotation="vertical")
        if row == n_rows-1:
            ax.text( \
                xlim[0] + 0.41*(xlim[1]-xlim[0]), \
                ylim[0] - 0.25*(ylim[1]-ylim[0]), \
                "$\hat{b}_{naive}$", fontsize=10)
            ax.text( \
                xlim[0] + 0.20*(xlim[1]-xlim[0]), \
                ylim[0] - 0.40*(ylim[1]-ylim[0]), \
                "$w_{social}=$" + str(round(p_social,3)).ljust(4,"0"), \
                fontsize=14)
        
        # Annotate the goal direction.
        ax.axvline(0, ls=":", lw=3, color="#999999", alpha=1.0)
        ax.fill_between([xlim[0], 0], [ylim[0],ylim[0]], [ylim[1],ylim[1]], \
            color="#000000", edgecolor=None, alpha=0.1)
        ax.annotate(r"goal $\mathsf{\Rightarrow}$",
            (0+0.03*(xlim[1]-xlim[0]), ylim[0]+0.03*(ylim[1]-ylim[0])), \
            color="#444444", fontsize=8)
        
        for p_memory in plot_parameters["p_memory"][::-1]:

            # Find the parameter indices in the ranges for this simulation.
            if COMPUTE_PLOT_BEARINGS_ONLY:
                pgi = plot_parameters["p_goal"].index(p_goal)
                psi = plot_parameters["p_social"].index(p_social)
                pmi = plot_parameters["p_memory"].index(p_memory)
            else:
                pgi = p_goal_range.index(p_goal)
                psi = p_social_range.index(p_social)
                pmi = p_memory_range.index(p_memory)
            # SD for goal and memory max set to first entry, which is currently
            # the only option. Flexibly written, though, in case future efforts
            # will focus on different SDs.
            sdgi = 0
            sd_goal = sd_goal_range[sdgi]
            sdmmi = 0
            sd_memory_max = sd_memory_max_range[sdmmi]
            
            # Check if this is a best-fit.
            current_fit = None
            for fit_type in best_fit.keys():
                if (p_goal == best_fit[fit_type]["p_goal"]) \
                    and (p_social == best_fit[fit_type]["p_social"]) \
                    and (p_memory == best_fit[fit_type]["p_memory"]):
                    current_fit = copy.deepcopy(fit_type)
            # Choose the appropriate colour and line style.
            if current_fit == "efficiency":
                colour = "#FF69B4"
                ls = "-"
                lbl = "Highest final efficiency"
            elif current_fit == "generational_improvement":
                colour = "#FF69B4"
                ls = "-"
                lbl = "Highest generational\nimprovement"
            else:
                ls = "-"
                colour = cmap(norm(p_memory))
                lbl = None
    
            # Create a histogram.
            notnan = numpy.invert(numpy.isnan( \
                b[pgi,psi,pmi,sdgi,sdmmi,:,:,:,:]))
            if numpy.sum(notnan) == 0:
                continue
            hist, bin_edges_ = numpy.histogram( \
                b[pgi,psi,pmi,sdgi,sdmmi,:,:,:,:][notnan], \
                density=True, bins=bin_edges)
            bin_centres = bin_edges_[:-1] + numpy.diff(bin_edges_)/2.0
            
            # Plot the histogram.
            ax.plot(bin_centres, hist, ls, lw=2, color=colour, label=lbl)
            # Add a legend.
            if lbl is not None:
                ax.legend(loc="upper left", fontsize=8)

        # Add the colour bar.
        cbar = matplotlib.colorbar.ColorbarBase(bax, \
            cmap=cmap, norm=norm, ticks=cticks, \
            orientation='vertical')
        if col == n_cols-1:
            cbar.ax.tick_params(labelsize=8)
            cbar.set_label("$w_{memory}$", fontsize=14)
        else:
            cbar.set_ticklabels([])

        # Set the limits.
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        # Remove tick labels.
        ax.set_xticks(x_)
        if row < n_rows-1:
            ax.set_xticklabels([])
        else:
            ax.set_xticklabels(xticklabels, fontsize=8, \
                rotation="vertical")
        yticks = numpy.linspace(ylim[0], ylim[1], 7)
        ax.set_yticks(yticks)
        if col > 0:
            ax.set_yticklabels([])
        else:
            ax.set_yticklabels(numpy.round(yticks,1), fontsize=8)

# Set the title on the middle-top axis.
if axes.shape[1] < 3:
    col = 0
else:
    col = 1
axes[0,col].set_title( \
    "Naive individuals can bias experienced navigators towards the goal", \
    fontsize=14)
# Save and close the figure.
fig.savefig(os.path.join(FIGDIR, \
    "figure-04_regression_to_the_goal_{}.png".format(parameter_range)))
pyplot.close(fig)
