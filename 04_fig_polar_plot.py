#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import numpy
import matplotlib
from matplotlib import gridspec, pyplot
from scipy.stats import vonmises

from agents import Agent, circular_mean, sd2kappa

from simulation_settings import n_generations, n_flights_per_generation, \
    start_pos, goal_pos, n_landmarks, start_heading, stepsize, max_steps, \
    goal_threshold, landmark_threshold

# Find the directory to draw in.
DIR = os.path.dirname(os.path.abspath(__file__))
TMPDIR = os.path.join(DIR, "output", "valenti")
if not os.path.isdir(TMPDIR):
    raise Exception("Missing pigeon data folder! Expected at: {}".format( \
        TMPDIR))
OUTDIR = os.path.join(DIR, "figures")
if not os.path.isdir(OUTDIR):
    os.mkdir(OUTDIR)

# Toggle to plot all paths, or only the last path in each generation.
PLOT_ALL_PATHS = False

# Set the components that should be drawn.
components = ["goal", "continuity", "social", "memory"]

# Set the component distribution centres.
m = { \
    "goal":         numpy.pi*-0.6, \
    "continuity":   0.0, \
    "social":       numpy.pi*0.25, \
    "memory":       numpy.pi*-0.3, \
    }

# Fitted values for a stepsize of 70.
param = { \
    "p_goal":           0.14, \
    "p_continuity":     0.58, \
    "p_social":         0.16, \
    "p_memory":         0.12, \
    "sd_goal":          1.00, \
    "sd_continuity":    0.36, \
    "sd_social":        0.82, \
    "sd_memory":        [1.98, 0.4, 5], \
    }

# Colours for each component.
plotcols = { \
    "goal":         "#4e9a06", \
    "continuity":   "#c4a000", \
    "social":       "#204a87", \
    "memory":       "#5c3566", \
    "combined":     "#2e3436", \
    }
# Choose colour maps for each generation to align with the original paper:
# 1: Orange
# 2: Red
# 3: Green
# 4: Blue
# 5: Black
cmap_per_generation = [ \
    "Oranges", \
    "Reds", \
    "Greens", \
    "Blues", \
    "Greys", \
    ]

# Create a function to spawn new agents with the above settings.
def new_agent():
    return Agent( \
        starting_heading=start_heading, \
        starting_position=start_pos, \
        goal_position=goal_pos, \
        p_goal=param["p_goal"], \
        sd_goal=param["sd_goal"], \
        p_social=param["p_social"], \
        sd_social=param["sd_social"], \
        p_continuity=param["p_continuity"], \
        sd_continuity=param["sd_continuity"], \
        p_memory=param["p_memory"], \
        sd_memory=param["sd_memory"], \
        stepsize=stepsize, \
        goal_threshold=goal_threshold, \
        landmark_threshold=landmark_threshold, \
        n_landmarks=n_landmarks, \
        prefer_first_path=True, \
        )

# Create distributions.
xstep = 0.01
x = numpy.arange(-numpy.pi, numpy.pi+xstep, xstep)
y = {}
y_prop = {}
y_combined = numpy.zeros(x.shape[0], dtype=numpy.float64)
for component in m.keys():
    # Compute the kappa parameter.
    if component == "memory":
        k = sd2kappa(param["sd_{}".format(component)][1])
    else:
        k = sd2kappa(param["sd_{}".format(component)])
    y[component] = vonmises.pdf(x, k, loc=m[component])
    y_prop[component] = y[component] * param["p_{}".format(component)]
    # Add to the combined distribution.
    y_combined += y_prop[component]

# Compute direction of the combined (weighted) distributions.
a = [m[component] for component in m.keys()]
w = [param["p_{}".format(component)] for component in m.keys()]
m_combined = circular_mean(a, weights=w)
# Compute the spread parameter of each of the memory distributions.
k_memory = sd2kappa(numpy.linspace(param["sd_memory"][0], param["sd_memory"][1], \
    param["sd_memory"][2]))

# Create a figure with two subplots.
fig = pyplot.figure(figsize=(8.268, 11.693), dpi=300.0)
gs_top = gridspec.GridSpec(2, 2, width_ratios=[1,1], height_ratios=[3.5,1])
gs_bottom = gridspec.GridSpec(2, 2, width_ratios=[1, 2], height_ratios=[3.5,1])
axes = [ \
    fig.add_subplot(gs_top[0]), \
    fig.add_subplot(gs_top[1]), \
    fig.add_subplot(gs_bottom[2], projection="polar"), \
    fig.add_subplot(gs_bottom[3]), \
    ]
fig.subplots_adjust(left=0.06, bottom=0.05, right=0.96, top=0.96, \
    wspace=0.15, hspace=0.2)

# COMPASS
ax = axes[2]
# Plot components in radial space.
ax.set_theta_zero_location("W")
ax.set_rorigin(-0.35)
for component in components:
    ax.fill_between(x, numpy.zeros(y[component].shape[0]), y[component], \
        color=plotcols[component], alpha=0.2)
    p = param["p_{}".format(component)]
    ax.arrow(m[component], 0, 0, p, lw=1.5, width=0.1, head_width=0.3, \
        length_includes_head=True, head_length=0.5*p, \
        color=plotcols[component], alpha=1.0)
rlim = ax.get_ylim()
ax.set_yticks(numpy.linspace(rlim[0], rlim[1], 5))
ax.set_yticklabels([])
ax.set_xticks(numpy.linspace(0, 1.5*numpy.pi, 4))
ax.set_xticklabels([])

# VON MISES MIXTURE
ax = axes[3]
# Plot underlying distributions.
for component in components:
    ax.plot(x, y_prop[component], "--", lw=3, color=plotcols[component], \
        alpha=0.5, label=component.capitalize())
ax.plot(x, y_combined, "-", lw=5, color=plotcols["combined"], alpha=0.8)
ax.set_xlim(-numpy.pi, numpy.pi)
ax.set_xticks([-numpy.pi, -0.5*numpy.pi, 0, 0.5*numpy.pi, numpy.pi])
ax.set_xticklabels(["$-\pi$", "-$\pi$/2", "0", "$\pi$/2", "$\pi$"], \
    fontsize=14)
ax.set_yticks([])
ax.set_ylabel("Probability density", fontsize=16)
ax.legend(loc="upper right", fontsize=12)

# AGENT PATHS
ax = axes[0]
ax.set_title("Artificial navigators", fontsize=16)
# Remove axis labels and ticks.
ax.set_xticks([])
ax.set_yticks([])
# Seed the number generator for reproducibility of the graph. (Set to
# everyone's favourite year, 2020; feel free to change to something
# less evil, like 666.)
numpy.random.seed(2020)
# Fly the agents.
path_shape = (2, n_generations, n_flights_per_generation, max_steps)
path = { \
    "x":numpy.zeros(path_shape, dtype=numpy.float64) * numpy.NaN, \
    "y":numpy.zeros(path_shape, dtype=numpy.float64) * numpy.NaN, \
    }
print("Simulating agent data.")
agents = [new_agent()]
for j in range(n_generations):
    # On the first journey, only one agent flies. On the second, one naive 
    # agent is added. On all following journeys, the most experienced agent
    # is replaced with a naive agent.
    if j > 0:
        while len(agents) >= 2:
            agents.pop(0)
        agents.append(new_agent())
    # Loop through all flights in this generation.
    for i in range(0, n_flights_per_generation):
        # Release the agent(s).
        for ai, agent in enumerate(agents):
            agent.release()
            path["x"][ai,j,i,0], path["y"][ai,j,i,0] = agent.get_position()
    
        # Make the journey.
        n_agents = len(agents)
        finished = numpy.zeros(n_agents, dtype=bool)
        for step in range(1, max_steps):
            # Determine the expected positions of all agents in paired
            # flights.
            if len(agents) == 1:
                expected_pos = [None, None]
            else:
                expected_pos = []
                for ai, agent in enumerate(agents):
                    expected_pos.append(agent.expected_next_position())
            # Loop through all agents.
            for ai, agent in enumerate(agents):
                # Move the agent forwards.
                pos, finished[ai] = agent.advance_position( \
                    other_position=expected_pos[1-ai])
                # Store the new position.
                path["x"][ai,j,i,step], path["y"][ai,j,i,step] = pos
            # Check if we're there yet.
            if numpy.sum(finished) == n_agents:
                break
        # Replace unfinished routes with NaN values.
        if numpy.sum(finished) != n_agents:
            # Save the path as NaNs
            path["x"][:,j,i,:] = numpy.NaN
            path["y"][:,j,i,:] = numpy.NaN
            # Reset the agents.
            for ai, agent in enumerate(agents):
                agent.reset()
            # Skip to the next iteration.
            print("\tGeneration {}, flight {} did not finish!".format(j,i))
            continue
# Plot the paths.
for j in range(n_generations):
    cmap = matplotlib.cm.get_cmap(cmap_per_generation[j])
    norm = matplotlib.colors.Normalize(vmin=-1, \
        vmax=n_flights_per_generation)
    # Plot all paths.
    if PLOT_ALL_PATHS:
        for i in range(n_flights_per_generation):
            if i < n_flights_per_generation - 1:
                alpha = 0.2
            else:
                alpha = 0.8
            ax.plot(path["x"][0,j,i,:], path["y"][0,j,i,:], "-", \
                color=cmap(norm(i)), alpha=alpha)
            ax.plot(path["x"][1,j,i,:], path["y"][1,j,i,:], ":", \
                color=cmap(norm(i)), alpha=alpha)
    else:
        i = n_flights_per_generation - 1
        ax.plot(path["x"][0,j,i,:], path["y"][0,j,i,:], "-", lw=3, \
            color=cmap(norm(i)), alpha=alpha)
        ax.plot(path["x"][1,j,i,:], path["y"][1,j,i,:], ":", lw=3, \
            color=cmap(norm(i)), alpha=alpha)
# Set the axis limits.
dx = abs(start_pos[0] - goal_pos[0])
dy = abs(start_pos[1] - goal_pos[1])
xlim = ( \
    min(start_pos[0], goal_pos[0]) - dx*0.3, \
    max(start_pos[0], goal_pos[0]) + dx*0.3, \
    )
ylim = ( \
    min(start_pos[1], goal_pos[1]) - dy*0.05, \
    max(start_pos[1], goal_pos[1]) + dy*0.05, \
    )
ax.set_xlim(xlim)
ax.set_ylim(ylim)

# PIGEON PATHS
ax = axes[1]
ax.set_title("Pigeons", fontsize=16)
# Remove axis labels and ticks.
ax.set_xticks([])
ax.set_yticks([])
# Load pigeon data.
# Compute the shape for NumPy arrays that will contain all the flight paths.
# Shape: conditions (exp, solo, pair), repetitions, birds, generations, 
# flights per generation, max samples.
path_shape = (3, 10, 2, 5, 12, 50000)
# Load memory-mapped arrays.
path = {}
print("Loading pigeon data.")
path["x"] = numpy.memmap(os.path.join(TMPDIR, "real_data_x.dat"), \
    dtype=numpy.float64, mode="r", shape=path_shape)
path["y"] = numpy.memmap(os.path.join(TMPDIR, "real_data_y.dat"), \
    dtype=numpy.float64, mode="r", shape=path_shape)
# Plot the paths. ci=0 for the experimental condition, and run=1 for a good
# example from the pigeon data (runs range from 0 to 9; try yourself!)
ci = 0
run = 1
for j in range(n_generations):
    cmap = matplotlib.cm.get_cmap(cmap_per_generation[j])
    norm = matplotlib.colors.Normalize(vmin=-1, \
        vmax=n_flights_per_generation)
    # Plot all paths.
    if PLOT_ALL_PATHS:
        for i in range(n_flights_per_generation):
            if i < n_flights_per_generation - 1:
                alpha = 0.2
            else:
                alpha = 0.8
            ax.plot(path["x"][ci,run,0,j,i,:], path["y"][ci,run,0,j,i,:], \
                "-", color=cmap(norm(i)), alpha=alpha)
            ax.plot(path["x"][ci,run,1,j,i,:], path["y"][ci,run,1,j,i,:], \
                ":", color=cmap(norm(i)), alpha=alpha)
    else:
        i = n_flights_per_generation - 1
        ax.plot(path["x"][ci,run,0,j,i,:], path["y"][ci,run,0,j,i,:], "-", \
            lw=3, color=cmap(norm(i)), alpha=alpha, \
            label="Generation {}".format(j+1))
        ax.plot(path["x"][ci,run,1,j,i,:], path["y"][ci,run,1,j,i,:], ":", \
            lw=3, color=cmap(norm(i)), alpha=alpha)
# Set the axis limits.
ax.set_xlim(xlim)
ax.set_ylim(ylim)
# Add the legend.
ax.legend(loc="lower right", fontsize=12)

# SAVE AND CLOSE
fig.savefig(os.path.join(OUTDIR, "figure-01_mixture_model.png"))
pyplot.close(fig)

