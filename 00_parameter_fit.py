#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import copy
import math

import matplotlib
from matplotlib import pyplot
import numpy
from scipy.optimize import minimize
from scipy.stats import vonmises

import utm

from agents import Agent, circular_distance, kappa2sd, sd2kappa

# Experimental settings.
N_REPETITIONS = 10
N_GENERATIONS = 5
N_FLIGHTS_PER_GENERATION = 12
# Maximum number of samples. Set to None to auto-compute.
MAX_STEPS = 50000
# Order in which to store the groups. Leave None to auto-decide.
GROUPS = ["experimental", "solo", "pair"]

# Route information (coordinates from Sasaki & Biro, 2017)
# START: N51°51'23.80" W1°17'3.0" (decimal degrees: 51.856611, -1.284167)
# GOAL:  N51°46'58.34" W1°19'2.4" (decimal degrees: 51.782872, -1.317333)
START_POS = (51.856611, -1.284167)
GOAL =      (51.782872, -1.317333)

# Convert to UTM coordinates.
START_UTM = utm.from_latlon(START_POS[0], START_POS[1])[:2]
GOAL_UTM = utm.from_latlon(GOAL[0], GOAL[1])[:2]
CENTRE_UTM = ((START_UTM[0]+GOAL_UTM[0]) / 2.0, \
    (START_UTM[1]+GOAL_UTM[1]) / 2.0)

    # Distance in meters (computed from GPS coordinates).
START_GOAL_DISTANCE = numpy.sqrt((START_UTM[0]-GOAL_UTM[0])**2 \
    + (START_UTM[1]-GOAL_UTM[1])**2)

# Compute the heading from start to goal (in radians).
START_HEADING = numpy.arctan2(GOAL_UTM[0]-START_UTM[0], \
    GOAL_UTM[1]-START_UTM[1])

# Set fixed fitting parameters. The stepsize is in meters, and based on the
# average distance between samples in the real data. The number of landmarks
# is set to 10, loosly based on Richard Mann's work on Gaussian processes 
# anchored at landmarks in modelling pigeon flight.
STEPSIZE = 3.5
N_LANDMARKS = 10

# Initial guess values for fitting procedure. These will be supplemented with
# various random guesses in parameter space.
P_GOAL = 0.15
SD_GOAL = 1.0
P_SOCIAL = 0.2
SD_SOCIAL = 0.8
P_MEMORY = 0.4
SD_MEMORY_MAX = 0.9
SD_MEMORY_MIN = 0.4
SD_MEMORY_STEPS = 5
P_CONTINUITY = numpy.round(1 - (P_GOAL + P_SOCIAL + P_MEMORY), 3)
SD_CONTINUITY = 0.35

# Files and folders
DIR = os.path.dirname(os.path.abspath(__file__))
PIGEON_DIR = os.path.join(DIR, "output", "valenti")
OUTDIR = os.path.join(DIR, "output", "valenti_fit")
for outdir in [OUTDIR]:
    if not os.path.isdir(outdir):
        os.mkdir(outdir)

# Functions
def gps_to_meters(lat, lon):
    utm_coordinates = utm.from_latlon(lat, lon)
    return utm_coordinates[0], utm_coordinates[1]

def update_kappa(dist_combined, dist_component, d_component, \
    k_lim=(6.7e-4, 100.5)):

    n = dist_combined.shape[0]
    rw = dist_combined / dist_component
    s = numpy.sin(d_component)
    c = numpy.cos(d_component)
    r = [numpy.sum(numpy.sum(s*rw, axis=0)), \
        numpy.sum(numpy.sum(c*rw, axis=0))]
    if numpy.sum(rw) == 0:
        k = 0.0
    else:
        R = numpy.sqrt(r[0]**2 + r[1]**2) / numpy.sum(rw)
        if numpy.isinf(R):
            return numpy.NaN
        if 0 <= R < 0.53:
            k = 2.0 * R + R**3.0 + (5.0 * R**5.0)/6.0
        elif R < 0.85:
            k = -0.4 + 1.39 * R + 0.43/(1.0 - R)
        else:
            k = 1.0/(R**3 - 4 * R**2 + 3 * R)
    if n < 15:
        if k < 2:
            k = max(k-2.0 / (n*k), 0.0)
        else:
            k = k * (n-1.0)**3.0 / (n**3.0+n)
    # Keep kappa values within the limits.
    k = max(min(k, k_lim[1]), k_lim[0])
    
    return k


# # # # #
# LOAD DATA

# Load pigeon data.
path_shape = (len(GROUPS), N_REPETITIONS, 2, N_GENERATIONS, \
    N_FLIGHTS_PER_GENERATION, MAX_STEPS)
pigeon_x = numpy.memmap(os.path.join(PIGEON_DIR, "real_data_x.dat"), \
    dtype=numpy.float64, mode="r", shape=path_shape)
pigeon_y = numpy.memmap(os.path.join(PIGEON_DIR, "real_data_y.dat"), \
    dtype=numpy.float64, mode="r", shape=path_shape)

# Downsample to increase the stepsize.
pigeon_x = pigeon_x[:,:,:,:,:,::100]
pigeon_y = pigeon_y[:,:,:,:,:,::100]

# Compute the empirical step size.
dx = numpy.diff(pigeon_x, axis=5)
dy = numpy.diff(pigeon_y, axis=5)
d = numpy.sqrt(dx**2 + dy**2)
stepsize = numpy.nanmean(d)
# Compute thresholds that depend on the average stepsize.
landmark_threshold = 10 * stepsize
goal_threshold = 4 * stepsize

# # # # #
# FIT DATA

# Keep track of the fitted values in each condition and run.
fitted_p = {}
fitted_sd = {}
fit_history = {}

# Go through all flights in the pair condition.
condition = "experimental"
ci = GROUPS.index(condition)
n_flights = N_GENERATIONS * N_FLIGHTS_PER_GENERATION
fitted_p[condition] = []
fitted_sd[condition] = []
fit_history[condition] = {}
k_lim = (0.00067,100.5)
for run in range(N_REPETITIONS):
    print("Fitting run {}".format(run+1))
    # Starting values for fitting.
    p = {}
    sd = {}
    for ai in range(2):
        p[ai] = {}
        sd[ai] = {}
        p[ai]["goal"] = copy.copy(P_GOAL)
        sd[ai]["goal"] = copy.copy(SD_GOAL)
        if condition == "solo":
            p[ai]["social"] = 0.0
            p[ai]["goal"] += P_SOCIAL
        else:
            p[ai]["social"] = copy.copy(P_SOCIAL)
        sd[ai]["social"] = copy.copy(SD_SOCIAL)
        p[ai]["memory"] = copy.copy(P_MEMORY)
        sd[ai]["memory"] = numpy.hstack([numpy.linspace(SD_MEMORY_MAX, SD_MEMORY_MIN, \
            SD_MEMORY_STEPS), numpy.ones(n_flights-SD_MEMORY_STEPS)*SD_MEMORY_MIN])
        p[ai]["continuity"] = copy.copy(P_CONTINUITY)
        sd[ai]["continuity"] = copy.copy(SD_CONTINUITY)

    # Create dictionaries to hold data from separate agents.
    ll = {}
    dll = {}
    ll_best = {}
    ll_history = {}
    p_best = {}
    sd_best = {}
    for ai in range(2):
        ll[ai] = numpy.NaN
        dll[ai] = numpy.NaN
        ll_best[ai] = -numpy.inf
        p_best[ai] = None
        sd_best[ai] = None
        
    # Run until our fit converges or crashes.
    w_history = []
    k_history = []
    min_dll = 10
    max_iter = 20
    iter_count = 0
    converged = False
    iter_limit = False
    nan_stuck = False
    while (not converged) and (not iter_limit) and (not nan_stuck):
        x = []
        y = []
        landmarks = []
        agents = []
        max_steps = 0
        for ai in range(2):
            # Add the paths for each pigeon.
            x.append(pigeon_x[ci,run,ai,:,:,:])
            y.append(pigeon_y[ci,run,ai,:,:,:])
            # Find the longest path.
            mx = numpy.nanmean(numpy.nanmean(x[ai], axis=0), axis=0)
            my = numpy.nanmean(numpy.nanmean(y[ai], axis=0), axis=0)
            # Find the end of the current path (==the last not-NaN).
            ei = numpy.where(numpy.isnan(mx))[0][0]
            if ei > max_steps:
                max_steps = ei
            # Create the indices within the average path of all landmarks.
            li = numpy.round(numpy.linspace(0, ei, N_LANDMARKS+2)).astype(int)
            # Skip the first and the last landmark, as these are the start and
            # the goal.
            li = li[1:-1]
            # Save the landmarks.
            landmarks.append({"x":mx[li], "y":my[li]})
            
            # Create an agent.
            agent = Agent( \
                starting_position=START_UTM, \
                goal_position=GOAL_UTM, \
                p_goal=p[ai]["goal"], \
                sd_goal=sd[ai]["goal"], \
                p_social=p[ai]["social"], \
                sd_social=sd[ai]["social"], \
                p_memory=p[ai]["memory"], \
                sd_memory=sd[ai]["memory"], \
                p_continuity=p[ai]["continuity"], \
                sd_continuity=sd[ai]["continuity"], \
                stepsize=stepsize, \
                goal_threshold=goal_threshold, \
                landmark_threshold=landmark_threshold, \
                n_landmarks=N_LANDMARKS, \
                landmarks=landmarks[ai], \
                alignment_distance_mean=stepsize*0.5, \
                alignment_distance_sd=stepsize*0.1)
            # Add agent to the list.
            agents.append(agent)
        
        # If the condition is solo or experimental, only retain the first
        # agent.
        if condition in ["solo", "experimental"]:
            agents.pop(1)
            ll.pop(1)
            dll.pop(1)
            ll_best.pop(1)
            p_best.pop(1)
            sd_best.pop(1)
            p.pop(1)
            sd.pop(1)
        
        # Make the agents follow the pigeon's paths.
        shape = (2, N_GENERATIONS, N_FLIGHTS_PER_GENERATION, max_steps)
        p_dist = numpy.zeros(shape, dtype=numpy.float64)*numpy.nan
        c_dist = { \
            "goal":numpy.zeros(shape, dtype=numpy.float64)*numpy.nan, \
            "social":numpy.zeros(shape, dtype=numpy.float64)*numpy.nan, \
            "memory":numpy.zeros(shape, dtype=numpy.float64)*numpy.nan, \
            "continuity":numpy.zeros(shape, dtype=numpy.float64)*numpy.nan, \
            }
        d_dist = { \
            "goal":numpy.zeros(shape, dtype=numpy.float64)*numpy.nan, \
            "social":numpy.zeros(shape, dtype=numpy.float64)*numpy.nan, \
            "memory":numpy.zeros(shape, dtype=numpy.float64)*numpy.nan, \
            "continuity":numpy.zeros(shape, dtype=numpy.float64)*numpy.nan, \
            }
        path_numbers = numpy.zeros(shape, dtype=numpy.int64) * numpy.nan
        
        flight_count = 0
        for gi in range(N_GENERATIONS):
            print("\tGeneration {}".format(gi+1))
            # In the experimental condition, eliminate the most experienced
            # agent if there is a pair. Also add a naive individual after the
            # very first generation.
            if condition == "experimental":
                if len(agents) >= 2:
                    agents.pop(0)
                    ll.pop(0)
                    dll.pop(0)
                    p.pop(0)
                    sd.pop(0)
                    ll_best.pop(0)
                    fit_history[condition]["{}_0".format(gi-1)] = { \
                        "p_best": p_best.pop(0), \
                        "sd_best": sd_best.pop(0), \
                        }
            if (condition == "experimental") & (gi > 0):
                # For landmarks, only use the first non-NaN path.
                mx = [numpy.nan]
                fi = 0
                while numpy.sum(numpy.invert(numpy.isnan(mx))) == 0:
                    mx = pigeon_x[ci,run,1,gi,fi,:]
                    my = pigeon_y[ci,run,1,gi,fi,:]
                    fi += 1
                    if fi >= N_FLIGHTS_PER_GENERATION:
                        break
                if fi >= N_FLIGHTS_PER_GENERATION:
                    break
                ei = numpy.where(numpy.isnan(mx))[0][0]
                # Create the indices within the average path of all landmarks.
                li = numpy.round(numpy.linspace(0, ei, N_LANDMARKS+2)).astype(int)
                # Skip the first and the last landmark, as these are the start and
                # the goal.
                li = li[1:-1]
                # Save the landmarks.
                landmarks.append({"x":mx[li], "y":my[li]})
                
                # Create an agent.
                agent = Agent( \
                    starting_position=START_UTM, \
                    goal_position=GOAL_UTM, \
                    p_goal=p[ai]["goal"], \
                    sd_goal=sd[ai]["goal"], \
                    p_social=p[ai]["social"], \
                    sd_social=sd[ai]["social"], \
                    p_memory=p[ai]["memory"], \
                    sd_memory=sd[ai]["memory"], \
                    p_continuity=p[ai]["continuity"], \
                    sd_continuity=sd[ai]["continuity"], \
                    stepsize=stepsize, \
                    goal_threshold=goal_threshold, \
                    landmark_threshold=landmark_threshold, \
                    n_landmarks=N_LANDMARKS, \
                    landmarks=landmarks[ai], \
                    alignment_distance_mean=30.0, \
                    alignment_distance_sd=10.0)
                # Add agent to the list.
                agents.append(agent)

            n_agents = len(agents)
            for fi in range(N_FLIGHTS_PER_GENERATION):
                print("\t\tFlight {}".format(fi+1))
                x_ = pigeon_x[ci,run,:,gi,fi,:]
                y_ = pigeon_y[ci,run,:,gi,fi,:]
                # Skip all-NaN flights.
                if numpy.sum(numpy.invert(numpy.isnan(x_[0,:]) \
                    | numpy.isnan(y_[0,:]))) == 0:
                    continue
                if n_agents > 1:
                    if numpy.sum(numpy.invert(numpy.isnan(x_[1,:]) \
                        | numpy.isnan(y_[1,:]))) == 0:
                        continue
                # Update the path count number.
                path_numbers[:,gi,fi,:] = flight_count
                # Release the agents, and move them to their first position.
                for ai, agent in enumerate(agents):
                    agent.release()
                    agent.advance_position( \
                        force_position=(x_[ai,0], y_[ai,0]))
                    agent.advance_position( \
                        force_position=(x_[ai,1], y_[ai,1]))
                # Loop until all agents are finished.
                i = 2
                finished = numpy.zeros(n_agents, dtype=bool)
                while numpy.sum(finished) < n_agents:
                    
                    # Determine the expected positions and current headings of 
                    # all agents in paired flights.
                    if len(agents) == 1:
                        expected_pos = [None, None]
                        expected_heading = [None, None]
                    else:
                        expected_pos = []
                        expected_heading = []
                        for ai, agent in enumerate(agents):
                            expected_pos.append(agent.expected_next_position())
                            expected_heading.append(agent.get_heading())
    
                    # Loop through all agents.
                    for ai, agent in enumerate(agents):
                        # If the next position is a NaN, move the agent to the
                        # goal instead.
                        if numpy.isnan(x_[ai,i]) or numpy.isnan(y_[ai,i]):
                            next_pos = GOAL_UTM
                        else:
                            next_pos = (x_[ai,i], y_[ai,i])
                        # Get the agents' distributions.
                        pdf, components, d = agent.get_current_distributions( \
                            other_position=expected_pos[1-ai], \
                            other_heading=expected_heading[1-ai], \
                            heading_centre=True)
                        # Move the agent.
                        pos, finished[ai] = agent.advance_position( \
                            force_position=next_pos)
                        # Save the outcomes.
                        p_dist[ai,gi,fi,i] = pdf
                        for component in components.keys():
                            c_dist[component][ai,gi,fi,i] = components[component]
                            d_dist[component][ai,gi,fi,i] = d[component]
                    
                    # Increment the counter.
                    i += 1
                # Increment the flight counter.
                flight_count += 1

                # Enforce the landmarks. (This should have gone well during the
                # first generation, but it's good to make sure.)
                for ai, agent in enumerate(agents):
                    agent._preferred_path_finished = True
                    agent._preferred_path = landmarks[ai]


        for ai, agent in enumerate(agents):
            # Flatten the outcomes, and compute the new kappa and weight values.
            kappa = {}
            w = {}
            p_dist_flat = p_dist[ai,:,:,:].flatten()
            path_numbers_flat = path_numbers[ai,:,:,:].flatten()
            notnan = numpy.invert(numpy.isnan(p_dist_flat))
            n = numpy.sum(notnan)
            if n == 0:
                print("Oh no, all NaNs!")
                nan_stuck = True
                break
            for component in c_dist.keys():
                c_dist_flat = c_dist[component][ai,:,:,:].flatten()
                d_dist_flat = d_dist[component][ai,:,:,:].flatten()
                include = notnan & numpy.invert(numpy.isnan(c_dist_flat)) \
                    & numpy.invert(numpy.isnan(d_dist_flat))
                w[component] = numpy.round(numpy.sum(c_dist_flat[include] \
                    / p_dist_flat[include]) / n, 3)
                if numpy.sum(include) == 0:
                    continue
                if component == "memory":
                    kappa[component] = numpy.zeros(n_flights, dtype=numpy.float64)
                    for mi in range(n_flights):
                        if mi == 0:
                            kappa[component][mi] = k_lim[0]
                            continue
                        sel = include & (path_numbers_flat == mi)
                        if numpy.sum(sel) > 1:
                            kappa[component][mi] = update_kappa( \
                                p_dist_flat[sel], \
                                c_dist_flat[sel], \
                                d_dist_flat[sel])
                        else:
                            kappa[component][mi] = numpy.nan
                else:
                    kappa[component] = update_kappa(p_dist_flat[include], \
                        c_dist_flat[include], d_dist_flat[include])
    
            # Keep a record of recent parameters.
            w_history.append(w)
            k_history.append(kappa)
    
            # Compute the new log likelihood.
            nll = numpy.sum(numpy.log(p_dist_flat[notnan]))
            if numpy.isnan(nll):
                print("Log likelihood is NaN!")
                nan_stuck = True
                break
            # Calculate the difference with the previous log likelihood.
            dll[ai] = nll - ll[ai]
            # Calculate the difference with any previous log likelihood.
            if len(ll_history[ai]) > 2:
                dll_any = numpy.min(numpy.abs(nll \
                    - numpy.array(ll_history[ai])))
            else:
                dll_any = numpy.inf
            # Store the new LL.
            ll[ai] = copy.deepcopy(nll)
            ll_history[ai].append(ll[ai])
            
            # Recompute standard deviations, and store them and the new weights.
            for component in kappa.keys():
                # Update the weights.
                if not numpy.isnan(w[component]):
                    p[ai][component] = w[component]
                # Update the precision parameters.
                if component == "memory":
                    notnan = numpy.invert(numpy.isnan(kappa[component]))
                    sd[ai][component][notnan] = kappa2sd(kappa[component][notnan])
                else:
                    if not numpy.isnan(kappa[component]):
                        sd[component] = kappa2sd(kappa[component])
                print("\t{}: p={}, sd={}, k={}".format(component, \
                    numpy.round(p[component],3), numpy.round(sd[component],3), \
                    numpy.round(kappa[component],3)))
    
            # Check if this is the best fit so far, and store parameters if it is.
            if ll[ai] > ll_best[ai]:
                p_best[ai] = copy.deepcopy(p[ai])
                sd_best[ai] = copy.deepcopy(sd[ai])
                ll_best[ai] = copy.deepcopy(ll[ai])
    
            # Stop the cycle if we are no longer making much progress in log 
            # likelihood improvement, or if we hit the maximum number of 
            # iterations.
            # if numpy.abs(dll) < min_dll:
            if dll_any < min_dll:
                converged[ai] = True
                break
            iter_count += 1
            if (iter_count > max_iter) or numpy.isnan(ll[ai]):
                iter_limit = True
                break

    fitted_p[condition].append(copy.deepcopy(p_best[ai]))
    fitted_sd[condition].append(copy.deepcopy(sd_best[ai]))

# Write best fits to a CSV.
fpath = os.path.join(OUTDIR, "pigeon_fits.csv")
with open(fpath, "w") as f:
    header = ["condition", "run", "agent", "p_goal", "sd_goal", "p_social", \
        "sd_social", "p_continuity", "sd_continuity", "p_memory"]
    header = header + ["sd_memory_{}".format(i+1) for i in range(n_flights)]
    f.write(",".join(header))
    for condition in fitted_p.keys():
        for run in range(len(fitted_p[condition])):
            if fitted_p[condition][run] is None:
                continue
            line = [condition, run+1, ai+1, \
                fitted_p[condition][run]["goal"], \
                fitted_sd[condition][run]["goal"], \
                fitted_p[condition][run]["social"], \
                fitted_sd[condition][run]["social"], \
                fitted_p[condition][run]["continuity"], \
                fitted_sd[condition][run]["continuity"], \
                fitted_p[condition][run]["memory"]]
            line = "\n" + ",".join(map(str, line))
            line = line + "," \
                + ",".join(map(str, fitted_sd[condition][run]["memory"]))
            f.write(line)

