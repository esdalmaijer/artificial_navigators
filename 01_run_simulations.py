#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from multiprocessing import cpu_count, Process
import time

import numpy
from simulations import run_simulations

from simulation_settings import parameter_range, simulation_types, \
    n_repetitions, n_generations, n_flights_per_generation, \
    p_goal_range, p_memory_range, p_social_range, p_continuity_range_used, \
    sd_goal_range, sd_continuity, sd_social, \
    sd_memory_max_range, sd_memory_min, sd_memory_steps, \
    start_pos, goal_pos, n_landmarks, start_heading, \
    stepsize, max_steps, goal_threshold, landmark_threshold, landmarks

# Skip simulations that already have existing folders?
skip_existing = False

# Set the directory to save data in.
this_directory = os.path.dirname(os.path.abspath(__file__))
data_directory = os.path.join(this_directory, "data_simulation_{}".format( \
    parameter_range))
if not os.path.isdir(data_directory):
    os.mkdir(data_directory)

# Create a list of all possible parameter combinations.
all_parameter_combinations = []
for p_goal in p_goal_range:
    for p_memory in p_memory_range:
        for p_social in p_social_range:
            p_continuity = 1.0 - (p_goal + p_memory + p_social)
            p_total = round(p_goal + p_social + p_memory + p_continuity, 3)
            if (p_continuity < 0) | (p_continuity > 1):
                continue
            if p_total != 1:
                continue
            for sd_goal in sd_goal_range:
                for sd_memory_max in sd_memory_max_range:
                    sd_memory = (sd_memory_max, sd_memory_min, sd_memory_steps)
                    
                    all_parameter_combinations.append({ \
                        "p_goal":           p_goal, \
                        "p_continuity":     p_continuity, \
                        "p_social":         p_social, \
                        "p_memory":         p_memory, \
                        "sd_goal":          sd_goal, \
                        "sd_continuity":    sd_continuity, \
                        "sd_social":        sd_social, \
                        "sd_memory":        sd_memory, \
                        })

# Count the number of parameter combinations.
n_combinations = len(all_parameter_combinations)

# Count the number of CPU cores.
n_cpu = cpu_count()

print("Running {} simulations on {} CPUs".format(n_combinations, n_cpu))

# Run through the parameter combinations, and spawn a new Process for each.
# Up to the number of CPUs minus one are allowed to run simultaneously.
current_processes = []
for i, params in enumerate(all_parameter_combinations):
    
    # Convenience renaming of the parameters.
    p_goal = params["p_goal"]
    p_continuity = params["p_continuity"]
    p_social = params["p_social"]
    p_memory = params["p_memory"]
    sd_goal = params["sd_goal"]
    sd_continuity = params["sd_continuity"]
    sd_social = params["sd_social"]
    sd_memory = params["sd_memory"]
    
    # Create a name for the directory.
    save_dir = os.path.join(data_directory, \
        "Pgoal-{}_SDgoal-{}_Pcontinuity-{}_SDcontinuity-{}_" \
        + "Psocial-{}_SDsocial-{}_Pmemory-{}_SDmemoryMax-{}_" \
        + "SDmemoryMin-{}_SDmemorySteps-{}")
    save_dir = save_dir.format( \
        round(p_goal*1000), round(sd_goal*1000), \
        round(p_continuity*1000), round(sd_continuity*1000), \
        round(p_social*1000), round(sd_social*1000), \
        round(p_memory*1000), round(sd_memory[0]*1000), \
        round(sd_memory[1]*1000), round(sd_memory[2]), \
        )
    
    # Skip if this folder already exists.
    if skip_existing and os.path.isdir(save_dir):
        print("Skipping sim_{}/{}; output folder already exists".format( \
            i+1, n_combinations))
        continue
    
    # Start a new Process.
    p = Process(target=run_simulations, args=(simulation_types, \
        p_goal, p_continuity, p_social, p_memory, \
        sd_goal, sd_continuity, sd_social, sd_memory, \
        n_repetitions, n_generations, n_flights_per_generation, \
        start_pos, goal_pos, start_heading, stepsize, \
        goal_threshold, landmark_threshold, n_landmarks, landmarks, \
        max_steps, save_dir))
    p.name = "sim_{}/{}".format(i+1, n_combinations)
    p.daemon = True
    p.start()
    
    # Add the process to the list.
    current_processes.append(p)
    
    # Wait until we have fewer than n_cpu-1 processes running.
    while len(current_processes) >= n_cpu-1:
        for p in current_processes:
            if not p.is_alive():
                print("Joining process {}...".format(p.name))
                p.join()
                current_processes.remove(p)
                print("\tProcess {} has finished!".format(p.name))
        time.sleep(2.0)
    
# Wait for all processes to finish.
while len(current_processes) > 0:
    for p in current_processes:
        if not p.is_alive():
            print("Joining process {}...".format(p.name))
            p.join()
            current_processes.remove(p)
            print("\tProcess {} has finished!".format(p.name))
    time.sleep(2.0)
