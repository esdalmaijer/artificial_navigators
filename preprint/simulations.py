#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy
from agents import Agent


def run_simulations(simulation_types, \
    p_goal, p_continuity, p_social, p_memory,
    sd_goal, sd_continuity, sd_social, sd_memory,
    n_repetitions, n_generations, n_flights_per_generation, \
    start_pos, goal_pos, start_heading, stepsize, goal_threshold, \
    landmark_threshold, n_landmarks, max_steps, data_directory, \
    overwrite=False):
    
    # Double-check simulation types.
    for simulation_type in simulation_types:
        if simulation_type not in ["experimental", "solo", "pair"]:
            raise Exception("Simulation type '{}' not recognised.")
    
    # Create a new directory if necessary.
    if not os.path.isdir(data_directory):
        os.mkdir(data_directory)

    # Run all the simulations.
    for simulation_type in simulation_types:
        for run in range(n_repetitions):
            # Start with the basic folder name.
            save_dir = os.path.join(data_directory, "{}_run-{}".format( \
                simulation_type, run+1))
            # Increment the run number by 1 until we found one that doesn't
            # exist yet.
            if not overwrite:
                i = 0
                while os.path.isdir(save_dir):
                    save_dir = os.path.join(data_directory, \
                        "{}_run-{}".format(simulation_type, run+1+i))
                    i += 1
            # Create the new directory.
            if not os.path.isdir(save_dir):
                os.mkdir(save_dir)
            # Simulate the flights.
            run_simulation(simulation_type, \
                p_goal, p_continuity, p_social, p_memory,
                sd_goal, sd_continuity, sd_social, sd_memory,
                n_generations, n_flights_per_generation, \
                start_pos, goal_pos, start_heading, stepsize, goal_threshold, \
                landmark_threshold, n_landmarks, max_steps, save_dir)


def run_simulation(simulation_type, \
    p_goal, p_continuity, p_social, p_memory,
    sd_goal, sd_continuity, sd_social, sd_memory,
    n_generations, n_flights_per_generation, \
    start_pos, goal_pos, start_heading, stepsize, goal_threshold, \
    landmark_threshold, n_landmarks, max_steps, save_dir):
    
    # Double-check simulation types.
    if simulation_type not in ["experimental", "solo", "pair"]:
        raise Exception("Simulation type '{}' not recognised.")
    
    # Seed the default RNG.
    numpy.random.seed()
    
    # Set the number of agents based on the simulation type.
    if simulation_type == "solo":
        n_agents = 1
    else:
        n_agents = 2

    # Compute distance between start and goal.
    start_goal_distance = numpy.sqrt((start_pos[0]-goal_pos[0])**2 \
        + (start_pos[1]-goal_pos[1])**2)

    # Create NumPy arrays that will contain all the flight paths.
    var_shape = (2, n_generations, n_flights_per_generation)
    path_shape = (2, n_generations, n_flights_per_generation, max_steps)
    x = numpy.zeros(path_shape, dtype=numpy.float64) * numpy.NaN
    y = numpy.zeros(path_shape, dtype=numpy.float64) * numpy.NaN
    efficiency = numpy.zeros(var_shape, dtype=numpy.float64) * numpy.NaN
    
    # Create a group of agents.
    agents = []
    for ai in range(n_agents):
        agents.append(Agent( \
            starting_heading=start_heading, \
            starting_position=start_pos, \
            goal_position=goal_pos, \
            p_goal=p_goal, \
            sd_goal=sd_goal, \
            p_social=p_social, \
            sd_social=sd_social, \
            p_continuity=p_continuity, \
            sd_continuity=sd_continuity, \
            p_memory=p_memory, \
            sd_memory=sd_memory, \
            stepsize=stepsize, \
            goal_threshold=goal_threshold, \
            landmark_threshold=goal_threshold, \
            n_landmarks=n_landmarks, \
            prefer_first_path=True, \
            ))
    
    # Run through all generations.
    for j in range(n_generations):

        # In the "experimental" group, the oldest agent is replaced by a
        # naive agent. Also, in the very first experimental run, agents
        # travel by themselves.
        if simulation_type == "experimental":
            if j == 0:
                agents.pop(1)
            else:
                while len(agents) >= n_agents:
                    agents.pop(0)
                agents.append(Agent( \
                    starting_heading=start_heading, \
                    starting_position=start_pos, \
                    goal_position=goal_pos, \
                    p_goal=p_goal, \
                    sd_goal=sd_goal, \
                    p_social=p_social, \
                    sd_social=sd_social, \
                    p_continuity=p_continuity, \
                    sd_continuity=sd_continuity, \
                    p_memory=p_memory, \
                    sd_memory=sd_memory, \
                    stepsize=stepsize, \
                    goal_threshold=goal_threshold, \
                    landmark_threshold=goal_threshold, \
                    n_landmarks=n_landmarks, \
                    prefer_first_path=True, \
                    ))
        
        # Make the journey.
        for i in range(0, n_flights_per_generation):
            
            # Count the number of agents in this flight.
            n_agents_this_flight = len(agents)
            
            # RELEASE THE HOUNDS!!
            for ai, agent in enumerate(agents):
                agent.release()
                x[ai,j,i,0], y[ai,j,i,0] = agent.get_position()
        
            # Make the journey.
            finished = numpy.zeros(n_agents_this_flight, dtype=bool)
            for step in range(1, max_steps):
                
                # Determine the expected positions of all agents in paired
                # flights.
                if len(agents) == 1:
                    expected_pos = [None, None]
                else:
                    expected_pos = []
                    for ai, agent in enumerate(agents):
                        expected_pos.append(agent.expected_next_position())

                # Move the agents forward, and save their steps.
                for ai, agent in enumerate(agents):
                    pos, finished[ai] = agent.advance_position( \
                        other_position=expected_pos[1-ai])
                    x[ai,j,i,step], y[ai,j,i,step] = pos
                
                # Check if we're there yet.
                if numpy.sum(finished) == n_agents_this_flight:
                    break
            
            # Replace unfinished routes with NaN values.
            if numpy.sum(finished) != n_agents_this_flight:
                # Save the path as NaNs
                x[:,j,i,:] = numpy.NaN
                y[:,j,i,:] = numpy.NaN
                # Reset the agents.
                for ai, agent in enumerate(agents):
                    agent.reset()
                # Skip to the next iteration.
                continue

            # Compute route efficiency. "Route efficiency was calculated
            # by dividing the direct straight-line distance from the 
            # release point to home  by the actual distance flown." 
            # (Sasaki & Biro, 2017)
            for ai in range(n_agents_this_flight):
                notnan = numpy.invert(numpy.isnan(x[ai,j,i,:]) | \
                    numpy.isnan(y[ai,j,i,:]))
                d = numpy.nansum(numpy.sqrt( \
                    numpy.diff(x[ai,j,i,notnan])**2 \
                    + numpy.diff(y[ai,j,i,notnan])**2))
                efficiency[ai,j,i] = start_goal_distance / d
    
    # Write efficiency to files.
    for j in range(n_generations):
        fpath = os.path.join(save_dir, "efficiency_gen-{}.csv".format(j))
        with open(fpath, "w") as f:
            header = ["flight", "efficiency_agent1", "efficiency_agent2"]
            f.write(",".join(header))
            for i in range(n_flights_per_generation):
                f.write("\n{},{},{}".format(i+1, efficiency[0,j,i], \
                    efficiency[1,j,i]))

    # Write paths to files.
    for j in range(n_generations):
        for i in range(n_flights_per_generation):
            all_nan = numpy.isnan(x[0,j,i]) & numpy.isnan(y[0,j,i]) \
                & numpy.isnan(x[1,j,i]) & numpy.isnan(y[1,j,i])
            nan_indices = numpy.where(all_nan)
            if nan_indices[0].shape[0] == 0:
                stopline = max_steps
            else:
                stopline = numpy.min(nan_indices)
            fpath = os.path.join(save_dir, \
                "xy_gen-{}_flight-{}.csv".format(j,i))
            with open(fpath, "w") as f:
                header = ["x_agent1", "y_agent1", "x_agent2", "y_agent2"]
                f.write(",".join(header))
                for line in range(max_steps):
                    if line == stopline:
                        break
                    f.write("\n{},{},{},{}".format(x[0,j,i,line], \
                        y[0,j,i,line], x[1,j,i,line], y[1,j,i,line]))

