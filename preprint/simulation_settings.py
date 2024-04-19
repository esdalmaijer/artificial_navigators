#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy

# Deteminde the parameter range.
parameter_range = "wide"

# Simulation settings.
simulation_types = ["experimental", "solo", "pair"]
n_repetitions = 10
n_generations = 5
n_flights_per_generation = 12

# Simulation parameter ranges.
if parameter_range == "wide":
    p_goal_range = numpy.arange(0.1, 0.75, 0.05)
    p_goal_range = list(numpy.round(p_goal_range, 3))
    p_continuity_range = numpy.arange(0.1, 0.75, 0.05)
    p_continuity_range = list(numpy.round(p_continuity_range, 3))
    p_social_range = numpy.arange(0.1, 0.75, 0.05)
    p_social_range = list(numpy.round(p_social_range, 3))
    p_memory_range = [0.04, 0.96]
elif parameter_range == "narrow":
    p_goal_range = numpy.arange(0.025, 0.26, 0.025)
    p_goal_range = list(numpy.round(p_goal_range, 3))
    p_continuity_range = numpy.arange(0.0, 1.01, 0.025)
    p_continuity_range = list(numpy.round(p_continuity_range, 3))
    p_social_range = numpy.arange(0.025, 0.26, 0.025)
    p_social_range = list(numpy.round(p_social_range, 3))
    p_memory_range = [0.01, 0.71]
else:
    raise Exception("Parameter range '{}' is not a valid input!".format( \
        parameter_range))

sd_goal_range = [1.0]
sd_continuity = 0.35
sd_social = 0.8
sd_memory_max_range = [2.0]
sd_memory_min = 0.4
sd_memory_steps = 5

# Compute the range of P(memory) parameter values that will be used, and count
# the number of simulations that the current settings will result in.
n_simulations = 0
p_memory_range_used = []
for p_goal in p_goal_range:
    for p_social in p_social_range:
        for p_continuity in p_continuity_range:
            p_memory = round(1.0 - (p_goal + p_social + p_continuity), 3)
            if (p_memory >= p_memory_range[0]) & (p_memory <= p_memory_range[1]):
                n_simulations += 1
                if p_memory not in p_memory_range_used:
                    p_memory_range_used.append(p_memory)
p_memory_range_used.sort()
n_simulations = n_simulations * len(sd_goal_range) * len(sd_memory_max_range)

# Path settings.
start_pos = (618164.67721362, 5746481.984207394)
goal_pos = (616070.1464758714, 5738228.319967276)
n_landmarks = 10

start_heading = numpy.arctan2(goal_pos[0]-start_pos[0], \
    goal_pos[1]-start_pos[1])

stepsize = 70.0
max_steps = int(numpy.ceil(((abs(start_pos[0]-goal_pos[0]) \
    + abs(start_pos[1]-goal_pos[1]))) / stepsize)) * 100

goal_threshold = 4.0 * stepsize
landmark_threshold = 10.0 * stepsize

